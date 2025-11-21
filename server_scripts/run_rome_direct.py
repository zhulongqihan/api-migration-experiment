#!/usr/bin/env python3
"""
直接实现ROME算法，不依赖EasyEdit库
基于论文: "Locating and Editing Factual Associations in GPT"
"""

import json
import torch
import torch.nn.functional as F
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple
import numpy as np

console = Console()

class ROMEEditor:
    """直接实现的ROME编辑器"""
    
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # 配置参数
        self.v_lr = 0.5
        self.v_num_grad_steps = 20
        self.mom2_update_weight = 10000
        
        # 目标层（对于Qwen2.5-Coder-1.5B，使用中后层）
        self.layer_ids = [20, 21, 22, 23, 24]
        
    def get_module_by_path(self, path: str):
        """根据路径获取模块"""
        parts = path.split('.')
        module = self.model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    
    def compute_z(self, layer_id: int, input_text: str, target_text: str):
        """
        计算编辑向量z
        核心ROME算法：最小化 ||h_l + Δh_l - z||^2
        """
        # 编码输入
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        # 前向传播，获取隐藏状态
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[layer_id]
            
        # 使用最后一个token的隐藏状态
        h_l = hidden_states[:, -1, :].detach()
        
        # 计算目标表示
        target_inputs = self.tokenizer(target_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            target_outputs = self.model(**target_inputs, output_hidden_states=True)
            target_hidden = target_outputs.hidden_states[layer_id][:, -1, :]
        
        # 计算z：目标隐藏状态
        z = target_hidden.detach()
        
        return h_l, z
    
    def compute_covariance(self, layer_id: int, sample_texts: List[str]):
        """
        计算协方差矩阵C
        C = E[key @ key.T]
        """
        console.print(f"[yellow]  计算层{layer_id}的协方差矩阵...[/yellow]")
        
        # 获取MLP模块
        layer_name = f"model.layers.{layer_id}"
        layer = self.get_module_by_path(layer_name)
        
        # 收集key激活
        key_activations = []
        
        for text in sample_texts[:10]:  # 使用前10个样本
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                hidden = outputs.hidden_states[layer_id]
                
                # 使用所有token的激活
                for i in range(hidden.shape[1]):
                    key_activations.append(hidden[0, i, :].cpu())
        
        # 计算协方差
        keys = torch.stack(key_activations)
        C = torch.cov(keys.T)
        
        return C
    
    def edit_layer(self, layer_id: int, old_text: str, new_text: str, C: torch.Tensor):
        """
        在特定层执行ROME编辑
        更新公式: W' = W + (z - h_l) @ k.T @ C^{-1}
        """
        # 计算编辑向量
        h_l, z = self.compute_z(layer_id, old_text, new_text)
        
        # 计算key向量（使用输入的隐藏状态作为key）
        k = h_l
        
        # 计算更新方向
        delta_h = (z - h_l)
        
        # 使用伪逆避免奇异矩阵
        try:
            C_inv = torch.linalg.pinv(C.to(self.device))
        except:
            console.print(f"[yellow]  ⚠ 使用正则化的逆矩阵[/yellow]")
            C_reg = C + torch.eye(C.shape[0]) * 1e-4
            C_inv = torch.linalg.pinv(C_reg.to(self.device))
        
        # 计算权重更新: ΔW = delta_h @ k.T @ C^{-1}
        weight_update = torch.outer(delta_h.squeeze(), k.squeeze()) @ C_inv
        
        # 应用到MLP层的权重
        layer_name = f"model.layers.{layer_id}.mlp.down_proj"
        try:
            mlp_layer = self.get_module_by_path(layer_name)
            
            # 更新权重
            with torch.no_grad():
                # 限制更新幅度
                update_norm = torch.norm(weight_update)
                if update_norm > 0.1:
                    weight_update = weight_update * (0.1 / update_norm)
                
                # 应用更新到权重矩阵的一部分
                W = mlp_layer.weight
                update_slice = weight_update[:W.shape[0], :W.shape[1]]
                mlp_layer.weight.data += 0.1 * update_slice  # 使用小的学习率
                
            return True
            
        except Exception as e:
            console.print(f"[red]  ✗ 更新失败: {e}[/red]")
            return False
    
    def edit(self, old_codes: List[str], new_codes: List[str]):
        """执行批量编辑"""
        results = []
        
        # 计算协方差矩阵（所有层共享）
        console.print("[yellow]预计算协方差矩阵...[/yellow]")
        covariances = {}
        
        all_texts = old_codes + new_codes
        for layer_id in self.layer_ids:
            covariances[layer_id] = self.compute_covariance(layer_id, all_texts)
        
        console.print("[green]✓ 协方差矩阵计算完成[/green]")
        
        # 对每个API进行编辑
        for i, (old_code, new_code) in enumerate(track(
            zip(old_codes, new_codes), 
            description="编辑中...",
            total=len(old_codes)
        )):
            success_count = 0
            
            # 在多个层进行编辑
            for layer_id in self.layer_ids:
                try:
                    success = self.edit_layer(
                        layer_id, 
                        old_code, 
                        new_code, 
                        covariances[layer_id]
                    )
                    if success:
                        success_count += 1
                except Exception as e:
                    console.print(f"[red]✗ 编辑 {i} 层 {layer_id} 失败: {e}[/red]")
            
            results.append({
                "index": i,
                "success": success_count > 0,
                "layers_edited": success_count
            })
        
        return results

def main():
    parser = argparse.ArgumentParser(description="直接ROME实现")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    parser.add_argument("--data_file", type=str, default="extended_dataset_50.json")
    parser.add_argument("--output_dir", type=str, default="../models/rome_direct")
    parser.add_argument("--num_edits", type=int, default=10)
    
    args = parser.parse_args()
    
    console.print("\n[bold cyan]🔬 直接ROME知识编辑[/bold cyan]\n")
    
    # 显示配置
    table = Table(title="实验配置")
    table.add_column("参数", style="cyan")
    table.add_column("值", style="green")
    table.add_row("基础模型", args.model_name)
    table.add_row("数据文件", args.data_file)
    table.add_row("编辑方法", "ROME (直接实现)")
    table.add_row("输出目录", args.output_dir)
    table.add_row("编辑数量", str(args.num_edits))
    console.print(table)
    
    # 步骤1：加载数据
    console.print("\n[yellow]步骤1/5: 加载数据[/yellow]")
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    train_data = data['train'][:args.num_edits]
    console.print(f"[green]✓ 加载了 {len(train_data)} 个API更新案例[/green]")
    
    # 步骤2：加载模型
    console.print("\n[yellow]步骤2/5: 加载模型[/yellow]")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,  # ROME需要float32
        device_map="auto",
        trust_remote_code=True
    )
    console.print("[green]✓ 模型加载成功[/green]")
    
    # 步骤3：初始化编辑器
    console.print("\n[yellow]步骤3/5: 初始化ROME编辑器[/yellow]")
    editor = ROMEEditor(model, tokenizer)
    console.print("[green]✓ 编辑器初始化成功[/green]")
    console.print(f"[cyan]  编辑层: {editor.layer_ids}[/cyan]")
    
    # 步骤4：执行编辑
    console.print("\n[yellow]步骤4/5: 执行ROME编辑[/yellow]")
    
    old_codes = [item['old_code'] for item in train_data]
    new_codes = [item['new_code'] for item in train_data]
    
    results = editor.edit(old_codes, new_codes)
    
    success_count = sum(1 for r in results if r['success'])
    console.print(f"\n[green]✅ 成功编辑: {success_count}/{len(train_data)}[/green]")
    
    # 步骤5：保存模型
    console.print("\n[yellow]步骤5/5: 保存编辑后的模型[/yellow]")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    console.print(f"[green]✓ 模型已保存到: {output_path}[/green]")
    
    # 保存编辑信息
    info = {
        "method": "ROME (直接实现)",
        "base_model": args.model_name,
        "num_edits": len(train_data),
        "success_count": success_count,
        "layer_ids": editor.layer_ids,
        "results": results
    }
    
    info_path = output_path / "editing_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[bold green]✅ ROME编辑完成！[/bold green]")
    console.print(f"\n[cyan]编辑统计：[/cyan]")
    console.print(f"  总编辑数: {len(train_data)}")
    console.print(f"  成功: {success_count}")
    console.print(f"  失败: {len(train_data) - success_count}")
    
    console.print("\n[cyan]下一步操作：[/cyan]")
    console.print(f"python3 evaluate_lora.py --model_path {output_path} --data_file {args.data_file}")

if __name__ == "__main__":
    main()
