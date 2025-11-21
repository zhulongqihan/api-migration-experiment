#!/usr/bin/env python3
"""
使用ROME (Rank-One Model Editing) 进行API知识编辑
更简单、更直接的实现
"""

import json
import torch
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track
import argparse

console = Console()

class SimpleROME:
    """简化的ROME实现，适用于代码API更新"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def locate_api_knowledge(self, old_api, new_api):
        """定位需要编辑的模型层"""
        # 简化实现：找到深层transformer层（通常知识存储在这里）
        # 对于Qwen2.5-Coder-1.5B，使用第20-25层
        target_layers = list(range(20, 26))
        return target_layers
    
    def compute_edit_vector(self, old_code, new_code):
        """计算编辑向量"""
        # 编码旧代码和新代码
        old_inputs = self.tokenizer(old_code, return_tensors="pt").to(self.device)
        new_inputs = self.tokenizer(new_code, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            old_outputs = self.model(**old_inputs, output_hidden_states=True)
            new_outputs = self.model(**new_inputs, output_hidden_states=True)
        
        # 计算深层hidden states的差异
        layer_idx = 24  # 使用倒数第5层
        old_hidden = old_outputs.hidden_states[layer_idx].mean(dim=1)
        new_hidden = new_outputs.hidden_states[layer_idx].mean(dim=1)
        
        edit_vector = new_hidden - old_hidden
        return edit_vector, layer_idx
    
    def apply_edit(self, edit_vector, layer_idx, strength=0.5):
        """应用编辑到模型"""
        # 获取目标层
        target_layer = self.model.model.layers[layer_idx]
        
        # 对MLP层进行秩1更新
        mlp = target_layer.mlp
        
        # 更新权重（简化版）
        with torch.no_grad():
            # 只更新输出投影
            if hasattr(mlp, 'down_proj'):
                weight = mlp.down_proj.weight
                # 秩1更新: W_new = W + strength * v * v^T
                update = strength * edit_vector.T @ edit_vector
                # 限制更新幅度
                update = torch.clamp(update, -0.01, 0.01)
                mlp.down_proj.weight.add_(update[:weight.shape[0], :weight.shape[1]])
        
        return True

def main():
    parser = argparse.ArgumentParser(description="使用简化ROME进行API知识编辑")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    parser.add_argument("--data_file", type=str, default="extended_dataset_50.json")
    parser.add_argument("--output_dir", type=str, default="../models/rome_edited")
    parser.add_argument("--num_edits", type=int, default=10, help="编辑的API数量")
    parser.add_argument("--strength", type=float, default=0.5, help="编辑强度")
    
    args = parser.parse_args()
    
    console.print("\n[bold cyan]🔬 ROME知识编辑实验[/bold cyan]\n")
    
    # 显示配置
    table = Table(title="实验配置")
    table.add_column("参数", style="cyan")
    table.add_column("值", style="green")
    table.add_row("基础模型", args.model_name)
    table.add_row("数据文件", args.data_file)
    table.add_row("编辑方法", "ROME (简化版)")
    table.add_row("输出目录", args.output_dir)
    table.add_row("编辑数量", str(args.num_edits))
    table.add_row("编辑强度", str(args.strength))
    console.print(table)
    
    # 步骤1：加载数据
    console.print("\n[yellow]步骤1/5: 加载数据[/yellow]")
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    train_data = data['train'][:args.num_edits]
    console.print(f"[green]✓ 加载了 {len(train_data)} 个API更新案例[/green]")
    
    # 步骤2：加载模型
    console.print("\n[yellow]步骤2/5: 加载模型[/yellow]")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,  # ROME需要float32
        device_map="auto",
        trust_remote_code=True
    )
    console.print("[green]✓ 模型加载成功[/green]")
    
    # 步骤3：初始化ROME编辑器
    console.print("\n[yellow]步骤3/5: 初始化ROME编辑器[/yellow]")
    editor = SimpleROME(model, tokenizer)
    console.print("[green]✓ 编辑器初始化成功[/green]")
    
    # 步骤4：执行编辑
    console.print("\n[yellow]步骤4/5: 执行知识编辑[/yellow]")
    
    edit_results = []
    for i, item in enumerate(track(train_data, description="编辑中...")):
        try:
            # 计算编辑向量
            edit_vector, layer_idx = editor.compute_edit_vector(
                item['old_code'], 
                item['new_code']
            )
            
            # 应用编辑
            success = editor.apply_edit(edit_vector, layer_idx, args.strength)
            
            edit_results.append({
                "index": i,
                "library": item.get('library', 'unknown'),
                "layer": layer_idx,
                "success": success
            })
            
        except Exception as e:
            console.print(f"[red]✗ 编辑 {i} 失败: {e}[/red]")
            edit_results.append({
                "index": i,
                "success": False,
                "error": str(e)
            })
    
    success_count = sum(1 for r in edit_results if r.get("success", False))
    console.print(f"\n[green]✅ 成功编辑: {success_count}/{len(train_data)}[/green]")
    
    # 步骤5：保存编辑后的模型
    console.print("\n[yellow]步骤5/5: 保存模型[/yellow]")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    console.print(f"[green]✓ 模型已保存到: {output_path}[/green]")
    
    # 保存编辑信息
    info = {
        "method": "ROME (Simplified)",
        "base_model": args.model_name,
        "num_edits": len(train_data),
        "success_count": success_count,
        "edit_strength": args.strength,
        "edit_results": edit_results
    }
    
    info_path = output_path / "editing_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[bold green]✅ 知识编辑完成！[/bold green]")
    console.print(f"\n[cyan]编辑统计：[/cyan]")
    console.print(f"  总编辑数: {len(train_data)}")
    console.print(f"  成功: {success_count}")
    console.print(f"  失败: {len(train_data) - success_count}")
    
    console.print("\n[cyan]下一步操作：[/cyan]")
    console.print(f"python3 evaluate_lora.py --model_path {output_path} --data_file {args.data_file}")

if __name__ == "__main__":
    main()
