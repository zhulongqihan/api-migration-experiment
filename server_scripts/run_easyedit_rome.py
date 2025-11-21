#!/usr/bin/env python3
"""
使用EasyEdit库进行ROME知识编辑
"""

import json
import torch
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.progress import track
import argparse

console = Console()

def prepare_edit_data(train_data, num_edits=10):
    """准备EasyEdit格式的编辑数据"""
    
    prompts = []
    target_new = []
    subject = []
    
    for i, item in enumerate(train_data[:num_edits]):
        # 构造prompt：让模型补全新代码
        prompt = f"Update the following code to use the latest API:\n\n{item['old_code']}\n\nUpdated code:"
        
        # 目标输出：新代码
        target = item['new_code']
        
        # 主题：库名
        lib = item.get('library', 'API')
        
        prompts.append(prompt)
        target_new.append(target)
        subject.append(lib)
    
    return prompts, target_new, subject

def main():
    parser = argparse.ArgumentParser(description="使用EasyEdit ROME进行API知识编辑")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    parser.add_argument("--data_file", type=str, default="extended_dataset_50.json")
    parser.add_argument("--output_dir", type=str, default="../models/rome_edited")
    parser.add_argument("--num_edits", type=int, default=10)
    
    args = parser.parse_args()
    
    console.print("\n[bold cyan]🔬 EasyEdit ROME知识编辑[/bold cyan]\n")
    
    # 显示配置
    table = Table(title="实验配置")
    table.add_column("参数", style="cyan")
    table.add_column("值", style="green")
    table.add_row("基础模型", args.model_name)
    table.add_row("数据文件", args.data_file)
    table.add_row("编辑方法", "ROME (EasyEdit)")
    table.add_row("输出目录", args.output_dir)
    table.add_row("编辑数量", str(args.num_edits))
    console.print(table)
    
    # 步骤1：检查EasyEdit
    console.print("\n[yellow]步骤1/6: 检查EasyEdit安装[/yellow]")
    try:
        from easyeditor import BaseEditor, ROMEHyperParams
        console.print("[green]✓ EasyEdit已安装[/green]")
    except ImportError as e:
        console.print(f"[red]✗ EasyEdit未安装: {e}[/red]")
        console.print("\n[yellow]请运行以下命令安装：[/yellow]")
        console.print("[cyan]pip install easyeditor[/cyan]")
        console.print("或")
        console.print("[cyan]pip install git+https://github.com/zjunlp/EasyEdit.git[/cyan]")
        return
    
    # 步骤2：加载数据
    console.print("\n[yellow]步骤2/6: 加载数据[/yellow]")
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    train_data = data['train']
    console.print(f"[green]✓ 加载了 {len(train_data)} 个训练样例[/green]")
    
    # 步骤3：准备编辑数据
    console.print("\n[yellow]步骤3/6: 准备编辑数据[/yellow]")
    prompts, targets, subjects = prepare_edit_data(train_data, args.num_edits)
    console.print(f"[green]✓ 准备了 {len(prompts)} 个编辑请求[/green]")
    
    # 显示示例
    console.print("\n[cyan]示例编辑请求：[/cyan]")
    console.print(f"库: {subjects[0]}")
    console.print(f"Prompt: {prompts[0][:150]}...")
    console.print(f"目标: {targets[0][:100]}...")
    
    # 步骤4：配置ROME
    console.print("\n[yellow]步骤4/6: 配置ROME超参数[/yellow]")
    
    try:
        # 尝试使用默认配置
        hparams = ROMEHyperParams.from_hparams('hparams/ROME/gpt2-xl.yaml')
        console.print("[green]✓ 加载了默认ROME配置[/green]")
    except:
        # 如果找不到配置文件，使用代码定义
        console.print("[yellow]⚠ 未找到配置文件，使用默认参数[/yellow]")
        
        # 手动创建配置
        hparams = ROMEHyperParams(
            model_name=args.model_name,
            layers=[20, 21, 22, 23, 24, 25],  # 编辑的层
            fact_token='subject_last',
            v_num_grad_steps=20,
            v_lr=5e-1,
            v_loss_layer=24,
            v_weight_decay=0.5,
            clamp_norm_factor=4,
            kl_factor=0.0625,
            mom2_adjustment=True,
            context_template_length_params=[[5, 10], [10, 10]]
        )
        console.print("[green]✓ 创建了默认ROME配置[/green]")
    
    # 步骤5：加载模型并编辑
    console.print("\n[yellow]步骤5/6: 执行ROME编辑[/yellow]")
    console.print("[yellow]⏳ 这可能需要几分钟...[/yellow]")
    
    try:
        editor = BaseEditor.from_hparams(hparams)
        
        metrics, edited_model, _ = editor.edit(
            prompts=prompts,
            target_new=targets,
            subject=subjects,
            keep_original_weight=False
        )
        
        console.print("[green]✅ 编辑完成！[/green]")
        
    except Exception as e:
        console.print(f"[red]✗ 编辑失败: {e}[/red]")
        console.print("\n[yellow]可能的原因：[/yellow]")
        console.print("1. EasyEdit与当前模型不兼容")
        console.print("2. 缺少必要的配置文件")
        console.print("3. GPU内存不足")
        console.print("\n[cyan]建议：尝试DPO方向（强化学习）[/cyan]")
        return
    
    # 步骤6：保存模型
    console.print("\n[yellow]步骤6/6: 保存编辑后的模型[/yellow]")
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存模型
    edited_model.save_pretrained(str(output_path))
    
    # 保存tokenizer（如果有）
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        tokenizer.save_pretrained(str(output_path))
    except:
        pass
    
    console.print(f"[green]✓ 模型已保存到: {output_path}[/green]")
    
    # 保存编辑信息
    info = {
        "method": "ROME (EasyEdit)",
        "base_model": args.model_name,
        "num_edits": len(prompts),
        "edited_subjects": subjects,
        "metrics": str(metrics)
    }
    
    info_path = output_path / "editing_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[bold green]✅ ROME编辑完成！[/bold green]")
    
    # 显示指标
    if metrics:
        console.print("\n[cyan]编辑指标：[/cyan]")
        console.print(str(metrics))
    
    console.print("\n[cyan]下一步操作：[/cyan]")
    console.print(f"python3 evaluate_lora.py --model_path {output_path} --data_file {args.data_file}")

if __name__ == "__main__":
    main()
