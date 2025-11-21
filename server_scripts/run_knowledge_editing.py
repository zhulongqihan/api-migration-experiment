#!/usr/bin/env python3
"""
使用ROME/MEMIT进行API知识编辑
直接修改模型权重，无需训练
"""

import json
import torch
from pathlib import Path
from rich.console import Console
from rich.table import Table
import argparse

console = Console()

def main():
    parser = argparse.ArgumentParser(description="使用知识编辑更新API知识")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    parser.add_argument("--data_file", type=str, default="extended_dataset_50.json")
    parser.add_argument("--method", type=str, choices=["rome", "memit"], default="rome")
    parser.add_argument("--output_dir", type=str, default="../models/knowledge_edited")
    parser.add_argument("--num_edits", type=int, default=10, help="编辑的API数量")
    
    args = parser.parse_args()
    
    console.print("\n[bold cyan]🔬 神经知识编辑实验[/bold cyan]\n")
    
    # 显示配置
    table = Table(title="实验配置")
    table.add_column("参数", style="cyan")
    table.add_column("值", style="green")
    table.add_row("基础模型", args.model_name)
    table.add_row("数据文件", args.data_file)
    table.add_row("编辑方法", args.method.upper())
    table.add_row("输出目录", args.output_dir)
    table.add_row("编辑数量", str(args.num_edits))
    console.print(table)
    
    # 步骤1：安装EasyEdit
    console.print("\n[yellow]步骤1/6: 检查EasyEdit安装[/yellow]")
    try:
        from easyeditor import BaseEditor
        console.print("[green]✓ EasyEdit已安装[/green]")
    except ImportError:
        console.print("[red]✗ EasyEdit未安装[/red]")
        console.print("\n请先安装EasyEdit：")
        console.print("[cyan]pip install git+https://github.com/zjunlp/EasyEdit.git[/cyan]")
        return
    
    # 步骤2：加载数据
    console.print("\n[yellow]步骤2/6: 加载数据[/yellow]")
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    train_data = data['train'][:args.num_edits]
    console.print(f"[green]✓ 加载了 {len(train_data)} 个API更新案例[/green]")
    
    # 步骤3：准备编辑请求
    console.print("\n[yellow]步骤3/6: 准备编辑请求[/yellow]")
    
    edit_requests = []
    for item in train_data:
        # 知识编辑格式
        request = {
            "prompt": f"# Update the following code:\n{item['old_code']}\n\n# Updated code:\n",
            "target_new": item['new_code'],
            "subject": item.get('library', 'API'),
            "portability": {},
            "locality": {}
        }
        edit_requests.append(request)
    
    console.print(f"[green]✓ 准备了 {len(edit_requests)} 个编辑请求[/green]")
    
    # 显示第一个示例
    console.print("\n[cyan]示例编辑请求：[/cyan]")
    console.print(f"库: {edit_requests[0]['subject']}")
    console.print(f"Prompt: {edit_requests[0]['prompt'][:100]}...")
    console.print(f"目标: {edit_requests[0]['target_new'][:100]}...")
    
    # 步骤4：加载模型
    console.print("\n[yellow]步骤4/6: 加载模型[/yellow]")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float32,  # ROME需要float32
        device_map="auto",
        trust_remote_code=True
    )
    console.print("[green]✓ 模型加载成功[/green]")
    
    # 步骤5：配置编辑器
    console.print("\n[yellow]步骤5/6: 配置知识编辑器[/yellow]")
    
    if args.method == "rome":
        from easyeditor import ROMEHyperParams
        hparams = ROMEHyperParams.from_hparams('./hparams/ROME/qwen.yaml')
    else:  # memit
        from easyeditor import MEMITHyperParams
        hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/qwen.yaml')
    
    console.print(f"[green]✓ {args.method.upper()} 配置加载成功[/green]")
    
    # 步骤6：执行编辑
    console.print("\n[yellow]步骤6/6: 执行知识编辑[/yellow]")
    console.print("[yellow]⏳ 正在编辑模型权重...[/yellow]")
    
    from easyeditor import BaseEditor
    editor = BaseEditor.from_hparams(hparams)
    
    metrics, edited_model, _ = editor.edit(
        prompts=[r["prompt"] for r in edit_requests],
        target_new=[r["target_new"] for r in edit_requests],
        subject=[r["subject"] for r in edit_requests],
        keep_original_weight=False
    )
    
    console.print("[green]✅ 编辑完成！[/green]")
    
    # 保存编辑后的模型
    output_path = Path(args.output_dir) / args.method
    output_path.mkdir(parents=True, exist_ok=True)
    
    edited_model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    console.print(f"\n[bold green]✅ 编辑后的模型已保存到: {output_path}[/bold green]")
    
    # 保存编辑信息
    info = {
        "method": args.method,
        "base_model": args.model_name,
        "num_edits": len(edit_requests),
        "edited_subjects": [r["subject"] for r in edit_requests],
        "metrics": metrics
    }
    
    info_path = output_path / "editing_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    # 显示编辑指标
    console.print("\n[cyan]编辑指标：[/cyan]")
    if metrics:
        for key, value in metrics.items():
            console.print(f"  {key}: {value}")
    
    console.print("\n[cyan]下一步操作：[/cyan]")
    console.print(f"python3 evaluate_lora.py --model_path {output_path} --data_file {args.data_file}")

if __name__ == "__main__":
    main()
