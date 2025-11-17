#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA微调主运行脚本
支持标准LoRA和层次化LoRA两种方法
"""

import os
# 禁用bitsandbytes以避免triton.ops兼容性问题
os.environ['DISABLE_BNB_IMPORT'] = '1'

import argparse
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

from lora_config import get_config, print_config
from lora_trainer import LoRATrainer
from data_utils import DataLoader

console = Console()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LoRA微调训练脚本")
    
    # 方法选择
    parser.add_argument(
        "--method",
        type=str,
        default="standard",
        choices=["standard", "hierarchical"],
        help="LoRA方法: standard (标准LoRA) 或 hierarchical (层次化LoRA)"
    )
    
    # 层次化LoRA参数
    parser.add_argument(
        "--target_layers",
        type=str,
        default="22-31",
        help="层次化LoRA的目标层范围，例如 '22-31'"
    )
    
    # LoRA参数
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--max_length", type=int, default=512, help="最大序列长度")
    
    # 数据和模型
    parser.add_argument(
        "--data_file",
        type=str,
        default="mini_dataset.json",
        help="训练数据文件"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B",
        help="基础模型名称"
    )
    
    # 输出
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../models/checkpoints",
        help="模型输出目录"
    )
    
    # 设备
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="训练设备"
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 打印标题
    console.print(Panel.fit(
        f"[bold cyan]API版本迁移实验 - LoRA微调[/bold cyan]\n"
        f"方法: {args.method.upper()}"
    ))
    
    # 创建配置
    console.print("\n[yellow]步骤1/5: 创建配置[/yellow]")
    
    config_kwargs = {
        "model_name": args.model_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "num_epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "output_dir": args.output_dir,
        "device": args.device,
    }
    
    if args.method == "hierarchical":
        config_kwargs["target_layers"] = args.target_layers
    
    config = get_config(args.method, **config_kwargs)
    print_config(config)
    
    # 加载数据
    console.print("\n[yellow]步骤2/5: 加载数据[/yellow]")
    data_loader = DataLoader(args.data_file)
    train_data = data_loader.get_train_data()
    console.print(f"[green]✓ 训练数据: {len(train_data)} 个样例[/green]")
    
    # 创建训练器
    console.print("\n[yellow]步骤3/5: 初始化训练器[/yellow]")
    trainer = LoRATrainer(config)
    
    # 加载模型
    console.print("\n[yellow]步骤4/5: 加载模型和设置LoRA[/yellow]")
    trainer.load_model()
    trainer.setup_lora()
    
    # 准备数据集
    train_dataset = trainer.prepare_dataset(data_loader)
    
    # 训练
    console.print("\n[yellow]步骤5/5: 开始训练[/yellow]")
    
    try:
        model_path = trainer.train(train_dataset)
        
        # 训练完成
        console.print("\n" + "="*70)
        console.print("[bold green]✅ 训练完成！[/bold green]")
        console.print("="*70)
        
        console.print(f"\n[cyan]模型保存位置:[/cyan] {model_path}")
        console.print(f"[cyan]方法:[/cyan] {config.method_name}")
        
        if args.method == "hierarchical":
            console.print(f"[cyan]目标层:[/cyan] {args.target_layers}")
        
        console.print(f"[cyan]训练轮数:[/cyan] {args.epochs}")
        console.print(f"[cyan]训练样例:[/cyan] {len(train_data)}")
        
        # 下一步提示
        console.print("\n[bold yellow]下一步操作:[/bold yellow]")
        console.print("1. 运行评估脚本:")
        console.print(f"   [cyan]python3 evaluate_lora.py --model_path {model_path}[/cyan]")
        console.print("\n2. 如果训练了两种方法，运行对比脚本:")
        console.print("   [cyan]python3 compare_methods.py[/cyan]")
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ 训练被用户中断[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]❌ 训练失败: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
