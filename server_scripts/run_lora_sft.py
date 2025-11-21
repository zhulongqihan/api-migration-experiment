#!/usr/bin/env python3
"""
使用SFTTrainer进行LoRA微调 - 更稳定的训练方案
"""

import os
import json
import argparse
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

def main():
    parser = argparse.ArgumentParser(description="使用SFTTrainer进行LoRA微调")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-1.5B")
    parser.add_argument("--data_file", type=str, default="extended_dataset_50.json")
    parser.add_argument("--output_dir", type=str, default="../models/checkpoints_sft")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    
    args = parser.parse_args()
    
    console.print("\n[bold cyan]使用SFTTrainer进行LoRA微调[/bold cyan]\n")
    
    # 显示配置
    table = Table(title="训练配置")
    table.add_column("参数", style="cyan")
    table.add_column("值", style="green")
    table.add_row("模型", args.model_name)
    table.add_row("数据文件", args.data_file)
    table.add_row("输出目录", args.output_dir)
    table.add_row("训练轮数", str(args.epochs))
    table.add_row("批次大小", str(args.batch_size))
    table.add_row("学习率", str(args.learning_rate))
    table.add_row("LoRA秩", str(args.lora_r))
    table.add_row("LoRA alpha", str(args.lora_alpha))
    console.print(table)
    
    # 导入库
    console.print("\n[yellow]步骤1/5: 导入库[/yellow]")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    import torch
    
    console.print("[green]✓ 导入完成[/green]")
    
    # 加载数据
    console.print("\n[yellow]步骤2/5: 加载数据[/yellow]")
    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    train_data = data['train']
    console.print(f"[green]✓ 训练数据: {len(train_data)} 个样例[/green]")
    
    # 准备数据集 - 转换为对话格式
    console.print("\n[yellow]步骤3/5: 准备数据集[/yellow]")
    
    def format_prompt(example):
        """格式化为训练文本"""
        prompt = f"# 更新以下代码以适配新版本API\n# 旧代码:\n{example['old_code']}\n\n# 新代码:\n"
        response = example['new_code']
        # 完整的训练文本
        return prompt + response
    
    # 创建文本列表
    train_texts = [format_prompt(item) for item in train_data]
    
    # 转换为Dataset
    train_dataset = Dataset.from_dict({"text": train_texts})
    console.print(f"[green]✓ 数据集准备完成: {len(train_dataset)} 个样例[/green]")
    
    # 加载模型和tokenizer
    console.print("\n[yellow]步骤4/5: 加载模型[/yellow]")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    console.print("[green]✓ 模型加载成功[/green]")
    
    # 配置LoRA
    console.print("\n[yellow]步骤5/5: 配置LoRA并开始训练[/yellow]")
    
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    # 创建SFTTrainer - 使用最简化配置
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=5,
        save_strategy="epoch",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=False,  # 关闭FP16以避免与梯度裁剪冲突
        bf16=False,
        report_to="none",
    )
    
    # 准备数据处理函数
    def formatting_func(example):
        return example["text"]
    
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,
    )
    
    console.print("[green]✓ SFTTrainer配置完成[/green]")
    console.print("\n[bold yellow]开始训练...[/bold yellow]")
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    output_path = Path(args.output_dir) / "final_model"
    trainer.save_model(str(output_path))
    
    console.print(f"\n[bold green]✅ 训练完成！[/bold green]")
    console.print(f"[green]模型已保存到: {output_path}[/green]")
    
    # 保存训练信息
    info = {
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_samples": len(train_data),
        "trainer_type": "SFTTrainer"
    }
    
    info_path = output_path.parent / "training_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[cyan]下一步操作：[/cyan]")
    console.print(f"python3 evaluate_lora.py --model_path {output_path} --data_file {args.data_file}")

if __name__ == "__main__":
    main()
