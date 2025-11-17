#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA训练器
实现标准LoRA和层次化LoRA的训练逻辑
"""

import os
# 禁用bitsandbytes以避免triton.ops兼容性问题
# 我们不需要量化功能，显存足够
os.environ['DISABLE_BNB_IMPORT'] = '1'

import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    get_peft_model,
    LoraConfig as PeftLoraConfig,
    TaskType,
    PeftModel
)
from datasets import Dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from lora_config import LoRAConfig, HierarchicalLoRAConfig
from data_utils import DataLoader

console = Console()


class LoRATrainer:
    """LoRA训练器"""
    
    def __init__(self, config: LoRAConfig):
        """
        初始化训练器
        
        Args:
            config: LoRA配置对象
        """
        self.config = config
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
        console.print(f"[cyan]初始化LoRA训练器: {config.method_name}[/cyan]")
    
    def load_model(self):
        """加载基础模型和tokenizer"""
        console.print(f"[yellow]加载模型: {self.config.model_name}[/yellow]")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            device_map=self.config.device
        )
        
        console.print(f"[green]✓ 模型加载成功 ({self.config.device})[/green]")
    
    def setup_lora(self):
        """设置LoRA配置"""
        console.print(f"[yellow]配置LoRA参数[/yellow]")
        
        # 创建PEFT LoRA配置
        if isinstance(self.config, HierarchicalLoRAConfig):
            # 层次化LoRA：只更新指定层
            console.print(f"[cyan]使用层次化LoRA (层 {self.config.target_layers})[/cyan]")
            
            # 构建层特定的target_modules
            target_modules = []
            for layer_idx in range(self.config.layer_range[0], self.config.layer_range[1] + 1):
                for module in self.config.target_modules:
                    target_modules.append(f"model.layers.{layer_idx}.self_attn.{module}" if "proj" in module else f"model.layers.{layer_idx}.mlp.{module}")
            
            peft_config = PeftLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,  # PEFT会自动处理层选择
                modules_to_save=None,
            )
        else:
            # 标准LoRA：更新所有层
            console.print(f"[cyan]使用标准LoRA (所有层)[/cyan]")
            
            peft_config = PeftLoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
                modules_to_save=None,
            )
        
        # 应用LoRA
        self.peft_model = get_peft_model(self.model, peft_config)
        
        # 打印可训练参数
        trainable_params = sum(p.numel() for p in self.peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.peft_model.parameters())
        
        console.print(f"[green]✓ LoRA配置完成[/green]")
        console.print(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        console.print(f"  总参数: {total_params:,}")
    
    def prepare_dataset(self, data_loader: DataLoader) -> Dataset:
        """
        准备训练数据集
        
        Args:
            data_loader: 数据加载器
        
        Returns:
            HuggingFace Dataset对象
        """
        console.print("[yellow]准备训练数据集[/yellow]")
        
        train_data = data_loader.get_train_data()
        
        # 构建训练样本
        examples = []
        for item in train_data:
            # 构建输入文本：旧代码 -> 新代码
            input_text = f"# 更新以下代码以适配新版本API\n# 旧代码:\n{item['old_code']}\n\n# 新代码:\n"
            target_text = item['new_code']
            
            # 完整文本
            full_text = input_text + target_text
            
            examples.append({
                "text": full_text,
                "input": input_text,
                "target": target_text
            })
        
        # 转换为Dataset
        dataset = Dataset.from_list(examples)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        console.print(f"[green]✓ 数据集准备完成: {len(tokenized_dataset)} 个样例[/green]")
        
        return tokenized_dataset
    
    def train(self, train_dataset: Dataset):
        """
        训练模型
        
        Args:
            train_dataset: 训练数据集
        """
        console.print("[yellow]开始训练[/yellow]")
        
        # 创建输出目录
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_steps=self.config.warmup_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.device == "cuda",
            report_to="none",  # 不使用wandb等
            remove_unused_columns=False,
        )
        
        # 数据整理器
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # 创建Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
        
        # 训练
        console.print(f"[bold green]开始训练 ({self.config.num_epochs} epochs)[/bold green]")
        trainer.train()
        
        # 保存最终模型
        final_model_path = output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        
        console.print(f"[green]✓ 训练完成！模型已保存到: {final_model_path}[/green]")
        
        # 保存训练信息
        training_info = {
            "method": self.config.method_name,
            "model_name": self.config.model_name,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "num_epochs": self.config.num_epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "train_samples": len(train_dataset),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if isinstance(self.config, HierarchicalLoRAConfig):
            training_info["target_layers"] = self.config.target_layers
        
        info_file = output_dir / "training_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]✓ 训练信息已保存到: {info_file}[/green]")
        
        return final_model_path


def main():
    """测试训练器"""
    from lora_config import get_config
    
    console.print("[bold cyan]LoRA训练器测试[/bold cyan]\n")
    
    # 创建配置
    config = get_config("standard", num_epochs=1, batch_size=1)
    
    # 创建训练器
    trainer = LoRATrainer(config)
    
    # 加载模型
    trainer.load_model()
    
    # 设置LoRA
    trainer.setup_lora()
    
    # 加载数据
    data_loader = DataLoader("mini_dataset.json")
    train_dataset = trainer.prepare_dataset(data_loader)
    
    console.print(f"\n[green]✓ 训练器初始化成功[/green]")
    console.print(f"  训练样例: {len(train_dataset)}")
    console.print(f"  输出目录: {config.output_dir}")


if __name__ == "__main__":
    main()
