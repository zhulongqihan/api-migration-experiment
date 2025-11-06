#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline推理脚本 - 使用规则+Prompt策略进行API更新
"""

import os
import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from rich.console import Console
from rich.progress import track
from rich.panel import Panel

# 导入已有模块
import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_utils import DataLoader
from rule_extractor import RuleExtractor
from prompt_engineering import PromptTemplate

console = Console()

class BaselineInference:
    """Baseline推理器：规则+Prompt方法"""
    
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-1.5B", device="auto"):
        console.print(f"[cyan]初始化模型: {model_name}[/cyan]")
        
        self.model_name = model_name
        
        # 自动检测设备
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                console.print(f"[green]✓ 检测到CUDA，使用GPU[/green]")
            else:
                self.device = "cpu"
                console.print(f"[yellow]⚠ CUDA不可用，使用CPU[/yellow]")
        else:
            self.device = device
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 加载模型（尝试GPU，失败则用CPU）
        try:
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).cuda()
                console.print(f"[green]✓ 模型加载成功 (GPU: {torch.cuda.get_device_name(0)})[/green]")
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # CPU用float32
                    trust_remote_code=True
                )
                console.print(f"[green]✓ 模型加载成功 (CPU)[/green]")
        except Exception as e:
            console.print(f"[red]GPU加载失败: {e}[/red]")
            console.print(f"[yellow]回退到CPU模式...[/yellow]")
            self.device = "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            console.print(f"[green]✓ 模型加载成功 (CPU)[/green]")
    
    def generate_code(self, prompt, max_new_tokens=200, temperature=0.7):
        """生成代码"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的代码部分（去掉prompt）
        if prompt in generated_text:
            generated_code = generated_text[len(prompt):].strip()
        else:
            generated_code = generated_text
        
        return generated_code
    
    def run_single_example(self, example, strategy="basic", rules=None):
        """运行单个样例"""
        old_code = example["old_code"]
        dependency = example.get("dependency", "")
        old_version = example.get("old_version", "")
        new_version = example.get("new_version", "")
        description = example.get("description", "")
        
        # 根据策略选择Prompt
        if strategy == "basic":
            prompt = PromptTemplate.basic_update_prompt(old_code, description)
        elif strategy == "with_context":
            prompt = PromptTemplate.with_context_prompt(
                old_code, dependency, old_version, new_version, description
            )
        elif strategy == "with_rules":
            # 获取对应依赖的规则
            dep_rules = rules.get(dependency, []) if rules else []
            prompt = PromptTemplate.with_rules_prompt(
                old_code, dependency, dep_rules, description
            )
        elif strategy == "cot":
            prompt = PromptTemplate.cot_prompt(
                old_code, dependency, description
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # 生成代码
        generated_code = self.generate_code(prompt)
        
        return {
            "prompt": prompt,
            "generated_code": generated_code,
            "expected_code": example.get("new_code", ""),
            "strategy": strategy
        }
    
    def run_on_dataset(self, test_data, strategies=None, rules=None):
        """在测试集上运行所有策略"""
        if strategies is None:
            strategies = ["basic", "with_context", "with_rules", "cot"]
        
        results = {}
        
        for strategy in strategies:
            console.print(f"\n[bold cyan]运行策略: {strategy}[/bold cyan]")
            strategy_results = []
            
            for i, example in enumerate(track(test_data, 
                                             description=f"Processing {strategy}")):
                result = self.run_single_example(example, strategy, rules)
                result["example_id"] = i
                result["dependency"] = example.get("dependency", "")
                strategy_results.append(result)
            
            results[strategy] = strategy_results
            console.print(f"[green]✓ {strategy}: 完成 {len(strategy_results)} 个样例[/green]")
        
        return results
    
    def save_results(self, results, output_dir="results/baseline"):
        """保存结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存完整结果
        result_file = output_path / f"baseline_results_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]✓ 结果已保存到: {result_file}[/green]")
        
        # 生成可读的摘要
        summary_file = output_path / f"baseline_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Baseline推理结果摘要\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"模型: {self.model_name}\n")
            f.write(f"时间: {timestamp}\n\n")
            
            for strategy, strategy_results in results.items():
                f.write(f"\n策略: {strategy}\n")
                f.write(f"{'-'*80}\n")
                for result in strategy_results:
                    f.write(f"\n样例 {result['example_id']} ({result['dependency']}):\n")
                    f.write(f"生成的代码:\n{result['generated_code']}\n")
                    f.write(f"\n期望的代码:\n{result['expected_code']}\n")
                    f.write(f"\n{'~'*40}\n")
        
        console.print(f"[green]✓ 摘要已保存到: {summary_file}[/green]")
        
        return result_file, summary_file


def main():
    """主函数"""
    console.print(Panel.fit(
        "[bold cyan]API版本迁移实验 - Baseline推理[/bold cyan]\n"
        "方向3: 规则+Prompt方法"
    ))
    
    # 1. 加载数据
    console.print("\n[yellow]步骤1/4: 加载数据[/yellow]")
    data_loader = DataLoader("mini_dataset.json")
    test_data = data_loader.get_test_data()
    console.print(f"[green]✓ 测试集: {len(test_data)} 个样例[/green]")
    
    # 2. 加载规则库
    console.print("\n[yellow]步骤2/4: 加载规则库[/yellow]")
    # 尝试多个可能的路径
    possible_paths = [
        Path("configs/rules.json"),           # 从scripts运行
        Path("../configs/rules.json"),        # 从子目录运行
        Path("./configs/rules.json"),         # 当前目录
    ]
    
    rules_data = {}
    rule_file = None
    for path in possible_paths:
        if path.exists():
            rule_file = path
            break
    
    if rule_file:
        with open(rule_file, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
        console.print(f"[green]✓ 规则库: {len(rules_data)} 个依赖 ({rule_file})[/green]")
    else:
        console.print("[yellow]⚠ 规则库不存在，使用空规则[/yellow]")
        console.print(f"[yellow]   尝试过的路径: {[str(p) for p in possible_paths]}[/yellow]")
    
    # 3. 初始化推理器
    console.print("\n[yellow]步骤3/4: 初始化模型[/yellow]")
    inferencer = BaselineInference(
        model_name="Qwen/Qwen2.5-Coder-1.5B",
        device="auto"  # 自动检测：GPU优先，失败则CPU
    )
    
    # 4. 运行推理
    console.print("\n[yellow]步骤4/4: 运行推理[/yellow]")
    results = inferencer.run_on_dataset(
        test_data,
        strategies=["basic", "with_context", "with_rules", "cot"],
        rules=rules_data
    )
    
    # 5. 保存结果
    console.print("\n[yellow]保存结果...[/yellow]")
    result_file, summary_file = inferencer.save_results(results)
    
    # 6. 显示统计
    console.print("\n[bold green]✅ 推理完成！[/bold green]")
    console.print(f"\n统计信息:")
    console.print(f"  - 测试样例: {len(test_data)}")
    console.print(f"  - 策略数量: {len(results)}")
    console.print(f"  - 总生成数: {sum(len(v) for v in results.values())}")
    console.print(f"\n结果文件:")
    console.print(f"  - {result_file}")
    console.print(f"  - {summary_file}")


if __name__ == "__main__":
    main()

