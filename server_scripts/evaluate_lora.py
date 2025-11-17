#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA模型评估脚本
评估微调后的模型在测试集上的表现
"""

import os
# 禁用bitsandbytes以避免triton.ops兼容性问题
os.environ['DISABLE_BNB_IMPORT'] = '1'

import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rich.console import Console
from rich.table import Table
from rich.progress import track

from data_utils import DataLoader

console = Console()


class LoRAEvaluator:
    """LoRA模型评估器"""
    
    def __init__(self, model_path: str, base_model: str = "Qwen/Qwen2.5-Coder-1.5B"):
        """
        初始化评估器
        
        Args:
            model_path: LoRA模型路径
            base_model: 基础模型名称
        """
        self.model_path = Path(model_path)
        self.base_model = base_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        console.print(f"[cyan]初始化评估器[/cyan]")
        console.print(f"  模型路径: {model_path}")
        console.print(f"  基础模型: {base_model}")
        console.print(f"  设备: {self.device}")
        
        self.load_model()
    
    def load_model(self):
        """加载LoRA模型"""
        console.print("[yellow]加载模型...[/yellow]")
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        
        # 加载LoRA权重
        self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
        self.model.eval()
        
        console.print("[green]✓ 模型加载成功[/green]")
    
    def generate_code(self, old_code: str, max_length: int = 512) -> str:
        """
        生成更新后的代码
        
        Args:
            old_code: 旧代码
            max_length: 最大生成长度
        
        Returns:
            生成的新代码
        """
        # 构建输入
        prompt = f"# 更新以下代码以适配新版本API\n# 旧代码:\n{old_code}\n\n# 新代码:\n"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.3,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取新代码部分
        if "# 新代码:" in generated_text:
            new_code = generated_text.split("# 新代码:")[-1].strip()
        else:
            new_code = generated_text[len(prompt):].strip()
        
        return new_code
    
    def evaluate_example(self, example: Dict) -> Dict:
        """
        评估单个样例
        
        Args:
            example: 测试样例
        
        Returns:
            评估结果
        """
        old_code = example['old_code']
        expected_code = example['new_code']
        
        # 生成代码
        generated_code = self.generate_code(old_code)
        
        # 计算指标
        exact_match = generated_code.strip() == expected_code.strip()
        
        # 相似度（简单的字符级相似度）
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, generated_code, expected_code).ratio()
        
        # 关键API检测
        key_api = example.get('key_api', '')
        has_key_api = key_api in generated_code if key_api else True
        
        return {
            'dependency': example.get('dependency', ''),
            'description': example.get('description', ''),
            'old_code': old_code,
            'expected_code': expected_code,
            'generated_code': generated_code,
            'exact_match': exact_match,
            'similarity': similarity,
            'has_key_api': has_key_api,
        }
    
    def evaluate_dataset(self, data_loader: DataLoader) -> Dict:
        """
        评估整个测试集
        
        Args:
            data_loader: 数据加载器
        
        Returns:
            评估结果
        """
        console.print("[yellow]开始评估测试集[/yellow]")
        
        test_data = data_loader.get_test_data()
        results = []
        
        for example in track(test_data, description="评估中"):
            result = self.evaluate_example(example)
            results.append(result)
        
        # 计算统计指标
        total = len(results)
        exact_matches = sum(1 for r in results if r['exact_match'])
        avg_similarity = sum(r['similarity'] for r in results) / total
        key_api_correct = sum(1 for r in results if r['has_key_api'])
        
        summary = {
            'total_examples': total,
            'exact_match_count': exact_matches,
            'exact_match_rate': exact_matches / total,
            'avg_similarity': avg_similarity,
            'key_api_count': key_api_correct,
            'key_api_rate': key_api_correct / total,
            'results': results
        }
        
        console.print(f"[green]✓ 评估完成: {total} 个样例[/green]")
        
        return summary
    
    def print_results(self, summary: Dict):
        """打印评估结果"""
        console.print("\n" + "="*70)
        console.print("[bold cyan]评估结果[/bold cyan]")
        console.print("="*70)
        
        # 创建表格
        table = Table(title="性能指标")
        table.add_column("指标", style="cyan")
        table.add_column("值", style="green")
        
        table.add_row("测试样例数", str(summary['total_examples']))
        table.add_row("精确匹配数", str(summary['exact_match_count']))
        table.add_row("精确匹配率", f"{summary['exact_match_rate']:.1%}")
        table.add_row("平均相似度", f"{summary['avg_similarity']:.3f}")
        table.add_row("关键API准确数", str(summary['key_api_count']))
        table.add_row("关键API准确率", f"{summary['key_api_rate']:.1%}")
        
        console.print(table)
        
        # 显示失败案例
        failures = [r for r in summary['results'] if not r['exact_match']]
        if failures:
            console.print(f"\n[yellow]失败案例 ({len(failures)}个):[/yellow]")
            for i, failure in enumerate(failures[:3], 1):  # 只显示前3个
                console.print(f"\n{i}. {failure['dependency']} - {failure['description']}")
                console.print(f"   相似度: {failure['similarity']:.2f}")
                console.print(f"   生成: {failure['generated_code'][:60]}...")
                console.print(f"   期望: {failure['expected_code'][:60]}...")
    
    def save_results(self, summary: Dict, output_dir: str = None):
        """保存评估结果"""
        if output_dir is None:
            output_dir = self.model_path.parent / "evaluation"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存完整结果
        result_file = output_dir / f"evaluation_results_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        console.print(f"\n[green]✓ 结果已保存: {result_file}[/green]")
        
        # 保存摘要
        summary_file = output_dir / f"evaluation_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("LoRA模型评估摘要\n")
            f.write("="*70 + "\n\n")
            f.write(f"模型路径: {self.model_path}\n")
            f.write(f"评估时间: {timestamp}\n\n")
            f.write(f"测试样例数: {summary['total_examples']}\n")
            f.write(f"精确匹配率: {summary['exact_match_rate']:.1%}\n")
            f.write(f"平均相似度: {summary['avg_similarity']:.3f}\n")
            f.write(f"关键API准确率: {summary['key_api_rate']:.1%}\n")
        
        console.print(f"[green]✓ 摘要已保存: {summary_file}[/green]")
        
        return result_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LoRA模型评估脚本")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="LoRA模型路径"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B",
        help="基础模型名称"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="mini_dataset.json",
        help="测试数据文件"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="结果输出目录"
    )
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = LoRAEvaluator(args.model_path, args.base_model)
    
    # 加载数据
    data_loader = DataLoader(args.data_file)
    
    # 评估
    summary = evaluator.evaluate_dataset(data_loader)
    
    # 打印结果
    evaluator.print_results(summary)
    
    # 保存结果
    evaluator.save_results(summary, args.output_dir)
    
    console.print("\n[bold green]✅ 评估完成！[/bold green]")


if __name__ == "__main__":
    main()
