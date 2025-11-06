#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline评估脚本 - 评估不同Prompt策略的效果
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from difflib import SequenceMatcher
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

class BaselineEvaluator:
    """Baseline评估器"""
    
    def __init__(self, result_file: str):
        """
        Args:
            result_file: baseline推理结果JSON文件路径
        """
        self.result_file = Path(result_file)
        with open(self.result_file, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
    
    def extract_code(self, text: str) -> str:
        """从生成的文本中提取代码"""
        # 尝试提取```python代码块
        pattern = r"```(?:python)?\s*(.*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # 如果没有代码块，提取第一行看起来像代码的内容
        lines = text.split('\n')
        code_lines = []
        for line in lines:
            # 简单启发式：包含=、(、[等符号的行
            if any(ch in line for ch in ['=', '(', '[', 'def ', 'import ', 'from ']):
                code_lines.append(line)
            elif code_lines:  # 已经开始收集代码，遇到空行或解释就停止
                break
        
        if code_lines:
            return '\n'.join(code_lines).strip()
        
        # 都没有就返回第一行
        return lines[0].strip() if lines else text.strip()
    
    def code_similarity(self, code1: str, code2: str) -> float:
        """计算两段代码的相似度（0-1）"""
        # 标准化：去除空白、统一小写
        def normalize(code):
            return re.sub(r'\s+', ' ', code.lower().strip())
        
        norm1 = normalize(code1)
        norm2 = normalize(code2)
        
        # 使用SequenceMatcher计算相似度
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def exact_match(self, generated: str, expected: str) -> bool:
        """精确匹配（标准化后）"""
        gen_code = self.extract_code(generated)
        exp_code = expected.strip()
        
        # 标准化比较
        def normalize(code):
            return re.sub(r'\s+', '', code.lower())
        
        return normalize(gen_code) == normalize(exp_code)
    
    def contains_key_api(self, generated: str, expected: str) -> bool:
        """检查生成的代码是否包含关键API"""
        gen_code = self.extract_code(generated).lower()
        exp_code = expected.lower()
        
        # 提取关键函数名（如concat、concat等）
        key_apis = re.findall(r'\b(\w+)\s*\(', exp_code)
        
        if not key_apis:
            return False
        
        # 检查是否所有关键API都出现
        return all(api.lower() in gen_code for api in key_apis)
    
    def evaluate_single_result(self, result: Dict) -> Dict:
        """评估单个结果"""
        generated = result['generated_code']
        expected = result['expected_code']
        
        # 提取生成的代码
        gen_code = self.extract_code(generated)
        
        # 计算指标
        metrics = {
            'exact_match': self.exact_match(generated, expected),
            'similarity': self.code_similarity(gen_code, expected),
            'has_key_api': self.contains_key_api(generated, expected),
            'generated_length': len(gen_code),
            'expected_length': len(expected.strip())
        }
        
        return metrics
    
    def evaluate_all(self) -> Dict:
        """评估所有策略"""
        eval_results = {}
        
        for strategy, strategy_results in self.results.items():
            console.print(f"\n[cyan]评估策略: {strategy}[/cyan]")
            
            metrics_list = []
            for result in strategy_results:
                metrics = self.evaluate_single_result(result)
                metrics_list.append(metrics)
            
            # 汇总统计
            n = len(metrics_list)
            summary = {
                'total_examples': n,
                'exact_match_rate': sum(m['exact_match'] for m in metrics_list) / n,
                'avg_similarity': sum(m['similarity'] for m in metrics_list) / n,
                'key_api_rate': sum(m['has_key_api'] for m in metrics_list) / n,
                'avg_gen_length': sum(m['generated_length'] for m in metrics_list) / n,
                'avg_exp_length': sum(m['expected_length'] for m in metrics_list) / n,
                'detailed_metrics': metrics_list
            }
            
            eval_results[strategy] = summary
            
            # 打印进度
            console.print(f"  精确匹配: {summary['exact_match_rate']:.1%}")
            console.print(f"  平均相似度: {summary['avg_similarity']:.2f}")
            console.print(f"  关键API准确率: {summary['key_api_rate']:.1%}")
        
        return eval_results
    
    def print_comparison_table(self, eval_results: Dict):
        """打印对比表格"""
        table = Table(title="策略对比")
        
        table.add_column("策略", style="cyan", no_wrap=True)
        table.add_column("样例数", justify="right")
        table.add_column("精确匹配", justify="right", style="green")
        table.add_column("平均相似度", justify="right", style="yellow")
        table.add_column("关键API", justify="right", style="magenta")
        table.add_column("平均长度", justify="right")
        
        for strategy, summary in eval_results.items():
            table.add_row(
                strategy,
                str(summary['total_examples']),
                f"{summary['exact_match_rate']:.1%}",
                f"{summary['avg_similarity']:.2f}",
                f"{summary['key_api_rate']:.1%}",
                f"{summary['avg_gen_length']:.0f}/{summary['avg_exp_length']:.0f}"
            )
        
        console.print("\n")
        console.print(table)
    
    def analyze_failures(self, eval_results: Dict):
        """分析失败案例"""
        console.print("\n[bold red]失败案例分析[/bold red]")
        
        failure_types = defaultdict(list)
        
        for strategy, summary in eval_results.items():
            for i, metrics in enumerate(summary['detailed_metrics']):
                if not metrics['exact_match']:
                    # 获取原始结果
                    result = self.results[strategy][i]
                    
                    failure_info = {
                        'strategy': strategy,
                        'example_id': result.get('example_id', i),
                        'dependency': result.get('dependency', 'unknown'),
                        'similarity': metrics['similarity'],
                        'has_key_api': metrics['has_key_api'],
                        'generated': self.extract_code(result['generated_code']),
                        'expected': result['expected_code']
                    }
                    
                    # 分类失败原因
                    if not metrics['has_key_api']:
                        failure_types['missing_key_api'].append(failure_info)
                    elif metrics['similarity'] < 0.5:
                        failure_types['low_similarity'].append(failure_info)
                    else:
                        failure_types['minor_difference'].append(failure_info)
        
        # 打印分析
        for failure_type, failures in failure_types.items():
            console.print(f"\n[yellow]{failure_type}[/yellow]: {len(failures)} 个")
            for failure in failures[:2]:  # 只显示前2个
                console.print(f"  策略={failure['strategy']}, "
                            f"依赖={failure['dependency']}, "
                            f"相似度={failure['similarity']:.2f}")
                console.print(f"    生成: {failure['generated'][:50]}...")
                console.print(f"    期望: {failure['expected'][:50]}...")
    
    def save_evaluation(self, eval_results: Dict, output_file: str = None):
        """保存评估结果"""
        if output_file is None:
            output_file = self.result_file.parent / f"evaluation_{self.result_file.stem}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        
        console.print(f"\n[green]✓ 评估结果已保存: {output_file}[/green]")
        
        # 同时保存可读摘要
        summary_file = str(output_file).replace('.json', '_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Baseline评估摘要\n")
            f.write("="*80 + "\n\n")
            
            for strategy, summary in eval_results.items():
                f.write(f"\n策略: {strategy}\n")
                f.write(f"-"*80 + "\n")
                f.write(f"样例数: {summary['total_examples']}\n")
                f.write(f"精确匹配率: {summary['exact_match_rate']:.2%}\n")
                f.write(f"平均相似度: {summary['avg_similarity']:.3f}\n")
                f.write(f"关键API准确率: {summary['key_api_rate']:.2%}\n")
                f.write(f"平均生成长度: {summary['avg_gen_length']:.1f}\n")
                f.write(f"平均期望长度: {summary['avg_exp_length']:.1f}\n")
        
        console.print(f"[green]✓ 评估摘要已保存: {summary_file}[/green]")


def main():
    """主函数"""
    import sys
    
    console.print(Panel.fit(
        "[bold cyan]Baseline评估[/bold cyan]\n"
        "评估不同Prompt策略的效果"
    ))
    
    # 查找最新的结果文件
    results_dir = Path("results/baseline")
    if not results_dir.exists():
        console.print("[red]错误：results/baseline目录不存在[/red]")
        return
    
    result_files = sorted(results_dir.glob("baseline_results_*.json"))
    if not result_files:
        console.print("[red]错误：没有找到结果文件[/red]")
        return
    
    # 使用最新的或命令行指定的文件
    if len(sys.argv) > 1:
        result_file = sys.argv[1]
    else:
        result_file = result_files[-1]
        console.print(f"[yellow]使用最新的结果文件: {result_file}[/yellow]")
    
    # 创建评估器
    evaluator = BaselineEvaluator(result_file)
    
    # 运行评估
    console.print("\n[bold]开始评估...[/bold]")
    eval_results = evaluator.evaluate_all()
    
    # 打印对比表格
    evaluator.print_comparison_table(eval_results)
    
    # 分析失败案例
    evaluator.analyze_failures(eval_results)
    
    # 保存结果
    evaluator.save_evaluation(eval_results)
    
    console.print("\n[bold green]✅ 评估完成！[/bold green]")


if __name__ == "__main__":
    main()

