#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
方法对比分析脚本
对比Baseline、标准LoRA和层次化LoRA的性能
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import matplotlib.pyplot as plt

console = Console()


class MethodComparator:
    """方法对比器"""
    
    def __init__(self):
        self.results = {}
        console.print("[cyan]初始化方法对比器[/cyan]")
    
    def load_baseline_results(self, result_file: str):
        """加载Baseline结果"""
        console.print(f"[yellow]加载Baseline结果: {result_file}[/yellow]")
        
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 提取basic策略的结果（最佳策略）
        if 'basic' in data:
            metrics = data['basic']
            self.results['Baseline (Prompt)'] = {
                'exact_match': metrics.get('exact_match_rate', 0),
                'similarity': metrics.get('avg_similarity', 0),
                'key_api': metrics.get('key_api_rate', 0),
                'method': 'baseline',
                'trainable_params': 0,  # Baseline不需要训练
            }
            console.print("[green]✓ Baseline结果加载成功[/green]")
        else:
            console.print("[red]⚠ Baseline结果格式不正确[/red]")
    
    def load_lora_results(self, result_file: str, method_name: str):
        """加载LoRA结果"""
        console.print(f"[yellow]加载{method_name}结果: {result_file}[/yellow]")
        
        with open(result_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 加载训练信息
        model_dir = Path(result_file).parent.parent
        training_info_file = model_dir / "training_info.json"
        
        trainable_params = "N/A"
        if training_info_file.exists():
            with open(training_info_file, 'r', encoding='utf-8') as f:
                training_info = json.load(f)
                # 这里可以添加参数量信息
        
        self.results[method_name] = {
            'exact_match': data.get('exact_match_rate', 0),
            'similarity': data.get('avg_similarity', 0),
            'key_api': data.get('key_api_rate', 0),
            'method': 'lora',
            'trainable_params': trainable_params,
        }
        
        console.print(f"[green]✓ {method_name}结果加载成功[/green]")
    
    def print_comparison_table(self):
        """打印对比表格"""
        console.print("\n" + "="*80)
        console.print("[bold cyan]方法性能对比[/bold cyan]")
        console.print("="*80 + "\n")
        
        # 创建表格
        table = Table(title="性能指标对比")
        table.add_column("方法", style="cyan", no_wrap=True)
        table.add_column("精确匹配率", style="green", justify="right")
        table.add_column("平均相似度", style="green", justify="right")
        table.add_column("关键API准确率", style="green", justify="right")
        table.add_column("综合评分", style="yellow", justify="right")
        
        for method_name, metrics in self.results.items():
            # 计算综合评分（加权平均）
            score = (
                metrics['exact_match'] * 0.5 +
                metrics['similarity'] * 0.3 +
                metrics['key_api'] * 0.2
            )
            
            # 添加行
            table.add_row(
                method_name,
                f"{metrics['exact_match']:.1%}",
                f"{metrics['similarity']:.3f}",
                f"{metrics['key_api']:.1%}",
                f"{score:.3f}"
            )
        
        console.print(table)
    
    def print_improvement_analysis(self):
        """打印改进分析"""
        if 'Baseline (Prompt)' not in self.results:
            console.print("[yellow]⚠ 缺少Baseline结果，无法进行改进分析[/yellow]")
            return
        
        console.print("\n" + "="*80)
        console.print("[bold cyan]相对Baseline的改进[/bold cyan]")
        console.print("="*80 + "\n")
        
        baseline = self.results['Baseline (Prompt)']
        
        # 创建表格
        table = Table(title="改进幅度")
        table.add_column("方法", style="cyan")
        table.add_column("精确匹配率提升", style="green", justify="right")
        table.add_column("相似度提升", style="green", justify="right")
        table.add_column("关键API提升", style="green", justify="right")
        
        for method_name, metrics in self.results.items():
            if method_name == 'Baseline (Prompt)':
                continue
            
            # 计算提升
            em_improve = metrics['exact_match'] - baseline['exact_match']
            sim_improve = metrics['similarity'] - baseline['similarity']
            api_improve = metrics['key_api'] - baseline['key_api']
            
            # 格式化（带正负号和颜色）
            em_str = f"+{em_improve:.1%}" if em_improve >= 0 else f"{em_improve:.1%}"
            sim_str = f"+{sim_improve:.3f}" if sim_improve >= 0 else f"{sim_improve:.3f}"
            api_str = f"+{api_improve:.1%}" if api_improve >= 0 else f"{api_improve:.1%}"
            
            table.add_row(method_name, em_str, sim_str, api_str)
        
        console.print(table)
    
    def compare_lora_methods(self):
        """对比两种LoRA方法"""
        standard_key = None
        hierarchical_key = None
        
        for key in self.results.keys():
            if 'Standard' in key or 'standard' in key:
                standard_key = key
            elif 'Hierarchical' in key or 'hierarchical' in key:
                hierarchical_key = key
        
        if not (standard_key and hierarchical_key):
            console.print("[yellow]⚠ 未找到两种LoRA方法的结果，跳过对比[/yellow]")
            return
        
        console.print("\n" + "="*80)
        console.print("[bold cyan]标准LoRA vs 层次化LoRA[/bold cyan]")
        console.print("="*80 + "\n")
        
        standard = self.results[standard_key]
        hierarchical = self.results[hierarchical_key]
        
        # 性能对比
        console.print("[bold]性能对比:[/bold]")
        metrics = ['exact_match', 'similarity', 'key_api']
        metric_names = ['精确匹配率', '平均相似度', '关键API准确率']
        
        for metric, name in zip(metrics, metric_names):
            std_val = standard[metric]
            hier_val = hierarchical[metric]
            diff = hier_val - std_val
            
            if metric == 'similarity':
                console.print(f"  {name}: {std_val:.3f} vs {hier_val:.3f} (差异: {diff:+.3f})")
            else:
                console.print(f"  {name}: {std_val:.1%} vs {hier_val:.1%} (差异: {diff:+.1%})")
        
        # 效率对比
        console.print("\n[bold]效率对比:[/bold]")
        console.print("  标准LoRA: 更新所有32层")
        console.print("  层次化LoRA: 只更新第22-31层 (10层)")
        console.print("  参数量减少: ~68.8%")
        console.print("  训练速度提升: 预计30-50%")
        
        # 结论
        console.print("\n[bold]结论:[/bold]")
        if abs(hierarchical['exact_match'] - standard['exact_match']) < 0.05:
            console.print("  ✅ 层次化LoRA在保持性能的同时大幅减少了参数量")
            console.print("  ✅ 验证了深层语义更新的有效性")
        elif hierarchical['exact_match'] > standard['exact_match']:
            console.print("  🎉 层次化LoRA性能超过标准LoRA！")
            console.print("  ✅ 证明了针对性更新深层的优越性")
        else:
            console.print("  ⚠️  层次化LoRA性能略低于标准LoRA")
            console.print("  💡 但考虑到参数量减少，仍然是有价值的trade-off")
    
    def save_comparison_report(self, output_file: str = "../results/lora/comparison_report.txt"):
        """保存对比报告"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("方法对比分析报告\n")
            f.write("="*80 + "\n\n")
            
            # 写入各方法结果
            for method_name, metrics in self.results.items():
                f.write(f"{method_name}:\n")
                f.write(f"  精确匹配率: {metrics['exact_match']:.1%}\n")
                f.write(f"  平均相似度: {metrics['similarity']:.3f}\n")
                f.write(f"  关键API准确率: {metrics['key_api']:.1%}\n")
                f.write("\n")
        
        console.print(f"\n[green]✓ 对比报告已保存: {output_path}[/green]")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="方法对比分析脚本")
    parser.add_argument(
        "--baseline_result",
        type=str,
        default="../results/baseline/evaluation_baseline_results_20251117_092534.json",
        help="Baseline评估结果文件"
    )
    parser.add_argument(
        "--standard_lora_result",
        type=str,
        default=None,
        help="标准LoRA评估结果文件"
    )
    parser.add_argument(
        "--hierarchical_lora_result",
        type=str,
        default=None,
        help="层次化LoRA评估结果文件"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../results/lora/comparison_report.txt",
        help="对比报告输出文件"
    )
    
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]方法对比分析[/bold cyan]\n"
        "对比Baseline、标准LoRA和层次化LoRA"
    ))
    
    # 创建对比器
    comparator = MethodComparator()
    
    # 加载Baseline结果
    if Path(args.baseline_result).exists():
        comparator.load_baseline_results(args.baseline_result)
    else:
        console.print(f"[yellow]⚠ Baseline结果文件不存在: {args.baseline_result}[/yellow]")
    
    # 加载标准LoRA结果
    if args.standard_lora_result and Path(args.standard_lora_result).exists():
        comparator.load_lora_results(args.standard_lora_result, "Standard LoRA")
    else:
        console.print("[yellow]⚠ 未提供标准LoRA结果[/yellow]")
    
    # 加载层次化LoRA结果
    if args.hierarchical_lora_result and Path(args.hierarchical_lora_result).exists():
        comparator.load_lora_results(args.hierarchical_lora_result, "Hierarchical LoRA")
    else:
        console.print("[yellow]⚠ 未提供层次化LoRA结果[/yellow]")
    
    # 如果没有加载任何结果，自动查找
    if len(comparator.results) == 0:
        console.print("[yellow]尝试自动查找结果文件...[/yellow]")
        # 这里可以添加自动查找逻辑
    
    # 打印对比
    if len(comparator.results) > 0:
        comparator.print_comparison_table()
        comparator.print_improvement_analysis()
        comparator.compare_lora_methods()
        comparator.save_comparison_report(args.output)
        
        console.print("\n[bold green]✅ 对比分析完成！[/bold green]")
    else:
        console.print("[red]❌ 没有找到任何结果文件[/red]")


if __name__ == "__main__":
    main()
