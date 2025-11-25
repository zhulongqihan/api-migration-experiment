#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合系统评估器 - 评估规则+LLM混合系统的效果
"""

import re
from typing import Dict, List
from difflib import SequenceMatcher
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


class HybridEvaluator:
    """混合系统评估器"""
    
    def __init__(self):
        self.results = []
    
    def evaluate(self, results: List[Dict]) -> Dict:
        """
        评估生成结果
        
        Args:
            results: 生成结果列表，每个包含：
                - old_code
                - generated_code
                - expected_code
                - strategy
                - confidence
                
        Returns:
            评估指标字典
        """
        self.results = results
        
        # 动态识别策略类型
        all_strategies = set(r['strategy'] for r in results)
        
        metrics = {
            'total': len(results),
            'exact_match': 0,
            'similarity': [],
            'key_api_correct': 0,
            'strategy_stats': {
                strategy: {'count': 0, 'correct': 0}
                for strategy in all_strategies
            }
        }
        
        for result in results:
            generated = result['generated_code']
            expected = result['expected_code']
            strategy = result['strategy']
            
            # 1. 精确匹配
            if self._exact_match(generated, expected):
                metrics['exact_match'] += 1
                metrics['strategy_stats'][strategy]['correct'] += 1
            
            # 2. 相似度
            sim = self._code_similarity(generated, expected)
            metrics['similarity'].append(sim)
            
            # 3. 关键API准确率
            if self._contains_key_api(generated, expected):
                metrics['key_api_correct'] += 1
            
            # 4. 策略统计
            metrics['strategy_stats'][strategy]['count'] += 1
        
        # 计算平均相似度
        if metrics['similarity']:
            metrics['avg_similarity'] = sum(metrics['similarity']) / len(metrics['similarity'])
        else:
            metrics['avg_similarity'] = 0.0
        
        # 计算准确率
        metrics['exact_match_rate'] = metrics['exact_match'] / metrics['total'] if metrics['total'] > 0 else 0
        metrics['key_api_accuracy'] = metrics['key_api_correct'] / metrics['total'] if metrics['total'] > 0 else 0
        
        return metrics
    
    def _exact_match(self, generated: str, expected: str) -> bool:
        """精确匹配（归一化后）"""
        gen_code = self._extract_code(generated)
        exp_code = expected.strip()
        
        # 归一化比较
        def normalize(code):
            return re.sub(r'\s+', '', code.lower())
        
        return normalize(gen_code) == normalize(exp_code)
    
    def _code_similarity(self, generated: str, expected: str) -> float:
        """代码相似度（0-1）"""
        gen_code = self._extract_code(generated)
        exp_code = expected.strip()
        
        # 归一化
        def normalize(code):
            return re.sub(r'\s+', ' ', code.lower().strip())
        
        norm_gen = normalize(gen_code)
        norm_exp = normalize(exp_code)
        
        return SequenceMatcher(None, norm_gen, norm_exp).ratio()
    
    def _contains_key_api(self, generated: str, expected: str) -> bool:
        """检查是否包含关键API"""
        gen_code = self._extract_code(generated).lower()
        exp_code = expected.lower()
        
        # 提取关键函数名
        key_apis = re.findall(r'\b(\w+)\s*\(', exp_code)
        
        if not key_apis:
            return False
        
        return all(api.lower() in gen_code for api in key_apis)
    
    def _extract_code(self, text: str) -> str:
        """从生成的文本中提取代码"""
        # 提取```代码块
        pattern = r"```(?:python)?\s*(.*?)\s*```"
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # 提取第一行代码
        lines = text.split('\n')
        for line in lines:
            if any(ch in line for ch in ['=', '(', '[', 'def ', 'import ']):
                return line.strip()
        
        return lines[0].strip() if lines else text.strip()
    
    def print_report(self, metrics: Dict):
        """打印评估报告"""
        console.print(Panel.fit(
            "[bold cyan]混合系统评估报告[/bold cyan]",
            border_style="cyan"
        ))
        
        # 总体指标
        console.print("\n[bold yellow]总体指标：[/bold yellow]")
        table = Table()
        table.add_column("指标", style="cyan")
        table.add_column("值", style="green")
        
        table.add_row("测试样本数", str(metrics['total']))
        table.add_row("精确匹配率", f"{metrics['exact_match_rate']*100:.1f}%")
        table.add_row("平均相似度", f"{metrics['avg_similarity']:.3f}")
        table.add_row("关键API准确率", f"{metrics['key_api_accuracy']*100:.1f}%")
        
        console.print(table)
        
        # 策略分析
        console.print("\n[bold yellow]策略效果分析：[/bold yellow]")
        strategy_table = Table()
        strategy_table.add_column("策略", style="cyan")
        strategy_table.add_column("使用次数", style="blue")
        strategy_table.add_column("正确次数", style="green")
        strategy_table.add_column("准确率", style="yellow")
        
        strategy_names = {
            'rule_direct': '规则直接应用',
            'rule_guided': '规则引导Prompt',
            'rule_applied': '规则应用',
            'llm_fallback': 'LLM兜底'
        }
        
        for strategy, stats in metrics['strategy_stats'].items():
            name = strategy_names.get(strategy, strategy)
            count = stats['count']
            correct = stats['correct']
            accuracy = f"{correct/count*100:.1f}%" if count > 0 else "N/A"
            
            strategy_table.add_row(name, str(count), str(correct), accuracy)
        
        console.print(strategy_table)
        
        # 成功案例展示
        console.print("\n[bold green]成功案例展示：[/bold green]")
        successes = []
        for i, result in enumerate(self.results):
            if self._exact_match(result['generated_code'], result['expected_code']):
                successes.append((i, result))
        
        if successes:
            console.print(f"成功案例数: {len(successes)}\n")
            for idx, (i, result) in enumerate(successes[:5], 1):  # 显示前5个成功案例
                console.print(f"[green]✓ 成功案例 {idx}:[/green]")
                console.print(f"  旧代码: {result['old_code']}")
                console.print(f"  生成: {result['generated_code']}")
                console.print(f"  期望: {result['expected_code']}")
                console.print(f"  策略: {result['strategy']}")
                console.print(f"  置信度: {result.get('confidence', 'N/A')}\n")
        else:
            console.print("[red]无成功案例[/red]\n")
        
        # 详细失败案例
        console.print("\n[bold yellow]失败案例分析：[/bold yellow]")
        failures = []
        for i, result in enumerate(self.results):
            if not self._exact_match(result['generated_code'], result['expected_code']):
                failures.append((i, result))
        
        if failures:
            console.print(f"失败案例数: {len(failures)}\n")
            for idx, (i, result) in enumerate(failures[:3], 1):  # 只显示前3个
                console.print(f"[red]失败案例 {idx}:[/red]")
                console.print(f"  旧代码: {result['old_code']}")
                console.print(f"  生成: {result['generated_code']}")
                console.print(f"  期望: {result['expected_code']}")
                console.print(f"  策略: {result['strategy']}")
                similarity = self._code_similarity(
                    result['generated_code'], result['expected_code']
                )
                console.print(f"  相似度: {similarity:.2f}\n")
        else:
            console.print("[green]✓ 全部正确！[/green]")
    
    def compare_with_baseline(
        self,
        hybrid_metrics: Dict,
        baseline_metrics: Dict
    ):
        """对比混合系统与Baseline"""
        console.print("\n[bold cyan]混合系统 vs Baseline 对比：[/bold cyan]\n")
        
        comparison_table = Table()
        comparison_table.add_column("指标", style="cyan")
        comparison_table.add_column("Baseline", style="yellow")
        comparison_table.add_column("混合系统", style="green")
        comparison_table.add_column("提升", style="magenta")
        
        # 精确匹配率
        base_em = baseline_metrics.get('exact_match_rate', 0.9) * 100
        hybrid_em = hybrid_metrics['exact_match_rate'] * 100
        improvement_em = hybrid_em - base_em
        
        comparison_table.add_row(
            "精确匹配率",
            f"{base_em:.1f}%",
            f"{hybrid_em:.1f}%",
            f"+{improvement_em:.1f}%" if improvement_em > 0 else f"{improvement_em:.1f}%"
        )
        
        # 平均相似度
        base_sim = baseline_metrics.get('avg_similarity', 0.98)
        hybrid_sim = hybrid_metrics['avg_similarity']
        improvement_sim = hybrid_sim - base_sim
        
        comparison_table.add_row(
            "平均相似度",
            f"{base_sim:.3f}",
            f"{hybrid_sim:.3f}",
            f"+{improvement_sim:.3f}" if improvement_sim > 0 else f"{improvement_sim:.3f}"
        )
        
        console.print(comparison_table)
        
        # 分析
        console.print("\n[bold]结论：[/bold]")
        if hybrid_em > base_em:
            console.print(f"[green]✓ 混合系统准确率提升 {improvement_em:.1f}%[/green]")
        else:
            console.print(f"[yellow]⚠ 混合系统准确率下降 {abs(improvement_em):.1f}%[/yellow]")


def main():
    """测试评估器"""
    # 模拟数据
    mock_results = [
        {
            'old_code': 'df.append(row)',
            'generated_code': 'pd.concat([df, row])',
            'expected_code': 'pd.concat([df, row])',
            'strategy': 'rule_direct',
            'confidence': 0.95
        },
        {
            'old_code': 'df.append(new_row, ignore_index=True)',
            'generated_code': 'pd.concat([df, new_row], ignore_index=True)',
            'expected_code': 'pd.concat([df, new_row], ignore_index=True)',
            'strategy': 'rule_guided',
            'confidence': 0.85
        },
        {
            'old_code': 'model.fit(X, y)',
            'generated_code': 'model.fit(X, y)',
            'expected_code': 'model.fit(X, y)',
            'strategy': 'llm_fallback',
            'confidence': 0.7
        }
    ]
    
    evaluator = HybridEvaluator()
    metrics = evaluator.evaluate(mock_results)
    evaluator.print_report(metrics)


if __name__ == "__main__":
    main()
