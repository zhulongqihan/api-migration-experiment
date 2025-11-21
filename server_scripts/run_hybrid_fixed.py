#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版混合系统运行脚本
主要修复：
1. 使用extended_dataset_50.json（40个训练样本）
2. 修复LLM生成问题
3. 改进规则应用逻辑
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel

from data_utils import DataLoader
from rule_learner import RuleLearner
from rule_matcher import RuleMatcher
from hybrid_generator import HybridGenerator
from evaluate_hybrid import HybridEvaluator

console = Console()


def main():
    parser = argparse.ArgumentParser(description="修复版混合API迁移系统")
    parser.add_argument(
        "--data_file",
        type=str,
        default="extended_dataset_50.json",  # 使用扩展数据集
        help="数据文件名"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B",
        help="LLM模型名称"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="运行设备"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.6,  # 降低阈值，更多使用规则
        help="规则匹配置信度阈值"
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="跳过规则学习"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../results/hybrid",
        help="结果输出目录"
    )
    parser.add_argument(
        "--test_only",
        type=int,
        default=None,
        help="只测试前N个样本"
    )
    
    args = parser.parse_args()
    
    console.print(Panel.fit(
        "[bold cyan]修复版混合API迁移系统[/bold cyan]\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "修复: 数据集 + LLM生成 + 规则应用",
        border_style="cyan"
    ))
    
    console.print(f"\n[yellow]配置信息：[/yellow]")
    console.print(f"  数据文件: {args.data_file}")
    console.print(f"  模型: {args.model_name}")
    console.print(f"  设备: {args.device}")
    console.print(f"  置信度阈值: {args.confidence_threshold}")
    
    # ========== 阶段1：加载数据 ==========
    console.print("\n[bold cyan]阶段1/5: 加载数据[/bold cyan]")
    
    try:
        data_loader = DataLoader(args.data_file)
        train_data = data_loader.get_train_data()
        test_data = data_loader.get_test_data()
    except FileNotFoundError:
        console.print(f"[red]错误: 找不到数据文件 {args.data_file}[/red]")
        console.print("[yellow]提示: 请确保数据文件在scripts目录下[/yellow]")
        return
    
    console.print(f"[green]✓ 训练集: {len(train_data)} 个样本[/green]")
    console.print(f"[green]✓ 测试集: {len(test_data)} 个样本[/green]")
    
    # 检查训练集大小
    if len(train_data) < 10:
        console.print(f"[yellow]⚠ 警告: 训练集只有 {len(train_data)} 个样本，建议至少20个[/yellow]")
    
    # 限制测试样本数
    if args.test_only:
        test_data = test_data[:args.test_only]
        console.print(f"[yellow]只测试前 {len(test_data)} 个样本[/yellow]")
    
    # ========== 阶段2：规则学习 ==========
    rule_file = Path("../configs/learned_rules_fixed.json")
    
    if args.skip_training and rule_file.exists():
        console.print(f"\n[bold cyan]阶段2/5: 加载已有规则[/bold cyan]")
        rules = RuleLearner.load_rules(str(rule_file))
    else:
        console.print(f"\n[bold cyan]阶段2/5: 规则学习[/bold cyan]")
        learner = RuleLearner()
        rules = learner.learn_from_data(train_data)
        
        # 保存规则
        output_path = learner.save_rules(str(rule_file))
        
        # 打印规则详情
        console.print(f"\n[yellow]规则详情：[/yellow]")
        for i, rule in enumerate(rules[:5], 1):  # 只显示前5条
            console.print(f"  {i}. {rule['type']}: ", end="")
            if rule['type'] == 'api_replacement':
                console.print(f"{rule['old_api']} → {rule['new_api']}")
            elif rule['type'] == 'parameter_migration':
                console.print(f"移除: {rule['removed_params']}")
            else:
                console.print(f"{rule.get('structure_type', 'N/A')}")
    
    console.print(f"[green]✓ 规则库包含 {len(rules)} 条规则[/green]")
    
    # ========== 阶段3：创建混合生成器 ==========
    console.print(f"\n[bold cyan]阶段3/5: 初始化混合生成器[/bold cyan]")
    
    matcher = RuleMatcher(rules, confidence_threshold=args.confidence_threshold)
    console.print(f"[green]✓ 规则匹配器就绪（阈值: {args.confidence_threshold}）[/green]")
    
    generator = HybridGenerator(matcher, model_name=args.model_name, device=args.device)
    console.print(f"[green]✓ 混合生成器就绪[/green]")
    
    # ========== 阶段4：批量生成 ==========
    console.print(f"\n[bold cyan]阶段4/5: 批量生成更新代码[/bold cyan]")
    results = generator.batch_generate(test_data)
    console.print(f"[green]✓ 完成 {len(results)} 个样本的生成[/green]")
    
    generator.print_stats()
    
    # ========== 阶段5：评估 ==========
    console.print(f"\n[bold cyan]阶段5/5: 评估效果[/bold cyan]")
    evaluator = HybridEvaluator()
    metrics = evaluator.evaluate(results)
    evaluator.print_report(metrics)
    
    # ========== 保存结果 ==========
    console.print(f"\n[yellow]保存结果...[/yellow]")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    result_file = output_dir / f"hybrid_fixed_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': vars(args),
            'metrics': metrics,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    console.print(f"[green]✓ 结果已保存到: {result_file}[/green]")
    
    # ========== 对比分析 ==========
    console.print(f"\n[bold yellow]━━━━ 关键发现 ━━━━[/bold yellow]")
    
    rule_coverage = (
        metrics['strategy_stats']['rule_direct']['count'] +
        metrics['strategy_stats']['rule_guided']['count']
    ) / metrics['total'] if metrics['total'] > 0 else 0
    
    console.print(f"[cyan]规则覆盖率: {rule_coverage*100:.1f}%[/cyan]")
    console.print(f"[cyan]规则数量: {len(rules)} 条[/cyan]")
    console.print(f"[cyan]训练样本: {len(train_data)} 个[/cyan]")
    
    # 判断是否改进
    if metrics['exact_match_rate'] >= 0.85:
        console.print(f"\n[bold green]✅ 成功！准确率 {metrics['exact_match_rate']*100:.1f}% >= 85%[/bold green]")
    elif metrics['exact_match_rate'] >= 0.70:
        console.print(f"\n[bold yellow]⚠ 部分成功：准确率 {metrics['exact_match_rate']*100:.1f}% [70-85%)[/bold yellow]")
    else:
        console.print(f"\n[bold red]❌ 需要改进：准确率 {metrics['exact_match_rate']*100:.1f}% < 70%[/bold red]")
    
    # 最终总结
    console.print(Panel.fit(
        f"[bold]混合系统运行完成[/bold]\n\n"
        f"精确匹配率: {metrics['exact_match_rate']*100:.1f}%\n"
        f"平均相似度: {metrics['avg_similarity']:.3f}\n"
        f"规则覆盖率: {rule_coverage*100:.1f}%\n\n"
        f"结果: {result_file}",
        border_style="green" if metrics['exact_match_rate'] >= 0.85 else "yellow"
    ))
    
    return {
        'metrics': metrics,
        'results': results,
        'generator': generator,
        'evaluator': evaluator
    }


if __name__ == "__main__":
    main()
