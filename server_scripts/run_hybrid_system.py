#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合系统完整运行脚本 - 方向3深化实现
规则学习 + 规则匹配 + 混合生成 + 评估
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.panel import Panel

# 导入自定义模块
from data_utils import DataLoader
from rule_learner import RuleLearner
from rule_matcher import RuleMatcher
from hybrid_generator import HybridGenerator
from evaluate_hybrid import HybridEvaluator

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="混合API迁移系统 - 方向3完整实现"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default="mini_dataset.json",
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
        default=0.7,
        help="规则匹配置信度阈值"
    )
    parser.add_argument(
        "--skip_training",
        action="store_true",
        help="跳过规则学习（使用已有规则库）"
    )
    parser.add_argument(
    console.print(Panel.fit(
        "[bold cyan]混合API迁移系统[/bold cyan]\n"
        "方向3：规则提取 + Prompt工程\n"
        "[dim]━━━━━━━━━━━━━━━━━━━━━━━━━[/dim]\n"
        "规则学习 → 规则匹配 → 混合生成 → 评估",
        border_style="cyan"
    ))

    # 配置（支持命令行参数）
    data_file = sys.argv[1] if len(sys.argv) > 1 else 'mini_dataset.json'

    config = {
        'data_file': data_file,
        'model_name': 'Qwen/Qwen2.5-Coder-1.5B',
        'device': 'auto',
        'confidence_threshold': 0.7,
        'output_dir': '../results/hybrid'
    }

    # 显示配置
    console.print(f"\n[yellow]配置信息：[/yellow]")
    console.print(f"  数据文件: {config['data_file']}")
    console.print(f"  模型: {config['model_name']}")
    console.print(f"  设备: {config['device']}")
    console.print(f"  置信度阈值: {config['confidence_threshold']}")
    console.print(f"  输出目录: {config['output_dir']}\n")

    # ========== 阶段1：加载数据 ==========
    console.print("[bold cyan]阶段1/5: 加载数据[/bold cyan]")
    data_loader = DataLoader(config['data_file'])
    train_data = data_loader.get_train_data()
    test_data = data_loader.get_test_data()
    
    console.print(f"[green]✓ 训练集: {len(train_data)} 个样本[/green]")
    console.print(f"[green]✓ 测试集: {len(test_data)} 个样本[/green]")
    
    # ========== 阶段2：规则学习 ==========
    rule_file = Path("../configs/learned_rules.json")
    
    if args.skip_training and rule_file.exists():
        console.print(f"\n[bold cyan]阶段2/5: 加载已有规则[/bold cyan]")
        rules = RuleLearner.load_rules(str(rule_file))
    else:
        console.print(f"\n[bold cyan]阶段2/5: 规则学习[/bold cyan]")
        learner = RuleLearner()
        rules = learner.learn_from_data(train_data)
        learner.save_rules(str(rule_file))
    
    console.print(f"[green]✓ 规则库包含 {len(rules)} 条规则[/green]")
    
    # ========== 阶段3：创建混合生成器 ==========
    console.print(f"\n[bold cyan]阶段3/5: 初始化混合生成器[/bold cyan]")
    
    # 创建规则匹配器
    matcher = RuleMatcher(rules, confidence_threshold=args.confidence_threshold)
    console.print(f"[green]✓ 规则匹配器就绪[/green]")
    
    # 创建混合生成器
    generator = HybridGenerator(matcher, model_name=args.model_name, device=args.device)
    console.print(f"[green]✓ 混合生成器就绪[/green]")
    
    # ========== 阶段4：批量生成 ==========
    console.print(f"\n[bold cyan]阶段4/5: 批量生成更新代码[/bold cyan]")
    results = generator.batch_generate(test_data)
    console.print(f"[green]✓ 完成 {len(results)} 个样本的生成[/green]")
    
    # 显示生成统计
    generator.print_stats()
    
    # ========== 阶段5：评估 ==========
    console.print(f"\n[bold cyan]阶段5/5: 评估效果[/bold cyan]")
    evaluator = HybridEvaluator()
    metrics = evaluator.evaluate(results)
    evaluator.print_report(metrics)
    
    # ========== 保存结果 ==========
    console.print(f"\n[yellow]保存结果...[/yellow]")
    
    # 确保输出目录存在
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存详细结果
    result_file = output_dir / f"hybrid_results_{timestamp}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': vars(args),
            'metrics': metrics,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    console.print(f"[green]✓ 结果已保存到: {result_file}[/green]")
    
    # 保存摘要
    summary_file = output_dir / f"hybrid_summary_{timestamp}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("混合API迁移系统 - 运行摘要\n")
        f.write("="*80 + "\n\n")
        f.write(f"时间: {timestamp}\n")
        f.write(f"数据文件: {args.data_file}\n")
        f.write(f"模型: {args.model_name}\n\n")
        
        f.write("评估结果:\n")
        f.write("-"*80 + "\n")
        f.write(f"测试样本数: {metrics['total']}\n")
        f.write(f"精确匹配率: {metrics['exact_match_rate']*100:.1f}%\n")
        f.write(f"平均相似度: {metrics['avg_similarity']:.3f}\n")
        f.write(f"关键API准确率: {metrics['key_api_accuracy']*100:.1f}%\n\n")
        
        f.write("策略统计:\n")
        f.write("-"*80 + "\n")
        for strategy, stats in metrics['strategy_stats'].items():
            count = stats['count']
            correct = stats['correct']
            f.write(f"{strategy}: {count} 次使用, {correct} 次正确\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("详细结果请查看: " + str(result_file) + "\n")
    
    console.print(f"[green]✓ 摘要已保存到: {summary_file}[/green]")
    
    # ========== 最终总结 ==========
    console.print(Panel.fit(
        f"[bold green]✅ 混合系统运行完成！[/bold green]\n\n"
        f"主要指标:\n"
        f"  • 精确匹配率: {metrics['exact_match_rate']*100:.1f}%\n"
        f"  • 平均相似度: {metrics['avg_similarity']:.3f}\n"
        f"  • 关键API准确率: {metrics['key_api_accuracy']*100:.1f}%\n\n"
        f"策略分布:\n"
        f"  • 规则直接: {metrics['strategy_stats']['rule_direct']['count']} 次\n"
        f"  • 规则引导: {metrics['strategy_stats']['rule_guided']['count']} 次\n"
        f"  • LLM兜底: {metrics['strategy_stats']['llm_fallback']['count']} 次\n\n"
        f"结果文件: {result_file}",
        border_style="green"
    ))
    
    # 返回结果供进一步分析
    return {
        'metrics': metrics,
        'results': results,
        'generator': generator,
        'evaluator': evaluator
    }


if __name__ == "__main__":
    main()
