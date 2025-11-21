#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试规则引导生成"""

from hybrid_generator import HybridGenerator
from rule_matcher import RuleMatcher
from rule_learner import RuleLearner
from rich.console import Console

console = Console()

def test_rule_guided_generation():
    """测试规则引导生成是否能正常工作"""
    
    console.print("[bold cyan]测试规则引导生成[/bold cyan]\n")
    
    # 1. 加载规则
    try:
        rules = RuleLearner.load_rules("../configs/learned_rules.json")
        console.print(f"✓ 加载了 {len(rules)} 条规则\n")
    except FileNotFoundError:
        console.print("[red]错误: 规则文件不存在[/red]")
        return
    
    # 2. 创建matcher和generator
    matcher = RuleMatcher(rules, confidence_threshold=0.6)
    generator = HybridGenerator(matcher, device="cuda")
    
    # 3. 测试案例
    test_cases = [
        {
            "old_code": "input.cuda()",
            "description": "设备迁移",
            "dependency": "torch",
            "expected": "input.to('cuda')"
        },
        {
            "old_code": "df.append(row)",
            "description": "DataFrame操作",
            "dependency": "pandas",
            "expected": "pd.concat([df, row])"
        },
        {
            "old_code": "df.sort('col')",
            "description": "排序",
            "dependency": "pandas",
            "expected": "df.sort_values('col')"
        },
    ]
    
    console.print("[yellow]测试案例：[/yellow]\n")
    
    for i, case in enumerate(test_cases, 1):
        console.print(f"[cyan]案例 {i}:[/cyan]")
        console.print(f"  旧代码: {case['old_code']}")
        console.print(f"  期望: {case['expected']}")
        
        # 生成
        new_code, strategy, confidence = generator.generate(
            case['old_code'],
            case['description'],
            case['dependency']
        )
        
        console.print(f"  生成: {new_code}")
        console.print(f"  策略: {strategy}")
        console.print(f"  置信度: {confidence:.2f}")
        
        # 判断是否成功
        if new_code == case['expected']:
            console.print(f"  [green]✓ 精确匹配[/green]\n")
        elif new_code and new_code != case['old_code']:
            console.print(f"  [yellow]~ 生成了新代码[/yellow]\n")
        else:
            console.print(f"  [red]✗ 生成失败（返回原代码或空）[/red]\n")
    
    # 4. 打印统计
    generator.print_stats()

if __name__ == "__main__":
    test_rule_guided_generation()
