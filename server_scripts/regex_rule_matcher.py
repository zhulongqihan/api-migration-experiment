#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于正则表达式的规则匹配器 - 使用手工编写的高质量规则
"""

import re
import json
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from rich.console import Console

console = Console()


class RegexRuleMatcher:
    """基于正则表达式的规则匹配器"""
    
    def __init__(self, rule_file: str = "../configs/manual_rules.json"):
        """加载手工编写的规则"""
        self.rules = self._load_rules(rule_file)
        console.print(f"[green]✓ 加载了 {len(self.rules)} 条手工规则[/green]")
    
    def _load_rules(self, rule_file: str) -> List[Dict]:
        """加载规则文件"""
        with open(rule_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def match_and_apply(self, code: str, dependency: str = "") -> Tuple[Optional[str], float]:
        """
        匹配并应用规则
        
        Returns:
            (new_code, confidence) 如果没有匹配返回 (None, 0)
        """
        best_match = None
        best_confidence = 0
        
        for rule in self.rules:
            # 过滤依赖
            if dependency and rule.get('dependency') and rule['dependency'] != dependency:
                continue
            
            pattern = rule['pattern']
            replacement = rule['replacement']
            
            # 尝试匹配
            if re.search(pattern, code):
                # 应用替换
                try:
                    new_code = re.sub(pattern, replacement, code)
                    
                    # 验证替换结果
                    if new_code != code and self._is_valid_code(new_code):
                        confidence = rule.get('confidence', 0.8)
                        
                        if confidence > best_confidence:
                            best_match = new_code
                            best_confidence = confidence
                            
                except Exception as e:
                    console.print(f"[yellow]规则应用失败: {e}[/yellow]")
                    continue
        
        return best_match, best_confidence
    
    def _is_valid_code(self, code: str) -> bool:
        """简单验证代码是否合法"""
        # 检查括号配对
        if code.count('(') != code.count(')'):
            return False
        if code.count('[') != code.count(']'):
            return False
        if code.count('{') != code.count('}'):
            return False
        
        # 不能太短或太长
        if len(code) < 5 or len(code) > 500:
            return False
        
        # 不能包含明显的错误模式
        if '..' in code or '((' in code or '))' in code:
            return False
        
        return True
    
    def can_handle(self, code: str, dependency: str = "") -> bool:
        """判断是否有规则能处理这段代码"""
        for rule in self.rules:
            if dependency and rule.get('dependency') and rule['dependency'] != dependency:
                continue
            
            if re.search(rule['pattern'], code):
                return True
        
        return False


def main():
    """测试正则规则匹配器"""
    matcher = RegexRuleMatcher()
    
    test_cases = [
        ("df.append(row)", "pandas"),
        ("matrix = np.matrix([[1, 2]])", "numpy"),
        ("flat = tf.contrib.layers.flatten(x)", "tensorflow"),
        ("X_scaled = scaler.fit_transform(X_train)", "sklearn"),
        ("torch.save(model, 'model.pth')", "torch"),
    ]
    
    console.print("\n[bold cyan]测试正则规则匹配器[/bold cyan]\n")
    
    for code, dep in test_cases:
        console.print(f"[yellow]原代码:[/yellow] {code}")
        new_code, conf = matcher.match_and_apply(code, dep)
        
        if new_code:
            console.print(f"[green]新代码:[/green] {new_code}")
            console.print(f"[cyan]置信度:[/cyan] {conf:.2f}\n")
        else:
            console.print(f"[red]无匹配规则[/red]\n")


if __name__ == "__main__":
    main()
