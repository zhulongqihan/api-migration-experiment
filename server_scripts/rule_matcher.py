#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
规则匹配器 - 匹配适用的规则并应用到新代码
"""

import ast
import re
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
from rich.console import Console

console = Console()


class RuleMatcher:
    """规则匹配与应用器"""
    
    def __init__(self, rules: List[Dict], confidence_threshold: float = 0.7):
        """
        Args:
            rules: 规则库
            confidence_threshold: 置信度阈值（低于此值不应用规则）
        """
        self.rules = rules
        self.confidence_threshold = confidence_threshold
        
        # 按类型组织规则
        self.api_rules = [r for r in rules if r['type'] == 'api_replacement']
        self.param_rules = [r for r in rules if r['type'] == 'parameter_migration']
        self.syntax_rules = [r for r in rules if r['type'] == 'syntax_pattern']
    
    def match_rules(self, old_code: str) -> List[Tuple[Dict, float]]:
        """
        匹配适用的规则
        
        Args:
            old_code: 旧代码
            
        Returns:
            [(rule, confidence), ...] 按置信度降序排列
        """
        matched_rules = []
        
        # 1. 匹配API替换规则
        for rule in self.api_rules:
            score = self._match_api_rule(old_code, rule)
            if score > 0:
                matched_rules.append((rule, score))
        
        # 2. 匹配参数迁移规则
        for rule in self.param_rules:
            score = self._match_param_rule(old_code, rule)
            if score > 0:
                matched_rules.append((rule, score))
        
        # 3. 匹配语法模式规则
        for rule in self.syntax_rules:
            score = self._match_syntax_rule(old_code, rule)
            if score > 0:
                matched_rules.append((rule, score))
        
        # 按置信度降序排序
        matched_rules.sort(key=lambda x: x[1], reverse=True)
        
        return matched_rules
    
    def _match_api_rule(self, code: str, rule: Dict) -> float:
        """匹配API替换规则"""
        old_api = rule['old_api']
        
        # 检查旧API是否出现在代码中
        if old_api in code:
            # 完全匹配 → 高置信度
            return rule['confidence']
        
        # 模糊匹配
        funcs = self._extract_function_calls(code)
        for func in funcs:
            similarity = SequenceMatcher(None, func, old_api).ratio()
            if similarity > 0.8:
                return rule['confidence'] * similarity
        
        return 0.0
    
    def _match_param_rule(self, code: str, rule: Dict) -> float:
        """匹配参数迁移规则"""
        removed_params = rule.get('removed_params', [])
        
        # 检查是否包含需要移除的参数
        for param in removed_params:
            if f"{param}=" in code:
                return rule['confidence']
        
        return 0.0
    
    def _match_syntax_rule(self, code: str, rule: Dict) -> float:
        """匹配语法模式规则"""
        old_pattern = rule['old_pattern']
        
        # 归一化代码
        normalized_code = self._normalize_code(code)
        
        # 计算相似度
        similarity = SequenceMatcher(None, normalized_code, old_pattern).ratio()
        
        if similarity > 0.6:
            return rule['confidence'] * similarity
        
        return 0.0
    
    def apply_rules(
        self, 
        old_code: str, 
        matched_rules: List[Tuple[Dict, float]]
    ) -> Optional[str]:
        """
        应用匹配的规则
        
        Args:
            old_code: 旧代码
            matched_rules: 匹配的规则列表
            
        Returns:
            更新后的代码，如果无法应用返回None
        """
        if not matched_rules:
            return None
        
        new_code = old_code
        
        for rule, confidence in matched_rules:
            if confidence < self.confidence_threshold:
                continue
            
            try:
                if rule['type'] == 'api_replacement':
                    new_code = self._apply_api_rule(new_code, rule)
                elif rule['type'] == 'parameter_migration':
                    new_code = self._apply_param_rule(new_code, rule)
                elif rule['type'] == 'structural_change':
                    new_code = self._apply_structural_rule(new_code, rule)
                elif rule['type'] == 'syntax_pattern':
                    new_code = self._apply_syntax_rule(new_code, rule)
            except Exception as e:
                console.print(f"[yellow]⚠ 规则应用失败: {e}[/yellow]")
                continue
        
        return new_code if new_code != old_code else None
    
    def _apply_api_rule(self, code: str, rule: Dict) -> str:
        """应用API替换规则（基于转换类型）"""
        import re
        
        old_api = rule['old_api']
        new_api = rule['new_api']
        transform_type = rule.get('transform_type', 'direct_replacement')
        
        # 根据转换类型应用不同的转换策略
        if transform_type == 'method_chain_split':
            # fit_transform() → fit().transform()
            return self._apply_chain_split(code, old_api, new_api)
        
        elif transform_type == 'call_wrapping':
            # flatten(x) → Flatten()(x)
            return self._apply_call_wrapping(code, rule)
        
        elif transform_type == 'path_simplification' or transform_type == 'path_expansion':
            # tf.contrib.layers.flatten → tf.keras.layers.Flatten
            return self._apply_path_change(code, old_api, new_api)
        
        elif transform_type == 'direct_replacement':
            # 直接替换完整路径
            return self._apply_direct_replacement(code, old_api, new_api)
        
        else:
            # 默认：尝试直接替换
            return self._apply_direct_replacement(code, old_api, new_api)
    
    def _apply_chain_split(self, code: str, old_api: str, new_api: str) -> str:
        """链式调用分离：fit_transform(X) → fit(X).transform(X)"""
        import re
        
        # 匹配 obj.fit_transform(args)
        pattern = r'(\w+)\.fit_transform\(([^)]+)\)'
        match = re.search(pattern, code)
        
        if match:
            obj_name = match.group(1)
            args = match.group(2)
            # 保留前缀
            prefix = code[:match.start()]
            suffix = code[match.end():]
            return f"{prefix}{obj_name}.fit({args}).transform({args}){suffix}"
        
        return code
    
    def _apply_call_wrapping(self, code: str, rule: Dict) -> str:
        """调用包装：flatten(x) → Flatten()(x)"""
        import re
        
        old_template = rule.get('old_code_template', '')
        new_template = rule.get('new_code_template', '')
        
        # 直接使用模板（如果代码结构相似）
        if old_template and new_template:
            # 计算相似度
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, code, old_template).ratio()
            
            if similarity > 0.6:
                # 直接返回新模板
                return new_template
        
        return code
    
    def _apply_path_change(self, code: str, old_api: str, new_api: str) -> str:
        """路径变化：完整API路径替换"""
        # 直接替换完整路径
        if old_api in code:
            return code.replace(old_api, new_api)
        return code
    
    def _apply_direct_replacement(self, code: str, old_api: str, new_api: str) -> str:
        """直接替换（考虑参数映射）"""
        import re
        
        # 特殊处理：df.append(row) → pd.concat([df, row])
        if 'append' in old_api and 'concat' in new_api:
            match = re.search(r'(\w+)\.append\((.*?)\)', code)
            if match:
                obj_name = match.group(1)
                args = match.group(2).strip()
                # 保留其他参数（如ignore_index）
                extra_params = ''
                if ',' in args:
                    parts = args.split(',', 1)
                    args = parts[0].strip()
                    extra_params = ', ' + parts[1].strip()
                return code[:match.start()] + f"pd.concat([{obj_name}, {args}]{extra_params})" + code[match.end():]
        
        # 优先完整路径替换
        if old_api in code:
            return code.replace(old_api, new_api)
        
        # 否则词边界替换
        pattern = r'\b' + re.escape(old_api.split('.')[-1]) + r'\b'
        return re.sub(pattern, new_api.split('.')[-1], code)
    
    def _apply_structural_rule(self, code: str, rule: Dict) -> str:
        """应用结构性变化规则"""
        change_type = rule.get('change_type', '')
        
        if change_type == 'parameter_rename':
            changes = rule.get('changes', {})
            removed = changes.get('removed', [])
            added = changes.get('added', [])
            
            # 简单的参数重命名（1对1映射）
            if len(removed) == 1 and len(added) == 1:
                old_param = removed[0]
                new_param = added[0]
                # 替换参数名
                pattern = rf'\b{old_param}\s*='
                replacement = f'{new_param}='
                return re.sub(pattern, replacement, code)
        
        return code
    
    def _apply_param_rule(self, code: str, rule: Dict) -> str:
        """应用参数迁移规则"""
        new_code = code
        
        # 移除废弃的参数
        removed_params = rule.get('removed_params', [])
        for param in removed_params:
            # 匹配 param=value 模式（包括可能的逗号）
            pattern = rf',?\s*{param}\s*=\s*[^,)]+,?'
            new_code = re.sub(pattern, '', new_code)
        
        # 清理多余的逗号和空格
        new_code = re.sub(r',\s*,', ',', new_code)
        new_code = re.sub(r'\(\s*,', '(', new_code)
        new_code = re.sub(r',\s*\)', ')', new_code)
        
        return new_code
    
    def _apply_syntax_rule(self, code: str, rule: Dict) -> str:
        """应用语法模式规则"""
        structure_type = rule.get('structure_type', 'unknown')
        
        # 根据结构类型应用不同的转换
        if structure_type == 'assignment_wrapping':
            # 提取变量名（假设是第一个标识符）
            match = re.match(r'(\w+)', code.strip())
            if match:
                var_name = match.group(1)
                # 如果还没有赋值，添加赋值
                if not code.strip().startswith(f"{var_name} ="):
                    return f"{var_name} = {code}"
        
        elif structure_type == 'list_wrapping':
            # 检查是否需要列表包装
            if '[' not in code:
                # 简单的列表包装（需要根据具体情况调整）
                pass
        
        return code
    
    def _extract_function_calls(self, code: str) -> List[str]:
        """提取代码中的函数调用"""
        funcs = []
        
        # 正则提取
        pattern = r'(\w+(?:\.\w+)*)\s*\('
        matches = re.findall(pattern, code)
        funcs.extend(matches)
        
        return list(set(funcs))
    
    def _normalize_code(self, code: str) -> str:
        """归一化代码"""
        code = re.sub(r'\s+', ' ', code).strip()
        code = re.sub(r'\b[a-z_]\w*\b', 'VAR', code)
        return code
    
    def get_best_rule(self, old_code: str) -> Optional[Tuple[Dict, float]]:
        """获取最佳匹配规则"""
        matched_rules = self.match_rules(old_code)
        
        if matched_rules and matched_rules[0][1] >= self.confidence_threshold:
            return matched_rules[0]
        
        return None
    
    def can_apply_directly(self, old_code: str) -> bool:
        """判断是否可以直接应用规则（高置信度）"""
        best_rule = self.get_best_rule(old_code)
        
        if best_rule and best_rule[1] >= 0.85:
            return True
        
        return False


def main():
    """测试规则匹配器"""
    from rule_learner import RuleLearner
    
    console.print("[bold cyan]规则匹配器测试[/bold cyan]\n")
    
    # 1. 加载规则
    try:
        rules = RuleLearner.load_rules("../configs/learned_rules.json")
    except FileNotFoundError:
        console.print("[red]错误: 请先运行 rule_learner.py 生成规则库[/red]")
        return
    
    # 2. 创建匹配器
    matcher = RuleMatcher(rules, confidence_threshold=0.7)
    console.print(f"✓ 加载了 {len(rules)} 条规则")
    console.print(f"  - API替换规则: {len(matcher.api_rules)}")
    console.print(f"  - 参数迁移规则: {len(matcher.param_rules)}")
    console.print(f"  - 语法模式规则: {len(matcher.syntax_rules)}\n")
    
    # 3. 测试匹配
    test_cases = [
        "df.append(row)",
        "df.append(new_row, ignore_index=True)",
        "result = model.fit(X, y, verbose=1)"
    ]
    
    for i, test_code in enumerate(test_cases, 1):
        console.print(f"[yellow]测试 {i}: {test_code}[/yellow]")
        
        # 匹配规则
        matched = matcher.match_rules(test_code)
        
        if matched:
            console.print(f"  匹配到 {len(matched)} 条规则:")
            for rule, score in matched[:3]:  # 只显示前3条
                console.print(f"    - {rule['type']}: {score:.2f}")
                if rule['type'] == 'api_replacement':
                    console.print(f"      {rule['old_api']} → {rule['new_api']}")
            
            # 应用规则
            new_code = matcher.apply_rules(test_code, matched)
            if new_code:
                console.print(f"  [green]→ {new_code}[/green]")
            else:
                console.print(f"  [yellow]→ 无法应用规则[/yellow]")
        else:
            console.print("  [yellow]未匹配到规则[/yellow]")
        
        console.print()


if __name__ == "__main__":
    main()
