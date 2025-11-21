#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
规则学习器 - 从训练数据中自动提取API迁移规则
"""

import ast
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from difflib import SequenceMatcher
from rich.console import Console
from rich.progress import track

console = Console()


class RuleLearner:
    """从数据中学习API迁移规则"""
    
    def __init__(self):
        self.rules = []
        self.api_replacements = defaultdict(int)
        self.parameter_changes = defaultdict(int)
        self.syntax_patterns = []
    
    def learn_from_data(self, training_data: List[Dict]) -> List[Dict]:
        """
        从训练数据学习规则
        
        Args:
            training_data: [(old_code, new_code, description), ...]
            
        Returns:
            学到的规则列表
        """
        console.print("\n[cyan]🎓 开始学习API迁移规则...[/cyan]")
        
        for example in track(training_data, description="规则学习"):
            old_code = example.get("old_code", "")
            new_code = example.get("new_code", "")
            description = example.get("description", "")
            dependency = example.get("dependency", "")
            
            try:
                # 1. 提取API替换规则
                api_rules = self._extract_api_replacement_rules(
                    old_code, new_code, dependency
                )
                for rule in api_rules:
                    self.rules.append(rule)
                
                # 2. 提取参数迁移规则
                param_rules = self._extract_parameter_rules(
                    old_code, new_code, dependency
                )
                for rule in param_rules:
                    self.rules.append(rule)
                
                # 3. 提取语法模式规则
                syntax_rule = self._extract_syntax_pattern(
                    old_code, new_code, description
                )
                if syntax_rule:
                    self.rules.append(syntax_rule)
                    
            except Exception as e:
                console.print(f"[yellow]⚠ 规则提取失败: {e}[/yellow]")
                continue
        
        # 4. 规则去重与泛化
        self.rules = self._deduplicate_rules(self.rules)
        
        console.print(f"\n[green]✓ 学到 {len(self.rules)} 条规则[/green]")
        self._print_rule_summary()
        
        return self.rules
    
    def _extract_api_replacement_rules(
        self, old_code: str, new_code: str, dependency: str
    ) -> List[Dict]:
        """提取API替换规则（基于代码对比）"""
        rules = []
        
        try:
            # 提取API调用
            old_funcs = self._extract_function_calls(old_code)
            new_funcs = self._extract_function_calls(new_code)
            
            # 策略1：直接替换（完整API路径匹配）
            for old_api in old_funcs:
                if old_api in old_code and old_api not in new_code:
                    # 寻找新代码中对应的替换
                    for new_api in new_funcs:
                        if new_api in new_code and new_api not in old_code:
                            # 构建转换模板
                            rule = self._build_transformation_rule(
                                old_code, new_code, old_api, new_api, dependency
                            )
                            if rule:
                                rules.append(rule)
                                self.api_replacements[(old_api, new_api)] += 1
                                break
            
            # 策略2：如果没找到直接替换，尝试识别结构性变化
            if not rules:
                structural_rule = self._extract_structural_change(
                    old_code, new_code, dependency
                )
                if structural_rule:
                    rules.append(structural_rule)
        except Exception as e:
            console.print(f"[yellow]API规则提取失败: {e}[/yellow]")
        
        return rules
    
    def _build_transformation_rule(
        self, old_code: str, new_code: str, old_api: str, new_api: str, dependency: str
    ) -> Dict:
        """构建转换规则（支持复杂模式和参数映射）"""
        # 分析转换类型
        transform_type = self._analyze_transform_type(old_code, new_code, old_api, new_api)
        
        # 提取参数映射（用于模板应用）
        param_mapping = self._extract_parameter_mapping(old_code, new_code, old_api, new_api)
        
        rule = {
            "type": "api_replacement",
            "dependency": dependency,
            "old_api": old_api,
            "new_api": new_api,
            "transform_type": transform_type,
            "old_code_template": old_code,
            "new_code_template": new_code,
            "param_mapping": param_mapping,  # 新增：参数映射
            "confidence": 0.9,
            "examples": [(old_code, new_code)]
        }
        
        return rule
    
    def _extract_parameter_mapping(self, old_code: str, new_code: str, old_api: str, new_api: str) -> Dict:
        """提取参数映射关系"""
        mapping = {}
        
        try:
            # 提取旧代码中的参数
            old_args_match = re.search(rf'{re.escape(old_api)}\((.*?)\)', old_code)
            if old_args_match:
                old_args = old_args_match.group(1).strip()
                
                # 提取新代码中的参数
                new_args_match = re.search(rf'{re.escape(new_api)}\((.*?)\)', new_code)
                if new_args_match:
                    new_args = new_args_match.group(1).strip()
                    
                    # 特殊处理：df.append(row) → pd.concat([df, row])
                    if 'append' in old_api and 'concat' in new_api:
                        # 提取对象名（df）
                        obj_match = re.search(r'(\w+)\.append', old_code)
                        if obj_match:
                            obj_name = obj_match.group(1)
                            mapping['obj'] = obj_name
                            mapping['args'] = old_args
                            mapping['pattern'] = 'append_to_concat'
                    
                    # 通用映射
                    mapping['old_args'] = old_args
                    mapping['new_args'] = new_args
        except:
            pass
        
        return mapping
    
    def _analyze_transform_type(self, old_code: str, new_code: str, old_api: str, new_api: str) -> str:
        """分析转换类型"""
        # 检测特殊模式
        if 'fit_transform' in old_api and 'fit' in new_api:
            return 'method_chain_split'  # fit_transform → fit().transform()
        
        if old_api.count('.') > new_api.count('.'):
            return 'path_simplification'  # tf.contrib.xxx → tf.keras.xxx
        
        if old_api.count('.') < new_api.count('.'):
            return 'path_expansion'
        
        if '(' in new_code and new_code.count('(') > old_code.count('('):
            return 'call_wrapping'  # flatten(x) → Flatten()(x)
        
        return 'direct_replacement'
    
    def _extract_structural_change(
        self, old_code: str, new_code: str, dependency: str
    ) -> Dict:
        """提取结构性变化规则"""
        # 识别参数名变化
        old_params = set(re.findall(r'(\w+)\s*=\s*', old_code))
        new_params = set(re.findall(r'(\w+)\s*=\s*', new_code))
        
        if old_params != new_params:
            changed = {
                'removed': list(old_params - new_params),
                'added': list(new_params - old_params)
            }
            
            return {
                "type": "structural_change",
                "dependency": dependency,
                "change_type": "parameter_rename",
                "changes": changed,
                "old_template": old_code,
                "new_template": new_code,
                "confidence": 0.85,
                "examples": [(old_code, new_code)]
            }
        
        return None
    
    def _extract_parameter_rules(
        self, old_code: str, new_code: str, dependency: str
    ) -> List[Dict]:
        """提取参数迁移规则"""
        rules = []
        
        try:
            # 使用正则提取参数
            old_params = set(re.findall(r'(\w+)\s*=', old_code))
            new_params = set(re.findall(r'(\w+)\s*=', new_code))
            
            # 找到消失的参数
            removed_params = old_params - new_params
            # 找到新增的参数
            added_params = new_params - old_params
            
            if removed_params or added_params:
                rule = {
                    "type": "parameter_migration",
                    "dependency": dependency,
                    "removed_params": list(removed_params),
                    "added_params": list(added_params),
                    "confidence": 0.8,
                    "examples": [(old_code, new_code)]
                }
                rules.append(rule)
        except:
            pass
        
        return rules
    
    def _extract_syntax_pattern(
        self, old_code: str, new_code: str, description: str
    ) -> Dict:
        """提取语法模式规则"""
        try:
            # 归一化代码（去除空格、变量名）
            old_normalized = self._normalize_code(old_code)
            new_normalized = self._normalize_code(new_code)
            
            if old_normalized != new_normalized:
                # 检测结构性变化
                structure_type = self._detect_structure_change(
                    old_code, new_code
                )
                
                rule = {
                    "type": "syntax_pattern",
                    "old_pattern": old_normalized,
                    "new_pattern": new_normalized,
                    "structure_type": structure_type,
                    "description": description,
                    "confidence": 0.75,
                    "examples": [(old_code, new_code)]
                }
                return rule
        except:
            pass
        
        return None
    
    def _extract_function_calls(self, code: str) -> List[str]:
        """提取函数调用（优先提取完整路径）"""
        funcs = []
        
        # 方法1：提取完整API路径（如 tf.contrib.layers.flatten, pd.concat）
        full_api_pattern = r'([a-zA-Z_][\w\.]+)\s*\('
        full_matches = re.findall(full_api_pattern, code)
        
        # 过滤：只保留包含点号的完整路径，或numpy/pandas/tf等库的调用
        for match in full_matches:
            if '.' in match:  # 完整路径
                funcs.append(match)
        
        # 方法2：AST提取（用于获取更准确的调用信息）
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # 提取完整调用路径
                    call_path = self._get_call_path(node.func)
                    if call_path:
                        funcs.append(call_path)
        except:
            pass
        
        return list(set(funcs))
    
    def _get_call_path(self, node) -> str:
        """从AST节点提取完整调用路径"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            parent = self._get_call_path(node.value)
            if parent:
                return f"{parent}.{node.attr}"
            return node.attr
        return ""
    
    def _normalize_code(self, code: str) -> str:
        """归一化代码（用于模式匹配）"""
        # 去除空格
        code = re.sub(r'\s+', ' ', code).strip()
        # 替换变量名为占位符
        code = re.sub(r'\b[a-z_]\w*\b', 'VAR', code)
        return code
    
    def _detect_structure_change(self, old_code: str, new_code: str) -> str:
        """检测结构性变化类型"""
        # 检测是否有赋值包装
        if '=' in new_code and '=' not in old_code:
            return "assignment_wrapping"
        
        # 检测是否有列表包装
        if '[' in new_code and '[' not in old_code:
            return "list_wrapping"
        
        # 检测是否有函数嵌套
        if new_code.count('(') > old_code.count('('):
            return "function_nesting"
        
        return "unknown"
    
    def _deduplicate_rules(self, rules: List[Dict]) -> List[Dict]:
        """规则去重与合并"""
        unique_rules = {}
        
        for rule in rules:
            rule_type = rule['type']
            
            if rule_type == 'api_replacement':
                key = (rule['old_api'], rule['new_api'])
                if key in unique_rules:
                    # 合并示例
                    unique_rules[key]['examples'].extend(rule['examples'])
                    # 更新置信度
                    unique_rules[key]['confidence'] = min(
                        1.0, unique_rules[key]['confidence'] + 0.05
                    )
                else:
                    unique_rules[key] = rule
            
            elif rule_type == 'parameter_migration':
                key = (
                    rule['dependency'],
                    tuple(sorted(rule['removed_params'])),
                    tuple(sorted(rule['added_params']))
                )
                if key in unique_rules:
                    unique_rules[key]['examples'].extend(rule['examples'])
                    unique_rules[key]['confidence'] = min(
                        1.0, unique_rules[key]['confidence'] + 0.05
                    )
                else:
                    unique_rules[key] = rule
            
            elif rule_type == 'syntax_pattern':
                key = (rule['old_pattern'], rule['new_pattern'])
                if key in unique_rules:
                    unique_rules[key]['examples'].extend(rule['examples'])
                    unique_rules[key]['confidence'] = min(
                        1.0, unique_rules[key]['confidence'] + 0.05
                    )
                else:
                    unique_rules[key] = rule
        
        return list(unique_rules.values())
    
    def _print_rule_summary(self):
        """打印规则摘要"""
        from rich.table import Table
        
        table = Table(title="学到的规则统计")
        table.add_column("规则类型", style="cyan")
        table.add_column("数量", style="green")
        table.add_column("示例", style="yellow")
        
        # 统计各类规则
        rule_counts = defaultdict(int)
        rule_examples = defaultdict(list)
        
        for rule in self.rules:
            rule_type = rule['type']
            rule_counts[rule_type] += 1
            
            if rule_type == 'api_replacement':
                example = f"{rule['old_api']} → {rule['new_api']}"
            elif rule_type == 'parameter_migration':
                example = f"移除: {rule['removed_params']}, 新增: {rule['added_params']}"
            elif rule_type == 'syntax_pattern':
                example = rule['structure_type']
            else:
                example = "N/A"
            
            if example not in rule_examples[rule_type]:
                rule_examples[rule_type].append(example)
        
        for rule_type, count in rule_counts.items():
            examples = rule_examples[rule_type][:2]  # 只显示前2个
            table.add_row(
                rule_type,
                str(count),
                "; ".join(examples)
            )
        
        console.print(table)
    
    def save_rules(self, output_file: str = "../configs/learned_rules.json"):
        """保存规则库"""
        # 确保输出目录存在
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备保存的数据（去除examples以减小文件大小）
        rules_to_save = []
        for rule in self.rules:
            rule_copy = rule.copy()
            # 只保留examples的数量
            rule_copy['example_count'] = len(rule.get('examples', []))
            rule_copy.pop('examples', None)
            rules_to_save.append(rule_copy)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rules_to_save, f, indent=2, ensure_ascii=False)
        
        console.print(f"\n[green]✓ 规则库已保存到: {output_path}[/green]")
        
        return output_path
    
    @staticmethod
    def load_rules(rule_file: str) -> List[Dict]:
        """加载规则库"""
        with open(rule_file, 'r', encoding='utf-8') as f:
            rules = json.load(f)
        
        console.print(f"[green]✓ 加载了 {len(rules)} 条规则[/green]")
        return rules


def main():
    """测试规则学习器"""
    from data_utils import DataLoader
    
    console.print("[bold cyan]规则学习器测试[/bold cyan]\n")
    
    # 1. 加载训练数据
    data_loader = DataLoader("mini_dataset.json")
    train_data = data_loader.get_train_data()
    console.print(f"✓ 加载了 {len(train_data)} 个训练样本\n")
    
    # 2. 学习规则
    learner = RuleLearner()
    rules = learner.learn_from_data(train_data)
    
    # 3. 保存规则
    learner.save_rules()
    
    # 4. 测试加载
    console.print("\n[yellow]测试规则加载...[/yellow]")
    loaded_rules = RuleLearner.load_rules("../configs/learned_rules.json")
    console.print(f"[green]✓ 成功加载 {len(loaded_rules)} 条规则[/green]")


if __name__ == "__main__":
    main()
