#!/usr/bin/env python3
"""规则提取模块 - 从代码对中提取API更新规则"""

from typing import Dict, List

class APIUpdateRule:
    """API更新规则"""
    
    def __init__(self, rule_type: str, pattern: Dict, replacement: Dict):
        self.rule_type = rule_type
        self.pattern = pattern
        self.replacement = replacement
    
    def to_dict(self) -> Dict:
        return {
            "type": self.rule_type,
            "pattern": self.pattern,
            "replacement": self.replacement,
        }
    
    def __repr__(self):
        return f"Rule({self.rule_type}: {self.pattern} -> {self.replacement})"

class RuleExtractor:
    """规则提取器"""
    
    def extract_from_pair(self, old_code: str, new_code: str) -> List[APIUpdateRule]:
        """从旧代码和新代码中提取规则"""
        rules = []
        
        # 规则1: 函数替换（append -> concat）
        if "append" in old_code and "concat" in new_code:
            rules.append(APIUpdateRule(
                rule_type="function_replacement",
                pattern={"function": "append"},
                replacement={"function": "concat"},
            ))
        
        # 规则2: 参数添加（timeout）
        if "timeout" in new_code and "timeout" not in old_code:
            rules.append(APIUpdateRule(
                rule_type="parameter_add",
                pattern={"param": None},
                replacement={"param": "timeout"},
            ))
        
        # 规则3: 参数变化（keepdims）
        if "keepdims" in old_code or "keepdims" in new_code:
            rules.append(APIUpdateRule(
                rule_type="parameter_change",
                pattern={"param": "keepdims"},
                replacement={"param": "keepdims", "default_changed": True},
            ))
        
        return rules
    
    def build_rule_library(self, dataset: List[Dict]) -> Dict[str, List[Dict]]:
        """构建规则库"""
        rule_library = {}
        
        for item in dataset:
            dependency = item.get("dependency", "unknown")
            old_code = item.get("old_code", "")
            new_code = item.get("new_code", "")
            
            rules = self.extract_from_pair(old_code, new_code)
            
            if dependency not in rule_library:
                rule_library[dependency] = []
            
            for rule in rules:
                rule_dict = rule.to_dict()
                # 避免重复规则
                if rule_dict not in rule_library[dependency]:
                    rule_library[dependency].append(rule_dict)
        
        return rule_library

if __name__ == "__main__":
    print("="*60)
    print("🔧 规则提取器测试")
    print("="*60)
    
    extractor = RuleExtractor()
    
    # 测试案例
    test_cases = [
        ("df.append(new_row)", "pd.concat([df, new_row])"),
        ("requests.get(url)", "requests.get(url, timeout=30)"),
        ("np.sum(arr, keepdims=True)", "np.sum(arr, keepdims=False)"),
    ]
    
    print("\n测试规则提取:")
    for i, (old, new) in enumerate(test_cases, 1):
        print(f"\n案例{i}:")
        print(f"  旧: {old}")
        print(f"  新: {new}")
        rules = extractor.extract_from_pair(old, new)
        print(f"  规则: {len(rules)} 条")
        for rule in rules:
            print(f"    - {rule}")
    
    print("\n" + "="*60)
    print("✅ 规则提取器测试完成")
    print("="*60)

