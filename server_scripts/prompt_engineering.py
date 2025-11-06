#!/usr/bin/env python3
"""Prompt工程模块"""

from typing import List, Dict

class PromptTemplate:
    """Prompt模板集合"""
    
    @staticmethod
    def basic_update_prompt(old_code: str, description: str) -> str:
        """基础更新Prompt"""
        return f"""### Task: Update deprecated API code

The following code uses deprecated APIs:
```python
{old_code}
```

Update requirement: {description}

IMPORTANT: Generate ONLY the updated code (one line), no explanations, no markdown formatting.

Updated code:
"""
    
    @staticmethod
    def with_context_prompt(old_code: str, dependency: str, old_version: str, new_version: str, description: str) -> str:
        """带上下文的Prompt"""
        return f"""### API Update Task

Library: {dependency}
Version: {old_version} → {new_version}
Change: {description}

Old Code:
{old_code}

IMPORTANT: Output ONLY the updated code (single line), no explanations.

Updated code:
"""
    
    @staticmethod
    def with_rules_prompt(old_code: str, dependency: str, rules: List[Dict], description: str) -> str:
        """带规则提示的Prompt"""
        rules_text = "\n".join([f"- {r.get('type', 'unknown')}: {r.get('pattern', {})} → {r.get('replacement', {})}" for r in rules])
        
        return f"""### API Update with Rules

Library: {dependency}
Rules:
{rules_text}

Old Code:
{old_code}

Task: {description}

IMPORTANT: Apply the rules above and output ONLY the updated code (one line), no explanations.

Updated code:
"""
    
    @staticmethod
    def cot_prompt(old_code: str, dependency: str, description: str) -> str:
        """Chain-of-Thought Prompt"""
        return f"""### API Update - Think Step by Step

Library: {dependency}
Task: {description}

Old Code:
{old_code}

Think step by step:
1. Identify the deprecated API
2. Find the replacement API
3. Adjust parameters if needed

IMPORTANT: After thinking, output ONLY the final updated code (one line).

Updated code:
"""

if __name__ == "__main__":
    print("="*60)
    print("📝 Prompt模板测试")
    print("="*60)
    
    template = PromptTemplate()
    
    # 测试数据
    old_code = "df.append(row)"
    dependency = "pandas"
    description = "Use concat instead of append"
    
    print("\n【模板1】基础Prompt:")
    prompt1 = template.basic_update_prompt(old_code, description)
    print(f"长度: {len(prompt1)} 字符")
    print("预览:")
    print(prompt1[:150] + "...")
    
    print("\n【模板2】上下文Prompt:")
    prompt2 = template.with_context_prompt(
        old_code, dependency, "1.3.0", "2.0.0", description
    )
    print(f"长度: {len(prompt2)} 字符")
    
    print("\n【模板3】带规则Prompt:")
    rules = [{"type": "function_replacement", "pattern": {"fn": "append"}, "replacement": {"fn": "concat"}}]
    prompt3 = template.with_rules_prompt(old_code, dependency, rules, description)
    print(f"长度: {len(prompt3)} 字符")
    
    print("\n【模板4】CoT Prompt:")
    prompt4 = template.cot_prompt(old_code, dependency, description)
    print(f"长度: {len(prompt4)} 字符")
    
    print("\n" + "="*60)
    print("✅ 所有模板测试完成")
    print("="*60)

