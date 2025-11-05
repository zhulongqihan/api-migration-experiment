#!/usr/bin/env python3
"""阶段2完整测试脚本"""

import sys
import os
import json
from pathlib import Path

# 导入模块
from data_utils import DataLoader
from rule_extractor import RuleExtractor
from prompt_engineering import PromptTemplate

def main():
    print("="*60)
    print("🧪 阶段2功能完整测试")
    print("="*60)
    
    # 检查数据文件是否存在
    data_file = "../data/processed/mini_dataset.json"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        print("   请先运行数据集创建命令")
        print("   当前目录:", os.getcwd())
        return False
    
    # ========== 测试1: 数据加载 ==========
    print("\n" + "="*60)
    print("【测试1】数据加载器")
    print("="*60)
    try:
        loader = DataLoader(data_file)
        loader.summary()
        
        train_count = len(loader.get_train_data())
        test_count = len(loader.get_test_data())
        
        if train_count > 0 and test_count > 0:
            print(f"✅ 数据加载器测试通过 ({train_count} train, {test_count} test)")
        else:
            print("❌ 数据集为空")
            return False
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========== 测试2: 规则提取 ==========
    print("\n" + "="*60)
    print("【测试2】规则提取器")
    print("="*60)
    try:
        extractor = RuleExtractor()
        train_data = loader.get_train_data()
        rules = extractor.build_rule_library(train_data)
        
        total_rules = sum(len(v) for v in rules.values())
        print(f"✅ 规则提取完成")
        print(f"   总规则数: {total_rules}")
        print(f"   涵盖库数: {len(rules)}")
        
        print("\n规则详情:")
        for dep, dep_rules in rules.items():
            print(f"  📦 {dep}: {len(dep_rules)} 条规则")
            for rule in dep_rules:
                print(f"     - {rule['type']}")
        
        if total_rules == 0:
            print("⚠️  警告: 未提取到任何规则")
    except Exception as e:
        print(f"❌ 规则提取器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========== 测试3: Prompt生成 ==========
    print("\n" + "="*60)
    print("【测试3】Prompt模板")
    print("="*60)
    try:
        template = PromptTemplate()
        sample = loader.get_sample(0)
        
        # 测试基础Prompt
        prompt1 = template.basic_update_prompt(
            sample['old_code'], 
            sample['description']
        )
        
        # 测试上下文Prompt
        prompt2 = template.with_context_prompt(
            sample['old_code'],
            sample['dependency'],
            sample.get('old_version', '1.0.0'),
            sample.get('new_version', '2.0.0'),
            sample['description']
        )
        
        # 测试CoT Prompt
        prompt3 = template.cot_prompt(
            sample['old_code'],
            sample['dependency'],
            sample['description']
        )
        
        print(f"✅ Prompt生成测试通过")
        print(f"   基础Prompt: {len(prompt1)} 字符")
        print(f"   上下文Prompt: {len(prompt2)} 字符")
        print(f"   CoT Prompt: {len(prompt3)} 字符")
        
    except Exception as e:
        print(f"❌ Prompt模板测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========== 测试4: 端到端流程 ==========
    print("\n" + "="*60)
    print("【测试4】端到端流程模拟")
    print("="*60)
    try:
        print("\n场景: 对测试集样例生成更新Prompt")
        
        test_sample = loader.get_test_data()[0]
        print(f"\n输入:")
        print(f"  库: {test_sample['dependency']}")
        print(f"  旧代码: {test_sample['old_code']}")
        print(f"  期望: {test_sample['new_code']}")
        
        # 获取该库的规则
        dep_rules = rules.get(test_sample['dependency'], [])
        
        # 根据规则选择Prompt策略
        if dep_rules:
            prompt = template.with_rules_prompt(
                test_sample['old_code'],
                test_sample['dependency'],
                dep_rules,
                test_sample.get('description', '')
            )
            strategy = "规则引导"
        else:
            prompt = template.basic_update_prompt(
                test_sample['old_code'],
                test_sample.get('description', 'Update to latest API')
            )
            strategy = "基础生成"
        
        print(f"\n生成Prompt:")
        print(f"  策略: {strategy}")
        print(f"  长度: {len(prompt)} 字符")
        print(f"  规则数: {len(dep_rules)}")
        
        print("\n✅ 端到端流程测试通过")
        
    except Exception as e:
        print(f"❌ 端到端测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ========== 保存规则库 ==========
    print("\n" + "="*60)
    print("【保存】规则库")
    print("="*60)
    try:
        os.makedirs('configs', exist_ok=True)
        rules_file = 'configs/rules.json'
        
        with open(rules_file, 'w', encoding='utf-8') as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)
        
        file_size = os.path.getsize(rules_file)
        print(f"✅ 规则库已保存")
        print(f"   位置: {rules_file}")
        print(f"   大小: {file_size} 字节")
        
    except Exception as e:
        print(f"❌ 保存失败: {e}")
        return False
    
    # ========== 最终总结 ==========
    print("\n" + "="*60)
    print("🎉 阶段2所有测试通过！")
    print("="*60)
    print("\n✅ 已完成:")
    print("  ✓ 数据集加载 (3 train, 1 test)")
    print(f"  ✓ 规则提取 ({total_rules} 条规则)")
    print("  ✓ Prompt模板 (4 种策略)")
    print("  ✓ 端到端流程验证")
    print("  ✓ 规则库保存")
    
    print("\n📂 生成的文件:")
    print("  - data/processed/mini_dataset.json")
    print("  - configs/rules.json")
    
    print("\n🚀 下一步选项:")
    print("  1. 进入阶段3（需要模型加载）")
    print("  2. 扩展数据集和规则")
    print("  3. 实现纯规则匹配baseline")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

