#!/bin/bash
# 阶段2：数据准备和框架搭建
# 用途：创建数据集、数据加载工具、Baseline框架
# 不需要网络，不需要模型

set -e

echo "=========================================="
echo "🚀 阶段2：数据准备和框架搭建"
echo "=========================================="

WORK_DIR=~/api_migration_exp

# 1. 创建数据集
echo -e "\n📊 步骤1: 创建最小数据集..."
mkdir -p $WORK_DIR/data/processed

cat > $WORK_DIR/data/processed/mini_dataset.json << 'DATAEOF'
{
  "train": [
    {
      "id": 1,
      "dependency": "pandas",
      "old_version": "1.3.0",
      "new_version": "2.0.0",
      "old_code": "df = df.append({'A': 3}, ignore_index=True)",
      "new_code": "df = pd.concat([df, pd.DataFrame({'A': [3]})], ignore_index=True)",
      "description": "DataFrame.append已废弃，使用pd.concat替代",
      "update_type": "function_replacement"
    },
    {
      "id": 2,
      "dependency": "numpy",
      "old_version": "1.20.0",
      "new_version": "1.24.0",
      "old_code": "result = np.sum(arr, keepdims=True)",
      "new_code": "result = np.sum(arr, keepdims=False)",
      "description": "keepdims默认值改变",
      "update_type": "parameter_change"
    },
    {
      "id": 3,
      "dependency": "requests",
      "old_version": "2.25.0",
      "new_version": "2.28.0",
      "old_code": "response = requests.get(url)",
      "new_code": "response = requests.get(url, timeout=30)",
      "description": "建议添加timeout参数",
      "update_type": "parameter_add"
    }
  ],
  "test": [
    {
      "id": 1,
      "dependency": "pandas",
      "old_code": "new_df = old_df.append(row)",
      "new_code": "new_df = pd.concat([old_df, row])",
      "description": "使用concat代替append"
    }
  ]
}
DATAEOF

python3 -c "import json; data = json.load(open('$WORK_DIR/data/processed/mini_dataset.json')); print(f'✅ 数据集创建成功: {len(data[\"train\"])} train, {len(data[\"test\"])} test')"

# 2. 创建数据加载工具
echo -e "\n📦 步骤2: 创建数据加载工具..."
mkdir -p $WORK_DIR/src/utils

cat > $WORK_DIR/src/utils/__init__.py << 'EOF'
# Utils package
EOF

cat > $WORK_DIR/src/utils/data_utils.py << 'UTILEOF'
#!/usr/bin/env python3
"""数据加载和处理工具"""

import json
from pathlib import Path
from typing import Dict, List

class DataLoader:
    """数据加载器"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.data = self._load_data()
    
    def _load_data(self) -> Dict:
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_train_data(self) -> List[Dict]:
        return self.data.get('train', [])
    
    def get_test_data(self) -> List[Dict]:
        return self.data.get('test', [])
    
    def get_sample(self, idx: int, split: str = 'train') -> Dict:
        data = self.get_train_data() if split == 'train' else self.get_test_data()
        return data[idx] if idx < len(data) else None
    
    def summary(self):
        train = self.get_train_data()
        test = self.get_test_data()
        
        print("="*60)
        print("📊 数据集摘要")
        print("="*60)
        print(f"训练集: {len(train)} 样例")
        print(f"测试集: {len(test)} 样例")
        
        deps = {}
        for item in train:
            dep = item.get('dependency', 'unknown')
            deps[dep] = deps.get(dep, 0) + 1
        
        print("\n库分布:")
        for dep, count in sorted(deps.items()):
            print(f"  - {dep}: {count}")
        print("="*60)
UTILEOF

# 3. 创建Baseline框架
echo -e "\n🔧 步骤3: 创建Baseline框架..."
mkdir -p $WORK_DIR/src/baseline

cat > $WORK_DIR/src/baseline/__init__.py << 'EOF'
# Baseline package
EOF

cat > $WORK_DIR/src/baseline/rule_extractor.py << 'RULEEOF'
#!/usr/bin/env python3
"""规则提取模块"""

from typing import Dict, List

class APIUpdateRule:
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

class RuleExtractor:
    def extract_from_pair(self, old_code: str, new_code: str) -> List[APIUpdateRule]:
        rules = []
        
        if "append" in old_code and "concat" in new_code:
            rules.append(APIUpdateRule(
                rule_type="function_replacement",
                pattern={"function": "append"},
                replacement={"function": "concat"},
            ))
        
        return rules
    
    def build_rule_library(self, dataset: List[Dict]) -> Dict[str, List[Dict]]:
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
                if rule_dict not in rule_library[dependency]:
                    rule_library[dependency].append(rule_dict)
        
        return rule_library
RULEEOF

cat > $WORK_DIR/src/baseline/prompt_engineering.py << 'PROMPTEOF'
#!/usr/bin/env python3
"""Prompt工程模块"""

class PromptTemplate:
    @staticmethod
    def basic_update_prompt(old_code: str, description: str) -> str:
        return f"""### Task: Update deprecated API code

The following code uses deprecated APIs:
```python
{old_code}
```

Update requirement: {description}

Generate the updated code:
```python
"""
    
    @staticmethod
    def with_context_prompt(old_code: str, dependency: str, description: str) -> str:
        return f"""### API Update Task

**Library**: {dependency}
**Change**: {description}

**Old Code**:
```python
{old_code}
```

**Updated Code**:
```python
"""
PROMPTEOF

# 4. 测试所有组件
echo -e "\n🧪 步骤4: 测试所有组件..."

python3 << 'TESTEOF'
import sys
sys.path.insert(0, '/home/zhangchangyu/api_migration_exp')

from src.utils.data_utils import DataLoader
from src.baseline.rule_extractor import RuleExtractor
from src.baseline.prompt_engineering import PromptTemplate

print("\n测试1: 数据加载...")
loader = DataLoader('data/processed/mini_dataset.json')
loader.summary()

print("\n测试2: 规则提取...")
extractor = RuleExtractor()
rules = extractor.build_rule_library(loader.get_train_data())
print(f"✅ 提取规则: {sum(len(v) for v in rules.values())} 条")

print("\n测试3: Prompt生成...")
template = PromptTemplate()
sample = loader.get_sample(0)
prompt = template.basic_update_prompt(sample['old_code'], sample['description'])
print(f"✅ Prompt长度: {len(prompt)} 字符")

print("\n" + "="*60)
print("🎉 所有测试通过！阶段2完成！")
print("="*60)
TESTEOF

echo -e "\n=========================================="
echo "✅ 阶段2完成！"
echo "=========================================="
echo ""
echo "已完成："
echo "  ✅ 数据集准备（3个训练样例，1个测试样例）"
echo "  ✅ 数据加载工具"
echo "  ✅ 规则提取框架"
echo "  ✅ Prompt模板设计"
echo ""
echo "下一步："
echo "  - 运行: cd ~/api_migration_exp && python -m src.utils.data_utils"
echo "  - 或继续阶段3（需要模型）"
echo ""

