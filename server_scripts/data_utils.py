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

if __name__ == "__main__":
    import sys
    import os
    
    # 测试数据加载器
    data_file = sys.argv[1] if len(sys.argv) > 1 else "data/processed/mini_dataset.json"
    
    print(f"加载数据: {data_file}")
    loader = DataLoader(data_file)
    loader.summary()
    
    print("\n📝 示例数据（第1个训练样例）:")
    sample = loader.get_sample(0)
    if sample:
        print(f"  依赖: {sample['dependency']}")
        print(f"  旧代码: {sample['old_code']}")
        print(f"  新代码: {sample['new_code']}")
        print(f"  说明: {sample.get('description', 'N/A')}")

