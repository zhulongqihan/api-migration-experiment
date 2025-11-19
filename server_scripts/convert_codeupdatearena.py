#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从CodeUpdateArena数据集准备训练数据
将CodeUpdateArena格式转换为我们的训练格式
"""

import json
import os
import random
from pathlib import Path
from collections import defaultdict


def extract_code_examples(doc_string):
    """从文档字符串中提取代码示例"""
    if not doc_string:
        return []
    
    examples = []
    lines = doc_string.split('\n')
    
    in_example = False
    current_example = []
    
    for line in lines:
        # 检测示例代码块（通常以>>>或...开头）
        if '>>>' in line:
            in_example = True
            # 提取代码（移除>>>和空格）
            code = line.split('>>>', 1)[1].strip()
            if code:
                current_example.append(code)
        elif in_example and line.strip().startswith('...'):
            # 继续多行代码
            code = line.split('...', 1)[1].strip()
            if code:
                current_example.append(code)
        elif in_example and current_example:
            # 示例结束
            examples.append('\n'.join(current_example))
            current_example = []
            in_example = False
    
    # 添加最后一个示例
    if current_example:
        examples.append('\n'.join(current_example))
    
    return examples


def load_codeupdatearena_data(source_dir):
    """加载CodeUpdateArena数据"""
    source_path = Path(source_dir)
    
    # 查找所有JSON文件
    json_files = list(source_path.glob("data/update_functions/*.json"))
    
    print(f"找到 {len(json_files)} 个数据文件")
    
    all_data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 提取函数名和包名
                function_name = data.get('function_name', '')
                package = data.get('package', 'unknown')
                
                # 方法1: 从old_doc和new_doc提取示例
                old_doc = data.get('old_doc', '')
                new_doc = data.get('new_doc', '')
                
                if old_doc and new_doc:
                    # 从文档中提取代码示例
                    old_examples = extract_code_examples(old_doc)
                    new_examples = extract_code_examples(new_doc)
                    
                    # 配对示例（假设顺序对应）
                    for i in range(min(len(old_examples), len(new_examples))):
                        if old_examples[i] != new_examples[i]:  # 只保留有变化的
                            all_data.append({
                                'dependency': package,
                                'old_code': old_examples[i],
                                'new_code': new_examples[i],
                                'description': f"{function_name} - 文档示例更新"
                            })
                
                # 方法2: 如果有summarized_doc，也提取示例
                summarized_doc = data.get('summarized_doc', '')
                if summarized_doc:
                    examples = extract_code_examples(summarized_doc)
                    # 使用总结文档中的示例作为新版本
                    for example in examples:
                        if len(example) > 10:  # 过滤太短的示例
                            all_data.append({
                                'dependency': package,
                                'old_code': example,  # 暂时用同样的代码
                                'new_code': example,
                                'description': f"{function_name} - API使用示例"
                            })
                    
        except Exception as e:
            print(f"跳过文件 {json_file.name}: {e}")
    
    print(f"成功加载 {len(all_data)} 条数据")
    return all_data


def create_dataset(all_data, train_size=50, test_size=50, libraries=None):
    """创建训练和测试数据集"""
    
    # 按库分组
    by_library = defaultdict(list)
    for item in all_data:
        lib = item['dependency']
        by_library[lib].append(item)
    
    print(f"\n数据分布:")
    for lib, items in sorted(by_library.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  {lib}: {len(items)} 条")
    
    # 如果指定了库，只使用这些库的数据
    if libraries:
        filtered_data = []
        for lib in libraries:
            if lib in by_library:
                filtered_data.extend(by_library[lib])
            # 也尝试部分匹配
            else:
                for key in by_library.keys():
                    if lib in key or key in lib:
                        filtered_data.extend(by_library[key])
                        break
        if filtered_data:
            all_data = filtered_data
            print(f"\n过滤后数据: {len(all_data)} 条")
        else:
            print(f"\n警告: 未找到指定库的数据，使用全部数据")
    
    # 打乱数据
    random.shuffle(all_data)
    
    # 分割训练集和测试集
    train_data = all_data[:train_size]
    test_data = all_data[train_size:train_size + test_size]
    
    # 如果数据不够，循环使用
    if len(train_data) < train_size:
        print(f"\n警告: 训练数据不足，从 {len(train_data)} 扩展到 {train_size}")
        while len(train_data) < train_size:
            train_data.extend(all_data[:min(len(all_data), train_size - len(train_data))])
    
    if len(test_data) < test_size:
        print(f"警告: 测试数据不足，从 {len(test_data)} 扩展到 {test_size}")
        # 使用不同的数据作为测试集
        remaining = [x for x in all_data if x not in train_data]
        if remaining:
            test_data = remaining[:test_size]
            while len(test_data) < test_size:
                test_data.extend(remaining[:min(len(remaining), test_size - len(test_data))])
        else:
            while len(test_data) < test_size:
                test_data.extend(all_data[:min(len(all_data), test_size - len(test_data))])
    
    return {
        'train': train_data[:train_size],
        'test': test_data[:test_size]
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='转换CodeUpdateArena数据')
    parser.add_argument('--source_dir', type=str, default='../CodeUpdateArena',
                        help='CodeUpdateArena目录')
    parser.add_argument('--output', type=str, default='extended_dataset_50.json',
                        help='输出文件名')
    parser.add_argument('--train_size', type=int, default=50,
                        help='训练集大小')
    parser.add_argument('--test_size', type=int, default=50,
                        help='测试集大小')
    parser.add_argument('--libraries', nargs='+', default=None,
                        help='指定库（可选），例如: pandas numpy torch')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    print(f"=== CodeUpdateArena数据转换 ===\n")
    print(f"源目录: {args.source_dir}")
    print(f"输出文件: {args.output}")
    print(f"训练集大小: {args.train_size}")
    print(f"测试集大小: {args.test_size}")
    if args.libraries:
        print(f"指定库: {', '.join(args.libraries)}")
    print()
    
    # 加载数据
    all_data = load_codeupdatearena_data(args.source_dir)
    
    if not all_data:
        print("❌ 没有找到有效数据")
        print("\n请检查:")
        print(f"  1. CodeUpdateArena目录是否存在: {args.source_dir}")
        print(f"  2. 数据文件是否在: {args.source_dir}/data/update_functions/")
        return
    
    # 创建数据集
    dataset = create_dataset(
        all_data,
        train_size=args.train_size,
        test_size=args.test_size,
        libraries=args.libraries
    )
    
    # 保存
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 数据集创建完成:")
    print(f"  训练集: {len(dataset['train'])} 条")
    print(f"  测试集: {len(dataset['test'])} 条")
    print(f"  保存到: {args.output}")
    
    # 统计库分布
    from collections import Counter
    train_libs = Counter([item['dependency'] for item in dataset['train']])
    test_libs = Counter([item['dependency'] for item in dataset['test']])
    
    print(f"\n训练集库分布:")
    for lib, count in train_libs.most_common(10):
        print(f"  {lib}: {count}")
    
    print(f"\n测试集库分布:")
    for lib, count in test_libs.most_common(10):
        print(f"  {lib}: {count}")
    
    # 显示前3个样例
    print(f"\n前3个训练样例:")
    for i, item in enumerate(dataset['train'][:3], 1):
        print(f"\n{i}. {item['dependency']}: {item['description']}")
        print(f"   旧代码: {item['old_code'][:100]}...")
        print(f"   新代码: {item['new_code'][:100]}...")


if __name__ == '__main__':
    main()
