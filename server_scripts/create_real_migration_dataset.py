#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建真实的API迁移数据集
包含常见Python库的真实API变更案例
"""

import json
import random


# 真实的API迁移案例库
REAL_API_MIGRATIONS = [
    # Pandas迁移案例
    {
        "dependency": "pandas",
        "old_code": "df1.append(df2)",
        "new_code": "pd.concat([df1, df2])",
        "description": "DataFrame.append已废弃，使用concat"
    },
    {
        "dependency": "pandas",
        "old_code": "df.append({'col1': 1, 'col2': 2}, ignore_index=True)",
        "new_code": "pd.concat([df, pd.DataFrame([{'col1': 1, 'col2': 2}])], ignore_index=True)",
        "description": "字典append改为concat"
    },
    {
        "dependency": "pandas",
        "old_code": "df.applymap(lambda x: x * 2)",
        "new_code": "df.map(lambda x: x * 2)",
        "description": "applymap改为map"
    },
    {
        "dependency": "pandas",
        "old_code": "df.ix[0]",
        "new_code": "df.iloc[0]",
        "description": "ix已废弃，使用iloc"
    },
    {
        "dependency": "pandas",
        "old_code": "df.ix[:, 'A']",
        "new_code": "df.loc[:, 'A']",
        "description": "ix改为loc用于标签索引"
    },
    {
        "dependency": "pandas",
        "old_code": "pd.DataFrame.from_items([('A', [1, 2]), ('B', [3, 4])])",
        "new_code": "pd.DataFrame({'A': [1, 2], 'B': [3, 4]})",
        "description": "from_items已废弃"
    },
    {
        "dependency": "pandas",
        "old_code": "df.sort(['col1', 'col2'])",
        "new_code": "df.sort_values(['col1', 'col2'])",
        "description": "sort改为sort_values"
    },
    {
        "dependency": "pandas",
        "old_code": "df.sort_index(by='col1')",
        "new_code": "df.sort_values(by='col1')",
        "description": "sort_index的by参数已移除"
    },
    
    # NumPy迁移案例
    {
        "dependency": "numpy",
        "old_code": "matrix = np.matrix([[1, 2], [3, 4]])",
        "new_code": "matrix = np.array([[1, 2], [3, 4]])",
        "description": "np.matrix已废弃，使用array"
    },
    {
        "dependency": "numpy",
        "old_code": "np.asscalar(arr[0])",
        "new_code": "arr[0].item()",
        "description": "asscalar已废弃，使用item()"
    },
    {
        "dependency": "numpy",
        "old_code": "np.rank(arr)",
        "new_code": "arr.ndim",
        "description": "np.rank已废弃，使用ndim"
    },
    {
        "dependency": "numpy",
        "old_code": "np.sum(arr, keepdims=1)",
        "new_code": "np.sum(arr, keepdims=True)",
        "description": "keepdims现在需要布尔值"
    },
    
    # TensorFlow迁移案例
    {
        "dependency": "tensorflow",
        "old_code": "flat = tf.contrib.layers.flatten(x)",
        "new_code": "flat = tf.keras.layers.Flatten()(x)",
        "description": "tf.contrib已移除，使用keras"
    },
    {
        "dependency": "tensorflow",
        "old_code": "tf.placeholder(tf.float32, shape=[None, 784])",
        "new_code": "tf.keras.Input(shape=(784,))",
        "description": "placeholder改为Input"
    },
    {
        "dependency": "tensorflow",
        "old_code": "tf.Session()",
        "new_code": "# TF2使用eager execution，不需要Session",
        "description": "TF2移除了Session"
    },
    {
        "dependency": "tensorflow",
        "old_code": "tf.global_variables_initializer()",
        "new_code": "# TF2不需要显式初始化变量",
        "description": "TF2自动初始化变量"
    },
    
    # Scikit-learn迁移案例
    {
        "dependency": "sklearn",
        "old_code": "from sklearn.cross_validation import train_test_split",
        "new_code": "from sklearn.model_selection import train_test_split",
        "description": "cross_validation模块已移动"
    },
    {
        "dependency": "sklearn",
        "old_code": "from sklearn.grid_search import GridSearchCV",
        "new_code": "from sklearn.model_selection import GridSearchCV",
        "description": "grid_search已移动到model_selection"
    },
    {
        "dependency": "sklearn",
        "old_code": "from sklearn.cross_validation import KFold",
        "new_code": "from sklearn.model_selection import KFold",
        "description": "KFold已移动"
    },
    {
        "dependency": "sklearn",
        "old_code": "from sklearn.learning_curve import learning_curve",
        "new_code": "from sklearn.model_selection import learning_curve",
        "description": "learning_curve已移动"
    },
    
    # PyTorch迁移案例
    {
        "dependency": "torch",
        "old_code": "torch.nn.functional.sigmoid(x)",
        "new_code": "torch.sigmoid(x)",
        "description": "sigmoid移到torch命名空间"
    },
    {
        "dependency": "torch",
        "old_code": "torch.nn.functional.tanh(x)",
        "new_code": "torch.tanh(x)",
        "description": "tanh移到torch命名空间"
    },
    {
        "dependency": "torch",
        "old_code": "torch.cuda.device_count()",
        "new_code": "torch.cuda.device_count()",
        "description": "CUDA API保持不变"
    },
    
    # Requests迁移案例
    {
        "dependency": "requests",
        "old_code": "r = requests.get(url, verify=False)",
        "new_code": "r = requests.get(url, verify=True)  # 建议启用SSL验证",
        "description": "建议启用SSL验证"
    },
    
    # Matplotlib迁移案例
    {
        "dependency": "matplotlib",
        "old_code": "plt.hold(True)",
        "new_code": "# hold已废弃，默认行为已改变",
        "description": "hold方法已移除"
    },
    {
        "dependency": "matplotlib",
        "old_code": "plt.ishold()",
        "new_code": "# ishold已废弃",
        "description": "ishold已移除"
    },
]


def create_dataset(train_size=50, test_size=50, seed=42):
    """创建训练和测试数据集"""
    random.seed(seed)
    
    # 扩展数据到所需大小
    all_data = REAL_API_MIGRATIONS.copy()
    
    # 如果数据不够，循环复制
    while len(all_data) < train_size + test_size:
        all_data.extend(REAL_API_MIGRATIONS)
    
    # 打乱数据
    random.shuffle(all_data)
    
    # 分割
    train_data = all_data[:train_size]
    test_data = all_data[train_size:train_size + test_size]
    
    return {
        'train': train_data,
        'test': test_data
    }


def main():
    import argparse
    from collections import Counter
    
    parser = argparse.ArgumentParser(description='创建真实API迁移数据集')
    parser.add_argument('--output', type=str, default='extended_dataset_50.json',
                        help='输出文件名')
    parser.add_argument('--train_size', type=int, default=50,
                        help='训练集大小')
    parser.add_argument('--test_size', type=int, default=50,
                        help='测试集大小')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    print(f"=== 创建真实API迁移数据集 ===\n")
    print(f"输出文件: {args.output}")
    print(f"训练集大小: {args.train_size}")
    print(f"测试集大小: {args.test_size}")
    print(f"基础案例数: {len(REAL_API_MIGRATIONS)}")
    print()
    
    # 创建数据集
    dataset = create_dataset(
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed
    )
    
    # 保存
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 数据集创建完成:")
    print(f"  训练集: {len(dataset['train'])} 条")
    print(f"  测试集: {len(dataset['test'])} 条")
    print(f"  保存到: {args.output}")
    
    # 统计库分布
    train_libs = Counter([item['dependency'] for item in dataset['train']])
    test_libs = Counter([item['dependency'] for item in dataset['test']])
    
    print(f"\n训练集库分布:")
    for lib, count in train_libs.most_common():
        print(f"  {lib}: {count}")
    
    print(f"\n测试集库分布:")
    for lib, count in test_libs.most_common():
        print(f"  {lib}: {count}")
    
    # 显示前5个样例
    print(f"\n前5个训练样例:")
    for i, item in enumerate(dataset['train'][:5], 1):
        print(f"\n{i}. {item['dependency']}: {item['description']}")
        print(f"   旧: {item['old_code'][:80]}")
        print(f"   新: {item['new_code'][:80]}")


if __name__ == '__main__':
    main()
