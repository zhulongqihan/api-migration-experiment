# 阶段2 Python脚本文件说明

## 📦 文件清单

| 文件名 | 行数 | 功能描述 |
|--------|------|----------|
| `data_utils.py` | 95 | 数据加载工具，读取和处理JSON数据集 |
| `rule_extractor.py` | 87 | 规则提取器，从代码对中提取API更新规则 |
| `prompt_engineering.py` | 107 | Prompt模板，提供4种不同的Prompt生成策略 |
| `test_phase2.py` | 256 | 完整测试脚本，验证所有模块功能 |
| `mini_dataset.json` | - | 最小数据集，包含3个训练样例和1个测试样例 |

## 🚀 使用方法

### 步骤1：上传到服务器

**方法A - 使用WinSCP/FileZilla**:
1. 连接到服务器
2. 导航到 `~/api_migration_exp/scripts/`
3. 拖拽上传所有文件

**方法B - 使用scp命令**:
```powershell
# 在Windows PowerShell中执行
scp F:\apirecode\api-migration-experiment\server_scripts\*.py 您的用户名@服务器:~/api_migration_exp/scripts/
scp F:\apirecode\api-migration-experiment\server_scripts\mini_dataset.json 您的用户名@服务器:~/api_migration_exp/scripts/
```

### 步骤2：在服务器上运行

```bash
# 进入项目目录
cd ~/api_migration_exp
conda activate apiupdate

# 创建目录
mkdir -p data/processed scripts configs

# 复制数据集
cp scripts/mini_dataset.json data/processed/

# 运行完整测试
cd scripts
python test_phase2.py
```

## 📝 单独测试各模块

```bash
cd ~/api_migration_exp/scripts

# 测试数据加载器
python data_utils.py ../data/processed/mini_dataset.json

# 测试规则提取器
python rule_extractor.py

# 测试Prompt模板
python prompt_engineering.py

# 完整测试
python test_phase2.py
```

## ✅ 预期输出

成功运行 `test_phase2.py` 后应该看到：

```
🎉 阶段2所有测试通过！
============================================================
✅ 已完成:
  ✓ 数据集加载 (3 train, 1 test)
  ✓ 规则提取 (3 条规则)
  ✓ Prompt模板 (4 种策略)
  ✓ 端到端流程验证
  ✓ 规则库保存
```

## 🔧 生成的文件

运行后会生成：
- `../data/processed/mini_dataset.json` - 数据集
- `../configs/rules.json` - 规则库

## 💡 注意事项

1. **无需网络**：此阶段不需要下载模型或访问外网
2. **Python环境**：需要激活 `apiupdate` conda环境
3. **工作目录**：确保在 `scripts` 目录下运行脚本
4. **依赖库**：只需要Python标准库（json, pathlib等）

## 🆘 常见问题

**Q: ModuleNotFoundError**
```bash
# 确保在正确的目录
cd ~/api_migration_exp/scripts
python test_phase2.py
```

**Q: FileNotFoundError: mini_dataset.json**
```bash
# 复制数据集文件
cp mini_dataset.json ../data/processed/
```

**Q: 某个测试失败**
```bash
# 单独运行每个模块找出问题
python data_utils.py ../data/processed/mini_dataset.json
python rule_extractor.py
python prompt_engineering.py
```

