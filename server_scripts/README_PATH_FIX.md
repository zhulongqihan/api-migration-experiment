# 路径修复说明

## 问题描述

原始脚本将结果文件保存在 `scripts/results/baseline/` 目录下，而不是项目根目录的 `results/baseline/`。

## 修复内容

### 1. `inference_baseline.py`

**修改位置**：第163-169行

**修改前**：
```python
def save_results(self, results, output_dir="results/baseline"):
 """保存结果"""
 output_path = Path(output_dir)
 output_path.mkdir(parents=True, exist_ok=True)
```

**修改后**：
```python
def save_results(self, results, output_dir="../results/baseline"):
 """保存结果"""
 # 确保使用项目根目录的results文件夹
 script_dir = Path(__file__).parent
 project_root = script_dir.parent
 output_path = project_root / "results" / "baseline"
 output_path.mkdir(parents=True, exist_ok=True)
```

### 2. `evaluate_baseline.py`

**修改位置**：第247-250行

**修改前**：
```python
# 查找最新的结果文件
results_dir = Path("results/baseline")
if not results_dir.exists():
 console.print("[red]错误：results/baseline目录不存在[/red]")
```

**修改后**：
```python
# 查找最新的结果文件（使用项目根目录）
script_dir = Path(__file__).parent
project_root = script_dir.parent
results_dir = project_root / "results" / "baseline"

if not results_dir.exists():
 console.print(f"[red]错误：{results_dir}目录不存在[/red]")
```

## 效果

现在所有结果文件都会正确保存到：
```
~/api_migration_exp/results/baseline/
├── baseline_results_*.json
├── baseline_summary_*.txt
├── evaluation_*.json
└── evaluation_*_summary.txt
```

而不是：
```
~/api_migration_exp/scripts/results/baseline/
```

## 使用方法

修复后，直接运行脚本即可：

```bash
cd ~/api_migration_exp/scripts

# 运行推理
python3 inference_baseline.py

# 运行评估
python3 evaluate_baseline.py
```

结果文件会自动保存到正确的位置。

---

**修复时间**：2025-11-17 
**修复人员**：Cascade AI Assistant
