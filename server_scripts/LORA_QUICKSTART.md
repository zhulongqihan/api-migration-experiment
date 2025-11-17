# LoRA微调快速开始指南

## 📋 目录结构

```
server_scripts/
├── lora_config.py          # LoRA配置（标准/层次化）
├── lora_trainer.py         # 训练器实现
├── run_lora.py            # 主运行脚本
├── evaluate_lora.py       # 评估脚本
├── compare_methods.py     # 对比分析脚本
└── mini_dataset.json      # 训练数据
```

## 🚀 快速开始

### 1. 环境检查

```bash
cd ~/api_migration_exp/scripts

# 检查环境
conda activate apiupdate
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import peft; print(f'PEFT: {peft.__version__}')"
```

如果PEFT未安装：
```bash
pip install peft
```

### 2. 测试配置

```bash
# 测试配置文件
python3 lora_config.py

# 预期输出：显示标准LoRA和层次化LoRA的配置对比
```

### 3. 训练标准LoRA

```bash
# 完整训练（3 epochs，约2-3小时）
python3 run_lora.py --method standard --epochs 3 --batch_size 2

# 快速测试（1 epoch，约1小时）
python3 run_lora.py --method standard --epochs 1 --batch_size 2
```

**输出**：
- 模型保存在：`../models/checkpoints/standard_lora/final_model/`
- 训练信息：`../models/checkpoints/standard_lora/training_info.json`

### 4. 训练层次化LoRA（创新方法）

```bash
# 完整训练（3 epochs，约1-2小时）
python3 run_lora.py --method hierarchical --target_layers 22-31 --epochs 3 --batch_size 2

# 快速测试（1 epoch，约30分钟）
python3 run_lora.py --method hierarchical --target_layers 22-31 --epochs 1 --batch_size 2
```

**输出**：
- 模型保存在：`../models/checkpoints/hierarchical_lora_layers_22-31/final_model/`
- 训练信息：`../models/checkpoints/hierarchical_lora_layers_22-31/training_info.json`

### 5. 评估模型

```bash
# 评估标准LoRA
python3 evaluate_lora.py \
    --model_path ../models/checkpoints/standard_lora/final_model

# 评估层次化LoRA
python3 evaluate_lora.py \
    --model_path ../models/checkpoints/hierarchical_lora_layers_22-31/final_model
```

**输出**：
- 评估结果：`../models/checkpoints/*/evaluation/evaluation_results_*.json`
- 评估摘要：`../models/checkpoints/*/evaluation/evaluation_summary_*.txt`

### 6. 对比分析

```bash
# 方法1：手动指定文件
python3 compare_methods.py \
    --baseline_result ../results/baseline/evaluation_baseline_results_20251117_092534.json \
    --standard_lora_result ../models/checkpoints/standard_lora/evaluation/evaluation_results_*.json \
    --hierarchical_lora_result ../models/checkpoints/hierarchical_lora_layers_22-31/evaluation/evaluation_results_*.json

# 方法2：使用默认路径（如果文件在标准位置）
python3 compare_methods.py
```

**输出**：
- 对比报告：`../results/lora/comparison_report.txt`
- 终端显示详细对比表格

---

## 📊 预期结果

### 标准LoRA
- **参数量**：更新所有32层
- **训练时间**：2-3小时（3 epochs）
- **预期性能**：精确匹配率 > 90%

### 层次化LoRA（创新点）
- **参数量**：只更新10层（22-31），减少~69%
- **训练时间**：1-2小时（3 epochs），快30-50%
- **预期性能**：精确匹配率 ≈ 标准LoRA（±5%）

### 核心假设验证
如果层次化LoRA性能接近标准LoRA，则证明：
1. ✅ 深层负责语义理解，更新深层足以适配API
2. ✅ 可以大幅减少参数量而不损失性能
3. ✅ 减少灾难性遗忘的风险

---

## 🔧 常见问题

### Q1: CUDA内存不足
```bash
# 减小batch_size
python3 run_lora.py --method standard --batch_size 1

# 或使用CPU（会很慢）
python3 run_lora.py --method standard --device cpu
```

### Q2: 训练中断后继续
训练器会自动保存checkpoint，可以从最新checkpoint继续：
```bash
# 查看已保存的checkpoint
ls -lh ../models/checkpoints/standard_lora/

# 从checkpoint继续训练（需要修改脚本支持）
```

### Q3: 修改LoRA参数
```bash
# 增大LoRA秩（更多参数，可能更好）
python3 run_lora.py --method standard --lora_r 16 --lora_alpha 32

# 减小LoRA秩（更少参数，更快）
python3 run_lora.py --method standard --lora_r 4 --lora_alpha 8
```

### Q4: 测试不同层范围
```bash
# 只更新最后5层
python3 run_lora.py --method hierarchical --target_layers 27-31

# 只更新中间层
python3 run_lora.py --method hierarchical --target_layers 16-23
```

---

## 📈 监控训练

### 查看训练日志
```bash
# 实时查看训练输出
tail -f ../models/checkpoints/standard_lora/training.log

# 查看GPU使用情况
watch -n 1 nvidia-smi
```

### 检查训练进度
```bash
# 查看已保存的checkpoint
ls -lht ../models/checkpoints/standard_lora/checkpoint-*/

# 查看训练信息
cat ../models/checkpoints/standard_lora/training_info.json
```

---

## 🎯 下一步

训练和评估完成后：

1. **分析结果**：查看对比报告，验证假设
2. **消融实验**：测试不同层范围（如16-23, 24-31等）
3. **撰写论文**：整理实验结果和发现
4. **GitHub同步**：提交代码和结果

---

## 📞 需要帮助？

## 🐛 常见问题及解决方案（2025-11-17更新）

### 问题1：bitsandbytes兼容性错误
**错误信息**：
```
ModuleNotFoundError: No module named 'triton.ops'
```

**解决方案**：
已在所有脚本中添加 `os.environ['DISABLE_BNB_IMPORT'] = '1'`，无需手动操作。

---

### 问题2：pandas/numpy版本冲突
**错误信息**：
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

**解决方案**：
```bash
pip uninstall pandas pyarrow datasets -y
pip install --only-binary=:all: pyarrow pandas datasets
```

**推荐版本**：
- pandas==2.3.3
- numpy==2.2.6
- pyarrow==20.0.0
- datasets==4.0.0

---

### 问题3：FP16混合精度训练错误
**错误信息**：
```
ValueError: Attempting to unscale FP16 gradients
```

**解决方案**：
已在 `lora_trainer.py` 中禁用FP16/BF16，无需手动操作。

---

### 问题4：训练卡住不动
**可能原因**：
- 使用了换行的nohup命令
- 进程在后台运行但未正确启动

**解决方案**：
```bash
# 使用一行命令
nohup python3 run_lora.py --method hierarchical --target_layers 22-31 --epochs 1 --batch_size 2 > lora_test.log 2>&1 &

# 检查进程
ps aux | grep run_lora

# 查看日志
tail -f lora_test.log
```

---

## 📞 获取帮助

如果遇到其他问题：
1. 查看 `ENVIRONMENT_SETUP.md` - 详细的环境配置记录
2. 查看 `PROGRESS_2025-11-17.md` - 今日问题解决记录
3. 检查错误日志并搜索相关错误信息
4. 查看README.md中的详细说明

---

**祝实验顺利！** 🚀

*最后更新: 2025-11-17*
