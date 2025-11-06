# 下一步操作 - GPU加速 & 评估分析

## 📝 当前状态

✅ **已完成**：
- 阶段1：环境配置
- 阶段2：数据和框架准备  
- 任务3.1.1：模型加载测试（CPU）
- 任务3.1.2：完整推理pipeline（4种策略，CPU运行）

🔄 **进行中**：
- 任务3.1.3：评估和分析

## 🎯 接下来要做的事

### 步骤1：测试GPU加速（5分钟）

**目的**：尝试使用GPU加速推理（速度提升10-100倍）

**操作**：
```bash
cd ~/api_migration_exp/scripts
python inference_baseline.py
```

**预期结果**：
- ✅ 成功：显示"检测到CUDA，使用GPU"，推理速度大幅提升
- ⚠️ 失败：自动回退到CPU，功能正常但速度较慢

**为什么安全**：
- 只修改了`inference_baseline.py`的设备检测逻辑
- 添加了try-except保护
- GPU失败会自动降级到CPU
- 不影响其他已完成的脚本（data_utils, rule_extractor等）

---

### 步骤2：运行评估脚本（2分钟）

**目的**：分析4种Prompt策略的效果

**操作**：
```bash
cd ~/api_migration_exp/scripts
python evaluate_baseline.py
```

**输出**：
1. **对比表格**：每种策略的精确匹配率、相似度、关键API准确率
2. **失败分析**：哪些样例失败了，失败原因是什么
3. **保存文件**：
   - `results/baseline/evaluation_baseline_results_*.json` - 完整评估数据
   - `results/baseline/evaluation_baseline_results_*_summary.txt` - 可读摘要

---

### 步骤3：查看和分析结果（5分钟）

**操作**：
```bash
# 查看评估摘要
cat ~/api_migration_exp/scripts/results/baseline/evaluation_*_summary.txt

# 或查看完整JSON
cat ~/api_migration_exp/scripts/results/baseline/evaluation_*.json
```

**分析要点**：
1. **哪种策略最好**？
   - 精确匹配率最高？
   - 代码相似度最高？
   - 关键API检测最准？

2. **为什么最好**？
   - basic：简单直接
   - with_context：有版本信息
   - with_rules：有规则指导
   - cot：有推理过程

3. **失败案例**：
   - 模型生成了什么？
   - 哪里出错了？
   - 如何改进？

---

## 📊 完成后的下一步

### 任务3.1.4（可选）：扩展数据集

**如果当前数据集太小**（只有1个测试样例），可以：
1. 手动添加更多API更新样例到`mini_dataset.json`
2. 重新运行推理和评估
3. 获得更可靠的统计结果

### 阶段3.2：方向1 - LoRA微调

**目标**：
- 收集更多训练数据（10-50个样例）
- 使用LoRA微调Qwen2.5-Coder-1.5B
- 对比微调前后的效果
- （创新点）尝试分层LoRA（只微调深层）

**预计时间**：2-3天

---

## 🔍 故障排查

### 如果GPU测试失败

**症状**：显示"GPU加载失败"，自动回退到CPU

**原因**：可能是CUDA版本不匹配

**解决**：
```bash
# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 如果是False，继续用CPU即可，不影响实验
# 如果是True但加载失败，检查显存
nvidia-smi
```

### 如果评估脚本报错

**症状**：找不到结果文件

**解决**：
```bash
# 检查结果文件是否存在
ls -lh ~/api_migration_exp/scripts/results/baseline/

# 如果不存在，先运行推理
cd ~/api_migration_exp/scripts
python inference_baseline.py
```

---

## 📂 文件清单

**新增/修改的文件**：
```
api-migration-experiment/
├── server_scripts/
│   ├── inference_baseline.py      # 修改：添加GPU自动检测
│   └── evaluate_baseline.py       # 新增：评估脚本
└── docs/
    └── 实验记录.md                # 更新：记录改进和评估计划
```

**上传到服务器**：
```bash
# 在本地（Windows）
# 使用WinSCP上传以下文件到服务器 ~/api_migration_exp/scripts/：
# - inference_baseline.py（覆盖）
# - evaluate_baseline.py（新建）
```

---

## ✅ 完成标志

当你完成以上步骤后，应该有：
1. ✅ GPU测试结果（成功或自动降级）
2. ✅ 评估报告（对比表格 + 失败分析）
3. ✅ 对4种策略的理解（哪种好，为什么）
4. ✅ 为下一阶段（LoRA微调）做好准备

然后就可以推送到GitHub并开始下一阶段！

---

**有问题随时问！** 🚀

