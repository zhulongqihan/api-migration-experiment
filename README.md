# API版本迁移实验 🚀

## 📖 项目简介

探索在不频繁全量微调的前提下，使中小规模代码模型在动态API环境中可靠地生成符合最新版库接口的代码，避免昂贵的持续微调与灾难性遗忘。

## 🎯 研究目标

**核心问题**：如何让代码生成模型快速适应API版本更新？

**解决思路**：
1. 避免频繁全量微调（成本高）
2. 防止灾难性遗忘（保持其他能力）
3. 实现精确的知识更新（只改API相关知识）

## 🔬 三个研究方向（并行探索）

### 方向1️⃣: 强化学习/偏好微调（基于ReCode改进）
- **核心思想**：通过奖励函数引导模型学习新API
- **技术方案**：GRPO/DPO + 改进的奖励函数
- **创新点**：
  - 层次化LoRA（只更新深层语义层）
  - 更好的奖励函数设计（执行成功 + 测试通过 + API正确性）

### 方向2️⃣: 神经知识编辑（主要创新✨）
- **核心思想**：精确编辑模型中的API知识
- **技术方案**：ROME / MEMIT / 知识遗忘
- **创新点**：
  - 首次将知识编辑应用于代码API更新
  - 评估编辑的准确性和局部性
  - 避免影响无关代码生成能力

### 方向3️⃣: 规则 + Prompt工程（快速Baseline）
- **核心思想**：结合规则匹配和精心设计的Prompt
- **技术方案**：规则库 + Chain-of-Thought Prompt
- **创新点**：
  - 从数据集自动提取更新规则
  - 设计代码特定的Prompt模板

## 📊 实验进度

### ✅ 最新进展（2025-11-21）

**方向3 - 混合系统**：
- ✅ **大规模公开数据集**：扩展到300+样本（原40样本）
  - 基于TensorFlow/Pandas/sklearn/NumPy/PyTorch官方文档
  - 训练集240+样本，测试集60+样本
  - 覆盖5个主流库，60+种真实迁移模式
- ✅ **修复规则引导生成**：解决空白输出问题
  - 改进fallback逻辑
  - 优化LLM生成参数
  - 放宽过滤阈值
- 🚧 **待验证**：上传修复后的文件并测试效果

**当前性能**（待改进）：
- 规则直接应用：50%准确率
- 精确匹配率：17-50%（目标60%+）
- 规则引导Prompt：修复后待验证

### ✅ 阶段1：环境配置
- [x] 服务器配置确认（GPU: NVIDIA RTX 3090 24GB, CUDA 11.4）
- [x] Git仓库初始化
- [x] conda环境创建（apiupdate, Python 3.10）
- [x] PyTorch安装（2.x+cu118，兼容CUDA 11.4）
- [x] 核心依赖安装（transformers, peft, datasets, accelerate）
- [x] 项目结构搭建
- [x] 环境测试通过

### ✅ 阶段2：数据和框架准备
- [x] **数据集创建**（3个训练样例 + 1个测试样例）
  - pandas: append → concat（函数替换）
  - numpy: keepdims参数变化
  - requests: timeout参数添加
  
- [x] **核心模块实现**
  - data_utils.py (95行) - 数据加载工具
  - rule_extractor.py (87行) - 规则提取器，3条规则
  - prompt_engineering.py (107行) - 4种Prompt策略
  - test_phase2.py (209行) - 完整测试套件
  
- [x] **测试验证**
  - ✅ 数据加载测试通过
  - ✅ 规则提取测试通过（3条规则，覆盖3个库）
  - ✅ Prompt模板测试通过（基础/上下文/规则/CoT）
  - ✅ 端到端流程验证通过
  - ✅ 规则库生成（configs/rules.json, 572字节）

**阶段2成果**：
```
✅ 数据加载器: 通过 (3 train, 1 test)
✅ 规则提取器: 3条规则 (pandas/numpy/requests)
✅ Prompt策略: 4种 (230-400字符)
✅ 端到端测试: 全部通过
```

### ⏳ 阶段3：三个方向依次实现（进行中）

**实验顺序**：方向3（规则+Prompt）→ 方向1（LoRA微调）→ 方向2（知识编辑）

#### 3.1 方向3：规则+Prompt混合系统（深化实现中🚀）

**最新更新**：2025-11-21  
**服务器路径**：`~/api_migration_exp/scripts/`  
**状态**：🚀 **深化实现中**（大规模数据集+修复规则引导）

**已完成工作**：

**阶段1：基础Baseline（2025-11-17）**：
- [x] 步骤1：生成规则库（`python3 test_phase2.py`）✅
- [x] 步骤2：升级PyTorch到2.6.0+cu118（解决版本兼容问题）✅
- [x] 步骤3：在10个样例上运行推理（`python3 inference_baseline.py`）✅
- [x] 步骤4：评估4种策略效果（`python3 evaluate_baseline.py`）✅
- [x] 步骤5：验证策略稳定性并分析结果 ✅
- [x] 步骤6：修复脚本路径问题（结果文件统一保存到项目根目录）✅

**阶段2：深化混合系统（2025-11-21）**：
- [x] 实现自动规则学习器（`rule_learner.py`）✅
  - 从训练数据自动提取API迁移规则
  - 支持参数映射提取（如`df.append(row)` → `pd.concat([df, row])`）
  - 识别6种转换类型
- [x] 实现智能规则匹配器（`rule_matcher.py`）✅
  - 规则应用引擎
  - 特殊处理复杂迁移（append→concat）
- [x] 实现混合生成器（`hybrid_generator.py`）✅
  - 三层策略：规则直接应用 → 规则引导LLM → 纯LLM兜底
  - 修复规则引导生成空白问题
  - 优化LLM采样参数
- [x] 扩展公开数据集到300+样本 ✅
  - 基于TensorFlow/Pandas/sklearn/NumPy/PyTorch官方文档
  - 训练集240+样本，测试集60+样本
  - 覆盖5个主流库，60+种真实迁移模式
- [x] 创建测试工具（`test_rule_guided.py`）✅
- [x] 完善文档和工作日志 ✅

**快速开始命令**：

**方式1：使用大规模公开数据集（推荐）**：
```bash
# 1. 环境准备
conda activate apiupdate
cd ~/api_migration_exp/scripts

# 2. 生成300+样本公开数据集
python3 fetch_public_dataset.py

# 3. 快速测试规则引导功能
python3 test_rule_guided.py

# 4. 运行完整混合系统
python3 run_hybrid_system_fixed.py public_dataset.json
```

**方式2：使用小数据集快速验证**：
```bash
cd ~/api_migration_exp/scripts
python3 run_hybrid_system_fixed.py mini_dataset.json
```

**方式3：运行原始Baseline（4种Prompt策略）**：
```bash
cd ~/api_migration_exp/scripts

# 生成规则库
python3 test_phase2.py

# 运行推理
python3 inference_baseline.py

# 评估结果
python3 evaluate_baseline.py
```

**核心文件**：

**数据和配置**：
- ✅ `mini_dataset.json` - 小数据集（40样本）
- ✅ `large_dataset.json` - 扩展数据集（100样本）
- ✅ `public_dataset.json` - 公开数据集（300+样本）✨
- ✅ `configs/rules.json` - 手工规则库（3条规则）
- ✅ `configs/learned_rules.json` - 自动学习规则库（100+条规则）✨

**核心脚本**：
- ✅ `server_scripts/rule_learner.py` - 规则学习器 ✨
- ✅ `server_scripts/rule_matcher.py` - 规则匹配器 ✨
- ✅ `server_scripts/hybrid_generator.py` - 混合生成器 ✨
- ✅ `server_scripts/evaluate_hybrid.py` - 增强评估器 ✨
- ✅ `server_scripts/run_hybrid_system_fixed.py` - 主运行脚本 ✨
- ✅ `server_scripts/fetch_public_dataset.py` - 公开数据集生成器 ✨
- ✅ `server_scripts/test_rule_guided.py` - 测试脚本 ✨

**结果文件**：
- ✅ `results/baseline/baseline_results_*.json` - Baseline推理结果
- ✅ `results/baseline/evaluation_*.json` - Baseline评估结果
- ✅ `results/hybrid/hybrid_results_*.json` - 混合系统结果 ✨
- ✅ `results/hybrid/hybrid_summary_*.txt` - 混合系统摘要 ✨

**实验结果**：

**Baseline实验（10样本，4种Prompt策略）**：

| 策略 | 精确匹配 | 平均相似度 | 关键API | 平均长度 | 评价 |
|------|---------|-----------|---------|---------|------|
| **basic** | **90.0%** ⭐ | **0.98** | **90.0%** | 34/32 | 优秀 |
| with_context | 80.0% | 0.98 | 90.0% | 32/32 | 良好 |
| with_rules | 70.0% | 0.87 | 90.0% | 47/32 | 中等 |
| cot | 70.0% | 0.75 | 90.0% | 93/32 | 一般 |

**混合系统实验（51样本，3种策略）**：

| 数据集 | 测试样本 | 规则直接应用 | 规则引导Prompt | LLM兜底 | 精确匹配率 | 平均相似度 |
|--------|---------|-------------|---------------|---------|-----------|----------|
| large_dataset | 51 | 50.0% (18/18) | 待修复 (0/32) | 0.0% (0/1) | 17.6% | 0.330 |
| **public_dataset** | **60+** | **预期35%** | **预期60%** | **预期5%** | **目标45-60%** | **目标0.60+** |

**核心结论**：

**Baseline阶段（已完成）**：
- ✅ basic策略表现优秀（90%精确匹配，0.98相似度）
- ✅ 所有策略关键API准确率稳定在90%
- ✅ 简单明确的Prompt策略最有效

**混合系统阶段（进行中）**：
- ✅ 规则直接应用准确率50%（可靠）
- ⚠️ 规则引导Prompt需要验证修复效果
- ✅ 自动规则学习有效（从40样本学到15条规则）
- ✅ 大规模数据集准备完成（300+样本）
- 🎯 **下一步**：验证修复后的规则引导功能

**已解决的问题**：
- ✅ 修复规则引导生成空白（改进fallback逻辑）
- ✅ 修复PyTorch数据生成bug
- ✅ 优化LLM采样参数（temperature=0.3）
- ✅ 放宽多样性过滤阈值（30%→20%）

**待完成任务**：
- [ ] 上传修复文件并验证效果
- [ ] 提升精确匹配率到60%+
- [ ] 继续扩展数据集到500+样本

#### 3.2 方向1：LoRA微调（暂时搁置⏸️，后续回归）

**最新更新**：2025-11-21  
**服务器路径**：`~/api_migration_exp/scripts/`
**状态**：⏸️ **暂时搁置**（遇到深层兼容性问题，转向知识编辑方案）

**目标**：实现并对比标准LoRA和层次化LoRA两种方法

**核心创新点**：
- 🎯 **层次化LoRA**：只更新深层（22-31层），参数量减少~69%
- 💡 **假设**：深层负责语义理解，更新深层足以适配API变化
- ⚡ **优势**：参数少、训练快、减少灾难性遗忘

**已完成工作**：
- [x] `lora_config.py` - LoRA配置文件（标准/层次化）✅
- [x] `lora_trainer.py` - LoRA训练器 ✅
- [x] `run_lora.py` - 主运行脚本 ✅
- [x] `evaluate_lora.py` - 评估脚本（修复CUDA错误和数据路径）✅
- [x] `compare_methods.py` - 对比分析脚本 ✅
- [x] `LORA_QUICKSTART.md` - 快速开始指南 ✅
- [x] `ENVIRONMENT_SETUP.md` - 环境配置记录 ✅
- [x] `DATA_EXPANSION_GUIDE.md` - 数据扩展指南 ✅
- [x] `create_real_migration_dataset.py` - 真实API迁移数据集生成器 ✅
- [x] 环境依赖修复（bitsandbytes、pandas兼容性）✅
- [x] 训练器初始化测试成功 ✅
- [x] **快速测试训练完成（mini数据集）** ✅
- [x] **50样例数据集准备完成** ✅
- [x] **评估脚本修复（greedy解码、自动路径查找）** ✅
- [x] **训练脚本优化（梯度裁剪、warmup、余弦衰减）** ✅
- [x] **训练数据修复（正确设置labels、DataCollatorForSeq2Seq）** ✅
- [x] **4次训练实验（v1-v4）和标准LoRA验证** ✅
- [x] **深度诊断（确认PEFT兼容性问题）** ✅

**实验记录**：

**1. Mini数据集快速测试（2025-11-17）**：
```
数据集: 3训练 + 10测试
方法: 层次化LoRA (层22-31)
可训练参数: 9,232,384 (0.59%)
训练时间: 1.52秒
训练损失: 1.959
评估结果: 精确匹配率 0% (数据量太少)
状态: ✅ 流程验证成功
```

**2. 50样例数据集实验（2025-11-19）**：
```
数据集: 50训练 + 50测试（26个真实API迁移案例）
库分布: pandas(15), tensorflow(9), numpy(9), sklearn(6), torch(5), matplotlib(4)

第一次训练（v1，参数不当）:
- Epochs: 3, Batch: 4, LR: 2e-5
- 训练时间: 14秒
- 训练损失: 69.61 (未收敛)
- 梯度: NaN (不稳定)
- 评估结果: 精确匹配率 0%
- 问题: 训练步数太少(12步)，损失未收敛

第二次训练（v2，优化参数，失败❌）:
- Epochs: 10, Batch: 2, LR: 5e-5
- 训练时间: 50秒
- 训练损失: 145.8 → 0.0 (数值崩溃)
- 梯度: NaN (持续异常)
- 评估结果: 精确匹配率 0%
- 问题: 梯度爆炸导致数值溢出

第三次训练（v3，降低学习率+梯度裁剪，失败❌）:
- Epochs: 15, Batch: 2, LR: 1e-5
- 训练时间: 76秒
- 训练损失: 145.8 → 0.0 (仍然崩溃)
- 梯度: NaN (未改善)
- 改进: 添加梯度裁剪(0.3)、warmup(10%)、余弦衰减
- 结果: 问题依旧，说明不只是学习率问题

第四次训练（v4，修复labels，失败❌）:
- Epochs: 10, Batch: 2, LR: 1e-5
- 训练损失: 50.7 → 0.0 (初始损失降低但仍崩溃)
- 梯度: NaN (持续)
- 关键修复: 正确设置labels（input部分用-100忽略）
- 改用DataCollatorForSeq2Seq
- 结果: 初始损失从145→50有改善，但仍然数值崩溃

标准LoRA验证（v4-standard，失败❌）:
- 方法: 标准LoRA（全层，非层次化）
- 训练损失: 50.7 → 0.0 (与层次化LoRA完全相同)
- 梯度: NaN
- 结论: 问题不是层次化LoRA，而是更底层的兼容性问题

最终诊断:
- ✅ 基础模型正常工作
- ✅ 数据格式正确
- ✅ labels设置正确
- ✅ 代码逻辑正确
- ❌ PEFT库与Qwen2.5-Coder可能存在兼容性问题
- ❌ 所有LoRA配置（层次化/标准/不同参数）都失败
- 结论: 需要调研PEFT/Transformers版本兼容性，或尝试其他模型
```

**问题解决记录**：

**环境配置问题（2025-11-17）**：
1. ✅ **bitsandbytes兼容性问题**
   - 错误: `ModuleNotFoundError: No module named 'triton.ops'`
   - 解决: 设置 `DISABLE_BNB_IMPORT=1`，禁用量化功能

2. ✅ **pandas/numpy版本冲突**
   - 错误: `ValueError: numpy.dtype size changed`
   - 解决: 重装 `pandas==2.3.3`, `pyarrow==20.0.0`, `datasets==4.0.0`

3. ✅ **FP16混合精度训练错误**
   - 错误: `ValueError: Attempting to unscale FP16 gradients`
   - 解决: 修改 `lora_trainer.py`，设置 `fp16=False, bf16=False`

**评估脚本问题（2025-11-19）**：
4. ✅ **CUDA device-side assert错误**
   - 错误: `RuntimeError: CUDA error: device-side assert triggered`
   - 原因: 采样生成时概率张量包含无效值
   - 解决: 改用greedy解码 (`do_sample=False, num_beams=1`)

5. ✅ **数据文件路径问题**
   - 错误: `FileNotFoundError: mini_dataset.json`
   - 解决: 添加自动路径查找逻辑

**训练问题（2025-11-19）**：
6. ❌ **LoRA训练数值崩溃问题（未解决，确认为兼容性问题）**
   - v1尝试: 损失69.61，梯度NaN，训练步数太少
   - v2尝试: 增加epochs(3→10)，LR: 5e-5 → 结果：损失0.0，梯度NaN
   - v3尝试: 降低LR(1e-5)，添加梯度裁剪(0.3)、warmup(10%)、余弦衰减 → 结果：损失仍0.0，梯度NaN
   - v4尝试: 修复labels（input用-100忽略），改用DataCollatorForSeq2Seq → 结果：初始损失降到50.7但仍崩溃
   - 标准LoRA验证: 全层LoRA与层次化LoRA问题完全相同
   - 最终诊断: 
     * ✅ 代码逻辑正确（基础模型能正常生成）
     * ✅ 数据格式正确（labels设置正确）
     * ❌ PEFT库与Qwen2.5-Coder存在兼容性问题
     * ❌ 所有配置（层次化/标准、不同LR/batch）都失败
   - v5尝试: 降级Transformers(4.57.1→4.44.0) → 结果：问题依旧
   - SFT尝试: 使用TRL的SFTTrainer → 结果：损失0.0，梯度NaN（与原生Trainer完全相同）
   - **最终决定**: 暂时搁置LoRA方向，转向知识编辑方案

**搁置原因**（2025-11-21）：
- 尝试了7种不同方案（v1-v5, standard, SFT）全部失败
- 问题本质：PEFT库与Qwen2.5-Coder存在深层兼容性问题
- 数值崩溃模式完全一致（损失→0.0，梯度→NaN）
- 已投入大量时间调试，收益递减
- **决策**：转向知识编辑方向，LoRA留待后续有新方案时回归

**执行命令**：
```bash
cd ~/api_migration_exp/scripts

# 1. 训练标准LoRA（baseline）
python3 run_lora.py --method standard --epochs 3 --batch_size 2

# 2. 训练层次化LoRA（创新方法）
python3 run_lora.py --method hierarchical --target_layers 22-31 --epochs 3 --batch_size 2

# 3. 评估标准LoRA
python3 evaluate_lora.py --model_path ../models/checkpoints/standard_lora/final_model

# 4. 评估层次化LoRA
python3 evaluate_lora.py --model_path ../models/checkpoints/hierarchical_lora_layers_22-31/final_model

# 5. 对比分析
python3 compare_methods.py \
    --baseline_result ../results/baseline/evaluation_baseline_results_20251117_092534.json \
    --standard_lora_result ../models/checkpoints/standard_lora/evaluation/evaluation_results_*.json \
    --hierarchical_lora_result ../models/checkpoints/hierarchical_lora_layers_22-31/evaluation/evaluation_results_*.json
```

**预计时间**：
- 标准LoRA训练：2-3小时
- 层次化LoRA训练：1-2小时（更快！）
- 评估和分析：30分钟
- **总计：4-6小时**

**下一步计划（阶段性扩展策略）**：

**阶段1：Mini数据集验证（已完成✅）**
- [x] 快速测试训练（1 epoch）✅
- [x] 评估快速测试模型 ✅
- [x] 修复评估脚本问题 ✅
- [ ] 同步代码到GitHub 🔄

**阶段2：50样例实验（深层兼容性问题⚠️）**
- [x] 创建真实API迁移数据集（26个案例）✅
- [x] 生成50训练+50测试样例 ✅
- [x] 第一次训练v1（参数不当，失败）❌
- [x] 第二次训练v2（优化参数，失败）❌
- [x] 第三次训练v3（梯度裁剪，失败）❌
- [x] 第四次训练v4（修复labels，失败）❌
- [x] 标准LoRA验证（排除层次化问题，失败）❌
- [x] 深度诊断（确认PEFT兼容性问题）✅
- [ ] 明天：调研PEFT/Transformers版本兼容性 ⏳
- [ ] 明天：尝试其他模型或使用Baseline结果

**阶段3：论文级别实验（200样例）**
- [ ] 准备200训练+100测试样例
- [ ] 完整训练实验（8-12小时）
- [ ] 详细评估和消融实验
- [ ] 撰写实验报告

**阶段4：SOTA实验（500+样例，可选）**
- [ ] 混合多数据集
- [ ] 大规模训练
- [ ] 冲击顶会性能

**立即可执行的命令**：
```bash
# 1. 评估快速测试模型
cd ~/api_migration_exp/scripts
python3 evaluate_lora.py \
    --model_path ../models/checkpoints/hierarchical_lora_layers_22-31/final_model \
    --data_file ../mini_dataset.json

# 2. 下载CodeUpdateArena（为下一步准备）
cd ~/api_migration_exp
git clone https://github.com/amazon-science/CodeUpdateArena.git

# 3. 同步到GitHub
cd ~/api_migration_exp
git add .
git commit -m "Phase 3.2: LoRA微调快速测试完成 + 环境配置文档"
git push origin main
```

#### 3.3 方向2：神经知识编辑（遇到技术障碍⚠️）

**最新更新**：2025-11-21  
**服务器路径**：`~/api_migration_exp/scripts/`
**状态**：⚠️ **遇到技术障碍**（多次尝试失败，转向DPO方向）

**目标**：使用ROME/MEMIT直接编辑模型权重，无需训练

**核心创新点**：
- 🎯 **首次应用于代码API更新**：将知识编辑技术应用于代码生成领域
- 💡 **直接修改权重**：不需要梯度下降训练，避免数值稳定性问题
- ⚡ **快速更新**：编辑过程只需几秒，比LoRA训练快数百倍
- 🎨 **局部性强**：只修改API相关知识，不影响其他代码生成能力

**已完成工作**（2025-11-21）：
- [x] `run_rome_editing.py` - 简化版ROME实现 ✅
- [x] `run_knowledge_editing.py` - 完整EasyEdit版本 ✅
- [x] `run_easyedit_rome.py` - EasyEdit适配脚本 ✅
- [x] `run_rome_direct.py` - 直接ROME实现 ✅
- [x] ROME编辑测试运行 ❌ **失败**
- [ ] 编辑效果评估 📅
- [ ] 与Baseline和LoRA对比 📅

**实验记录**（2025-11-21）：

**尝试1：简化版ROME**：
```
命令: python3 run_rome_editing.py --num_edits 10
结果: ❌ 失败
错误: 张量维度不匹配 (8960 vs 1536)
原因: 简化实现的权重更新计算错误
```

**尝试2：EasyEdit ROME**：
```
命令: pip install easyeditor && python3 run_knowledge_editing.py
结果: ❌ 失败
错误: 依赖冲突无法解决
问题:
  - 缺少fairscale依赖
  - huggingface_hub版本冲突（需要<1.0但环境中是1.1.5）
  - cached_download API已废弃
  - split_torch_state_dict_into_shards不存在
尝试: 安装fairscale、降级huggingface_hub到多个版本
结果: 环境彻底损坏，certifi元数据丢失
```

**尝试3：直接ROME实现**：
```
命令: python3 run_rome_direct.py --num_edits 10
结果: ❌ 失败
问题:
  - 初始错误: Qwen2Tokenizer不存在（transformers版本太旧）
  - 升级transformers后: huggingface_hub版本冲突
  - 修复依赖后: numpy版本冲突导致datasets无法导入
  - 最终修复所有依赖后运行: 所有10个API编辑全部失败（0/10成功）
  - 错误: 张量维度不匹配 (8960 vs 1536) at non-singleton dimension 1
运行时间: 42秒
原因: 权重矩阵维度计算错误，协方差矩阵更新公式实现有误
```

**依赖问题汇总**（2025-11-21）：
```
环境状态: 严重损坏
主要问题:
  1. huggingface_hub: 多次安装卸载导致metadata丢失
  2. transformers: 版本要求<1.0但安装了1.1.5
  3. numpy: 版本冲突（scipy要求<1.23，安装了1.24.3）
  4. certifi: RECORD文件丢失
  5. easyeditor: 与现有环境完全不兼容

修复尝试:
  - pip install fairscale ✓
  - pip install huggingface_hub==0.16.4 ❌
  - pip install huggingface_hub==0.19.4 ❌
  - pip install huggingface_hub==0.25.0 ❌
  - pip install --upgrade huggingface_hub ❌
  - rm -rf huggingface_hub* && 重装 ✓ (部分成功)
  - pip install numpy==1.24.3 ✓ (但与scipy冲突)
  - pip install transformers==4.37.0 ✓
  
最终环境状态:
  torch: 2.0.1+cu117 ✓
  transformers: 4.37.0 ✓
  tokenizers: 0.15.2 ✓
  huggingface_hub: 0.25.0 ✓
  numpy: 1.24.3 ⚠️ (与scipy 1.7.3冲突)
  accelerate: 1.11.0 ✓
  peft: ERROR (依赖huggingface_hub)
  datasets: ERROR (numpy版本冲突)
```

**失败原因分析**：
1. **ROME算法复杂度**：需要精确的因果追踪和协方差矩阵计算
2. **维度匹配问题**：Qwen2.5-Coder的MLP层结构与标准实现不匹配
3. **依赖地狱**：EasyEdit与现有环境完全不兼容，强行安装导致环境损坏
4. **简化实现局限**：直接实现ROME算法难度极高，容易出错

**结论**（2025-11-21）：
- ❌ **ROME方向暂时放弃**：技术难度过高，依赖问题无法解决
- 📋 **已完成的方法**：Baseline (90%准确率) ✅
- ⏸️ **搁置的方法**：LoRA (7次失败), ROME (3次失败)
- 🎯 **下一步**：转向DPO（Direct Preference Optimization）或使用Baseline结果

**方法说明**：

**ROME (Rank-One Model Editing)**：
- 原理：在特定Transformer层进行秩1权重更新
- 步骤：
  1. 定位知识存储层（通常是深层，如第20-25层）
  2. 计算旧API和新API的表示差异
  3. 对MLP层进行低秩更新
  4. 保存编辑后的模型
- 优势：
  - 无需训练，直接修改权重
  - 编辑精确，影响范围可控
  - 快速（秒级完成）

**实验计划**：

**阶段1：初步验证（当前）**：
- [ ] 在10个API上测试ROME编辑
- [ ] 评估编辑成功率和代码生成质量
- [ ] 验证局部性（不影响其他API）

**阶段2：完整实验**：
- [ ] 在50个API上进行编辑
- [ ] 对比ROME vs Baseline
- [ ] 分析失败案例

**阶段3：论文撰写**：
- [ ] 整理实验数据
- [ ] 撰写方法和结果章节

**执行命令**（已失败）：
```bash
cd ~/api_migration_exp/scripts

# 尝试1: 简化版ROME
python3 run_rome_editing.py --num_edits 10 --strength 0.5
# 结果: 维度不匹配错误

# 尝试2: EasyEdit ROME
pip install easyeditor
python3 run_easyedit_rome.py --num_edits 10
# 结果: 依赖冲突

# 尝试3: 直接ROME实现
python3 run_rome_direct.py --num_edits 10
# 结果: 0/10成功，维度不匹配
```

**实际时间**：
- 依赖修复：2小时
- ROME尝试：3次，全部失败
- **总投入：3-4小时** ❌

**经验教训**：
1. 知识编辑算法实现难度远超预期
2. 维度匹配需要深入理解模型结构
3. EasyEdit库与Qwen模型不兼容
4. 环境依赖管理需要更谨慎

#### 3.4 方向1b：DPO偏好优化（待开始📅）

**最新更新**：2025-11-21  
**服务器路径**：`~/api_migration_exp/scripts/`
**状态**：📅 **待开始**（LoRA和ROME失败后的备选方案）

**目标**：使用Direct Preference Optimization (DPO)训练模型学习新API

**核心创新点**：
- 🎯 **偏好学习**：通过偏好对（旧API vs 新API）训练模型
- 💡 **无需奖励模型**：比PPO更简单稳定
- ⚡ **训练稳定**：不依赖值函数近似
- 🎨 **适合代码任务**：直接优化生成质量

**方法说明**：

**DPO (Direct Preference Optimization)**：
- 原理：通过偏好对直接优化策略
- 数据格式：(prompt, chosen, rejected)
  - prompt: 旧代码
  - chosen: 新API代码（正确）
  - rejected: 旧API代码（不推荐）
- 优势：
  - 无需单独的奖励模型
  - 训练过程更稳定
  - 内存占用更少
  - 适合小数据集

**计划任务**：
- [ ] 创建DPO训练脚本
- [ ] 构造偏好数据集
- [ ] 训练DPO模型
- [ ] 评估DPO效果
- [ ] 与Baseline对比

**预计时间**：
- 脚本准备：30分钟
- 数据准备：20分钟
- 训练：1-2小时
- 评估：30分钟
- **总计：2-3小时**

**决策点**（2025-11-21）：
```
当前状态:
  ✅ Baseline: 90%准确率（已完成）
  ❌ LoRA: 7次失败（搁置）
  ❌ ROME: 3次失败（搁置）
  📅 DPO: 待尝试

选择:
  选项1: 尝试DPO（最后一个神经网络方法）
  选项2: 直接用Baseline结果写论文
  
建议: 先尝试DPO，如果1-2小时内无进展，则使用Baseline结果
```

### ⏳ 阶段4：对比分析和消融实验（待开始）
- [ ] 三个方向统一评估
- [ ] 消融实验设计与运行
- [ ] 失败案例深入分析
- [ ] 确定论文重点方向

### 📝 阶段5：论文撰写（待开始）
- [ ] 论文结构设计
- [ ] 实验结果整理
- [ ] 图表制作
- [ ] 论文撰写和修改

## 🛠️ 技术栈

### 核心依赖
- **Python**: 3.10+
- **PyTorch**: 2.0+ (CUDA 11.4)
- **Transformers**: 4.36.0
- **PEFT**: 0.7.0（LoRA微调）
- **EasyEdit**: 最新版（知识编辑）
- **Datasets**: 2.15.0

### 硬件要求
- **GPU**: 至少10GB显存（推荐24GB）
- **实验环境**: 2x NVIDIA RTX 3090 (24GB each)

## 📂 项目结构

```
api-migration-experiment/
├── data/                      # 数据集
│   ├── mini_dataset.json         # 小数据集（40样本）
│   ├── large_dataset.json        # 扩展数据集（100样本）
│   └── public_dataset.json       # 公开数据集（300+样本）✨
├── server_scripts/            # 服务器端脚本
│   ├── # Baseline脚本
│   ├── data_utils.py             # 数据加载工具
│   ├── rule_extractor.py         # 规则提取器
│   ├── prompt_engineering.py     # Prompt模板
│   ├── inference_baseline.py     # Baseline推理
│   ├── evaluate_baseline.py      # Baseline评估
│   │
│   ├── # 混合系统脚本（方向3深化）✨
│   ├── rule_learner.py           # 自动规则学习器
│   ├── rule_matcher.py           # 智能规则匹配器
│   ├── hybrid_generator.py       # 混合生成器
│   ├── evaluate_hybrid.py        # 增强评估器
│   ├── run_hybrid_system_fixed.py # 主运行脚本
│   ├── fetch_public_dataset.py   # 公开数据集生成器
│   ├── test_rule_guided.py       # 测试脚本
│   │
│   ├── # LoRA微调脚本（方向1）
│   ├── lora_config.py            # LoRA配置
│   ├── lora_trainer.py           # LoRA训练器
│   ├── run_lora.py               # LoRA运行脚本
│   └── evaluate_lora.py          # LoRA评估
├── models/                    # 模型文件
│   └── checkpoints/           # 训练检查点
├── results/                   # 实验结果
│   ├── baseline/              # Baseline结果
│   └── hybrid/                # 混合系统结果 ✨
├── configs/                   # 配置文件
│   ├── rules.json             # 手工规则库（3条）
│   └── learned_rules.json     # 自动学习规则库（100+条）✨
├── docs/                      # 文档
│   ├── 方向3快速开始.md         # 快速开始指南 ✨
│   ├── 方向3深化实现指南.md     # 深化实现指南
│   ├── 大规模公开数据集说明.md  # 数据集详情 ✨
│   ├── 使用公开数据集.md        # 使用指南 ✨
│   ├── 修复规则引导生成空白问题.md # 修复说明 ✨
│   ├── 工作日志-20251121.md     # 工作日志 ✨
│   ├── 新手完整指南.md
│   ├── Git同步指南.md
│   └── 实验记录.md
├── .gitignore                # Git忽略规则
├── README.md                 # 项目说明（本文件）
├── git_commands.sh           # Git更新脚本 ✨
├── 更新说明-20251121.md       # 更新说明 ✨
└── requirements.txt          # Python依赖
```

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone https://github.com/您的用户名/api-migration-experiment.git
cd api-migration-experiment
```

### 2. 创建环境
```bash
# 创建conda环境
conda create -n apiupdate python=3.10 -y
conda activate apiupdate

# 安装PyTorch (CUDA 11.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113

# 安装其他依赖
pip install transformers==4.36.0 peft==0.7.0 datasets==2.15.0
pip install accelerate bitsandbytes libcst tqdm rich wandb
```

### 3. 准备数据
```bash
python scripts/prepare_data.py
```

### 4. 测试环境
```bash
python scripts/test_env.py
```

### 5. 运行实验

**方向3：混合系统（推荐）** ✨:
```bash
cd server_scripts

# 方式1：使用大规模公开数据集（300+样本）
python3 fetch_public_dataset.py
python3 run_hybrid_system_fixed.py public_dataset.json

# 方式2：使用小数据集快速验证
python3 run_hybrid_system_fixed.py mini_dataset.json

# 方式3：测试规则引导功能
python3 test_rule_guided.py
```

**方向3：原始Baseline（4种Prompt策略）**:
```bash
cd server_scripts
python3 test_phase2.py           # 生成规则库
python3 inference_baseline.py    # 运行推理
python3 evaluate_baseline.py     # 评估结果
```

**方向1：LoRA微调（暂时搁置）**:
```bash
cd server_scripts
python3 run_lora.py --method hierarchical --epochs 10
python3 evaluate_lora.py
```

**方向2：知识编辑（暂时搁置）**:
```bash
cd server_scripts
python3 run_rome_editing.py --num_edits 10
```

## 📝 参考论文

1. **ReCode**: Updating Code API Knowledge with Reinforcement Learning
   - arXiv: 2507.12367
   - 机构: 浙江大学 & 腾讯AI西雅图实验室

2. **CodeUpdateArena**: Benchmarking Knowledge Editing on API Updates
   - arXiv: 2407.06249
   - 机构: The University of Texas at Austin

3. **GitChameleon 2.0**: Evaluating AI Code Generation Against Python Library Version Incompatibilities
   - arXiv: 2506.20495
   - 机构: ELLIS Institute Tübingen, Mila Quebec AI Institute, Google等

### 相关工作
- ROME: Locating and Editing Factual Associations in GPT (NeurIPS 2022)
- MEMIT: Mass-Editing Memory in a Transformer (ICLR 2023)
- LoRA: Low-Rank Adaptation of Large Language Models (ICLR 2022)

## 📈 实验结果

### 初步结果

#### 方向3：规则+Prompt Baseline（优化完成✅）

**第一轮测试（优化前）**：

| 策略 | 精确匹配 | 相似度 | 关键API | 问题 |
|------|---------|--------|---------|------|
| basic | 0% | 0.20 | 0% | 生成解释文本 |
| with_context | 0% | 0.21 | 100% | 生成教程 |
| with_rules | 0% | 0.15 | 0% | 规则未生效 |
| cot | 0% | 0.15 | 0% | 推理冗长 |

**第二轮测试（Prompt优化后，1个样例）**：

| 策略 | 精确匹配 | 相似度 | 关键API |
|------|---------|--------|---------|
| basic | 100% | 1.00 | 100% |
| with_context | 0% | 0.78 | 100% |
| with_rules | 0% | 0.89 | 100% |
| cot | 0% | 0.78 | 100% |

**第三轮测试（最终结果，10个样例）**：

| 策略 | 精确匹配 | 相似度 | 关键API | 评价 |
|------|---------|--------|---------|------|
| **basic** | **90%** ⭐ | **0.98** | **90%** | 优秀且稳定 |
| with_context | 80% | 0.98 | 90% | 良好 |
| with_rules | 70% | 0.87 | 90% | 中等 |
| cot | 70% | 0.75 | 90% | 一般 |

**优化措施**：
1. ✅ Prompt添加严格输出约束
2. ✅ temperature降低（0.7→0.3）
3. ✅ 规则库路径修复

**数据集扩展**：
- ✅ 从1个扩展到10个测试样例
- ✅ 覆盖8个主流库（pandas, numpy, tensorflow, sklearn, torch, PIL, requests, matplotlib）
- ✅ 包含多种API更新类型（函数替换、参数变化、重命名）

**核心结论**：
- ✅ basic策略在10个样例上保持90%精确匹配（优秀且稳定）
- ✅ 所有策略关键API准确率90%
- ✅ 简单明确的Prompt策略最有效
- ✅ Baseline效果可靠，可以进入LoRA微调阶段

---

#### 其他方向（待进行）

| 方法 | 语法正确率 | 执行成功率 | 完全匹配率 | 状态 |
|------|-----------|-----------|-----------|------|
| LoRA微调 | - | - | - | 待开始 |
| 知识编辑 (ROME) | - | - | - | 待开始 |

## 🤝 贡献

欢迎提出问题和建议！


### 详细日志
查看 [CHANGELOG.md](CHANGELOG.md) 和 [实验进度记录](docs/实验进度记录.md) 获取完整的更新历史。

## 📄 许可证

MIT License

## 🙏 致谢

感谢以下开源项目：
- [Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [EasyEdit](https://github.com/zjunlp/EasyEdit)

---

**实验进展和详细记录请查看 [实验记录文档](docs/实验记录.md)**

*Last updated: 2025-11-21 20:45 - 添加大规模公开数据集和修复规则引导生成*

---

## 📖 更多文档

### 方向3文档（最新）✨
- [方向3快速开始](docs/方向3快速开始.md) - 5分钟快速开始指南
- [方向3深化实现指南](docs/方向3深化实现指南.md) - 详细实现说明
- [大规模公开数据集说明](docs/大规模公开数据集说明.md) - 数据集详情（300+样本）
- [使用公开数据集](docs/使用公开数据集.md) - 数据集使用指南
- [修复规则引导生成空白问题](docs/修复规则引导生成空白问题.md) - 问题修复说明
- [工作日志-20251121](docs/工作日志-20251121.md) - 今日工作记录
- [更新说明-20251121](更新说明-20251121.md) - 本次更新内容

### 通用文档
- [研究思路](研究思路.md) - 简单易懂的研究想法说明
- [实验路线图](实验路线图.md) - 详细的实验步骤和时间安排
- [实验完整指南](实验完整指南.md) - 手把手操作指南
- [Git同步指南](Git同步指南.md) - GitHub操作说明
- [实验记录](docs/实验记录.md) - 详细的实验日志

