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

#### 3.1 方向3：规则+Prompt baseline（已完成首轮✅）
- [x] 规则库构建
- [x] Prompt模板设计（4种策略）
- [x] 模型加载测试（Qwen2.5-Coder-1.5B，CPU运行）
- [x] 完整推理pipeline实现（4种策略测试成功）
- [x] GPU自动检测和降级机制添加
- [x] 评估脚本运行和结果分析
- [x] 失败案例总结
- [ ] **改进Prompt和参数**（生成质量优化）
- [ ] 扩展测试数据集（5-10个样例）

#### 3.2 方向1：LoRA微调（待开始📅）
- [ ] 标准LoRA实现
- [ ] 层次化LoRA实现（核心创新）
- [ ] 消融实验（不同层范围）

#### 3.3 方向2：知识编辑（待开始📅）
- [ ] ROME/MEMIT实现
- [ ] 代码领域适配
- [ ] 编辑效果评估

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
│   ├── raw/                   # 原始数据
│   ├── processed/             # 处理后数据
│   └── test/                  # 测试集
├── src/                       # 源代码
│   ├── baseline/              # 方向3: 规则+Prompt
│   │   ├── rule_extractor.py      # 规则提取
│   │   ├── prompt_engineering.py  # Prompt设计
│   │   └── run_baseline.py        # Baseline运行
│   ├── rl_finetuning/         # 方向1: RL微调
│   │   ├── lora_finetune.py       # LoRA微调
│   │   └── reward_function.py     # 奖励函数
│   ├── knowledge_editing/     # 方向2: 知识编辑
│   │   ├── run_editing.py         # 知识编辑
│   │   └── evaluate_editing.py    # 编辑评估
│   └── utils/                 # 工具函数
│       ├── evaluate.py            # 统一评估
│       └── data_utils.py          # 数据处理
├── models/                    # 模型文件
│   └── checkpoints/           # 训练检查点
├── results/                   # 实验结果
│   ├── baseline/              # Baseline结果
│   ├── rl/                    # RL结果
│   └── editing/               # 编辑结果
├── configs/                   # 配置文件
│   └── rules.json            # API更新规则库
├── scripts/                   # 运行脚本
│   ├── setup_env.sh          # 环境配置
│   ├── prepare_data.py       # 数据准备
│   └── test_env.py           # 环境测试
├── logs/                      # 日志文件
├── docs/                      # 文档
│   ├── 新手完整指南.md
│   ├── Git同步指南.md
│   └── 实验记录.md
├── .gitignore                # Git忽略规则
├── README.md                 # 项目说明（本文件）
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

**Baseline方法**:
```bash
cd src/baseline
python run_baseline.py
```

**LoRA微调**:
```bash
cd src/rl_finetuning
python lora_finetune.py
```

**知识编辑**:
```bash
cd src/knowledge_editing
python run_editing.py
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

**第二轮测试（Prompt优化后）**：

| 策略 | 精确匹配 | 相似度 | 关键API | 改进 |
|------|---------|--------|---------|------|
| **basic** | **100%** ⭐ | **1.00** | **100%** | **完美匹配** |
| with_context | 0% | **0.78** | **100%** | +271% |
| with_rules | 0% | **0.89** | **100%** | +493% |
| cot | 0% | **0.78** | **100%** | +420% |

**优化措施**：
1. ✅ Prompt添加严格输出约束
2. ✅ temperature降低（0.7→0.3）
3. ✅ 规则库路径修复

**关键发现**：
- ✅ basic策略达到完美匹配（100%精确、1.00相似度）
- ✅ 所有策略关键API准确率100%
- ✅ 相似度平均提升370%
- 💡 简单明确的Prompt最有效

**下一步**：
- 扩展测试数据集（5-10个样例）验证效果
- 进入LoRA微调阶段

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

*Last updated: 2025-11-06*

---

## 📖 更多文档

- [研究思路](研究思路.md) - 简单易懂的研究想法说明
- [实验路线图](实验路线图.md) - 详细的实验步骤和时间安排
- [实验完整指南](实验完整指南.md) - 手把手操作指南
- [Git同步指南](Git同步指南.md) - GitHub操作说明
- [实验记录](docs/实验记录.md) - 详细的实验日志

