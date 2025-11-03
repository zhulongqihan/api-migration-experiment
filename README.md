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

### Week 1: 环境准备与Baseline
- [x] 环境配置（2x RTX 3090, Python 3.10.18）
- [x] Git仓库初始化
- [x] conda环境创建（apiupdate）
- [ ] 项目结构搭建
- [ ] 数据集准备
- [ ] Baseline实现（方向3）

### Week 2: 核心方法实现
- [ ] LoRA微调（方向1）
- [ ] 知识编辑（方向2）
- [ ] 三个方向对比

### Week 3: 评估与分析
- [ ] 统一评估框架
- [ ] 失败案例分析
- [ ] 实验报告撰写

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

### 初步结果（更新中...）

| 方法 | 语法正确率 | 执行成功率 | 完全匹配率 | 训练时间 |
|------|-----------|-----------|-----------|---------|
| Baseline (规则+Prompt) | - | - | - | 0min |
| LoRA微调 | - | - | - | -min |
| 知识编辑 (ROME) | - | - | - | -min |

*注：结果将在实验完成后更新*

## 🤝 贡献

欢迎提出问题和建议！

## 📅 时间线

- ✅ 项目初始化
- ✅ 服务器环境配置（2x RTX 3090）
- ✅ conda环境创建（Python 3.10.18）
- ✅ Git仓库初始化

### 待更新...

## 📄 许可证

MIT License

## 🙏 致谢

感谢以下开源项目：
- [Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [EasyEdit](https://github.com/zjunlp/EasyEdit)

---

**实验进展和详细记录请查看 [实验记录文档](docs/实验记录.md)**

*Last updated: 2025-11-02*

