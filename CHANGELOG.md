# 更新日志

## [2025-11-25] - 混合系统数值稳定性修复

### 修复的问题
- **LLM生成数值崩溃**：修复`RuntimeError: probability tensor contains either inf, nan or element < 0`
 - 根本原因：Qwen2.5-Coder + transformers 4.44 + 采样模式(temperature>0)导致数值溢出
 - 解决方案：采样模式全部改用greedy解码(temperature=0.0, do_sample=False)
 - 避免使用`repetition_penalty`和`no_repeat_ngram_size`

### 性能优化
- **策略阈值调整**：
 - 规则直接应用阈值：0.85 → 0.5（覆盖更多样本）
 - 规则引导阈值：0.6 → 0.3
 - 增加安全检查：确保生成代码与原代码不同
- **预期提升**：
 - 规则直接应用覆盖：21次 → 35-40次
 - 整体精确匹配率：23.5% → 35-45%

### 新增功能
- **成功案例展示**：在评估输出中显示前5个成功案例
 - 展示旧代码、生成代码、期望代码
 - 显示使用的策略和置信度
- **调试模式**：HybridGenerator支持debug参数

### 修改文件
- `server_scripts/hybrid_generator.py`：
 - 修改`_llm_generate`默认temperature=0.0
 - 移除greedy分支的repetition_penalty
 - 提高采样分支最低温度到0.7
 - 降低策略阈值
 - 添加debug支持
- `server_scripts/evaluate_hybrid.py`：
 - 新增成功案例展示部分
 - 显示前5个成功案例详情

### 当前性能
- 测试集：51个样本
- 精确匹配率：23.5%
- 平均相似度：0.806
- 策略分布：
 - 规则直接应用：21次（57.1%准确率）
 - 规则引导Prompt：29次（0.0%准确率，greedy复制问题）
 - LLM兜底：1次（0.0%准确率）

### 下一步优化
- 验证阈值调整后的效果
- 解决采样模式数值问题（升级transformers或换模型）
- 扩大数据集规模

---

## [2025-11-24] - Baseline失败排查

### 发现的问题
- **Baseline准确率归零**：之前90%的准确率降为0%
- **生成异常输出**：生成重复特殊字符(如`!!"!#!$...`)
- **数值稳定性问题**：与混合系统类似的`RuntimeError`

### 尝试的解决方案
- 调整generation参数：temperature从0.7到0.0，移除repetition_penalty
- 环境对比：怀疑是transformers版本或环境变化导致
- **结论**：暂时搁置baseline问题，专注混合系统优化

---

## [2025-11-05] - 阶段2完成

### 已完成
- **阶段2：数据和框架准备**
 - 创建数据集（3训练样例+1测试样例）
 - 实现数据加载工具（data_utils.py, 95行）
 - 实现规则提取器（rule_extractor.py, 87行）
 - 设计Prompt模板（prompt_engineering.py, 107行，4种策略）
 - 完整测试套件（test_phase2.py, 209行）
 - 规则库生成（configs/rules.json, 572字节）

### 测试结果
```
 数据加载器测试: 通过 (3 train, 1 test)
 规则提取测试: 通过 (3条规则，3个库)
 Prompt模板测试: 通过 (4种策略)
 端到端流程: 通过
 规则库保存: 成功
```

### 修复的问题
- 修复test_phase2.py中的路径错误（data/processed/ → ../data/processed/）

### 新增文件
- `server_scripts/data_utils.py` - 数据加载工具
- `server_scripts/rule_extractor.py` - 规则提取器
- `server_scripts/prompt_engineering.py` - Prompt模板
- `server_scripts/test_phase2.py` - 测试脚本
- `server_scripts/mini_dataset.json` - 数据集
- `server_scripts/README.md` - 使用说明
- `docs/实验进度记录.md` - 进度跟踪

### 更新文档
- `实验完整指南.md` - 更新阶段2完成状态
- `README.md` - 更新进度追踪

---

## [2025-11-05] - 阶段1完成

### 已完成
- **阶段1：环境配置**
 - 服务器配置: GPU RTX 3090 (24GB), CUDA 11.4
 - conda环境: apiupdate (Python 3.10)
 - PyTorch: 2.x+cu118（兼容CUDA 11.4）
 - 核心依赖: transformers 4.36.0, peft 0.7.0, datasets, accelerate 0.25.0

### 解决的问题
1. PyTorch安装: cu113 → cu118索引（兼容CUDA 11.4）
2. CMake版本: 使用conda安装datasets解决依赖
3. 网络问题: 识别无法访问HuggingFace，计划使用镜像

### 项目结构
```
~/api_migration_exp/
├── data/
├── src/
├── models/
├── results/
├── configs/
├── scripts/
└── logs/
```

---

## 下一步计划

### 阶段3选项
1. **方案A**: 解决网络问题，加载小型模型测试
2. **方案B**: 实现纯规则匹配baseline（不需要模型）
3. **方案C**: 扩展数据集规模

**状态**: 等待选择下一步方向

