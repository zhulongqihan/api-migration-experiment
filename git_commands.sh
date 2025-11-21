#!/bin/bash
# GitHub 更新指令 - 2025-11-21

# ================================
# 1. 检查当前状态
# ================================
cd ~/api_migration_exp
git status

# ================================
# 2. 添加所有修改的文件
# ================================

# 核心代码文件（修复）
git add server_scripts/hybrid_generator.py
git add server_scripts/rule_learner.py
git add server_scripts/rule_matcher.py
git add server_scripts/run_hybrid_system_fixed.py
git add server_scripts/evaluate_hybrid.py

# 新增文件
git add server_scripts/fetch_public_dataset.py
git add server_scripts/test_rule_guided.py

# 文档更新
git add docs/方向3快速开始.md
git add docs/大规模公开数据集说明.md
git add docs/使用公开数据集.md
git add docs/修复规则引导生成空白问题.md
git add docs/工作日志-20251121.md

# Git指令文件、README和更新说明
git add git_commands.sh
git add README.md
git add 更新说明-20251121.md

# ================================
# 3. 提交更改
# ================================
git commit -m "feat: 扩展公开数据集到300+样本并修复规则引导生成

主要更新：

1. 扩展公开数据集
   - 新增 fetch_public_dataset.py：基于官方文档生成300+样本
   - 数据来源：TensorFlow/Pandas/sklearn/NumPy/PyTorch
   - 训练集240+样本，测试集60+样本
   - 80/20划分，固定随机种子保证可复现

2. 修复规则引导生成空白问题
   - 改进fallback逻辑：规则应用 → LLM生成 → 返回原代码
   - 放宽LLM过滤阈值：30% → 20%
   - 温和采样参数：temperature=0.3, top_p=0.9
   - 修复PyTorch数据生成bug

3. 新增测试工具
   - test_rule_guided.py：快速验证规则引导功能
   - 测试常见迁移案例

4. 文档更新
   - 更新快速开始指南
   - 添加公开数据集使用指南
   - 添加修复说明文档
   - 记录工作日志

待完成：
- 验证修复效果
- 提升精确匹配率到60%+
- 继续扩展数据集到500+样本

相关issue: #扩展数据集 #修复规则引导
"

# ================================
# 4. 推送到远程仓库
# ================================
git push origin main

# 如果是其他分支，使用：
# git push origin <your-branch-name>

# ================================
# 5. 查看提交历史
# ================================
git log --oneline -5

echo ""
echo "✅ Git 提交和推送完成！"
echo ""
echo "下一步："
echo "1. 在GitHub上查看提交"
echo "2. 上传修改的文件到服务器"
echo "3. 运行 test_rule_guided.py 验证修复"
echo "4. 运行完整系统测试"
