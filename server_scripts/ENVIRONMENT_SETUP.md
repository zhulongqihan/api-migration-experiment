# 环境配置记录

## 📅 配置时间
2025-11-17

## 🖥️ 服务器信息
- **主机名**: 3090
- **用户**: zhangchangyu
- **GPU**: 2 × RTX 3090 Ti (24GB each)
- **CUDA**: 11.8
- **Python**: 3.10
- **Conda环境**: apiupdate

## ✅ 已安装的核心依赖

### 深度学习框架
```
torch==2.6.0+cu118
transformers==4.48.0
peft==0.7.0
accelerate==1.11.0
```

### 数据处理
```
pandas==2.3.3
numpy==2.2.6
datasets==4.0.0
pyarrow==20.0.0
scipy==1.15.3
```

### 其他工具
```
rich==13.9.4
```

## ⚠️ 已解决的兼容性问题

### 1. bitsandbytes与triton.ops冲突

**问题**：
```
ModuleNotFoundError: No module named 'triton.ops'
```

**原因**：
- PEFT依赖bitsandbytes
- bitsandbytes需要triton.ops（已在新版本triton中移除）

**解决方案**：
- 卸载bitsandbytes
- 在所有脚本开头设置：`os.environ['DISABLE_BNB_IMPORT'] = '1'`
- 我们不需要量化功能（显存充足）

**影响**：
- ✅ 无负面影响
- ✅ LoRA训练正常工作
- ❌ 无法使用8bit/4bit量化（但我们不需要）

### 2. pandas与numpy版本不兼容

**问题**：
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility
```

**原因**：
- pandas用旧版本numpy编译
- 环境中有新版本numpy

**解决方案**：
```bash
pip uninstall pandas pyarrow datasets -y
pip install --only-binary=:all: pyarrow pandas datasets
```

**最终版本**：
- pandas==2.3.3
- numpy==2.2.6
- pyarrow==20.0.0
- datasets==4.0.0

## 🔧 环境变量设置

在所有LoRA相关脚本中添加：
```python
import os
os.environ['DISABLE_BNB_IMPORT'] = '1'
```

或在`~/.bashrc`中添加：
```bash
export DISABLE_BNB_IMPORT=1
```

## ✅ 验证结果

### 训练器初始化测试
```bash
python3 lora_trainer.py
```

**输出**：
```
✓ 模型加载成功 (cuda)
✓ LoRA配置完成
  可训练参数: 9,232,384 (0.59%)
  总参数: 1,552,946,688
✓ 数据集准备完成: 3 个样例
✓ 训练器初始化成功
```

### PEFT功能测试
```bash
python3 -c "import peft; from peft import LoraConfig, TaskType; print('✅ PEFT正常')"
```

### GPU可用性测试
```bash
python3 -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}, 设备数: {torch.cuda.device_count()}')"
```

## 📝 完整安装命令（参考）

如果需要在新环境中重新配置：

```bash
# 创建conda环境
conda create -n apiupdate python=3.10 -y
conda activate apiupdate

# 安装PyTorch（CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装transformers和PEFT
pip install transformers peft accelerate

# 安装数据处理库（使用预编译版本）
pip install --only-binary=:all: pandas pyarrow datasets scipy

# 安装其他工具
pip install rich

# 验证安装
python3 -c "import torch, transformers, peft; print('✅ 环境配置成功')"
```

## 🚀 下一步

环境配置完成后：
1. 测试训练器：`python3 lora_trainer.py`
2. 快速测试训练：`python3 run_lora.py --method hierarchical --epochs 1 --batch_size 2`
3. 完整训练实验

## 📞 问题排查

### 如果遇到导入错误
```bash
# 检查PEFT
python3 -c "import peft; print(peft.__version__)"

# 检查环境变量
echo $DISABLE_BNB_IMPORT

# 重新设置
export DISABLE_BNB_IMPORT=1
```

### 如果遇到CUDA错误
```bash
# 检查CUDA
nvidia-smi

# 检查PyTorch CUDA
python3 -c "import torch; print(torch.cuda.is_available())"
```

### 如果遇到内存不足
```bash
# 减小batch_size
python3 run_lora.py --method hierarchical --batch_size 1

# 或使用CPU（会很慢）
python3 run_lora.py --method hierarchical --device cpu
```

---

**最后更新**: 2025-11-17  
**状态**: ✅ 环境配置完成，可以开始训练
