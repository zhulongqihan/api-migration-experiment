#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA微调配置文件
定义标准LoRA和层次化LoRA的配置参数
"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class LoRAConfig:
    """LoRA配置基类"""
    
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B"
    
    # LoRA参数
    lora_r: int = 8  # LoRA秩
    lora_alpha: int = 16  # LoRA缩放因子
    lora_dropout: float = 0.05
    
    # 训练参数
    num_epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_length: int = 512
    
    # 优化器参数
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    
    # 保存和日志
    output_dir: str = "../models/checkpoints"
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    
    # 设备
    device: str = "auto"  # auto, cuda, cpu
    
    def __post_init__(self):
        """初始化后处理"""
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class StandardLoRAConfig(LoRAConfig):
    """标准LoRA配置：更新所有层"""
    
    method_name: str = "standard_lora"
    target_modules: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        # 标准LoRA：更新所有注意力层和MLP层
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",  # Query投影
                "k_proj",  # Key投影
                "v_proj",  # Value投影
                "o_proj",  # Output投影
                "gate_proj",  # MLP门控
                "up_proj",    # MLP上投影
                "down_proj"   # MLP下投影
            ]
        
        self.output_dir = f"{self.output_dir}/{self.method_name}"


@dataclass
class HierarchicalLoRAConfig(LoRAConfig):
    """层次化LoRA配置：只更新深层（创新点）"""
    
    method_name: str = "hierarchical_lora"
    target_modules: List[str] = None
    target_layers: str = "22-31"  # 只更新第22-31层（深层）
    
    def __post_init__(self):
        super().__post_init__()
        # 层次化LoRA：只更新深层的注意力和MLP
        if self.target_modules is None:
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ]
        
        # 解析目标层范围
        start, end = map(int, self.target_layers.split('-'))
        self.layer_range = (start, end)
        
        self.output_dir = f"{self.output_dir}/{self.method_name}_layers_{self.target_layers}"
    
    def should_apply_lora(self, layer_idx: int) -> bool:
        """判断是否应该对该层应用LoRA"""
        return self.layer_range[0] <= layer_idx <= self.layer_range[1]


def get_config(method: str = "standard", **kwargs) -> LoRAConfig:
    """
    获取配置对象
    
    Args:
        method: 方法名称，'standard' 或 'hierarchical'
        **kwargs: 额外的配置参数
    
    Returns:
        配置对象
    """
    if method == "standard":
        return StandardLoRAConfig(**kwargs)
    elif method == "hierarchical":
        return HierarchicalLoRAConfig(**kwargs)
    else:
        raise ValueError(f"未知的方法: {method}")


def print_config(config: LoRAConfig):
    """打印配置信息"""
    from rich.console import Console
    from rich.table import Table
    
    console = Console()
    
    table = Table(title=f"LoRA配置 - {config.method_name}")
    table.add_column("参数", style="cyan")
    table.add_column("值", style="green")
    
    # 基本配置
    table.add_row("模型", config.model_name)
    table.add_row("方法", config.method_name)
    
    # LoRA参数
    table.add_row("LoRA秩 (r)", str(config.lora_r))
    table.add_row("LoRA alpha", str(config.lora_alpha))
    table.add_row("LoRA dropout", str(config.lora_dropout))
    
    # 训练参数
    table.add_row("训练轮数", str(config.num_epochs))
    table.add_row("批次大小", str(config.batch_size))
    table.add_row("学习率", str(config.learning_rate))
    table.add_row("最大长度", str(config.max_length))
    
    # 特殊配置
    if isinstance(config, HierarchicalLoRAConfig):
        table.add_row("目标层范围", config.target_layers)
        table.add_row("层数", str(config.layer_range[1] - config.layer_range[0] + 1))
    
    table.add_row("目标模块", ", ".join(config.target_modules[:3]) + "...")
    table.add_row("输出目录", config.output_dir)
    table.add_row("设备", config.device)
    
    console.print(table)


if __name__ == "__main__":
    """测试配置"""
    from rich.console import Console
    
    console = Console()
    
    # 测试标准LoRA配置
    console.print("\n[bold cyan]标准LoRA配置[/bold cyan]")
    standard_config = get_config("standard")
    print_config(standard_config)
    
    # 测试层次化LoRA配置
    console.print("\n[bold cyan]层次化LoRA配置[/bold cyan]")
    hierarchical_config = get_config("hierarchical", target_layers="22-31")
    print_config(hierarchical_config)
    
    # 计算参数量差异
    console.print("\n[bold yellow]参数量对比[/bold yellow]")
    total_layers = 32  # Qwen2.5-Coder-1.5B有32层
    hierarchical_layers = hierarchical_config.layer_range[1] - hierarchical_config.layer_range[0] + 1
    
    console.print(f"标准LoRA: 更新所有 {total_layers} 层")
    console.print(f"层次化LoRA: 只更新 {hierarchical_layers} 层 (第{hierarchical_config.target_layers}层)")
    console.print(f"参数量减少: {(1 - hierarchical_layers/total_layers)*100:.1f}%")
