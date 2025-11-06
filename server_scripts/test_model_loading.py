#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试代码生成模型加载和推理
推荐模型：Qwen/Qwen2.5-Coder-1.5B（较小，1.5B参数）
"""

import os
import ssl
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rich.console import Console
from rich.panel import Panel

# 禁用SSL证书验证（解决自签名证书问题）
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

console = Console()

def test_model_loading():
    """测试模型加载"""
    
    # 推荐使用的小模型（容易加载）
    model_name = "Qwen/Qwen2.5-Coder-1.5B"
    
    console.print(Panel.fit(
        f"[bold cyan]测试模型加载和推理[/bold cyan]\n"
        f"模型: {model_name}\n"
        f"镜像: {os.getenv('HF_ENDPOINT', '未配置')}"
    ))
    
    try:
        console.print("\n[yellow]步骤1/3: 加载Tokenizer...[/yellow]")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        console.print("[green]✓ Tokenizer加载成功[/green]")
        
        console.print("\n[yellow]步骤2/3: 加载模型...[/yellow]")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).cuda()  # 直接加载到GPU，不使用device_map
        console.print(f"[green]✓ 模型加载成功[/green]")
        console.print(f"  - 参数量: {model.num_parameters() / 1e9:.2f}B")
        console.print(f"  - 设备: {model.device}")
        
        console.print("\n[yellow]步骤3/3: 测试推理...[/yellow]")
        
        # 简单的代码生成测试
        test_prompt = "# 用Python写一个函数，计算两个数的和\ndef add(a, b):"
        
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )
        
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        console.print("[green]✓ 推理测试成功[/green]")
        console.print("\n[bold]生成的代码：[/bold]")
        console.print(Panel(generated_code, border_style="green"))
        
        console.print("\n[bold green]✅ 所有测试通过！模型可以正常使用[/bold green]")
        
        return True
        
    except Exception as e:
        console.print(f"\n[bold red]❌ 错误：{e}[/bold red]")
        console.print("\n[yellow]可能的解决方案：[/yellow]")
        console.print("1. 检查网络连接：ping hf-mirror.com")
        console.print("2. 检查HF_ENDPOINT环境变量：echo $HF_ENDPOINT")
        console.print("3. 尝试使用ModelScope（安装：pip install modelscope）")
        console.print("4. 手动下载模型到本地")
        
        return False

if __name__ == "__main__":
    success = test_model_loading()
    exit(0 if success else 1)