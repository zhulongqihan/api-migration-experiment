#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从CodeUpdateArena数据集准备训练数据
将原始数据转换为我们的格式
"""

import json
import os
from pathlib import Path
from typing import List, Dict
from rich.console import Console
from rich.progress import track

console = Console()


class CodeUpdateArenaConverter:
    """CodeUpdateArena数据集转换器"""
    
    def __init__(self, source_dir: str, output_file: str = "extended_dataset.json"):
        """
        初始化转换器
        
        Args:
            source_dir: CodeUpdateArena数据集目录
            output_file: 输出文件名
        """
        self.source_dir = Path(source_dir)
        self.output_file = output_file
        
        console.print(f"[cyan]初始化数据转换器[/cyan]")
        console.print(f"  源目录: {source_dir}")
        console.print(f"  输出文件: {output_file}")
    
    def load_codeupdatearena_data(self) -> List[Dict]:
        """加载CodeUpdateArena原始数据"""
        console.print("[yellow]加载CodeUpdateArena数据...[/yellow]")
        
        # 这里需要根据实际的CodeUpdateArena数据格式调整
        # 假设数据在 data/python/ 目录下
        data_files = list(self.source_dir.glob("data/python/*.json"))
        
        all_data = []
        for file in track(data_files, description="读取文件"):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                else:
                    all_data.append(data)
        
        console.print(f"[green]✓ 加载了 {len(all_data)} 条原始数据[/green]")
        return all_data
    
    def convert_to_our_format(self, raw_data: List[Dict]) -> Dict:
        """
        转换为我们的数据格式
        
        Args:
            raw_data: 原始数据
        
        Returns:
            转换后的数据
        """
        console.print("[yellow]转换数据格式...[/yellow]")
        
        converted_data = {
            "train": [],
            "test": []
        }
        
        for idx, item in enumerate(track(raw_data, description="转换中")):
            # 根据CodeUpdateArena的实际格式调整
            # 这里是示例格式
            converted_item = {
                "dependency": item.get("library", "unknown"),
                "old_code": item.get("old_code", "").strip(),
                "new_code": item.get("new_code", "").strip(),
                "description": item.get("description", f"API update for {item.get('library', 'unknown')}"),
                "old_version": item.get("old_version", ""),
                "new_version": item.get("new_version", ""),
                "change_type": item.get("change_type", "api_change")
            }
            
            # 过滤无效数据
            if not converted_item["old_code"] or not converted_item["new_code"]:
                continue
            
            # 8:2 划分训练集和测试集
            if idx % 5 == 0:  # 20% 测试集
                converted_data["test"].append(converted_item)
            else:  # 80% 训练集
                converted_data["train"].append(converted_item)
        
        console.print(f"[green]✓ 转换完成[/green]")
        console.print(f"  训练集: {len(converted_data['train'])} 条")
        console.print(f"  测试集: {len(converted_data['test'])} 条")
        
        return converted_data
    
    def filter_by_library(self, data: Dict, libraries: List[str]) -> Dict:
        """
        按库过滤数据
        
        Args:
            data: 数据
            libraries: 要保留的库列表
        
        Returns:
            过滤后的数据
        """
        console.print(f"[yellow]按库过滤数据: {', '.join(libraries)}[/yellow]")
        
        filtered_data = {
            "train": [item for item in data["train"] if item["dependency"] in libraries],
            "test": [item for item in data["test"] if item["dependency"] in libraries]
        }
        
        console.print(f"[green]✓ 过滤后[/green]")
        console.print(f"  训练集: {len(filtered_data['train'])} 条")
        console.print(f"  测试集: {len(filtered_data['test'])} 条")
        
        return filtered_data
    
    def sample_data(self, data: Dict, train_size: int = 50, test_size: int = 50) -> Dict:
        """
        采样数据
        
        Args:
            data: 数据
            train_size: 训练集大小
            test_size: 测试集大小
        
        Returns:
            采样后的数据
        """
        import random
        
        console.print(f"[yellow]采样数据: 训练{train_size}, 测试{test_size}[/yellow]")
        
        sampled_data = {
            "train": random.sample(data["train"], min(train_size, len(data["train"]))),
            "test": random.sample(data["test"], min(test_size, len(data["test"])))
        }
        
        console.print(f"[green]✓ 采样完成[/green]")
        console.print(f"  训练集: {len(sampled_data['train'])} 条")
        console.print(f"  测试集: {len(sampled_data['test'])} 条")
        
        return sampled_data
    
    def save_data(self, data: Dict):
        """保存数据"""
        console.print(f"[yellow]保存数据到 {self.output_file}...[/yellow]")
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        console.print(f"[green]✓ 数据已保存[/green]")
        
        # 打印统计信息
        self.print_statistics(data)
    
    def print_statistics(self, data: Dict):
        """打印数据统计"""
        console.print("\n" + "="*70)
        console.print("[bold cyan]数据集统计[/bold cyan]")
        console.print("="*70)
        
        # 训练集统计
        train_libs = {}
        for item in data["train"]:
            lib = item["dependency"]
            train_libs[lib] = train_libs.get(lib, 0) + 1
        
        console.print("\n[bold]训练集分布:[/bold]")
        for lib, count in sorted(train_libs.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  {lib}: {count}")
        
        # 测试集统计
        test_libs = {}
        for item in data["test"]:
            lib = item["dependency"]
            test_libs[lib] = test_libs.get(lib, 0) + 1
        
        console.print("\n[bold]测试集分布:[/bold]")
        for lib, count in sorted(test_libs.items(), key=lambda x: x[1], reverse=True):
            console.print(f"  {lib}: {count}")
        
        console.print(f"\n[bold]总计:[/bold]")
        console.print(f"  训练集: {len(data['train'])} 条")
        console.print(f"  测试集: {len(data['test'])} 条")
        console.print(f"  总计: {len(data['train']) + len(data['test'])} 条")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CodeUpdateArena数据转换脚本")
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="CodeUpdateArena数据集目录"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="extended_dataset.json",
        help="输出文件名"
    )
    parser.add_argument(
        "--libraries",
        type=str,
        nargs="+",
        default=["pandas", "numpy", "tensorflow", "sklearn", "torch", "requests"],
        help="要保留的库"
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=50,
        help="训练集大小"
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=50,
        help="测试集大小"
    )
    
    args = parser.parse_args()
    
    # 检查源目录
    if not Path(args.source_dir).exists():
        console.print(f"[red]错误: 源目录不存在: {args.source_dir}[/red]")
        console.print("\n[yellow]请先下载CodeUpdateArena数据集:[/yellow]")
        console.print("  git clone https://github.com/amazon-science/CodeUpdateArena.git")
        return
    
    # 创建转换器
    converter = CodeUpdateArenaConverter(args.source_dir, args.output)
    
    # 加载数据
    raw_data = converter.load_codeupdatearena_data()
    
    # 转换格式
    converted_data = converter.convert_to_our_format(raw_data)
    
    # 按库过滤
    if args.libraries:
        converted_data = converter.filter_by_library(converted_data, args.libraries)
    
    # 采样
    if args.train_size or args.test_size:
        converted_data = converter.sample_data(
            converted_data,
            args.train_size,
            args.test_size
        )
    
    # 保存
    converter.save_data(converted_data)
    
    console.print("\n[bold green]✅ 数据准备完成！[/bold green]")
    console.print(f"\n[cyan]下一步:[/cyan]")
    console.print(f"  1. 使用新数据集训练: python3 run_lora.py --data_file {args.output}")
    console.print(f"  2. 评估模型: python3 evaluate_lora.py")


if __name__ == "__main__":
    main()
