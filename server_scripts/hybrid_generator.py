#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合生成器 - 规则优先 + LLM兜底的API迁移系统
"""

import torch
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from rich.console import Console
from rich.progress import track

from rule_matcher import RuleMatcher
from prompt_engineering import PromptTemplate

console = Console()


class HybridGenerator:
    """混合API迁移生成器：规则 + LLM"""
    
    def __init__(
        self,
        rule_matcher: RuleMatcher,
        model_name: str = "Qwen/Qwen2.5-Coder-1.5B",
        device: str = "auto"
    ):
        """
        Args:
            rule_matcher: 规则匹配器
            model_name: LLM模型名称
            device: 设备（auto/cpu/cuda）
        """
        self.matcher = rule_matcher
        self.model_name = model_name
        
        # 统计信息
        self.stats = {
            'rule_direct': 0,      # 规则直接应用
            'rule_guided': 0,      # 规则引导Prompt
            'llm_fallback': 0,     # 纯LLM兜底
            'total': 0
        }
        
        # 加载LLM
        console.print(f"\n[cyan]初始化LLM: {model_name}[/cyan]")
        self._load_model(device)
    
    def _load_model(self, device: str):
        """加载语言模型"""
        # 自动检测设备
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                console.print(f"[green]✓ 使用GPU: {torch.cuda.get_device_name(0)}[/green]")
            else:
                self.device = "cpu"
                console.print(f"[yellow]⚠ 使用CPU（GPU不可用）[/yellow]")
        else:
            self.device = device
        
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # 加载模型
        try:
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                ).cuda()
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
            console.print(f"[green]✓ 模型加载成功[/green]")
        except Exception as e:
            console.print(f"[red]模型加载失败: {e}[/red]")
            raise
    
    def generate(
        self, 
        old_code: str, 
        description: str = "",
        dependency: str = ""
    ) -> Tuple[str, str, float]:
        """
        生成更新后的代码
        
        Args:
            old_code: 旧代码
            description: 更新描述
            dependency: 依赖库名称
            
        Returns:
            (new_code, strategy, confidence)
            - new_code: 生成的新代码
            - strategy: 使用的策略（rule_direct/rule_guided/llm_fallback）
            - confidence: 置信度
        """
        self.stats['total'] += 1
        
        # 1. 尝试规则匹配
        matched_rules = self.matcher.match_rules(old_code)
        
        if matched_rules:
            best_rule, best_score = matched_rules[0]
            
            # 策略A: 高置信度 → 规则直接应用
            if best_score >= 0.85:
                new_code = self.matcher.apply_rules(old_code, [(best_rule, best_score)])
                if new_code:
                    self.stats['rule_direct'] += 1
                    return new_code, 'rule_direct', best_score
            
            # 策略B: 中置信度 → 规则引导Prompt
            if best_score >= 0.6:
                new_code = self._generate_with_rule_guidance(
                    old_code, best_rule, description
                )
                self.stats['rule_guided'] += 1
                return new_code, 'rule_guided', best_score * 0.9
        
        # 策略C: 无规则或低置信度 → LLM兜底
        new_code = self._generate_with_llm(old_code, description, dependency)
        self.stats['llm_fallback'] += 1
        return new_code, 'llm_fallback', 0.7
    
    def _generate_with_rule_guidance(
        self, 
        old_code: str, 
        rule: Dict,
        description: str
    ) -> str:
        """使用规则引导的LLM生成（先尝试直接应用）"""
        # 先尝试用当前matcher直接应用规则
        try:
            direct_result = self.matcher.apply_rules(old_code, [(rule, 1.0)])
            if direct_result and direct_result != old_code:
                # 规则能直接应用，直接返回
                return direct_result
        except Exception as e:
            console.print(f"[dim]规则应用失败: {e}[/dim]")
        
        # 规则无法直接应用，使用简化的LLM生成
        if rule['type'] == 'api_replacement':
            old_api = rule['old_api']
            new_api = rule['new_api']
            
            # 简化prompt，直接替换
            prompt = f"""{old_code}
Replace {old_api} with {new_api}:"""
        
        elif rule['type'] == 'parameter_migration':
            # 参数迁移直接返回原代码（太复杂）
            return old_code
        
        else:
            # 其他类型直接返回原代码
            return old_code
        
        # 生成新代码
        generated = self._llm_generate(prompt, max_new_tokens=50)
        
        # 如果生成失败，返回原代码
        if not generated or generated.strip() == "":
            return old_code
        
        return generated
    
    def _generate_with_llm(
        self, 
        old_code: str, 
        description: str,
        dependency: str
    ) -> str:
        """纯LLM生成（兜底策略）"""
        # 使用基础Prompt模板
        if dependency:
            prompt = PromptTemplate.with_context_prompt(
                old_code, dependency, "", "", description
            )
        else:
            prompt = PromptTemplate.basic_update_prompt(old_code, description)
        
        return self._llm_generate(prompt)
    
    def _llm_generate(self, prompt: str, max_new_tokens: int = 80) -> str:
        """LLM生成（底层调用）"""
        # 不截断prompt（规则引导的prompt已经很短）
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,  # 使用采样
                temperature=0.3,  # 低温度，更确定性
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,  # 适度防重复
                no_repeat_ngram_size=3
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的代码部分
        if prompt in generated_text:
            generated_code = generated_text[len(prompt):].strip()
        else:
            generated_code = generated_text.strip()
        
        # 清理输出
        lines = generated_code.split('\n')
        code_lines = []
        
        for line in lines:
            line = line.strip()
            
            # 跳过空行
            if not line:
                continue
            
            # 跳过注释
            if line.startswith('#'):
                continue
            
            # 跳过明显的解释文本
            if line.startswith('Updated') or line.startswith('Task') or line.startswith('Replace'):
                continue
            
            # 检测极端重复（如!!!!!!）
            if len(line) > 15:
                unique_chars = len(set(line))
                total_chars = len(line)
                diversity = unique_chars / total_chars
                
                # 只过滤极低多样性（<20%），放宽到允许正常代码通过
                if diversity < 0.2:
                    console.print(f"[dim]跳过极低多样性: {line[:40]}...[/dim]")
                    continue
                
                # 检测过多特殊字符
                if line.count('!') > 5 or line.count('?') > 5:
                    continue
            
            # 接受第一个合理的代码行
            code_lines.append(line)
            break
        
        # 返回生成的代码，如果没有则返回空字符串
        return code_lines[0] if code_lines else ""
    
    def batch_generate(
        self, 
        examples: List[Dict]
    ) -> List[Dict]:
        """批量生成"""
        results = []
        
        for example in track(examples, description="混合生成"):
            old_code = example.get('old_code', '')
            description = example.get('description', '')
            dependency = example.get('dependency', '')
            
            new_code, strategy, confidence = self.generate(
                old_code, description, dependency
            )
            
            result = {
                'old_code': old_code,
                'generated_code': new_code,
                'expected_code': example.get('new_code', ''),
                'strategy': strategy,
                'confidence': confidence,
                'dependency': dependency
            }
            results.append(result)
        
        return results
    
    def print_stats(self):
        """打印统计信息"""
        from rich.table import Table
        
        total = self.stats['total']
        if total == 0:
            console.print("[yellow]还没有生成任何代码[/yellow]")
            return
        
        table = Table(title="混合生成统计")
        table.add_column("策略", style="cyan")
        table.add_column("数量", style="green")
        table.add_column("比例", style="yellow")
        
        for strategy, count in self.stats.items():
            if strategy == 'total':
                continue
            percentage = f"{count/total*100:.1f}%"
            
            strategy_name = {
                'rule_direct': '规则直接应用',
                'rule_guided': '规则引导Prompt',
                'llm_fallback': 'LLM兜底'
            }.get(strategy, strategy)
            
            table.add_row(strategy_name, str(count), percentage)
        
        table.add_row("总计", str(total), "100%", style="bold")
        
        console.print(table)


def main():
    """测试混合生成器"""
    from rule_learner import RuleLearner
    from data_utils import DataLoader
    
    console.print("[bold cyan]混合生成器测试[/bold cyan]\n")
    
    # 1. 加载规则
    try:
        rules = RuleLearner.load_rules("../configs/learned_rules.json")
    except FileNotFoundError:
        console.print("[red]错误: 请先运行 rule_learner.py 生成规则库[/red]")
        return
    
    # 2. 创建匹配器
    matcher = RuleMatcher(rules, confidence_threshold=0.7)
    
    # 3. 创建混合生成器
    generator = HybridGenerator(matcher, device="auto")
    
    # 4. 加载测试数据
    data_loader = DataLoader("mini_dataset.json")
    test_data = data_loader.get_test_data()
    console.print(f"\n✓ 加载了 {len(test_data)} 个测试样本\n")
    
    # 5. 批量生成
    results = generator.batch_generate(test_data[:5])  # 测试前5个
    
    # 6. 显示结果
    console.print("\n[bold green]生成结果：[/bold green]\n")
    for i, result in enumerate(results, 1):
        console.print(f"[yellow]样本 {i}:[/yellow]")
        console.print(f"  旧代码: {result['old_code']}")
        console.print(f"  生成: {result['generated_code']}")
        console.print(f"  期望: {result['expected_code']}")
        console.print(f"  策略: {result['strategy']} (置信度: {result['confidence']:.2f})")
        console.print()
    
    # 7. 打印统计
    generator.print_stats()


if __name__ == "__main__":
    main()
