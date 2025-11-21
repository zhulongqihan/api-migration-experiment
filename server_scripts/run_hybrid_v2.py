#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
混合系统V2 - 使用手工规则 + LLM
"""

import json
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from rich.console import Console
from rich.progress import track
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_utils import DataLoader
from regex_rule_matcher import RegexRuleMatcher
from evaluate_hybrid import HybridEvaluator
from prompt_engineering import PromptTemplate

console = Console()


class HybridGeneratorV2:
    """混合生成器V2：手工规则优先 + LLM兜底"""
    
    def __init__(
        self,
        rule_matcher: RegexRuleMatcher,
        model_name: str = "Qwen/Qwen2.5-Coder-1.5B",
        device: str = "auto"
    ):
        self.matcher = rule_matcher
        self.model_name = model_name
        
        self.stats = {
            'rule_applied': 0,
            'llm_fallback': 0,
            'total': 0
        }
        
        console.print(f"\n[cyan]初始化LLM: {model_name}[/cyan]")
        self._load_model(device)
    
    def _load_model(self, device: str):
        """加载LLM"""
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        console.print(f"[green]✓ 使用设备: {self.device}[/green]")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        
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
    
    def generate(
        self,
        old_code: str,
        description: str = "",
        dependency: str = ""
    ) -> Tuple[str, str, float]:
        """
        生成新代码
        
        Returns:
            (new_code, strategy, confidence)
        """
        self.stats['total'] += 1
        
        # 策略1：尝试规则匹配
        new_code, confidence = self.matcher.match_and_apply(old_code, dependency)
        
        if new_code and confidence >= 0.85:
            self.stats['rule_applied'] += 1
            return new_code, 'rule_applied', confidence
        
        # 策略2：LLM生成
        llm_code = self._generate_with_llm(old_code, description, dependency)
        self.stats['llm_fallback'] += 1
        return llm_code, 'llm_fallback', 0.7
    
    def _generate_with_llm(
        self,
        old_code: str,
        description: str,
        dependency: str
    ) -> str:
        """LLM生成"""
        # 使用简洁的prompt
        prompt = f"""Update this API code:
Old: {old_code}
Task: {description}
Output only the updated code (one line):
"""
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.3,
                no_repeat_ngram_size=4
            )
        
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取代码
        if prompt in generated:
            code = generated[len(prompt):].strip()
        else:
            code = generated.strip()
        
        # 清理输出
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # 过滤重复字符
            if len(set(line)) >= 5:
                return line
        
        return code.split('\n')[0].strip() if code else old_code
    
    def batch_generate(self, examples: List[Dict]) -> List[Dict]:
        """批量生成"""
        results = []
        
        for ex in track(examples, description="混合生成V2"):
            old_code = ex.get('old_code', '')
            description = ex.get('description', '')
            dependency = ex.get('dependency', '')
            
            new_code, strategy, conf = self.generate(old_code, description, dependency)
            
            results.append({
                'old_code': old_code,
                'generated_code': new_code,
                'expected_code': ex.get('new_code', ''),
                'strategy': strategy,
                'confidence': conf,
                'dependency': dependency
            })
        
        return results
    
    def print_stats(self):
        """打印统计"""
        from rich.table import Table
        
        total = self.stats['total']
        if total == 0:
            return
        
        table = Table(title="混合生成V2统计")
        table.add_column("策略", style="cyan")
        table.add_column("数量", style="green")
        table.add_column("比例", style="yellow")
        
        table.add_row(
            "规则应用",
            str(self.stats['rule_applied']),
            f"{self.stats['rule_applied']/total*100:.1f}%"
        )
        table.add_row(
            "LLM兜底",
            str(self.stats['llm_fallback']),
            f"{self.stats['llm_fallback']/total*100:.1f}%"
        )
        table.add_row("总计", str(total), "100%", style="bold")
        
        console.print(table)


def main():
    from rich.panel import Panel
    
    console.print(Panel.fit(
        "[bold cyan]混合系统V2[/bold cyan]\n"
        "手工规则 + LLM兜底",
        border_style="cyan"
    ))
    
    # 1. 加载数据
    console.print("\n[cyan]加载数据...[/cyan]")
    data_loader = DataLoader("mini_dataset.json")
    test_data = data_loader.get_test_data()
    console.print(f"[green]✓ 测试集: {len(test_data)} 个样本[/green]")
    
    # 2. 创建规则匹配器
    console.print("\n[cyan]加载手工规则...[/cyan]")
    matcher = RegexRuleMatcher()
    
    # 3. 创建生成器
    console.print("\n[cyan]初始化生成器...[/cyan]")
    generator = HybridGeneratorV2(matcher, device="auto")
    
    # 4. 批量生成
    console.print("\n[cyan]批量生成...[/cyan]")
    results = generator.batch_generate(test_data)
    console.print(f"[green]✓ 完成 {len(results)} 个样本[/green]\n")
    
    # 5. 统计
    generator.print_stats()
    
    # 6. 评估
    console.print("\n[cyan]评估效果...[/cyan]")
    evaluator = HybridEvaluator()
    metrics = evaluator.evaluate(results)
    evaluator.print_report(metrics)
    
    # 7. 保存结果
    output_dir = Path("../results/hybrid_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"results_{timestamp}.json"
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'metrics': metrics,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[green]✓ 结果已保存: {result_file}[/green]")
    
    # 8. 总结
    console.print(Panel.fit(
        f"[bold]混合系统V2运行完成[/bold]\n\n"
        f"精确匹配率: {metrics['exact_match_rate']*100:.1f}%\n"
        f"平均相似度: {metrics['avg_similarity']:.3f}\n"
        f"规则覆盖率: {generator.stats['rule_applied']/generator.stats['total']*100:.1f}%",
        border_style="green" if metrics['exact_match_rate'] >= 0.7 else "yellow"
    ))


if __name__ == "__main__":
    main()
