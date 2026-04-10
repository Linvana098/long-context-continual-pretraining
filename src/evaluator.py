"""
多维度评估模块
==============
1. 长上下文 PPL 评估（不同 seq_len 的 PPL 对比）
2. 遗忘分析（领域 PPL 下降 vs 通用 PPL 上升）
3. 混合 vs 纯领域 对比
"""

import os
import json
import math
import numpy as np
import torch
from typing import Dict, List
from torch.amp import autocast

from src.utils import setup_logging


class ContextLengthEvaluator:
    """不同上下文长度的 PPL 评估。"""

    def __init__(self, device=None):
        self.logger = setup_logging()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def evaluate_at_length(self, model, dataloader, use_bf16=True, max_batches=50):
        model.eval()
        total_loss, total_tokens, n = 0.0, 0, 0
        for batch in dataloader:
            if n >= max_batches:
                break
            batch = batch.to(self.device)
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                _, loss = model(input_ids=batch, targets=batch)
            if loss is not None:
                tk = batch.shape[0] * (batch.shape[1] - 1)
                total_loss += loss.item() * tk
                total_tokens += tk
                n += 1
        avg = total_loss / max(total_tokens, 1)
        return {"loss": avg, "ppl": math.exp(min(avg, 20)), "tokens": total_tokens}


def generate_forgetting_report(
    results: Dict,
    output_dir: str = "reports"
) -> str:
    """
    生成遗忘分析报告。

    对比：
    1. 基座模型 vs 继续预训练后模型 在通用验证集上的 PPL
    2. 纯领域训练 vs 混合训练 的遗忘程度
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "continual_pretrain_report.md")

    lines = [
        "# 长上下文扩展 + 金融领域继续预训练报告",
        "",
        "## 一、长上下文扩展",
        "",
    ]

    # 长上下文结果
    if "context_extension" in results:
        lines += ["| 阶段 | seq_len | 验证 PPL | 训练步数 | 显存(MB) |",
                  "|------|---------|---------|---------|---------|"]
        for stage in results["context_extension"]:
            lines.append(
                f"| {stage.get('name','?')} | {stage.get('seq_len','?')} | "
                f"{stage.get('val_ppl','N/A')} | {stage.get('steps','?')} | "
                f"{stage.get('peak_mem','?')} |")

    # 继续预训练结果
    lines += ["", "## 二、金融领域继续预训练", ""]

    for exp_name in ["domain_only", "domain_mixed"]:
        if exp_name in results:
            r = results[exp_name]
            desc = "纯领域训练" if "only" in exp_name else "混合训练（70%领域+30%通用）"
            lines += [f"### {desc}", ""]

            baseline = r.get("baseline_eval", {})
            final = r.get("final_eval", {})

            lines += ["| 验证集 | 训练前 PPL | 训练后 PPL | 变化 |",
                      "|--------|-----------|-----------|------|"]

            for vset in ["domain", "general"]:
                b = baseline.get(vset, {}).get("ppl", "N/A")
                f = final.get(vset, {}).get("ppl", "N/A")
                if isinstance(b, float) and isinstance(f, float):
                    delta = f - b
                    sign = "↑" if delta > 0 else "↓"
                    lines.append(f"| {vset} | {b:.1f} | {f:.1f} | {sign}{abs(delta):.1f} |")
                else:
                    lines.append(f"| {vset} | {b} | {f} | - |")

            lines.append("")

    # 遗忘分析
    lines += [
        "## 三、遗忘分析",
        "",
        "### 判断标准",
        "- 通用 PPL 上升 < 5%: 无明显遗忘",
        "- 通用 PPL 上升 5%~15%: 轻微遗忘（可接受）",
        "- 通用 PPL 上升 > 15%: 严重遗忘（需要增加通用语料比例）",
        "",
        "### 混合训练的价值",
        "- 对比纯领域训练和混合训练的通用 PPL 变化",
        "- 如果混合训练的通用 PPL 上升显著小于纯领域训练，",
        "  说明混入 30% 通用语料有效缓解了遗忘",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path