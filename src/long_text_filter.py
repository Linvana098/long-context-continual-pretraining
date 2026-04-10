"""
长文本筛选模块
==============
从项目1的通用语料中筛选 >= min_tokens 的长文档，
用于长上下文扩展训练和混合数据构建。
"""

import os
import json
import glob
import numpy as np
import sentencepiece as spm
from typing import Dict, List
from tqdm import tqdm

from src.utils import setup_logging
from src.domain_processor import tokenize_and_pack


def filter_long_texts(config: Dict) -> Dict:
    """
    从项目1的 decontaminated 语料中筛选长文本。

    长文本对 context extension 训练至关重要：
    - 模型需要在长序列上学习长距离依赖
    - 短文本 pack 到 4K/8K 只是拼接，不涉及跨文档的长距离建模
    - 真正的长文档（一篇文章 > 2K tokens）才能训练 RoPE 对远距离位置的泛化
    """
    logger = setup_logging()
    logger.info("筛选长文本")

    tokenizer_path = config["base_model"]["tokenizer_path"]
    min_tokens = config["long_text"]["min_tokens"]
    output_dir = config["long_text"]["output_dir"]

    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)

    # 从项目1的去污染后语料读取
    long_texts = []
    total_checked = 0

    for lang in ["zh", "en"]:
        data_dir = config["general_data"].get(f"decontaminated_{lang}", "")
        if not os.path.exists(data_dir):
            logger.warning(f"  {lang} 语料路径不存在: {data_dir}")
            continue

        files = glob.glob(os.path.join(data_dir, "*.jsonl"))
        for fp in tqdm(files, desc=f"扫描 {lang} 长文本"):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = obj.get("text", "")
                    except json.JSONDecodeError:
                        continue

                    total_checked += 1
                    # 快速估算 token 数（避免对每条都做完整 tokenize）
                    # 中文: ~1.5 tokens/char, 英文: ~1.3 tokens/word
                    est_tokens = len(text) * 0.7 if any('\u4e00' <= c <= '\u9fff' for c in text[:100]) else len(text.split()) * 1.3

                    if est_tokens >= min_tokens * 0.8:  # 留余量
                        # 精确 tokenize 确认
                        ids = sp.Encode(text, out_type=int)
                        if len(ids) >= min_tokens:
                            long_texts.append(text)

    logger.info(f"  扫描 {total_checked} 条，筛选出 {len(long_texts)} 条长文本 (>= {min_tokens} tokens)")

    # Tokenize + Pack
    stats = {"total_checked": total_checked, "long_texts": len(long_texts)}

    for seq_len in [4096, 8192]:
        logger.info(f"\n  Pack 长文本 (seq_len={seq_len})")
        out_bin = os.path.join(output_dir, f"general_long_{seq_len}.bin")
        out_idx = os.path.join(output_dir, f"general_long_{seq_len}.idx")
        n = tokenize_and_pack(long_texts, tokenizer_path, seq_len, out_bin, out_idx)
        stats[f"seqs_{seq_len}"] = n

    return stats