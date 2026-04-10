"""
金融研报预处理模块
==================
将 600MB 金融研报清洗、tokenize、pack 成可训练的 bin/idx 格式。

金融研报的特殊噪音：
- 页眉页脚（"第 X 页 / 共 Y 页"、公司名称反复出现）
- 免责声明（每篇末尾的法律声明，大量重复）
- 数字密集表格（财务数据表，对语言建模无意义）
- PDF 转文本的乱码（特殊符号、断行错误）

清洗策略：
1. 编码统一（UTF-8）
2. 去页眉页脚模板
3. 去免责声明段落
4. 去纯数字/表格段落
5. 合并断行
6. 长度过滤
7. Tokenize + Pack（多种 seq_len）
"""

import os
import re
import json
import glob
import numpy as np
import sentencepiece as spm
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from src.utils import setup_logging


class FinancialReportCleaner:
    """金融研报清洗器。"""

    def __init__(self, config: Dict):
        self.min_length = config.get("min_length", 200)
        self.max_length = config.get("max_length", 500000)
        self.remove_headers = config.get("remove_headers", True)
        self.remove_tables = config.get("remove_tables", True)
        self.remove_disclaimers = config.get("remove_disclaimers", True)

        # 页眉页脚模式
        self.header_patterns = [
            re.compile(r"第\s*\d+\s*页\s*/?\s*共\s*\d+\s*页"),
            re.compile(r"^\s*\d+\s*$", re.MULTILINE),  # 单独的页码行
            re.compile(r"请务必阅读正文之后的.*?声明"),
            re.compile(r"证券研究报告"),
        ]

        # 免责声明关键句
        self.disclaimer_keywords = [
            "免责声明", "风险提示", "重要声明", "法律声明",
            "本报告仅供", "不构成投资建议", "据此操作",
            "分析师声明", "评级说明", "投资评级",
        ]

        # 数字密集度检测（表格特征）
        self.digit_ratio_threshold = 0.4  # 数字+标点占比超过40%视为表格

    def clean_single(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        清洗单篇研报。

        Returns:
            (清洗后文本, 丢弃原因)
        """
        if not text or len(text.strip()) < self.min_length:
            return None, f"too_short:{len(text)}"

        # 1. 合并断行（PDF转文本常见问题：每行都有换行符）
        #    保留段落间的空行，合并段落内的断行
        text = re.sub(r"(?<!\n)\n(?!\n)", "", text)  # 单个换行→合并
        text = re.sub(r"\n{3,}", "\n\n", text)        # 多个空行→两个

        # 2. 去页眉页脚
        if self.remove_headers:
            for pattern in self.header_patterns:
                text = pattern.sub("", text)

        # 3. 去免责声明（通常在文末，检测到关键句后截断）
        if self.remove_disclaimers:
            lines = text.split("\n")
            cut_idx = len(lines)
            for i, line in enumerate(lines):
                # 如果在文档后半部分出现免责关键句，截断
                if i > len(lines) * 0.7:
                    if any(kw in line for kw in self.disclaimer_keywords):
                        cut_idx = i
                        break
            text = "\n".join(lines[:cut_idx])

        # 4. 去数字密集段落（表格）
        if self.remove_tables:
            paragraphs = text.split("\n\n")
            filtered = []
            for para in paragraphs:
                if not para.strip():
                    continue
                # 统计数字+标点占比
                digits_and_punct = sum(1 for c in para if c.isdigit() or c in ".,;:|-+%()（）")
                ratio = digits_and_punct / max(len(para), 1)
                if ratio < self.digit_ratio_threshold:
                    filtered.append(para)
            text = "\n\n".join(filtered)

        # 5. 清理多余空白
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        # 6. 长度过滤
        if len(text) < self.min_length:
            return None, f"too_short_after_clean:{len(text)}"
        if len(text) > self.max_length:
            text = text[:self.max_length]

        return text, None


def tokenize_and_pack(
    texts: List[str],
    tokenizer_path: str,
    seq_len: int,
    output_bin: str,
    output_idx: str,
):
    """
    将文本列表 tokenize + pack 成 memmap 格式。

    与项目1的 packing 逻辑一致：
    - 多个文档拼接到 seq_len 长度
    - 文档间用 <eos> 分隔
    """
    logger = setup_logging()
    sp = spm.SentencePieceProcessor()
    sp.Load(tokenizer_path)
    eos_id = sp.eos_id()

    # tokenize 所有文本
    all_ids = []
    for text in tqdm(texts, desc="Tokenize"):
        ids = sp.Encode(text, out_type=int)
        ids.append(eos_id)
        all_ids.extend(ids)

    total_tokens = len(all_ids)
    logger.info(f"  总 tokens: {total_tokens:,}")

    # pack 到固定 seq_len
    n_seqs = total_tokens // seq_len
    if n_seqs == 0:
        logger.warning(f"  数据不足一个 seq_len={seq_len} 的序列")
        return 0

    # 截断到整数倍
    all_ids = all_ids[:n_seqs * seq_len]
    data = np.array(all_ids, dtype=np.uint16).reshape(n_seqs, seq_len)

    # 保存
    os.makedirs(os.path.dirname(output_bin), exist_ok=True)
    fp = np.memmap(output_bin, dtype=np.uint16, mode="w+", shape=data.shape)
    fp[:] = data[:]
    fp.flush()
    del fp

    idx = {"num_sequences": n_seqs, "seq_len": seq_len,
           "dtype": "uint16", "shape": list(data.shape),
           "total_tokens": int(data.size)}
    with open(output_idx, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2)

    logger.info(f"  保存: {n_seqs} 序列 × {seq_len} = {data.size:,} tokens")
    return n_seqs


def run_domain_processing(config: Dict) -> Dict:
    """
    金融研报预处理主流程。

    1. 读取 raw/ 下的所有文本文件
    2. 清洗
    3. 保存清洗后的 jsonl
    4. Tokenize + Pack（生成多种 seq_len 版本）
    """
    logger = setup_logging()
    logger.info("金融研报预处理")

    cleaning_config = config.get("domain_cleaning", {})
    cleaner = FinancialReportCleaner(cleaning_config)

    raw_dir = config["domain_data"]["raw_dir"]
    cleaned_dir = config["domain_data"]["cleaned_dir"]
    tokenized_dir = config["domain_data"]["tokenized_dir"]
    tokenizer_path = config["base_model"]["tokenizer_path"]

    # 读取所有文件（支持 txt, jsonl, md）
    raw_files = []
    for ext in ["*.txt", "*.jsonl", "*.json", "*.md", "*.csv"]:
        raw_files.extend(glob.glob(os.path.join(raw_dir, "**", ext), recursive=True))

    if not raw_files:
        logger.warning(f"未找到原始文件: {raw_dir}")
        logger.info(f"请将金融研报文本文件放入 {raw_dir}/")
        return {"status": "no_data"}

    logger.info(f"  找到 {len(raw_files)} 个文件")

    # 清洗
    all_texts = []
    stats = {"total_files": len(raw_files), "total_docs": 0, "kept": 0, "discarded": 0}

    for fp in tqdm(raw_files, desc="读取+清洗"):
        try:
            # 尝试按 json 读取
            if fp.endswith(".json"):
                with open(fp, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                obj = json.loads(line)
                                text = obj.get("text", obj.get("content", ""))
                            except json.JSONDecodeError:
                                text = line
                            stats["total_docs"] += 1
                            cleaned, reason = cleaner.clean_single(text)
                            if cleaned:
                                all_texts.append(cleaned)
                                stats["kept"] += 1
                            else:
                                stats["discarded"] += 1
            else:
                # 纯文本文件：整个文件作为一篇文档
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                stats["total_docs"] += 1
                cleaned, reason = cleaner.clean_single(text)
                if cleaned:
                    all_texts.append(cleaned)
                    stats["kept"] += 1
                else:
                    stats["discarded"] += 1
        except Exception as e:
            logger.warning(f"  读取失败: {fp} ({e})")

    logger.info(f"  清洗结果: {stats['kept']}/{stats['total_docs']} 保留")

    # 保存清洗后的 jsonl
    cleaned_path = os.path.join(cleaned_dir, "domain_cleaned.jsonl")
    os.makedirs(cleaned_dir, exist_ok=True)
    with open(cleaned_path, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write(json.dumps({"text": text, "source": "financial_report"}, ensure_ascii=False) + "\n")

    # Tokenize + Pack（生成多种 seq_len）
    for seq_len in [2048, 4096, 8192]:
        logger.info(f"\n  Tokenize + Pack (seq_len={seq_len})")
        out_bin = os.path.join(tokenized_dir, f"domain_{seq_len}.bin")
        out_idx = os.path.join(tokenized_dir, f"domain_{seq_len}.idx")
        n = tokenize_and_pack(all_texts, tokenizer_path, seq_len, out_bin, out_idx)
        stats[f"seqs_{seq_len}"] = n

    return stats