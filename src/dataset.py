"""
预训练数据集加载模块
====================
从项目1输出的 memmap (.bin + .idx) 文件中加载已 tokenized 和 packed 的数据。

数据格式：
- .bin 文件: numpy memmap，shape = (num_sequences, seq_len)，dtype = uint16
- .idx 文件: JSON 元数据（shape、dtype、mixture 名称等）

关键设计：
- 使用 memmap 懒加载，不需要一次性读入内存
- DataLoader 使用 pin_memory + prefetch 加速 GPU 传输
- 每个 epoch 随机打乱序列顺序
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional


class PretrainDataset(Dataset):
    """
    预训练数据集（读取 memmap 格式）。

    每个样本是一个长度为 seq_len 的 token id 序列（已 packed）。
    训练时：input_ids = tokens[:-1], targets = tokens[1:]
    但为了效率，我们返回完整序列，在模型内部做 shift。
    """

    def __init__(self, bin_path: str, idx_path: str):
        """
        Args:
            bin_path: .bin 文件路径（memmap 数据）
            idx_path: .idx 文件路径（元数据 JSON）
        """
        # 读取元数据
        with open(idx_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        shape = tuple(self.meta["shape"])
        dtype_str = self.meta.get("dtype", "uint16")
        dtype = np.uint16 if "16" in dtype_str else np.uint32

        # memmap 懒加载：不会一次性读入内存
        # data shape: (num_sequences, seq_len)
        self.data = np.memmap(bin_path, dtype=dtype, mode="r", shape=shape)
        self.num_sequences = shape[0]
        self.seq_len = shape[1] if len(shape) > 1 else self.meta.get("seq_len", 2048)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        返回一个序列的 token ids。

        Args:
            idx: 序列索引

        Returns:
            tokens: (seq_len,) int64 tensor
        """
        # 从 memmap 读取一行（零拷贝，按需加载）
        tokens = self.data[idx].astype(np.int64)
        return torch.from_numpy(tokens)


def build_dataloader(
    bin_path: str,
    idx_path: str,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,   # Windows 下建议设为 0
    pin_memory: bool = True,
    drop_last: bool = True
) -> DataLoader:
    """
    构建 DataLoader。

    Args:
        bin_path: .bin 文件路径
        idx_path: .idx 文件路径
        batch_size: 每批样本数
        shuffle: 是否打乱（训练时 True，验证时 False）
        num_workers: 数据加载进程数（Windows 设 0）
        pin_memory: 是否 pin 内存（加速 CPU→GPU 传输）
        drop_last: 丢弃最后不完整的 batch

    Returns:
        DataLoader 实例
    """
    dataset = PretrainDataset(bin_path, idx_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
    )


def load_mixture_data(
    config: Dict,
    mix_name: str,
    batch_size: int
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    加载指定 Mixture 的训练和验证 DataLoader。

    Args:
        config: 全局配置
        mix_name: "mix_a" / "mix_b" / "mix_c"
        batch_size: micro batch size

    Returns:
        (train_loader, val_loader, meta_info)
    """
    mix_cfg = config["data"]["mixtures"][mix_name]

    train_loader = build_dataloader(
        mix_cfg["train_bin"], mix_cfg["train_idx"],
        batch_size=batch_size, shuffle=True
    )
    val_loader = build_dataloader(
        mix_cfg["valid_bin"], mix_cfg["valid_idx"],
        batch_size=batch_size, shuffle=False
    )

    meta = {
        "mix_name": mix_name,
        "train_sequences": len(train_loader.dataset),
        "val_sequences": len(val_loader.dataset),
        "train_tokens": len(train_loader.dataset) * train_loader.dataset.seq_len,
    }

    return train_loader, val_loader, meta