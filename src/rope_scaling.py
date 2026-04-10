"""
RoPE 缩放方法
=============
将预训练在 2K 上下文的模型扩展到 4K / 8K / 16K。
"""

import math
import torch
from typing import Tuple, Optional


def compute_rope_frequencies_linear(
    head_dim: int,
    max_seq_len: int,
    original_max_len: int = 2048,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Linear Interpolation（位置插值）。

    将位置索引按比例缩放到原始训练范围。
    位置 m → m * (original_max_len / max_seq_len)

    Args:
        head_dim:         每个注意力头的维度
        max_seq_len:      目标序列长度（如 8192）
        original_max_len: 原始训练长度（2048）
        theta:            RoPE base frequency
    """
    # 频率基底不变
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    # 位置索引做线性插值（压缩到原始范围）
    scale = original_max_len / max_seq_len  # 2048/8192 = 0.25
    positions = torch.arange(max_seq_len, device=device).float() * scale
    # 位置 0 → 0, 位置 4000 → 1000, 位置 8191 → 2047.75

    angles = torch.outer(positions, freqs)
    return torch.cos(angles), torch.sin(angles)


def compute_rope_frequencies_ntk(
    head_dim: int,
    max_seq_len: int,
    original_max_len: int = 2048,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NTK-aware Scaling。

    调高 base theta 使旋转频率降低，
    让更大的位置索引产生和训练时类似的旋转角。

    公式：theta_new = theta × (target_len / train_len) ^ (d / (d-2))

    直觉：如果目标长度是训练长度的 4 倍，
    我们把 theta 放大约 4^(d/(d-2)) 倍，
    这样位置 m 的旋转角 m/theta_new 约等于 (m/4)/theta，
    落入训练时的分布范围。
    """
    # 计算缩放后的 theta
    scale = max_seq_len / original_max_len  # 8192/2048 = 4.0
    d = head_dim
    # NTK scaling factor
    theta_scaled = theta * (scale ** (d / (d - 2)))

    freqs = 1.0 / (theta_scaled ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    # 位置索引不缩放（原样使用）
    positions = torch.arange(max_seq_len, device=device).float()

    angles = torch.outer(positions, freqs)
    return torch.cos(angles), torch.sin(angles)


def compute_rope_frequencies_yarn(
    head_dim: int,
    max_seq_len: int,
    original_max_len: int = 2048,
    theta: float = 10000.0,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    YaRN (Yet another RoPE extensioN)。

    分段策略：
    - 高频维度（i 小，θ_i 大）：这些维度的旋转周期短，
      在原始训练长度内已经转了很多圈，外推时不会出问题 → 不缩放
    - 低频维度（i 大，θ_i 小）：旋转周期长，
      原始长度内可能不到一圈，外推时角度超出分布 → 做线性插值

    通过 ramp 函数在高频和低频之间平滑过渡。

    beta_fast: 高频边界（周期 < 2π/beta_fast 的维度不缩放）
    beta_slow: 低频边界（周期 > 2π/beta_slow 的维度做全插值）
    """
    scale = max_seq_len / original_max_len

    # 计算每个维度的"波长"
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    wavelengths = 2 * math.pi / freqs  # 每个维度的旋转周期

    # 计算 ramp 函数（0~1 之间的插值系数）
    # ramp = 0 → 不缩放（高频）
    # ramp = 1 → 完全线性插值（低频）
    low = original_max_len / beta_fast
    high = original_max_len / beta_slow
    ramp = ((wavelengths - low) / (high - low)).clamp(0.0, 1.0)

    # 对低频维度做线性插值，高频维度保持不变
    # 插值后的频率 = freqs / lerp(1, scale, ramp)
    interpolation_factor = 1.0 - ramp + ramp / scale
    scaled_freqs = freqs * interpolation_factor

    positions = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(positions, scaled_freqs)

    # YaRN 还加了一个 attention scaling factor（补偿 softmax 温度变化）
    # attention_scale = 0.1 * ln(scale) + 1.0
    # 这个在 attention 计算时乘到 scale factor 上，这里不处理

    return torch.cos(angles), torch.sin(angles)


def get_rope_frequencies(
    method: str,
    head_dim: int,
    max_seq_len: int,
    original_max_len: int = 2048,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    统一入口：根据方法名返回对应的 RoPE 频率表。
    """
    if method == "linear":
        return compute_rope_frequencies_linear(head_dim, max_seq_len, original_max_len, theta, device)
    elif method == "ntk":
        return compute_rope_frequencies_ntk(head_dim, max_seq_len, original_max_len, theta, device)
    elif method == "yarn":
        return compute_rope_frequencies_yarn(head_dim, max_seq_len, original_max_len, theta, device=device)
    else:
        raise ValueError(f"未知的 RoPE scaling 方法: {method}. 可选: linear, ntk, yarn")