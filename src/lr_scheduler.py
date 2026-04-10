"""
学习率调度器
============
实现 warmup + cosine decay 策略
"""

import math


def get_cosine_schedule_with_warmup(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int
) -> float:
    """
    计算当前步数对应的学习率。

    Args:
        step: 当前训练步数
        max_lr: 峰值学习率
        min_lr: 最低学习率
        warmup_steps: warmup 步数
        max_steps: 总训练步数

    Returns:
        当前学习率
    """
    if step < warmup_steps:
        # Warmup: 线性增加
        return max_lr * (step / max(warmup_steps, 1))
    elif step >= max_steps:
        # 训练结束后保持最低学习率
        return min_lr
    else:
        # Cosine Decay
        # 进度: 0 → 1
        progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
        # cosine: 1 → 0（半个周期）
        cosine_value = 0.5 * (1.0 + math.cos(math.pi * progress))
        # 映射到 [min_lr, max_lr]
        return min_lr + (max_lr - min_lr) * cosine_value