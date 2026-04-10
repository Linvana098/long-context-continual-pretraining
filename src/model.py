"""
手写 Decoder-Only Transformer
==============================
从零实现 LLaMA 风格的 causal language model。

架构选择（对齐主流基座模型）：
- Pre-LayerNorm：用 RMSNorm 替代 LayerNorm（LLaMA/Qwen 的标配）
- RoPE：旋转位置编码（替代绝对位置编码，支持长度外推）
- SwiGLU：替代标准 FFN 的激活函数（提升表达能力）
- Weight Tying：输入 Embedding 与输出 Head 共享权重（减少参数）
- Causal Mask：自回归注意力掩码

参数量对照：
- Baseline: 8 层, hidden=512, heads=8, FFN=1408  → ~42M
- Main:    12 层, hidden=768, heads=12, FFN=2048  → ~110M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


# ==========================================
# 1. 模型配置（数据类）
# ==========================================

@dataclass
class ModelConfig:
    """
    模型超参数配置。

    Attributes:
        n_layers:          Transformer 块的数量
        hidden_size:       隐藏层维度 d_model
        n_heads:           注意力头数
        head_dim:          每个头的维度 = hidden_size / n_heads
        intermediate_size: SwiGLU FFN 的中间层维度
        vocab_size:        词表大小
        max_seq_len:       最大序列长度
        dropout:           Dropout 概率（预训练时通常为 0）
        weight_tying:      是否共享 Embedding 和 Output Head 权重
        rope_theta:        RoPE 的基础频率（默认 10000）
    """
    n_layers: int = 8
    hidden_size: int = 512
    n_heads: int = 8
    head_dim: int = 64
    intermediate_size: int = 1408
    vocab_size: int = 32000
    max_seq_len: int = 2048
    dropout: float = 0.0
    weight_tying: bool = True
    rope_theta: float = 10000.0

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """从字典创建配置。"""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ==========================================
# 2. RMSNorm（替代 LayerNorm）
# ==========================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization。

    与 LayerNorm 的区别：
    - 不减均值（没有 centering），只做缩放
    - 计算更快（少一次均值计算）
    - 实践中效果与 LayerNorm 相当

    公式: output = x * weight / sqrt(mean(x²) + eps)

    为什么 LLaMA 用 RMSNorm？
    - 比 LayerNorm 快 ~10%
    - 训练稳定性不受影响
    - 简化实现
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Args:
            hidden_size: 归一化维度
            eps: 数值稳定性常数
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # weight 形状: (hidden_size,)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_size)

        Returns:
            normalized: (batch_size, seq_len, hidden_size)
        """
        # x² 的均值，沿最后一维计算
        # variance: (batch_size, seq_len, 1)
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)

        # x * rsqrt(var + eps)  →  归一化
        # 先转 float32 计算（数值稳定），再转回原精度
        x_normed = x.float() * torch.rsqrt(variance + self.eps)

        # 乘以可学习的缩放参数 weight
        # output: (batch_size, seq_len, hidden_size)
        return (x_normed * self.weight).type_as(x)


# ==========================================
# 3. RoPE 旋转位置编码
# ==========================================

def precompute_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    预计算 RoPE 的 cos 和 sin 频率表。

    RoPE 的核心思想：
    - 不给每个位置一个固定的 embedding 向量
    - 而是在注意力计算时，按位置对 Q 和 K 向量做"旋转"
    - 旋转角度与位置成正比，不同维度的旋转频率不同
    - 这使得两个 token 的注意力分数只依赖于它们的相对距离

    频率公式：
    θ_i = theta^(-2i/d)，其中 i = 0, 1, ..., d/2-1

    对于位置 m，旋转角为：m * θ_i

    Args:
        head_dim: 每个注意力头的维度
        max_seq_len: 最大序列长度
        theta: 基础频率（LLaMA 默认 10000）

    Returns:
        cos_freq: (max_seq_len, head_dim / 2)  每个位置每个维度对的 cos 值
        sin_freq: (max_seq_len, head_dim / 2)  每个位置每个维度对的 sin 值
    """
    # 计算频率基底：θ_i = theta^(-2i/d)
    # i = [0, 1, 2, ..., head_dim/2 - 1]
    # freqs: (head_dim / 2,)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

    # 位置索引：m = [0, 1, 2, ..., max_seq_len - 1]
    # positions: (max_seq_len,)
    positions = torch.arange(max_seq_len, device=device).float()

    # 外积：每个位置 × 每个频率 = 旋转角矩阵
    # angles: (max_seq_len, head_dim / 2)
    angles = torch.outer(positions, freqs)

    # cos 和 sin 缓存
    cos_freq = torch.cos(angles)  # (max_seq_len, head_dim / 2)
    sin_freq = torch.sin(angles)  # (max_seq_len, head_dim / 2)

    return cos_freq, sin_freq


def apply_rope(
    x: torch.Tensor,
    cos_freq: torch.Tensor,
    sin_freq: torch.Tensor
) -> torch.Tensor:
    """
    对 Q 或 K 张量施加 RoPE 旋转。

    旋转公式（对每对相邻维度 [x_2i, x_{2i+1}]）：
    x_2i'    = x_2i * cos(mθ_i) - x_{2i+1} * sin(mθ_i)
    x_{2i+1}' = x_2i * sin(mθ_i) + x_{2i+1} * cos(mθ_i)

    这等价于一个 2D 旋转矩阵：
    [cos  -sin] [x_2i    ]
    [sin   cos] [x_{2i+1}]

    Args:
        x: (batch_size, n_heads, seq_len, head_dim)  Q 或 K
        cos_freq: (max_seq_len, head_dim / 2)  预计算的 cos 值
        sin_freq: (max_seq_len, head_dim / 2)  预计算的 sin 值

    Returns:
        rotated: (batch_size, n_heads, seq_len, head_dim)
    """
    B, H, S, D = x.shape
    # 将 x 的最后一维拆成两半：(x_even, x_odd)
    # x 形状: (B, H, S, D) → 拆成 (B, H, S, D/2, 2)
    x_reshaped = x.float().reshape(B, H, S, -1, 2)
    # x_even: (B, H, S, D/2) — 偶数维度
    # x_odd:  (B, H, S, D/2) — 奇数维度
    x_even = x_reshaped[..., 0]
    x_odd = x_reshaped[..., 1]

    # 扩展 cos/sin 以匹配 batch 和 heads 维度
    # cos_freq: (S, D/2) → (1, 1, S, D/2)
    cos_f = cos_freq[None, None, :S, :]
    sin_f = sin_freq[None, None, :S, :]

    # 应用旋转
    # rotated_even = x_even * cos - x_odd * sin
    # rotated_odd  = x_even * sin + x_odd * cos
    # (B, H, S, D/2)
    rotated_even = x_even * cos_f - x_odd * sin_f
    rotated_odd = x_even * sin_f + x_odd * cos_f

    # 重新交错合并: (B, H, S, D/2) → (B, H, S, D/2, 2) → (B, H, S, D)
    rotated = torch.stack([rotated_even, rotated_odd], dim=-1)
    return rotated.reshape(B, H, S, D).type_as(x)


# ==========================================
# 4. 多头因果自注意力
# ==========================================

class CausalSelfAttention(nn.Module):
    """
    多头因果自注意力（带 RoPE）。

    计算流程：
    1. 线性投影：x → Q, K, V
    2. 拆头：(B, S, D) → (B, H, S, D_head)
    3. RoPE：对 Q, K 施加旋转位置编码
    4. 注意力计算：softmax(QK^T / √d) · V
    5. 合并头：(B, H, S, D_head) → (B, S, D)
    6. 输出投影：o_proj
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        # Q, K, V, O 线性投影
        # 每个投影: (hidden_size, hidden_size)  因为 n_heads × head_dim = hidden_size
        self.q_proj = nn.Linear(config.hidden_size, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.n_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.n_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos_freq: torch.Tensor,
        sin_freq: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x:        (batch_size, seq_len, hidden_size)
            cos_freq: (max_seq_len, head_dim / 2)
            sin_freq: (max_seq_len, head_dim / 2)
            mask:     (max_seq_len, max_seq_len) causal mask，上三角为 -inf

        Returns:
            output:   (batch_size, seq_len, hidden_size)
        """
        B, S, D = x.shape  # B=batch, S=seq_len, D=hidden_size

        # ---- 步骤1：线性投影 ----
        # Q, K, V: (B, S, n_heads * head_dim) = (B, S, D)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ---- 步骤2：拆头 ----
        # (B, S, D) → (B, S, H, D_head) → (B, H, S, D_head)
        q = q.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # 现在 Q, K, V: (B, H, S, D_head)

        # ---- 步骤3：RoPE ----
        # 对 Q 和 K 施加旋转位置编码（V 不做旋转）
        q = apply_rope(q, cos_freq, sin_freq)
        k = apply_rope(k, cos_freq, sin_freq)
        # Q, K 仍为: (B, H, S, D_head)

        # ★★★ 替换：用 SDPA 自动调用 FlashAttention 2 ★★★
        # F.scaled_dot_product_attention 会自动选择最优 backend:
        #   1. FlashAttention 2（如果可用：Ampere+ GPU + bf16/fp16）
        #   2. Memory-Efficient Attention（xformers 风格）
        #   3. 数学实现（fallback）
        #
        # 显存节省：O(N²) → O(N)（不存储完整 attention 矩阵）
        # 速度提升：长序列 2~4x（减少 HBM 读写）
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )   # attn_output: (B, H, S, D_head)

        # ---- 步骤5：合并头 ----
        # (B, H, S, D_head) → (B, S, H, D_head) → (B, S, D)
        attn_output = attn_output.transpose(1, 2).reshape(B, S, D)

        # ---- 步骤6：输出投影 ----
        # output: (B, S, D)
        output = self.o_proj(attn_output)

        return output


# ==========================================
# 5. SwiGLU FFN
# ==========================================

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network。

    标准 FFN:  output = W_down · ReLU(W_up · x)
    SwiGLU:    output = W_down · (Swish(W_gate · x) ⊙ (W_up · x))

    其中 Swish(x) = x · σ(x)，σ 是 sigmoid。
    ⊙ 表示逐元素乘法（门控机制）。

    为什么 SwiGLU 更好？
    - 门控机制让网络可以选择性地激活/抑制不同维度
    - 实验表明比 ReLU/GELU FFN 在相同参数量下效果更好
    - LLaMA、PaLM、Gemma 等都使用 SwiGLU

    参数量：3 × hidden × intermediate（比标准 FFN 多 50%）
    为了保持总参数量一致，intermediate 通常设为 2/3 × 4 × hidden ≈ 8/3 × hidden
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # gate_proj: x → 门控信号    (hidden_size → intermediate_size)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # up_proj: x → 值信号        (hidden_size → intermediate_size)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        # down_proj: 降维回原始维度   (intermediate_size → hidden_size)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_size)

        Returns:
            output: (batch_size, seq_len, hidden_size)

        中间张量维度：
            gate:   (B, S, intermediate_size) — 门控信号
            up:     (B, S, intermediate_size) — 值信号
            hidden: (B, S, intermediate_size) — Swish(gate) ⊙ up
            output: (B, S, hidden_size)       — 降维回原始维度
        """
        # gate: (B, S, hidden_size) → (B, S, intermediate_size)
        gate = self.gate_proj(x)
        # up:   (B, S, hidden_size) → (B, S, intermediate_size)
        up = self.up_proj(x)

        # Swish(gate) ⊙ up — 门控激活
        # F.silu(gate) = gate * sigmoid(gate) = Swish
        # hidden: (B, S, intermediate_size)
        hidden = F.silu(gate) * up

        # down: (B, S, intermediate_size) → (B, S, hidden_size)
        output = self.down_proj(hidden)
        output = self.dropout(output)

        return output


# ==========================================
# 6. Transformer 块
# ==========================================

class TransformerBlock(nn.Module):
    """
    单个 Transformer 块（Pre-LayerNorm 风格）。

    计算流程：
    x → RMSNorm → Attention → + 残差
      → RMSNorm → SwiGLU FFN → + 残差

    Pre-LN vs Post-LN：
    - Post-LN（原始 Transformer）: x + Norm(Attn(x))  → 深层时容易梯度爆炸
    - Pre-LN（LLaMA 等）:  x + Attn(Norm(x))  → 训练更稳定，无需 warmup 也能收敛
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        # Pre-LN: 在 Attention / FFN 之前做归一化
        self.attn_norm = RMSNorm(config.hidden_size)
        self.ffn_norm = RMSNorm(config.hidden_size)

        self.attention = CausalSelfAttention(config)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        cos_freq: torch.Tensor,
        sin_freq: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_size)

        Returns:
            output: (batch_size, seq_len, hidden_size)
        """
        # ---- 自注意力子层 ----
        # residual = x
        # x = x + Attention(RMSNorm(x))
        residual = x
        x = self.attn_norm(x)       # (B, S, D) → (B, S, D)
        x = self.attention(x, cos_freq, sin_freq, mask)  # (B, S, D) → (B, S, D)
        x = residual + x            # 残差连接

        # ---- FFN 子层 ----
        # x = x + FFN(RMSNorm(x))
        residual = x
        x = self.ffn_norm(x)        # (B, S, D) → (B, S, D)
        x = self.ffn(x)             # (B, S, D) → (B, S, D)
        x = residual + x            # 残差连接

        return x


# ==========================================
# 7. 完整模型
# ==========================================

class MiniLM(nn.Module):
    """
    完整的 Mini Language Model（Decoder-Only）。

    结构：
    Token Embedding → N × TransformerBlock → RMSNorm → LM Head

    特性：
    - Weight Tying: LM Head 与 Token Embedding 共享权重
    - 预计算 RoPE 频率表和因果掩码（不随训练更新）
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token Embedding: token_id → 向量
        # embed_tokens 权重: (vocab_size, hidden_size)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # N 个 Transformer 块
        self.layers = nn.ModuleList([
            TransformerBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # 最终 RMSNorm
        self.final_norm = RMSNorm(config.hidden_size)

        # LM Head: hidden → vocab logits
        if config.weight_tying:
            # 共享权重：直接用 embed_tokens.weight 做输出投影
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # ---- 预计算不可训练的缓存 ----
        # RoPE 频率表
        cos_freq, sin_freq = precompute_rope_frequencies(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta
        )
        # 注册为 buffer（不参与梯度计算，但会随模型移动设备）
        self.register_buffer("cos_freq", cos_freq, persistent=False)
        self.register_buffer("sin_freq", sin_freq, persistent=False)

        # 因果掩码（上三角为 -inf）
        # mask: (max_seq_len, max_seq_len)
        mask = torch.full((config.max_seq_len, config.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)  # 上三角为 -inf，对角线及下方为 0
        self.register_buffer("causal_mask", mask, persistent=False)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """
        权重初始化策略：

        - Embedding: N(0, 0.02)
        - Linear: N(0, 0.02)
        - 残差路径的最后一层线性层: N(0, 0.02 / √(2 * n_layers))
          这是 GPT-2 的做法，确保深层网络的残差不会随层数线性增长

        为什么用 0.02 而不是 Xavier/He 初始化？
        - 经验值。GPT 系列、LLaMA 都用类似的值
        - 太小会导致初始输出接近零，太大会导致训练初期不稳定
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # 残差路径的输出投影使用更小的初始化
        for layer in self.layers:
            torch.nn.init.normal_(
                layer.attention.o_proj.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.config.n_layers)
            )
            torch.nn.init.normal_(
                layer.ffn.down_proj.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * self.config.n_layers)
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播。

        Args:
            input_ids: (batch_size, seq_len)  token id 序列
            targets:   (batch_size, seq_len)  目标序列（用于计算 loss）
                       通常 targets = input_ids（自回归：预测下一个 token）

        Returns:
            logits: (batch_size, seq_len, vocab_size)  每个位置的词表分布
            loss:   标量，交叉熵损失（仅当 targets 不为 None 时返回）

        计算流程：
            input_ids (B, S)
                ↓ Embedding
            x (B, S, D)
                ↓ N × TransformerBlock
            x (B, S, D)
                ↓ RMSNorm
            x (B, S, D)
                ↓ LM Head (D → V)
            logits (B, S, V)
                ↓ CrossEntropyLoss(logits, targets)
            loss (scalar)
        """
        B, S = input_ids.shape

        # Token Embedding
        # input_ids: (B, S) → x: (B, S, hidden_size)
        x = self.embed_tokens(input_ids)

        # 通过所有 Transformer 块
        for layer in self.layers:
            x = layer(x, self.cos_freq, self.sin_freq, self.causal_mask)
        # x: (B, S, hidden_size)

        # 最终归一化
        x = self.final_norm(x)  # (B, S, hidden_size)

        # LM Head → logits
        if self.lm_head is not None:
            # 独立的输出投影
            logits = self.lm_head(x)
        else:
            # Weight Tying: 用 Embedding 权重的转置做投影
            # x @ embed_tokens.weight^T
            # (B, S, D) @ (D, V) → (B, S, V)
            logits = F.linear(x, self.embed_tokens.weight)
        # logits: (B, S, vocab_size)

        # 计算损失
        loss = None
        if targets is not None:
            # 自回归 loss：用位置 i 的 logits 预测位置 i+1 的 token
            # shift: logits[:-1] 预测 targets[1:]
            shift_logits = logits[:, :-1, :].contiguous()  # (B, S-1, V)
            shift_targets = targets[:, 1:].contiguous()    # (B, S-1)

            # 交叉熵损失
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),  # (B*(S-1), V)
                shift_targets.view(-1),                        # (B*(S-1),)
                ignore_index=-1  # 忽略 padding（如果有的话）
            )

        return logits, loss


def build_model(config_dict: dict) -> MiniLM:
    """从配置字典构建模型。"""
    model_config = ModelConfig.from_dict(config_dict)
    return MiniLM(model_config)