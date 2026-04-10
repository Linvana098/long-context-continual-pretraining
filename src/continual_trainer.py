"""
继续预训练训练器
================
基于项目2的 Trainer 修改，增加以下功能：
1. 从项目2的 checkpoint 加载模型
2. 支持动态修改 RoPE 频率表（上下文扩展）
3. 支持混合数据训练（领域 + 通用）
4. 双验证集评估（领域 PPL + 通用 PPL，检测遗忘）
5. Activation Checkpointing（长序列必须）

与项目2 Trainer 的区别：
- 不从头训练，从 checkpoint 恢复
- 学习率更低（微调级别）
- 增加遗忘监控
"""

import os
import sys
import time
import math
import json
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm

from src.utils import setup_logging, set_seed
from src.rope_scaling import get_rope_frequencies

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def load_base_model(config: Dict, target_seq_len: int = 2048, rope_method: str = "ntk"):
    """
    从项目2的 checkpoint 加载模型，并修改 RoPE 频率表支持新的 seq_len。

    步骤：
    1. 用项目2的 ModelConfig 创建模型结构
    2. 加载 checkpoint 的 state_dict
    3. 替换 RoPE 频率表和 causal mask 以支持新的 seq_len
    """
    logger = setup_logging()

    # 创建模型结构（必须与项目2一致）
    # 先导入项目2的模型类
    from src.model import MiniLM, ModelConfig

    model_cfg = config["base_model"]
    model_config = ModelConfig(
        n_layers=model_cfg["n_layers"],
        hidden_size=model_cfg["hidden_size"],
        n_heads=model_cfg["n_heads"],
        head_dim=model_cfg["head_dim"],
        intermediate_size=model_cfg["intermediate_size"],
        vocab_size=model_cfg["vocab_size"],
        max_seq_len=target_seq_len,  # ★ 设为目标长度
        dropout=model_cfg.get("dropout", 0.0),
        weight_tying=model_cfg.get("weight_tying", True),
        rope_theta=model_cfg.get("rope_theta", 10000.0),
    )

    model = MiniLM(model_config)

    # 加载 checkpoint
    ckpt_path = model_cfg["checkpoint_path"]
    if os.path.exists(ckpt_path):
        logger.info(f"加载基座模型: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        # 忽略 shape 不匹配的 buffer（RoPE 频率表和 causal mask）
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"  模型权重加载完成")
    else:
        logger.warning(f"  未找到 checkpoint: {ckpt_path}，使用随机初始化")

    # ★★★ 替换 RoPE 频率表（扩展到目标长度）★★★
    original_len = model_cfg.get("max_seq_len", 2048)
    if target_seq_len > original_len:
        logger.info(f"  RoPE 扩展: {original_len} → {target_seq_len} (方法={rope_method})")
        cos_freq, sin_freq = get_rope_frequencies(
            method=rope_method,
            head_dim=model_cfg["head_dim"],
            max_seq_len=target_seq_len,
            original_max_len=original_len,
            theta=model_cfg.get("rope_theta", 10000.0),
        )
        model.cos_freq = cos_freq
        model.sin_freq = sin_freq

        # 替换 causal mask
        mask = torch.full((target_seq_len, target_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        model.causal_mask = mask

    return model


class ContinualTrainer:
    """继续预训练训练器。"""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loaders: Dict,  # {"domain": loader, "general": loader}
        config: Dict,
        run_name: str = "continual",
        wandb_config: Optional[Dict] = None,
    ):
        self.logger = setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.run_name = run_name

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loaders = val_loaders

        # 开启 activation checkpointing
        if config.get("use_activation_checkpoint", True):
            self._enable_activation_checkpointing()

        # 训练超参
        self.micro_batch_size = config["micro_batch_size"]
        self.grad_accum = config["gradient_accumulation"]
        self.max_steps = config["max_steps"]
        self.eval_interval = config.get("eval_interval", 100)
        self.save_interval = config.get("save_interval", 500)
        self.log_interval = config.get("log_interval", 10)
        self.grad_clip = config.get("grad_clip", 1.0)
        self.use_bf16 = config.get("use_bf16", True) and torch.cuda.is_available()

        # 优化器
        decay_params = [p for n, p in self.model.named_parameters() if p.dim() >= 2]
        no_decay_params = [p for n, p in self.model.named_parameters() if p.dim() < 2]
        self.optimizer = torch.optim.AdamW([
            {"params": decay_params, "weight_decay": config.get("weight_decay", 0.1)},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=config["learning_rate"],
           betas=(config.get("beta1", 0.9), config.get("beta2", 0.95)),
           eps=config.get("eps", 1e-8))

        self.scaler = GradScaler(enabled=self.use_bf16)

        # 状态
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.loss_history = []
        self.eval_history = {"domain": [], "general": []}

        # W&B
        self.use_wandb = HAS_WANDB and wandb_config and wandb_config.get("enabled", False)
        if self.use_wandb:
            wandb.init(project=wandb_config.get("project", "long-context"),
                       entity=wandb_config.get("entity"),
                       name=run_name, config=config, reinit=True)

        n_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"模型参数: {n_params/1e6:.1f}M | 设备: {self.device} | bf16: {self.use_bf16}")

    def _enable_activation_checkpointing(self):
        """开启 activation checkpointing（长序列必须）。"""
        for layer in self.model.layers:
            layer._original_forward = layer.forward
            def make_ckpt(module):
                def ckpt_fwd(x, cos_freq, sin_freq, mask=None):
                    return activation_checkpoint(
                        module._original_forward, x, cos_freq, sin_freq, mask,
                        use_reentrant=False)
                return ckpt_fwd
            layer.forward = make_ckpt(layer)
        self.logger.info("Activation Checkpointing 已开启")

    def _get_lr(self, step):
        from src.lr_scheduler import get_cosine_schedule_with_warmup
        return get_cosine_schedule_with_warmup(
            step, self.config["learning_rate"], self.config["min_lr"],
            self.config["warmup_steps"], self.max_steps)

    @torch.no_grad()
    def evaluate(self, name: str, loader) -> Dict:
        """在指定验证集上评估 PPL。"""
        self.model.eval()
        total_loss, total_tokens, n_batches = 0.0, 0, 0

        for batch in loader:
            batch = batch.to(self.device)
            with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                _, loss = self.model(input_ids=batch, targets=batch)
            if loss is not None:
                n = batch.shape[0] * (batch.shape[1] - 1)
                total_loss += loss.item() * n
                total_tokens += n
                n_batches += 1
            if n_batches >= 50:
                break

        self.model.train()
        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))
        return {"name": name, "loss": avg_loss, "ppl": ppl}

    def evaluate_all(self) -> Dict:
        """在所有验证集上评估。"""
        results = {}
        for name, loader in self.val_loaders.items():
            if loader is not None:
                r = self.evaluate(name, loader)
                results[name] = r
                self.eval_history[name].append((self.global_step, r["ppl"]))
        return results

    def _save_checkpoint(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "loss_history": self.loss_history[-1000:],
            "eval_history": self.eval_history,
            "config": self.config,
        }, path)

    def train(self, checkpoint_dir: str) -> Dict:
        """继续预训练主循环。"""
        self.model.train()
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"继续预训练开始: {self.run_name}")
        self.logger.info(f"{'='*60}")

        # 训练前评估（基线）
        baseline = self.evaluate_all()
        self.logger.info(f"  训练前基线:")
        for name, r in baseline.items():
            self.logger.info(f"    {name}: PPL={r['ppl']:.1f}")

        data_iter = iter(self.train_loader)
        train_start = time.time()
        step_start = time.time()

        while self.global_step < self.max_steps:
            lr = self._get_lr(self.global_step)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            for micro in range(self.grad_accum):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)

                batch = batch.to(self.device, non_blocking=True)
                with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                    _, loss = self.model(input_ids=batch, targets=batch)
                    loss = loss / self.grad_accum

                self.scaler.scale(loss).backward()
                step_loss += loss.item()

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            self.loss_history.append(step_loss)
            self.global_step += 1

            if self.global_step % self.log_interval == 0:
                elapsed = time.time() - step_start

                def get_seq_len(dataset):
                    """万能获取 seq_len，支持 Subset / ConcatDataset / 任意嵌套 """
                    from torch.utils.data import Subset, ConcatDataset

                    # 如果是 Subset，剥一层
                    if isinstance(dataset, Subset):
                        return get_seq_len(dataset.dataset)

                    # 如果是 ConcatDataset，取第一个子数据集（所有数据集 seq_len 一定相同）
                    elif isinstance(dataset, ConcatDataset):
                        return get_seq_len(dataset.datasets[0])

                    # 终于到原始数据集，直接返回 seq_len
                    else:
                        return dataset.seq_len

                seq_len = get_seq_len(self.train_loader.dataset)

                tok_s = self.micro_batch_size * self.grad_accum * seq_len / elapsed
                mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
                self.logger.info(
                    f"  step={self.global_step:>5d}/{self.max_steps} | loss={step_loss:.4f} | "
                    f"lr={lr:.2e} | grad={grad_norm:.2f} | tok/s={tok_s:.0f} | mem={mem:.0f}MB")

                if self.use_wandb:
                    wandb.log({"train/loss": step_loss, "train/lr": lr,
                               "train/grad_norm": float(grad_norm)}, step=self.global_step)
                step_start = time.time()

            if self.global_step % self.eval_interval == 0:
                eval_res = self.evaluate_all()
                for name, r in eval_res.items():
                    self.logger.info(f"    {name}: PPL={r['ppl']:.1f}")
                    if self.use_wandb:
                        wandb.log({f"val/{name}_ppl": r["ppl"]}, step=self.global_step)

            if self.global_step % self.save_interval == 0:
                path = os.path.join(checkpoint_dir, f"step_{self.global_step}.pt")
                self._save_checkpoint(path)

        # 最终评估
        final_eval = self.evaluate_all()
        total_time = time.time() - train_start

        self._save_checkpoint(os.path.join(checkpoint_dir, "final.pt"))

        if self.use_wandb:
            wandb.finish()

        result = {
            "run_name": self.run_name,
            "total_time_s": total_time,
            "baseline_eval": {k: v for k, v in baseline.items()},
            "final_eval": {k: v for k, v in final_eval.items()},
            "eval_history": self.eval_history,
            "loss_history": self.loss_history,
        }

        self.logger.info(f"\n  完成: {self.run_name} | {total_time:.0f}s")
        for name in final_eval:
            b_ppl = baseline[name]["ppl"] if name in baseline else "N/A"
            f_ppl = final_eval[name]["ppl"]
            self.logger.info(f"    {name}: {b_ppl:.1f} → {f_ppl:.1f}")

        return result