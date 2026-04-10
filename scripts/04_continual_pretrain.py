"""
步骤4: 金融领域继续预训练
========================
两组对比实验：
1. 纯领域训练（100% 金融研报）
2. 混合训练（70% 金融 + 30% 通用）
"""
import os, sys, json
import torch
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging, set_seed, ensure_dirs
from src.continual_trainer import load_base_model, ContinualTrainer
from src.dataset import PretrainDataset
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

if __name__ == "__main__":
    logger = setup_logging(log_file="logs/04_continual.log")
    config = load_config()
    ensure_dirs(config)

    cont_cfg = config["continual_pretrain"]
    seq_len = cont_cfg["seq_len"]
    rope_method = config["rope_scaling"]["method"]

    # 加载上下文扩展后的模型（如果有 4K 的 checkpoint）
    ctx_ckpt = os.path.join(config["output"]["checkpoint_dir"], "context_4k", "final.pt")
    if not os.path.exists(ctx_ckpt):
        ctx_ckpt = config["base_model"]["checkpoint_path"]  # 回退到原始模型
        logger.info(f"未找到 4K checkpoint，使用原始基座模型")

    # 准备验证集
    val_loaders_base = {}
    # 领域验证集（从领域数据中取一部分）
    domain_bin = f"data/domain/tokenized/domain_{seq_len}.bin"
    domain_idx = f"data/domain/tokenized/domain_{seq_len}.idx"
    if os.path.exists(domain_bin):
        ds = PretrainDataset(domain_bin, domain_idx)
        # 取最后 5% 作为验证集
        val_size = max(int(len(ds) * 0.05), 1)
        train_size = len(ds) - val_size

        train_indices = range(0, train_size)
        train_ds = torch.utils.data.Subset(ds, train_indices)

        val_indices = range(train_size, len(ds))
        val_ds = torch.utils.data.Subset(ds, val_indices)

        val_loaders_base["domain"] = DataLoader(
            val_ds, batch_size=cont_cfg["micro_batch_size"], shuffle=False, drop_last=True)

    # 通用验证集
    gen_valid_bin = config["general_data"].get("general_valid_bin")
    gen_valid_idx = config["general_data"].get("general_valid_idx")
    if gen_valid_bin and os.path.exists(gen_valid_bin):
        val_ds = PretrainDataset(gen_valid_bin, gen_valid_idx)
        val_loaders_base["general"] = DataLoader(
            val_ds, batch_size=cont_cfg["micro_batch_size"], shuffle=False, drop_last=True)

    all_results = {}

    # ============================================================
    # 实验1: 纯领域训练
    # ============================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"实验1: 纯领域继续预训练")
    logger.info(f"{'='*60}")

    set_seed(cont_cfg["seed"])
    model = load_base_model(config, target_seq_len=seq_len, rope_method=rope_method)

    if os.path.exists(ctx_ckpt) and ctx_ckpt != config["base_model"]["checkpoint_path"]:
        import torch
        ckpt = torch.load(ctx_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    if os.path.exists(domain_bin):
        train_loader = DataLoader(train_ds, batch_size=cont_cfg["micro_batch_size"],
                                   shuffle=True, num_workers=0, drop_last=True)

        trainer = ContinualTrainer(
            model=model, train_loader=train_loader, val_loaders=val_loaders_base,
            config=cont_cfg, run_name="domain_only", wandb_config=config.get("wandb"))

        result = trainer.train(os.path.join(config["output"]["checkpoint_dir"], "domain_only"))
        all_results["domain_only"] = result

    # ============================================================
    # 实验2: 混合训练（85% 领域 + 15% 通用）
    # ============================================================
    logger.info(f"\n{'='*60}")
    logger.info(f"实验2: 混合继续预训练 (70% 领域 + 30% 通用)")
    logger.info(f"{'='*60}")

    set_seed(cont_cfg["seed"])
    model = load_base_model(config, target_seq_len=seq_len, rope_method=rope_method)

    if os.path.exists(ctx_ckpt) and ctx_ckpt != config["base_model"]["checkpoint_path"]:
        import torch
        ckpt = torch.load(ctx_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # 构建混合数据集
    general_long_bin = f"data/general_long/tokenized/general_long_{seq_len}.bin"
    general_long_idx = f"data/general_long/tokenized/general_long_{seq_len}.idx"

    if os.path.exists(domain_bin) and os.path.exists(general_long_bin):
        domain_ds = PretrainDataset(domain_bin, domain_idx)
        general_ds = PretrainDataset(general_long_bin, general_long_idx)

        # 按比例采样：用 ConcatDataset + WeightedRandomSampler 更精确
        len_general_ds = int(len(domain_ds) / 0.7 * 0.3)

        # 随机打乱索引，然后取前 85%
        torch.manual_seed(42)
        indices = torch.randperm(len(domain_ds)).tolist()
        indices = indices[:len_general_ds]

        general_cut_ds = torch.utils.data.Subset(general_ds, indices)

        domain_weights = [0.7] * len(domain_ds)
        general_weights = [0.3] * len_general_ds
        all_weights = domain_weights + general_weights

        sampler = WeightedRandomSampler(
            weights=all_weights,
            num_samples=len(domain_ds) + len_general_ds,
            replacement=True
        )

        mixed_ds = ConcatDataset([domain_ds, general_cut_ds])
        # 使用sampler代替shuffle
        train_loader = DataLoader(mixed_ds, batch_size=cont_cfg["micro_batch_size"],
                                   sampler=sampler, shuffle=False, num_workers=0, drop_last=True)

        logger.info(f"  领域: {len(domain_ds)} 序列, 通用: {len_general_ds} 序列")

        trainer = ContinualTrainer(
            model=model, train_loader=train_loader, val_loaders=val_loaders_base,
            config=cont_cfg, run_name="domain_mixed", wandb_config=config.get("wandb"))

        result = trainer.train(os.path.join(config["output"]["checkpoint_dir"], "domain_mixed"))
        all_results["domain_mixed"] = result
    else:
        logger.warning("通用长文本数据不足，跳过混合训练实验")

    # 保存
    with open("reports/continual_pretrain_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\n继续预训练完成！结果: reports/continual_pretrain_results.json")