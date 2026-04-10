"""
步骤3: 长上下文扩展（Curriculum: 2K → 4K → 8K）
================================================
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging, set_seed, ensure_dirs
from src.continual_trainer import load_base_model, ContinualTrainer
from src.dataset import PretrainDataset  # 复用项目2的 Dataset

from torch.utils.data import DataLoader

if __name__ == "__main__":
    logger = setup_logging(log_file="logs/03_context.log")
    config = load_config()
    ensure_dirs(config)

    rope_method = config["rope_scaling"]["method"]
    results = []

    # Curriculum: 逐阶段扩展
    prev_ckpt = config["base_model"]["checkpoint_path"]

    for stage in config["context_extension"]["stages"]:
        name = stage["name"]
        seq_len = stage["seq_len"]
        logger.info(f"\n{'='*60}")
        logger.info(f"上下文扩展: {name} (seq_len={seq_len})")
        logger.info(f"{'='*60}")

        set_seed(config["context_extension"]["seed"])

        # 加载模型并替换 RoPE
        model = load_base_model(config, target_seq_len=seq_len, rope_method=rope_method)

        # 如果有前一阶段的 checkpoint，加载权重
        if prev_ckpt and os.path.exists(prev_ckpt) and prev_ckpt != config["base_model"]["checkpoint_path"]:
            import torch
            ckpt = torch.load(prev_ckpt, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            logger.info(f"  从上一阶段加载: {prev_ckpt}")

        # 数据：长文本 + 领域数据
        # 优先用 general_long 数据（真正的长文档）
        long_bin = f"data/general_long/tokenized/general_long_{seq_len}.bin"
        long_idx = f"data/general_long/tokenized/general_long_{seq_len}.idx"

        if not os.path.exists(long_bin):
            # 没有长文本，用领域数据（金融研报通常够长）
            long_bin = f"data/domain/tokenized/domain_{seq_len}.bin"
            long_idx = f"data/domain/tokenized/domain_{seq_len}.idx"

        if not os.path.exists(long_bin):
            logger.error(f"  未找到 seq_len={seq_len} 的训练数据")
            continue

        train_dataset = PretrainDataset(long_bin, long_idx)
        train_loader = DataLoader(train_dataset, batch_size=stage["micro_batch_size"],
                                   shuffle=True, num_workers=0, drop_last=True)

        # 验证集：用 2048 的通用验证集（检测基础能力是否保持）
        val_loaders = {}
        gen_valid_bin = config["general_data"].get("general_valid_bin")
        gen_valid_idx = config["general_data"].get("general_valid_idx")
        if gen_valid_bin and os.path.exists(gen_valid_bin):
            val_ds = PretrainDataset(gen_valid_bin, gen_valid_idx)
            val_loaders["general"] = DataLoader(val_ds, batch_size=2, shuffle=False, drop_last=True)

        # 训练
        trainer = ContinualTrainer(
            model=model, train_loader=train_loader, val_loaders=val_loaders,
            config=stage, run_name=f"context_{name}",
            wandb_config=config.get("wandb"),
        )

        ckpt_dir = os.path.join(config["output"]["checkpoint_dir"], f"context_{name}")
        result = trainer.train(checkpoint_dir=ckpt_dir)

        result["name"] = name
        result["seq_len"] = seq_len
        results.append(result)

        prev_ckpt = os.path.join(ckpt_dir, "final.pt")

    # 保存
    with open("reports/context_extension_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n上下文扩展完成！结果: reports/context_extension_results.json")