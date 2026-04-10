"""通用工具模块"""
import os, sys, yaml, random, logging
import numpy as np
import torch
from typing import Dict, Any, Optional


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger("long_context")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def load_config(path: str = "configs/continual_config.yaml") -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs(config: Dict):
    for key in ["checkpoint_dir", "log_dir", "report_dir"]:
        os.makedirs(config["output"].get(key, key), exist_ok=True)
    os.makedirs(os.path.join(config["output"]["report_dir"], "figures"), exist_ok=True)
    for sub in ["context_4k", "context_8k", "domain_only", "domain_mixed"]:
        os.makedirs(os.path.join(config["output"]["checkpoint_dir"], sub), exist_ok=True)
    for d in ["data/domain/raw", "data/domain/cleaned", "data/domain/tokenized",
              "data/general_long/tokenized", "data/mixed/tokenized"]:
        os.makedirs(d, exist_ok=True)