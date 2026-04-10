"""步骤1: 金融研报预处理"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging, ensure_dirs
from src.domain_processor import run_domain_processing

if __name__ == "__main__":
    logger = setup_logging(log_file="logs/01_domain.log")
    config = load_config()
    ensure_dirs(config)
    logger.info("步骤1: 金融研报预处理")
    stats = run_domain_processing(config)
    logger.info(f"完成: {stats}")