"""步骤2: 筛选长文本"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging, ensure_dirs
from src.long_text_filter import filter_long_texts

if __name__ == "__main__":
    logger = setup_logging(log_file="logs/02_long_text.log")
    config = load_config()
    ensure_dirs(config)
    logger.info("步骤2: 筛选长文本")
    stats = filter_long_texts(config)
    logger.info(f"完成: {stats}")