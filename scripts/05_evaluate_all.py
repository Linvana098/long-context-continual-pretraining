"""步骤5: 全维度评估"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging
from src.evaluator import generate_forgetting_report

if __name__ == "__main__":
    logger = setup_logging()
    config = load_config()

    # 汇总所有结果
    all_results = {}

    ctx_path = "reports/context_extension_results.json"
    if os.path.exists(ctx_path):
        with open(ctx_path) as f:
            all_results["context_extension"] = json.load(f)

    cont_path = "reports/continual_pretrain_results.json"
    if os.path.exists(cont_path):
        with open(cont_path) as f:
            all_results.update(json.load(f))

    report_path = generate_forgetting_report(all_results, "reports")
    logger.info(f"报告: {report_path}")