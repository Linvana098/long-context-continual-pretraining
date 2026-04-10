"""
long_context_pretrain.src
=========================
长上下文扩展 + 金融领域继续预训练 核心模块包

模块列表：
- utils:              通用工具
- domain_processor:   金融研报清洗与 tokenize
- long_text_filter:   长文本筛选
- rope_scaling:       RoPE 缩放方法（线性插值/NTK/YaRN）
- continual_trainer:  继续预训练训练器
- evaluator:          多维度评估（遗忘分析/长上下文 PPL）
"""