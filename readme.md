```
long_context_pretrain/
│
├── README.md
├── requirements.txt
│
├── configs/
│   └── continual_config.yaml          # 全局配置
│
├── scripts/
│   ├── 01_prepare_domain_data.py      # 金融研报预处理
│   ├── 02_prepare_general_data.py     # 通用语料筛选（长文本）
│   ├── 03_extend_context.py           # RoPE scaling + 长上下文微调
│   ├── 04_continual_pretrain.py       # 金融领域继续预训练
│   └── 05_evaluate_all.py             # 全维度评估
│
├── src/
│   ├── __init__.py
│   ├── utils.py                       # 通用工具
│   ├── domain_processor.py            # 金融研报清洗与 tokenize
│   ├── long_text_filter.py            # 长文本筛选
│   ├── rope_scaling.py                # RoPE 缩放方法（核心）
│   ├── continual_trainer.py           # 继续预训练训练器
│   ├── lr_scheduler.py                # 自定义调度器
│   ├── model.py                       # Mini模型
│   ├── dataset.py                     # 数据集加载模块
│   └── evaluator.py                   # 多维度评估
│
├── data/
│   ├── domain/
│   │   ├── raw/                       # 金融研报原始数据
│   │   ├── cleaned/                   # 清洗后
│   │   └── tokenized/                 # tokenize + pack 后
│   ├── general_long/                  # 从项目1筛选的长文本
│   │   └── tokenized/
│   └── mixed/                         # 领域+通用混合数据
│       └── tokenized/
│
├── checkpoints/
│   ├── context_4k/
│   ├── context_8k/
│   ├── domain_only/                   # 纯领域继续预训练
│   └── domain_mixed/                  # 混合通用语料继续预训练
│
├── logs/                              # 日志
└── reports/                           # 生成的报告
```