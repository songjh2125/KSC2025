transformer-xl/
├─ txl/
│  ├─ mem_transformer.py     # 원본 그대로(복사본). 수정 금지
│  ├─ mem_transformer_ta.py       # Topic-Aware 수정본
│  └─ __init__.py
├─ txl_hf/
│  ├─ train_hf.py
│  ├─ collator_stream.py
│  ├─ build_dataset.py
│  ├─ build_tokenizer.py
│  ├─ utils_logging.py
│  ├─ mem_baseline.py             # HF용 래퍼(원본)
│  ├─ mem_ta.py                   # HF용 래퍼(수정본)
│  └─ __init__.py
├─ data/
│  ├─ aihub/
│  ├─ kowiki/
│  └─ kodial/
├─ artifacts/
│  └─ tokenizer/ko_bpe.json
└─ logs/