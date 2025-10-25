```bash
KSC2025/
├── README.md
├── requirements.txt
├── .env            # 사용자가 설정해야 함
├── .gitignore
│
├── scripts/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── data_utils.py
│   │   ├── mem_modules.py
│   │   └── summarizer_local.py
│   │
│   ├── eval.py           
│   ├── infer.py          
│   ├── memLLM_QLoRA_QWEN_train.py
│   └── preprocess.py
│
├── configs/
│   ├── eval_config.yaml
│   ├── get_data.yaml
│   ├── preprocess.yaml
│   ├── train_config.base.yaml
│   └── train_config.mem.yaml
│
├── out/
│   └── solar-mem-qlora/
│       └── .gitkeep
│
├── log/
│   └── .gitkeep
│
├── cache/
│   └── .gitkeep
│
├── data/
│   └── .gitkeep
│
└── Makefile 
```

### 학습
```bash
# 베이스라인: 텍스트만 
python scripts/QLoRA_QWEN_train.py \
  --cfg configs/train_config.base.yaml

# 메모리: 경계/요약 임베딩 + aux.pt 저장
python scripts/memLLM_QLoRA_QWEN_train.py \
  --cfg configs/train_config.mem.yaml
```

### 추론
```bash
# 베이스라인
python mem_infer_demo.py \
  --ckpt out/qwen-qlora-base \
  --mode baseline \
  --load_4bit

# 메모리
python mem_infer_demo.py \
  --ckpt out/qwen-qlora-mem \
  --mode mem \
  --load_4bit \
  --sum_4bit
```

### 평가
```bash
# 베이스라인: Judge만
python scripts/eval_mem.py \
  --ckpt out/qwen-qlora-base \
  --data cache/labels_ko_samsum.jsonl \
  --mode base \
  --judge --judge_model gpt-4o-mini --sample_n 50 \
  --load_4bit

# 메모리: Boundary/Summary + Judge
python scripts/eval_mem.py \
  --ckpt out/qwen-qlora-mem \
  --data cache/labels_ko_samsum.jsonl \
  --mode mem \
  --boundary_thr 0.5 \
  --judge --judge_model gpt-4o-mini --sample_n 50 \
  --load_4bit
```
