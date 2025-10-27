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

### 1. 학습
```bash
# 베이스라인: 텍스트만 
python scripts/QLoRA_QWEN_train.py \
  --cfg configs/train_config.base.yaml

# 메모리: 경계/요약 임베딩 + aux.pt 저장
python scripts/memLLM_QLoRA_QWEN_train.py \
  --cfg configs/train_config.mem.yaml
```

### 2. 평가
```bash
# 메모리: Dev에서 임계치 찾기 (--thr_sweep 0.55,0.60,0.65 \)
python scripts/eval.py \
  --mode mem \
  --ckpt_mem out/qwen-qlora-mem \
  --data_mem data/val_data.jsonl \
  --load_4bit \
  --judge --judge_model gpt-4o-mini --judge_gate_hit_rate 0.0 \
  --sample_n 80 --gen_max_new 192 --temperature 0.6 --top_p 0.92 \
  --bthr_sweep 0.10:0.90:0.02 \
  --thr_sweep 0.40:0.85:0.01 \
  --save_best_thresholds thresholds/mem_best.json \
  --plot_dir plots/run_mem_sweep \
  --export_csv plots/run_mem_sweep/per_dialog.csv

# 베이스라인 + 메모리 (권장)
python scripts/eval.py \
  --mode both \
  --ckpt_base out/qwen-qlora-base \
  --ckpt_mem  out/qwen-qlora-mem \
  --data_base data/val_data.jsonl \
  --data_mem  data/val_data.jsonl \
  --load_4bit \
  --judge --judge_model gpt-4o-mini --judge_gate_hit_rate 0.0 \
  --sample_n 80 --gen_max_new 192 --temperature 0.6 --top_p 0.92 \
  --bthr_sweep 0.10:0.90:0.02 \
  --thr_sweep 0.40:0.85:0.01 \
  --save_best_thresholds thresholds/mem_best.json \
  --plot_dir plots/run_both \
  --export_csv plots/run_both
```

### 3. 추론 (only-memLLM)
```bash
# 메모리
python scripts/infer.py \
  --ckpt out/qwen-qlora-mem \
  --load_4bit \
  --sum_4bit \
  --thresholds_json thresholds/mem_best.json
```

