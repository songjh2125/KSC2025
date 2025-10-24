```bash
KSC2025/
├── README.md
├── requirements.txt              # 필수 패키지 목록 (Transformers / PEFT / Accelerate / DeepSpeed 등)
├── .env.example                  # 예시 환경 변수 파일 (예: OPENAI_API_KEY)
├── .gitignore
│
├── scripts/                      # 실행 스크립트 (직접 실행)
│   ├── src/                      # Python 모듈 (import path: scripts/src)
│   │   ├── __init__.py
│   │   ├── mem_modules.py        # (제공) 메모리/보조 헤드/Proj_mem 정의
│   │   ├── summarizer_local.py   # (제공) 로컬 SOLAR 요약기
│   │   └── data_utils.py         # (제공) 정규화 및 데이터 유틸
│   │
│   ├── make_labels.py            # (제공) GPT-4o-mini 기반 라벨 생성 및 JSONL 캐시
│   ├── memLLM_QLoRA_SOLAR_koSamsum_train.py  # (제공) 학습 스크립트
│   ├── eval_mem.py               # (제공) 평가 스크립트
│   └── mem_infer_demo.py         # (제공) 추론 데모
│
├── cache/                        # make_labels.py 결과 캐시(JSONL 등)
│   └── .gitkeep
│
├── out/                          # 학습 산출물 (모델, 어댑터, aux_mem.pt 등)
│   ├── solar-mem-qlora/          # 기본 체크포인트 디렉터리
│   │   └── .gitkeep
│   └── .gitkeep
│
├── logs/                         # 학습 및 평가 로그
│   └── .gitkeep
│
├── configs/                      # (옵션) 설정 파일
│   ├── train_config.yaml         # 하이퍼파라미터 및 경로 외부화
│   └── eval_config.yaml
│
└── Makefile                      # 자주 쓰는 명령어 단축용 (옵션)
```