# 🧠 Transformer-XL Topic-Aware (TXL-TA)

이 프로젝트는 **Transformer-XL (TXL)**을 기반으로 **Topic-Aware Memory 구조**를 실험적으로 확장한 코드베이스입니다.  
Hugging Face Datasets 및 Tokenizers를 이용해 데이터 파이프라인을 단순화하고,  
세션 단위 기억 유지(Recall) 및 망각(Forgetting) 곡선을 평가할 수 있도록 구성되어 있습니다.

---

## 📁 프로젝트 구조
```plaintext
transformer-xl/
├─ txl/
│  ├─ mem_transformer.py     # 원본 그대로(복사본). 수정 금지
│  ├─ mem_transformer_ta.py  # Topic-Aware 수정본
│  └─ __init__.py
│
├─ txl_hf/
│  ├─ train_hf.py            # HF Trainer 기반 학습/평가 스크립트
│  ├─ collator_stream.py     # 스트리밍 Collator (세션 단위 처리)
│  ├─ build_dataset.py       # HF Dataset 생성 및 전처리
│  ├─ build_tokenizer.py     # BPE 기반 Tokenizer 빌드
│  ├─ utils_logging.py       # 로그 및 체크포인트 유틸리티
│  ├─ mem_baseline.py        # HF용 TXL 래퍼 (baseline)
│  ├─ mem_ta.py              # Topic-Aware Memory 적용 HF 래퍼
│  └─ __init__.py
│
├─ data/
│  ├─ aihub/                 # AI Hub 일상대화 데이터셋
│  ├─ kowiki/                # 한국어 Wikipedia 문서 데이터
│  └─ kodial/                # Ko-Dial (KoNLP 공개 대화 corpus)
│
├─ artifacts/
│  └─ tokenizer/
│     └─ ko_bpe.json         # 학습용 BPE 토크나이저 저장 파일
│
└─ logs/                     # 학습 및 평가 로그 저장 경로
```

---

## ⚙️ 주요 구성 요소 설명

| 모듈 | 역할 |
|------|------|
| `txl/mem_transformer.py` | Transformer-XL 원본 (Google 공식 구현 기반) |
| `txl/mem_transformer_ta.py` | Topic-Aware Memory 구조 추가 버전 |
| `txl_hf/train_hf.py` | Hugging Face Trainer 기반 학습 루프 |
| `txl_hf/mem_baseline.py` | HF 포맷용 Transformer-XL 래퍼 |
| `txl_hf/mem_ta.py` | Topic-Aware Memory 적용 HF 래퍼 |
| `txl_hf/build_dataset.py` | 데이터셋 전처리 및 HF Dataset 객체 생성 |
| `txl_hf/build_tokenizer.py` | BPE 기반 한국어 토크나이저 빌드 |
| `txl_hf/utils_logging.py` | 학습 로그, 체크포인트 관리 |
| `txl_hf/collator_stream.py` | 세션 단위로 시퀀스를 스트리밍 처리 |

---

## 🚀 실행 예시

1️⃣ 데이터 전처리

```bash
python txl_hf/build_dataset.py --source aihub
```

2️⃣ 토크나이저 생성
```bash
python txl_hf/build_tokenizer.py \
  --data_dir data/aihub \
  --output_dir artifacts/tokenizer
```

3️⃣ 학습 (Baseline TXL)
```bash
python txl_hf/train_hf.py \
  --model mem_baseline \
  --dataset aihub \
  --batch_size 4 \
  --lr 1e-4 \
  --epochs 5
```

4️⃣ 학습 (Topic-Aware TXL)
```bash
python txl_hf/train_hf.py \
  --model mem_ta \
  --dataset aihub \
  --batch_size 4 \
  --lr 1e-4 \
  --epochs 5
```

5️⃣ 로그 및 시각화
```bash
tensorboard --logdir logs/
```

## 🧩 Topic-Aware Memory 구조 요약

| 항목 | Transformer-XL (TXL) | Topic-Aware TXL (TA-TXL) |
|------|-----------------------|---------------------------|
| Memory 전달 | 모든 시퀀스의 hidden state를 단순히 이어붙임 | 중요 topic memory만 선별 유지 |
| 기억 단위 | 시퀀스 (sequence) | 세션 + 토픽 단위 |
| 갱신 시점 | 시퀀스 단위 (매 batch) | 토픽 경계 감지 시 selective update |
| 목적 | 문맥 길이 확장 | 주제 지속성 유지 및 망각 제어 |

---

## 🧠 구조 개념도 (Mermaid)
```mermaid
flowchart LR

%% =========================
%% RMT — Sequential memory passing
%% =========================
subgraph RMT[RMT — 순차 메모리 전달]
  direction LR
  R1[Segment 1] --> RM1[Memory_1]
  R2[Segment 2 + Memory_1] --> RM2[Memory_2]
  R3[Segment 3 + Memory_2] --> RM3[Memory_3]
  Rellipsis[(...)] --> RMellipsis[(...)]
end

%% =========================
%% Memformer — Global memory update
%% =========================
subgraph MEM[Memformer — Global Memory Update]
  direction TB
  M1[Segment 1] --> GMU[Global Memory Update]
  M2[Segment 2] --> GMU
  M3[Segment 3] --> GMU
  GMU --> GM[Global Memory]
end

%% =========================
%% HMT --- Sensory --> Short --> Long (periodic accumulation)
%% =========================
subgraph HMT[HMT --- 감각→단기→장기(주기적 누적)]
  direction TB
  H1S[Segment 1] --> H1x[sensory_1] --> H1s[short_memory_1] --> HL[long_memory]
  H2S[Segment 2] --> H2x[sensory_2] --> H2s[short_memory_2] --> HL
  H3S[Segment 3] --> H3x[sensory_3] --> H3s[short_memory_3] --> HL
  H4S[Segment 4] --> H4x[sensory_4] --> H4s[short_memory_4] --> HL
end

%% =========================
%% Ours — Topic-aware summary & selective reuse
%% =========================
subgraph OURS[Ours — 토픽 요약 + 선택적 참조(routing)]
  direction TB
  O1[Sequence 1] --> O1m[seq_memory_1]
  O2[Sequence 2] --> O2m[seq_memory_2]
  O3[Sequence 3] --> O3m[seq_memory_3]

  %% Topic Block A summary → topic_memory_A
  O1m --> TBA[Topic Block A 요약]
  O2m --> TBA
  O3m --> TBA
  TBA --> TMA[topic_memory_A]

  NewTopic[(사용자가 새로운 화제 제시)]
  NewTopic --> O4[Sequence 4]
  O4 --> O4m[seq_memory_4]

  %% Selective reuse (only relevant topic memory)
  TMA -. 선택적 참조 .-> O4

  O5[Sequence 5] --> O5m[seq_memory_5]
end

%% =========================
%% Legend
%% =========================
classDef note fill:#f7f7f7,stroke:#aaa,color:#333,font-size:11px;
```
