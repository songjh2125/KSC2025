# memLLM_QLoRA_QWEN_train.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, yaml, datetime, math, random
from typing import Dict, Any, List
from pathlib import Path

from bitsandbytes.optim import Adam8bit

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    get_linear_schedule_with_warmup, AutoModel
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator

from src.mem_modules import MemConfig, MemoryManager
from src.data_utils import basic_normalize, spell_normalize


# ---- 보조 헤드(경계/요약 임베딩) ----
class AuxHeads(nn.Module):
    def __init__(self, hidden_size: int, embed_dim: int = 768):
        super().__init__()
        self.boundary = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
        self.summary_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, embed_dim),
        )
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, pooled: torch.Tensor):
        b_logit = self.boundary(pooled).squeeze(-1)
        e_pred = self.summary_head(pooled)
        return b_logit, e_pred


# ---- 설정 ----
class TrainConfig:
    def __init__(self, cfg_path="configs/train_config.yaml"):
        p = Path(cfg_path)
        assert p.exists(), f"Config file not found: {p}"
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for k, v in (cfg or {}).items():
            setattr(self, k, v)

        # 타입 보정
        def _as_float(x, default):
            try:
                return float(x)
            except Exception:
                return float(default)

        def _as_int(x, default):
            try:
                return int(x)
            except Exception:
                return int(default)

        def _as_bool(x, default):
            if isinstance(x, bool):
                return x
            if isinstance(x, str):
                return x.strip().lower() in {"1", "true", "yes", "y", "t"}
            return bool(default)
        
        # 기본값
        self.base_model = getattr(self, "base_model", "Qwen/Qwen2.5-1.5B-Instruct")
        self.embedding_model = getattr(self, "embedding_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.train_path = getattr(self, "train_path", "data/train_data.jsonl")
        self.output_dir = getattr(self, "output_dir", "out/mem-qwen2.5-1.5b-qlora")
        self.max_samples = getattr(self, "max_samples", 0)
        self.normalize = getattr(self, "normalize", "basic")
        self.max_train_steps = _as_int(getattr(self, "max_train_steps", 0), 0)  # 0이면 제한 없음
        self.use_sum_if_boundary = _as_bool(getattr(self, "use_sum_if_boundary", True), True)

        # 샘플링 관련
        self.sample_mode = getattr(self, "sample_mode", "pos_plus_k_neg")  # "pos_plus_k_neg" | "all"
        self.negatives_per_pos = _as_int(getattr(self, "negatives_per_pos", 3), 3)  # 가짜 경계 3개
        self.include_last = _as_bool(getattr(self, "include_last", True), True)

        # 풀링: 0이면 전체 평균, >0이면 마지막 n토큰 평균
        self.pool_last_n_tokens = _as_int(getattr(self, "pool_last_n_tokens", 0), 0)

        # 시퀀스/배치
        self.max_length = _as_int(getattr(self, "max_length", 1024), 1024)
        self.batch_size = _as_int(getattr(self, "batch_size", 1), 1)      # micro-batch per process
        self.grad_accum = _as_int(getattr(self, "grad_accum", 16), 16)
        self.epochs = _as_int(getattr(self, "epochs", 1), 1)

        # Optim
        self.lr = _as_float(getattr(self, "lr", 2e-4), 2e-4)
        self.warmup_ratio = _as_float(getattr(self, "warmup_ratio", 0.03), 0.03)

        # QLoRA
        self.bnb_4bit_compute_dtype = getattr(self, "bnb_4bit_compute_dtype", "float16")
        self.bnb_4bit_use_double_quant = _as_bool(getattr(self, "bnb_4bit_use_double_quant", True), True)
        self.bnb_4bit_quant_type = getattr(self, "bnb_4bit_quant_type", "nf4")

        # LoRA
        self.lora_r = _as_int(getattr(self, "lora_r", 16), 16)
        self.lora_alpha = _as_int(getattr(self, "lora_alpha", 32), 32)
        self.lora_dropout = _as_float(getattr(self, "lora_dropout", 0.05), 0.05)

        # target_modules
        tm = getattr(self, "target_modules", ("q_proj", "v_proj"))
        if isinstance(tm, str):
            tm = [t.strip() for t in tm.split(",") if t.strip()]
        self.target_modules = tuple(tm)

        # 기타
        self.try_flash_attn = _as_bool(getattr(self, "try_flash_attn", True), True)


# ---- 데이터셋 ----
class AIHubDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        jsonl_path,
        max_length=1024,
        max_samples=0,
        normalizer="basic",
        use_sum_if_boundary=True,
        sample_mode="pos_plus_k_neg",
        negatives_per_pos=3,
        include_last=True,
    ):
        self.tok = tokenizer
        self.max_length = max_length
        self.samples = []
        self.use_sum_if_boundary = use_sum_if_boundary
        self.sample_mode = sample_mode
        self.negatives_per_pos = int(negatives_per_pos)
        self.include_last = bool(include_last)
        assert os.path.exists(jsonl_path), f"JSONL not found: {jsonl_path}"

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)

                turns = ex.get("text", []) or []
                if normalizer == "basic":
                    turns = [basic_normalize(t) for t in turns]
                elif normalizer == "spell":
                    turns = [spell_normalize(t) for t in turns]

                boundaries = ex.get("boundaries", [])
                seg_summaries = ex.get("seg_summaries", [""] * len(boundaries))

                # 길이 맞추기
                L = len(turns)
                if len(boundaries) < L:
                    boundaries += [0] * (L - len(boundaries))
                if len(seg_summaries) < L:
                    seg_summaries += [""] * (L - len(seg_summaries))
                boundaries = boundaries[:L]
                seg_summaries = seg_summaries[:L]

                did = ex.get("id")

                # 인덱스 분리
                pos_idx = [i for i in range(L) if int(boundaries[i]) == 1]
                neg_idx = [i for i in range(L) if int(boundaries[i]) == 0]

                if self.sample_mode == "pos_plus_k_neg":
                    # 후보 선정: 진짜 경계(b==1) + 가짜 경계 일부(b==0). 후보에만 BCE 적용
                    selected_tuples: List[tuple[int, int]] = [(i, 1) for i in pos_idx]  # (index, bce_cand)

                    # 가짜 경계: 경계 인접(hard negatives) 우선 + 부족분 랜덤
                    hard_negs: List[int] = []
                    for p in pos_idx:
                        if p - 1 >= 0 and int(boundaries[p - 1]) == 0:
                            hard_negs.append(p - 1)
                        if p + 1 < L and int(boundaries[p + 1]) == 0:
                            hard_negs.append(p + 1)
                    # unique 유지
                    hard_negs = list(dict.fromkeys(hard_negs))

                    K_total = self.negatives_per_pos * max(1, len(pos_idx))
                    chosen_negs = hard_negs[:K_total]
                    remain = K_total - len(chosen_negs)
                    if remain > 0:
                        pool = [i for i in neg_idx if i not in set(hard_negs)]
                        if len(pool) > 0:
                            chosen_negs += random.sample(pool, min(remain, len(pool)))
                    selected_tuples += [(i, 1) for i in chosen_negs]

                    # 마지막 인덱스는 표현/롱컨텍스트용(BCE 비계산 후보)
                    if self.include_last and L > 0:
                        i_last = L - 1
                        already = [x for x, _ in selected_tuples]
                        if i_last not in already:
                            selected_tuples.append((i_last, 0))

                    # 정렬/중복 제거
                    seen = set()
                    selected: List[tuple[int, int]] = []
                    for i, c in sorted(selected_tuples, key=lambda x: x[0]):
                        if i in seen:
                            continue
                        seen.add(i)
                        selected.append((i, c))
                else:
                    # "all": 모든 i 사용 (스텝 폭증 주의) - 전부 BCE 후보
                    selected = [(i, 1) for i in range(L)]

                # 샘플 생성
                for sel in selected:
                    i, is_cand = sel
                    sub_dialogue = "\n".join(turns[: i + 1])
                    b = int(boundaries[i])
                    s = (seg_summaries[i] or "").strip()
                    if self.use_sum_if_boundary:
                        s = s if (b == 1 and len(s) > 0) else ""
                    self.samples.append(
                        {
                            "id": f"{did}_{i}",
                            "dialogue": sub_dialogue,
                            "b_label": b,
                            "sum_text": s,
                            "bce_cand": int(is_cand),
                        }
                    )
                    if max_samples and len(self.samples) >= max_samples:
                        break
                if max_samples and len(self.samples) >= max_samples:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        it = self.samples[i]
        enc = self.tok(
            it["dialogue"],
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        ids = enc["input_ids"].squeeze(0)
        labels = ids.clone()
        return {
            "input_ids": ids,
            "labels": labels,
            "b_label": torch.tensor(it["b_label"], dtype=torch.float),
            "sum_text": it.get("sum_text", ""),
            "bce_cand": torch.tensor(it.get("bce_cand", 1), dtype=torch.float),
            "id": it["id"],
        }


# ---- 트레이너 ----
class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.acc = Accelerator(mixed_precision="bf16")
        self.device = self.acc.device
        self._last_hidden = None

        # Tokenizer
        self.tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=False)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        # 4bit
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        )

        # Base model
        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=qconf,
            dtype=torch.bfloat16,
            device_map=None,
            attn_implementation="flash_attention_2" if cfg.try_flash_attn else None,
        )
        # Flash-Attn 폴백
        try:
            if cfg.try_flash_attn:
                base.config.attn_implementation = "flash_attention_2"
        except Exception:
            pass

        # 4bit 학습 준비 + Checkpointing
        base = prepare_model_for_kbit_training(base)
        base.gradient_checkpointing_enable()
        base.config.use_cache = False

        hidden = base.config.hidden_size

        # LoRA
        lconf = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.target_modules),
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, lconf)
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        # 임베딩 모델(고정)
        self.embed_tok = AutoTokenizer.from_pretrained(cfg.embedding_model)
        self.embed_model = AutoModel.from_pretrained(cfg.embedding_model).to("cpu")
        for p in self.embed_model.parameters():
            p.requires_grad = False
        self.emb_dim = getattr(self.embed_model.config, "hidden_size", 768)

        # AuxHeads
        self.aux = AuxHeads(hidden_size=hidden, embed_dim=self.emb_dim).to(self.device)
        self.aux.to(dtype=torch.bfloat16)

        # MemoryManager (encode 유틸)
        self.mem = MemoryManager(MemConfig(), self.embed_model, self.embed_tok)

        # Dataset/Loader
        self.ds = AIHubDataset(
            self.tok,
            cfg.train_path,
            cfg.max_length,
            cfg.max_samples,
            normalizer=cfg.normalize,
            use_sum_if_boundary=cfg.use_sum_if_boundary,
            sample_mode=cfg.sample_mode,
            negatives_per_pos=cfg.negatives_per_pos,
            include_last=cfg.include_last,
        )
        self.loader = DataLoader(
            self.ds, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=self._collate
        )

        # step estimate (정보 출력)
        n_total = len(self.ds)
        batches_per_epoch = math.ceil(n_total / self.cfg.batch_size)
        est_steps = self.cfg.epochs * batches_per_epoch
        print(f"[EST] samples={n_total}, batch={self.cfg.batch_size}, epochs={self.cfg.epochs} "
            f"-> steps≈{math.ceil(n_total / self.cfg.batch_size) * self.cfg.epochs} "
            f"(cap={self.cfg.max_train_steps or '∞'})")

        # 통계/가중치
        n_pos = sum(int(s["b_label"]) for s in self.ds.samples)
        n_neg = max(0, n_total - n_pos)
        beta = (n_neg / max(1, n_pos)) if n_pos > 0 else 1.0
        self.pos_weight = torch.tensor(beta, dtype=torch.float32, device=self.acc.device)

        # Optim/Sched
        self.optimizer = Adam8bit(list(self.model.parameters()) + list(self.aux.parameters()), lr=cfg.lr)
        computed_total = max(1, (len(self.loader) * self.cfg.epochs) // max(1, self.cfg.grad_accum))
        if self.cfg.max_train_steps and self.cfg.max_train_steps > 0:
            total_steps = self.cfg.max_train_steps
        else:
            total_steps = computed_total
            
        self.sched = get_linear_schedule_with_warmup(
            self.optimizer, int(total_steps * self.cfg.warmup_ratio), total_steps
        )

        # Accelerate prepare
        (self.model, self.aux, self.optimizer, self.loader, self.sched) = self.acc.prepare(
            self.model, self.aux, self.optimizer, self.loader, self.sched
        )

        # ---- hook: last hidden capture ----
        def _capture_last_hidden(module, inputs, output):
            hs = output[0] if isinstance(output, (tuple, list)) else output  # [B,T,H]
            T = hs.size(1)
            n = getattr(self.cfg, "pool_last_n_tokens", 64)
            if n and n > 0:
                n = min(n, T)
                # pooled = hs[:, -n:, :].mean(dim=1)
                pooled = hs.mean(dim=1)
            else:
                pooled = hs.mean(dim=1)  # 전체 평균 → 세션 전체 맥락 반영
            self._last_hidden = pooled

        def _unwrap(m):
            while hasattr(m, "module"):
                m = m.module
            return m

        def _maybe_base(m):
            m = _unwrap(m)
            try:
                from peft import PeftModel
                if isinstance(m, PeftModel):
                    m = _unwrap(m.get_base_model())
            except Exception:
                pass
            return getattr(m, "base_model", m)

        def _find_tr(backbone):
            for k in ("model", "transformer", "backbone"):
                tr = getattr(backbone, k, None)
                if tr is not None:
                    return tr
            return backbone

        def _find_layers(tr):
            for k in ("layers", "h", "blocks"):
                ml = getattr(tr, k, None)
                if isinstance(ml, nn.ModuleList) and len(ml) > 0:
                    return ml
            for root in ("decoder", "encoder"):
                r = getattr(tr, root, None)
                if r is not None:
                    for k in ("layers", "h", "blocks"):
                        ml = getattr(r, k, None)
                        if isinstance(ml, nn.ModuleList) and len(ml) > 0:
                            return ml
            best = None
            n = 0
            for _, mod in tr.named_modules():
                if isinstance(mod, nn.ModuleList) and len(mod) > n:
                    best = mod
                    n = len(mod)
            return best

        bb = _maybe_base(self.model)
        tr = _find_tr(bb)
        layers = _find_layers(tr)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            layers[-1].register_forward_hook(_capture_last_hidden)
        else:
            for cand in ("norm", "ln_f", "final_layernorm", "final_norm"):
                mod = getattr(tr, cand, None)
                if isinstance(mod, nn.Module):
                    mod.register_forward_hook(_capture_last_hidden)
                    break

        # ---- logging ----
        subdir = "mem"
        os.makedirs(os.path.join("log", subdir), exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join("log", subdir, f"train_mem_{ts}.txt")
        with open(self.log_path, "a", encoding="utf-8") as f:
            meta = {
                "event": "train_start",
                "timestamp": ts,
                "cfg": {k: getattr(self.cfg, k) for k in vars(self.cfg)},
            }
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    # Collate: 오른쪽 패딩 + attention_mask
    def _collate(self, batch: List[Dict[str, Any]]):
        maxlen = max(len(x["input_ids"]) for x in batch)
        ids_list = []
        labels_list = []
        attn_list = []
        b_list = []
        sum_texts = []
        bce_cands = []
        for x in batch:
            pad = maxlen - len(x["input_ids"])
            ids = torch.cat([x["input_ids"], torch.full((pad,), self.tok.pad_token_id, dtype=torch.long)])
            labs = torch.cat([x["labels"], torch.full((pad,), -100, dtype=torch.long)])
            attn = ids != self.tok.pad_token_id
            ids_list.append(ids)
            labels_list.append(labs)
            attn_list.append(attn)
            b_list.append(x["b_label"])
            sum_texts.append(x.get("sum_text", ""))
            bce_cands.append(x["bce_cand"])
        return {
            "input_ids": torch.stack(ids_list),
            "labels": torch.stack(labels_list),
            "attention_mask": torch.stack(attn_list),
            "b_label": torch.stack(b_list),
            "sum_text": sum_texts,
            "bce_cand": torch.stack(bce_cands),
        }

    # 안전 임베딩 인코더
    @torch.no_grad()
    def _encode_text_to_device(self, text: str) -> torch.Tensor:
        if not text or not text.strip():
            return torch.zeros(self.emb_dim, device=self.acc.device, dtype=torch.float32)
        emb = self.mem.encode(text)
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb)
        emb = emb.to(self.acc.device, dtype=torch.float32)
        emb = F.normalize(emb.float(), dim=-1).to(torch.bfloat16)
        return emb

    # 인배치 InfoNCE
    def _contrastive_infonce(self, e_pred: torch.Tensor, e_t: torch.Tensor, has_sum_mask: torch.Tensor, tau: float = 0.07):
        idx = (has_sum_mask > 0.5).nonzero(as_tuple=False).squeeze(-1)
        if idx.numel() <= 1:
            return e_pred.new_zeros(())
        p = F.normalize(e_pred[idx], dim=-1)  # [N, D]
        t = F.normalize(e_t[idx], dim=-1)     # [N, D]
        logits_pt = (p @ t.t()) / tau
        logits_tp = (t @ p.t()) / tau
        labels = torch.arange(logits_pt.size(0), device=logits_pt.device)
        loss_pt = F.cross_entropy(logits_pt, labels)
        loss_tp = F.cross_entropy(logits_tp, labels)
        return 0.5 * (loss_pt + loss_tp)

    # 손실
    def _compute_losses(self, logits, labels, pooled, b_label, e_t, has_sum_mask, bce_cand_mask):
        # LM
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        lm = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Aux
        b_logit, e_pred = self.aux(pooled)  # bf16

        # masked BCE: 경계 후보에서만
        pw = self.pos_weight.to(b_logit.device)
        bce_per = F.binary_cross_entropy_with_logits(
            b_logit.float(), b_label.float(), pos_weight=pw, reduction="none"
        )
        m = (bce_cand_mask > 0.5).float()
        denom = m.sum().clamp_min(1.0)
        bce = (bce_per * m).sum() / denom

        # 요약(경계=1에서만 has_sum_mask=1)
        if has_sum_mask.sum().item() > 0:
            e_pred_f = e_pred.float()
            e_t_f = e_t.float()
            e_pred_n = F.normalize(e_pred_f, dim=-1)
            e_t_n = F.normalize(e_t_f, dim=-1)

            cos = 1 - (e_pred_n * e_t_n).sum(dim=-1)
            cos = (cos * has_sum_mask).sum() / (has_sum_mask.sum() + 1e-9)

            mse = F.mse_loss(e_pred_f, e_t_f, reduction="none").mean(dim=-1)
            mse = (mse * has_sum_mask).sum() / (has_sum_mask.sum() + 1e-9)
        else:
            cos = logits.new_zeros(())
            mse = logits.new_zeros(())

        info_nce = self._contrastive_infonce(e_pred.float(), e_t.float(), has_sum_mask, tau=0.07)

        # loss = lm + bce + 0.25 * (cos + mse) + 0.5 * info_nce
        loss = lm + bce + 0.5*(cos + mse) + 1.0*info_nce
        logs = {
            "lm": lm.item(),
            "bce": bce.item(),
            "cos": float(cos),
            "mse": float(mse),
            "info_nce": float(info_nce),
        }
        return loss, logs

    # 학습
    def train(self):
        self.model.train()
        self.aux.train()
        step = 0
        max_steps = int(getattr(self.cfg, "max_train_steps", 0) or 0)
        for ep in range(self.cfg.epochs):
            for batch in self.loader:
                ids = batch["input_ids"].to(self.acc.device)
                labels = batch["labels"].to(self.acc.device)
                attn = batch["attention_mask"].to(self.acc.device)
                b_label = batch["b_label"].to(self.acc.device)
                sum_text = batch["sum_text"]

                out = self.model(input_ids=ids, attention_mask=attn)
                logits = out.logits

                assert self._last_hidden is not None, "last_hidden 캡처 실패"
                pooled = self._last_hidden
                self._last_hidden = None

                has_sum = torch.tensor(
                    [1.0 if (s and len(s.strip()) > 0) else 0.0 for s in sum_text],
                    device=self.acc.device,
                    dtype=torch.float32,
                )
                bce_cand_mask = batch["bce_cand"].to(self.acc.device)

                with torch.no_grad():
                    e_list = [self._encode_text_to_device(s) for s in sum_text]
                e_t = torch.stack(e_list)  # [B, emb_dim]

                loss, logs = self._compute_losses(logits, labels, pooled, b_label, e_t, has_sum, bce_cand_mask)

                self.acc.backward(loss)
                if (step + 1) % self.cfg.grad_accum == 0:
                    self.optimizer.step(); self.sched.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()

                if self.acc.is_main_process and step % 20 == 0:
                    rec = {"event": "train_log", "step": step, **logs}
                    print(rec)
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                step += 1

                if max_steps and step >= max_steps:
                    break
            if max_steps and step >= max_steps:
                break

        # 추론 대비 설정
        self.model.eval()
        self.model.config.use_cache = True

        # ---- Save model & write log ----
        if self.acc.is_main_process:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "saving", "output_dir": self.cfg.output_dir}, ensure_ascii=False) + "\n")

            os.makedirs(self.cfg.output_dir, exist_ok=True)
            self.model.save_pretrained(self.cfg.output_dir)
            torch.save(
                {"aux": self.aux.state_dict(), "embed_model_name": self.cfg.embedding_model, "emb_dim": int(self.emb_dim)},
                os.path.join(self.cfg.output_dir, "aux.pt"),
            )

            with open(os.path.join(self.cfg.output_dir, "embedding_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"embedding_model": self.cfg.embedding_model, "emb_dim": int(self.emb_dim)}, f, ensure_ascii=False, indent=2)

            print(f"Saved to: {self.cfg.output_dir}")

            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "saved", "output_dir": self.cfg.output_dir}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="configs/train_config.yaml")
    args = ap.parse_args()
    cfg = TrainConfig(args.cfg)
    Trainer(cfg).train()
