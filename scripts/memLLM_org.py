# memLLM_QLoRA_QWEN_train.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, yaml, datetime
from typing import Dict, Any, List
from pathlib import Path
import re
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
    def __init__(self, hidden_size: int, embed_dim: int=768):
        super().__init__()
        self.boundary = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.Tanh(),
            nn.Linear(hidden_size//2, 1)
        )
        self.summary_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, embed_dim)
        )
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); 
            if m.bias is not None: nn.init.zeros_(m.bias)
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

        # 필수/기본값 보정 (YAML에 없으면 아래 값 사용)
        self.base_model = getattr(self, "base_model", "Qwen/Qwen2.5-1.5B-Instruct")
        self.embedding_model = getattr(self, "embedding_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        self.train_path = getattr(self, "train_path", "data/train_data.jsonl")
        self.output_dir = getattr(self, "output_dir", "out/mem-qwen2.5-1.5b-qlora")
        self.max_samples = getattr(self, "max_samples", 0)
        self.normalize = getattr(self, "normalize", "basic")

        # ---- 타입 보정(문자열로 들어온 값 대비) ----
        def _as_float(x, default):
            try: return float(x)
            except Exception: return float(default)

        def _as_int(x, default):
            try: return int(x)
            except Exception: return int(default)

        def _as_bool(x, default):
            if isinstance(x, bool): return x
            if isinstance(x, str):
                return x.strip().lower() in {"1","true","yes","y","t"}
            return bool(default)

        # 시퀀스/배치
        self.max_length = _as_int(getattr(self, "max_length", 1024), 1024)
        self.batch_size = _as_int(getattr(self, "batch_size", 1), 1)       # micro-batch per process
        self.grad_accum = _as_int(getattr(self, "grad_accum", 16), 16)
        self.epochs     = _as_int(getattr(self, "epochs", 1), 1)

        # Optim
        self.lr           = _as_float(getattr(self, "lr", 2e-4), 2e-4)
        self.warmup_ratio = _as_float(getattr(self, "warmup_ratio", 0.03), 0.03)

        # QLoRA (dtype 문자열은 그대로 사용)
        self.bnb_4bit_compute_dtype   = getattr(self, "bnb_4bit_compute_dtype", "float16")
        self.bnb_4bit_use_double_quant= _as_bool(getattr(self, "bnb_4bit_use_double_quant", True), True)
        self.bnb_4bit_quant_type      = getattr(self, "bnb_4bit_quant_type", "nf4")

        # LoRA
        self.lora_r       = _as_int(getattr(self, "lora_r", 16), 16)
        self.lora_alpha   = _as_int(getattr(self, "lora_alpha", 32), 32)
        self.lora_dropout = _as_float(getattr(self, "lora_dropout", 0.05), 0.05)

        # target_modules는 리스트/튜플/콤마문자열 모두 허용
        tm = getattr(self, "target_modules", ("q_proj", "v_proj"))
        if isinstance(tm, str):
            tm = [t.strip() for t in tm.split(",") if t.strip()]
        self.target_modules = tuple(tm)

        # 기타
        self.try_flash_attn = _as_bool(getattr(self, "try_flash_attn", True), True)

# ---- 데이터셋 ----
class AIHubDataset(Dataset):
    def __init__(self, tokenizer, jsonl_path, max_length=1024, max_samples=0, normalizer="basic"):
        self.tok = tokenizer
        self.max_length = max_length        
        self.samples = []
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
                dialogue = "\n".join(turns)

                boundaries = ex.get("boundaries", [])
                seg_summaries = ex.get("seg_summaries", [""] * len(boundaries))

                # 길이 맞추기
                if len(boundaries) != len(turns):
                    diff = len(turns) - len(boundaries)
                    if diff > 0:
                        boundaries += [0] * diff
                        seg_summaries += [""] * diff
                    else:
                        boundaries = boundaries[:len(turns)]
                        seg_summaries = seg_summaries[:len(turns)]

                self.samples.append({
                    "id": ex.get("id"),
                    "dialogue": dialogue,
                    "boundaries": boundaries,
                    "seg_summaries": seg_summaries,
                })
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

        b_label = torch.tensor(it["boundaries"][-1] if it["boundaries"] else 0, dtype=torch.float)

        # 마지막 요약을 boundary==1일 때만 target으로
        last_sum = ""
        for s in it["seg_summaries"]:
            if s and s.strip():
                last_sum = s
        sum_text = last_sum if b_label.item() == 1.0 else ""

        return {
            "input_ids": ids,
            "labels": labels,
            "b_label": b_label,
            "seg_summaries": it["seg_summaries"],
            "sum_text": sum_text,
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
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type
        )

        # Base model (Accelerate가 배치 관리)
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
        self.ds = AIHubDataset(self.tok, cfg.train_path, cfg.max_length, cfg.max_samples, normalizer=cfg.normalize)
        self.loader = DataLoader(self.ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=self._collate)

        # Optim/Sched
        self.optimizer = Adam8bit(
            list(self.model.parameters()) + list(self.aux.parameters()),
            lr=cfg.lr
        )
        total_steps = max(1, (len(self.loader) * cfg.epochs) // max(1, cfg.grad_accum))
        self.sched = get_linear_schedule_with_warmup(
            self.optimizer, int(total_steps * cfg.warmup_ratio), total_steps
        )

        # Accelerate prepare
        (self.model, self.aux, self.optimizer, self.loader, self.sched) = self.acc.prepare(
            self.model, self.aux, self.optimizer, self.loader, self.sched
        )

        # ---- hook ----
        def _capture_last_hidden(module, inputs, output):
            hs = output[0] if isinstance(output, (tuple, list)) else output
            n_last = getattr(self.cfg, "pool_last_n_tokens", 64)
            pooled = hs[:, -n_last:, :].mean(dim=1).detach()
            self._last_hidden = pooled

        def _unwrap(m):
            while hasattr(m, "module"): m = m.module
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
            for k in ("model","transformer","backbone"):
                tr = getattr(backbone, k, None)
                if tr is not None: return tr
            return backbone

        def _find_layers(tr):
            for k in ("layers","h","blocks"):
                ml = getattr(tr, k, None)
                if isinstance(ml, nn.ModuleList) and len(ml)>0: return ml
            for root in ("decoder","encoder"):
                r = getattr(tr, root, None)
                if r is not None:
                    for k in ("layers","h","blocks"):
                        ml = getattr(r, k, None)
                        if isinstance(ml, nn.ModuleList) and len(ml)>0: return ml
            best=None; n=0
            for _, mod in tr.named_modules():
                if isinstance(mod, nn.ModuleList) and len(mod)>n:
                    best=mod; n=len(mod)
            return best

        bb = _maybe_base(self.model)
        tr = _find_tr(bb)
        layers = _find_layers(tr)
        if isinstance(layers, nn.ModuleList) and len(layers)>0:
            layers[-1].register_forward_hook(_capture_last_hidden)
        else:
            for cand in ("norm","ln_f","final_layernorm","final_norm"):
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
    def _collate(self, batch: List[Dict[str,Any]]):
        maxlen = max(len(x['input_ids']) for x in batch)
        ids_list=[]; labels_list=[]; attn_list=[]; b_list=[]; sum_texts=[]
        for x in batch:
            pad = maxlen - len(x['input_ids'])
            ids  = torch.cat([x['input_ids'], torch.full((pad,), self.tok.pad_token_id)])
            labs = torch.cat([x['labels'], torch.full((pad,), -100)])
            ids  = torch.cat([x["input_ids"], torch.full((pad,), self.tok.pad_token_id, dtype=torch.long)])
            labs = torch.cat([x["labels"], torch.full((pad,), -100, dtype=torch.long)])
            attn = (ids != self.tok.pad_token_id)
            ids_list.append(ids); labels_list.append(labs); attn_list.append(attn)
            b_list.append(x['b_label']); sum_texts.append(x.get('sum_text', ""))
        return {
            'input_ids': torch.stack(ids_list),
            'labels': torch.stack(labels_list),
            'attention_mask': torch.stack(attn_list),
            'b_label': torch.stack(b_list),
            'sum_text': sum_texts,
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
    def _contrastive_infonce(self, e_pred: torch.Tensor, e_t: torch.Tensor,
                             has_sum_mask: torch.Tensor, tau: float = 0.07):
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
    def _compute_losses(self, logits, labels, pooled, b_label, e_t, has_sum_mask):
        # LM
        shift_logits = logits[:, :-1].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        lm = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        # Aux
        b_logit, e_pred = self.aux(pooled)  # bf16
        bce = F.binary_cross_entropy_with_logits(b_logit.float(), b_label.float())

        if has_sum_mask.sum().item() > 0:
            e_pred_f = e_pred.float(); e_t_f = e_t.float()
            e_pred_n = F.normalize(e_pred_f, dim=-1)
            e_t_n    = F.normalize(e_t_f, dim=-1)
            cos = 1 - (e_pred_n * e_t_n).sum(dim=-1)
            cos = (cos * has_sum_mask).sum() / (has_sum_mask.sum() + 1e-9)

            mse = F.mse_loss(e_pred_f, e_t_f, reduction='none').mean(dim=-1)
            mse = (mse * has_sum_mask).sum() / (has_sum_mask.sum() + 1e-9)
        else:
            cos = logits.new_zeros(())
            mse = logits.new_zeros(())

        info_nce = self._contrastive_infonce(e_pred.float(), e_t.float(), has_sum_mask, tau=0.07)

        loss = lm + bce + 0.25 * (cos + mse) + 0.5 * info_nce
        logs = {'lm': lm.item(), 'bce': bce.item(), 'cos': float(cos), 'mse': float(mse), 'info_nce': float(info_nce)}
        return loss, logs

    # 학습
    def train(self):
        self.model.train(); self.aux.train()
        step = 0
        for ep in range(self.cfg.epochs):
            for batch in self.loader:
                ids    = batch['input_ids'].to(self.acc.device)
                labels = batch['labels'].to(self.acc.device)
                attn   = batch['attention_mask'].to(self.acc.device)
                b_label= batch['b_label'].to(self.acc.device)
                sum_text = batch['sum_text']

                out = self.model(input_ids=ids, attention_mask=attn)
                logits = out.logits
                assert self._last_hidden is not None, "last_hidden 캡처 실패"
                n_last = getattr(self.cfg, "pool_last_n_tokens", 64)
                pooled = self._last_hidden
                if pooled.dim() == 3:
                    pooled = pooled[:, -n_last:, :].mean(dim=1)  # [B, T, H] -> [B, H]
                elif pooled.dim() != 2:
                    raise ValueError(f"Unexpected pooled dim: {pooled.shape}")
                self._last_hidden = None

                has_sum = torch.tensor(
                    [1.0 if (s and len(s.strip())>0) else 0.0 for s in sum_text],
                    device=self.acc.device, dtype=torch.float32
                )
                with torch.no_grad():
                    e_list = [self._encode_text_to_device(s) for s in sum_text]
                e_t = torch.stack(e_list)  # [B, emb_dim]

                loss, logs = self._compute_losses(logits, labels, pooled, b_label, e_t, has_sum)

                self.acc.backward(loss)
                if (step+1) % self.cfg.grad_accum == 0:
                    self.optimizer.step(); self.sched.step(); self.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                if self.acc.is_main_process and step % 20 == 0:
                    rec = {'event': 'train_log', 'step': step, **logs}
                    print(rec)
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                step += 1

        # 추론 대비 설정
        self.model.eval()
        self.model.config.use_cache = True

        # ---- Save model & write log ----
        if self.acc.is_main_process:
            # 저장 시작 로그
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "saving",
                                    "output_dir": self.cfg.output_dir},
                                   ensure_ascii=False) + "\n")

            os.makedirs(self.cfg.output_dir, exist_ok=True)
            self.model.save_pretrained(self.cfg.output_dir)
            torch.save({'aux': self.aux.state_dict()},
                       os.path.join(self.cfg.output_dir, 'aux.pt'))

            print(f"Saved to: {self.cfg.output_dir}")

            # 저장 완료 로그
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "saved",
                                    "output_dir": self.cfg.output_dir},
                                   ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="configs/train_config.yaml")
    args = ap.parse_args()
    cfg = TrainConfig(args.cfg)
    Trainer(cfg).train()