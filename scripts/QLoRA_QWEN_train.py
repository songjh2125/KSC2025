# QLoRA_QWEN_train.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, yaml, datetime
from pathlib import Path
from typing import List, Dict, Any

import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate import Accelerator

# ---- 설정 ----
class TrainConfig:
    def __init__(self, cfg_path="configs/train_config.yaml"):
        p = Path(cfg_path); assert p.exists(), f"Config file not found: {p}"
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items(): setattr(self, k, v)

        # 기본값 (베이스라인은 메모리/보조헤드 미사용)
        self.base_model   = getattr(self, "base_model", "Qwen/Qwen2.5-1.5B-Instruct")
        self.train_path   = getattr(self, "train_path", "data/train_data.jsonl")
        self.output_dir   = getattr(self, "output_dir", "out/qwen2.5-1.5b-qlora")  # 베이스라인 출력
        self.max_samples  = getattr(self, "max_samples", 0)

        # ---- 타입 보정(문자열 대비) ----
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
        self.batch_size = _as_int(getattr(self, "batch_size", 1), 1)
        self.grad_accum = _as_int(getattr(self, "grad_accum", 16), 16)
        self.epochs     = _as_int(getattr(self, "epochs", 1), 1)

        # Optim
        self.lr           = _as_float(getattr(self, "lr", 2e-4), 2e-4)
        self.warmup_ratio = _as_float(getattr(self, "warmup_ratio", 0.03), 0.03)

        # QLoRA
        self.bnb_4bit_compute_dtype    = getattr(self, "bnb_4bit_compute_dtype", "float16")  # 문자열 유지
        self.bnb_4bit_use_double_quant = _as_bool(getattr(self, "bnb_4bit_use_double_quant", True), True)
        self.bnb_4bit_quant_type       = getattr(self, "bnb_4bit_quant_type", "nf4")         # 문자열 유지

        # LoRA
        self.lora_r       = _as_int(getattr(self, "lora_r", 16), 16)
        self.lora_alpha   = _as_int(getattr(self, "lora_alpha", 32), 32)
        self.lora_dropout = _as_float(getattr(self, "lora_dropout", 0.05), 0.05)

        # target_modules: 리스트/튜플/콤마문자열 모두 허용
        tm = getattr(self, "target_modules", ("q_proj", "v_proj"))
        if isinstance(tm, str):
            tm = [t.strip() for t in tm.split(",") if t.strip()]
        self.target_modules = tuple(tm)

        # 기타
        self.try_flash_attn = _as_bool(getattr(self, "try_flash_attn", True), True)

# ---- 데이터셋 (텍스트만 사용) ----
class DialogueDataset(Dataset):
    def __init__(self, tokenizer, jsonl_path, max_length=1024, max_samples=0):
        self.tok = tokenizer
        self.max_length = max_length
        self.items: List[str] = []
        assert os.path.exists(jsonl_path), f"JSONL not found: {jsonl_path}"
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                ex = json.loads(line)
                turns = ex.get("text", []) or []
                if not turns: continue
                dialogue = "\n".join(turns)
                self.items.append(dialogue)
                if max_samples and len(self.items) >= max_samples: break

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        enc = self.tok(
            self.items[i],
            truncation=True, max_length=self.max_length,
            return_tensors="pt"
        )
        ids = enc["input_ids"].squeeze(0)
        labels = ids.clone()
        return {"input_ids": ids, "labels": labels}

def collate_fn(batch: List[Dict[str, Any]], pad_id: int):
    maxlen = max(len(x["input_ids"]) for x in batch)
    ids_list, labels_list, attn_list = [], [], []
    for x in batch:
        pad = maxlen - len(x["input_ids"])
        ids  = torch.cat([x["input_ids"], torch.full((pad,), pad_id)])
        labs = torch.cat([x["labels"], torch.full((pad,), -100)])
        attn = (ids != pad_id).long()
        ids_list.append(ids); labels_list.append(labs); attn_list.append(attn)
    return {
        "input_ids": torch.stack(ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(attn_list),
    }

# ---- 트레이너 ----
class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.acc = Accelerator()
        self.device = self.acc.device

        self.tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=False)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type
        )

        base = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            quantization_config=qconf,
            dtype=torch.float16,
            device_map=None,
            attn_implementation="flash_attention_2" if cfg.try_flash_attn else None,
        )
        try:
            if cfg.try_flash_attn:
                base.config.attn_implementation = "flash_attention_2"
        except Exception:
            pass

        base = prepare_model_for_kbit_training(base)
        base.gradient_checkpointing_enable()
        base.config.use_cache = False

        lconf = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.target_modules), task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(base, lconf)
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        self.ds = DialogueDataset(self.tok, cfg.train_path, cfg.max_length, cfg.max_samples)
        self.loader = DataLoader(
            self.ds, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=lambda b: collate_fn(b, self.tok.pad_token_id)
        )

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        total_steps = max(1, (len(self.loader) * cfg.epochs) // max(1, cfg.grad_accum))
        self.sched = get_linear_schedule_with_warmup(
            self.optimizer, int(total_steps * cfg.warmup_ratio), total_steps
        )

        (self.model, self.optimizer, self.loader, self.sched) = self.acc.prepare(
            self.model, self.optimizer, self.loader, self.sched
        )

        subdir = "base"
        os.makedirs(os.path.join("log", subdir), exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join("log", subdir, f"train_baseline_{ts}.txt")
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event":"train_start","cfg":vars(self.cfg)}, ensure_ascii=False) + "\n")

    def train(self):
        self.model.train()
        step = 0
        for ep in range(self.cfg.epochs):
            for batch in self.loader:
                ids    = batch['input_ids'].to(self.acc.device)
                labels = batch['labels'].to(self.acc.device)
                attn   = batch['attention_mask'].to(self.acc.device)

                out = self.model(input_ids=ids, attention_mask=attn)
                shift_logits = out.logits[:, :-1].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )

                self.acc.backward(loss)
                if (step+1) % self.cfg.grad_accum == 0:
                    self.optimizer.step(); self.sched.step(); self.optimizer.zero_grad()
                if self.acc.is_main_process and step % 20 == 0:
                    rec = {'event': 'train_log', 'step': step, 'lm': loss.item()}
                    print(rec)
                    with open(self.log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                step += 1

        self.model.eval(); self.model.config.use_cache = True

        if self.acc.is_main_process:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event":"saving","output_dir":self.cfg.output_dir}, ensure_ascii=False) + "\n")
            os.makedirs(self.cfg.output_dir, exist_ok=True)
            self.model.save_pretrained(self.cfg.output_dir)
            print(f"Saved baseline QLoRA to: {self.cfg.output_dir}")
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event":"saved","output_dir":self.cfg.output_dir}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="configs/train_config.yaml")
    args = ap.parse_args()
    cfg = TrainConfig(args.cfg)
    Trainer(cfg).train()