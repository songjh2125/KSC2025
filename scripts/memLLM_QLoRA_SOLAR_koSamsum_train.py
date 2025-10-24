#!/usr/bin/env python3
# trainer
from __future__ import annotations
import os, json, math, random
from typing import Dict, Any, List

import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator

from transformers import AutoModel  # ko-sroberta
from mem_modules import MemConfig, MemoryManager, AuxHeads
from summarizer_local import LocalSummarizer
from data_utils import basic_normalize, spell_normalize

class TrainConfig:
    base_model = "Upstage/SOLAR-10.7B-Instruct-v1.0"
    embedding_model = "jhgan/ko-sroberta-multitask"
    dataset_name = "heegyu/ko-samsum"
    output_dir = "out/solar-mem-qlora"
    lr = 2e-4; batch_size = 1; grad_accum = 16; epochs = 1
    max_length = 1024; max_samples = 0; warmup_ratio = 0.03; seed = 42
    labels_cache = "cache/labels_ko_samsum.jsonl"
    normalize = "basic"  # none|basic|spell

    lora_r=8; lora_alpha=16; lora_dropout=0.05
    target_modules = ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj")
    bnb_4bit_compute_dtype="float16"; bnb_4bit_use_double_quant=True; bnb_4bit_quant_type="nf4"

class KoSamsumDataset(Dataset):
    def __init__(self, tokenizer, max_length, max_samples, labels_cache, normalizer='basic'):
        self.tok = tokenizer
        ds = load_dataset("heegyu/ko-samsum", split="train")
        if max_samples and max_samples>0:
            ds = ds.select(range(min(max_samples, len(ds))))
        # load cache
        cache = []
        if os.path.exists(labels_cache):
            with open(labels_cache, 'r', encoding='utf-8') as f:
                for line in f:
                    cache.append(json.loads(line))
        cache_by_id = {c['id']: c for c in cache if 'id' in c}
        self.samples = []
        for idx, ex in enumerate(ds):
            dialog = ex.get('dialogue') or ex.get('conversation') or ''
            summary = ex.get('summary') or ''
            if len(summary) < 100:
                continue
            if normalizer=='basic':
                dialog = '\n'.join(basic_normalize(u) for u in dialog.split('\n'))
                summary = basic_normalize(summary)
            elif normalizer=='spell':
                dialog = '\n'.join(spell_normalize(u) for u in dialog.split('\n'))
                summary = spell_normalize(summary)
            meta = cache_by_id.get(idx, None)
            boundaries = meta['boundaries'] if meta else [0]*len([u for u in dialog.split('\n') if u.strip()])
            seg_summ = meta['seg_summaries'] if meta else [""]*len(boundaries)
            self.samples.append({
                'id': idx,
                'dialogue': dialog,
                'summary': summary,
                'boundaries': boundaries,
                'seg_summaries': seg_summ,
            })
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        it = self.samples[i]
        enc = self.tok(it['dialogue'], truncation=True, max_length=self.max_length, return_tensors='pt')
        ids = enc['input_ids'].squeeze(0)
        labels = ids.clone()
        b_label = torch.tensor(it['boundaries'][-1] if it['boundaries'] else 0, dtype=torch.float)
        return {
            'input_ids': ids,
            'labels': labels,
            'b_label': b_label,
            'seg_summaries': it['seg_summaries'],
        }

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.acc = Accelerator(); self.device = self.acc.device
        self.tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=False)
        if self.tok.pad_token is None: self.tok.pad_token = self.tok.eos_token
        qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=getattr(torch, cfg.bnb_4bit_compute_dtype),
                                   bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant, bnb_4bit_quant_type=cfg.bnb_4bit_quant_type)
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model, quantization_config=qconf, torch_dtype=torch.float16, device_map={'': self.device})
        hidden = base.config.hidden_size
        lconf = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, target_modules=list(cfg.target_modules), task_type="CAUSAL_LM")
        self.model = get_peft_model(base, lconf)
        self.aux = AuxHeads(hidden_size=hidden, embed_dim=768).to(self.device)
        self.embed_tok = AutoTokenizer.from_pretrained(cfg.embedding_model)
        self.embed_model = AutoModel.from_pretrained(cfg.embedding_model).to(self.device)
        self.mem = MemoryManager(MemConfig(), self.embed_model, self.embed_tok, hidden)
        self.summarizer = LocalSummarizer(model_name=cfg.base_model)
        self.ds = KoSamsumDataset(self.tok, cfg.max_length, cfg.max_samples, cfg.labels_cache, normalizer=cfg.normalize)
        self.loader = DataLoader(self.ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=self._collate)
        self.optimizer = torch.optim.AdamW(list(self.model.parameters())+list(self.aux.parameters())+list(self.mem.parameters()), lr=cfg.lr)
        total_steps = (len(self.loader)*cfg.epochs)//cfg.grad_accum
        self.sched = get_linear_schedule_with_warmup(self.optimizer, int(total_steps*cfg.warmup_ratio), total_steps)
        self.cfg = cfg
        (self.model,self.aux,self.optimizer,self.loader,self.sched) = self.acc.prepare(self.model,self.aux,self.optimizer,self.loader,self.sched)

    def _collate(self, batch: List[Dict[str,Any]]):
        maxlen = max(len(x['input_ids']) for x in batch)
        ids_list=[]; labels_list=[]; b_list=[]; seg_list=[]
        for x in batch:
            pad = maxlen - len(x['input_ids'])
            ids = torch.cat([torch.full((pad,), self.tok.pad_token_id), x['input_ids']])
            labs = torch.cat([torch.full((pad,), -100), x['labels']])
            ids_list.append(ids); labels_list.append(labs)
            b_list.append(x['b_label']); seg_list.append(x['seg_summaries'])
        return {
            'input_ids': torch.stack(ids_list),
            'labels': torch.stack(labels_list),
            'b_label': torch.stack(b_list),
            'seg_summaries': seg_list,
        }

    def _compute_losses(self, logits, labels, pooled, b_label, e_t):
        shift_logits = logits[:, :-1].contiguous(); shift_labels = labels[:,1:].contiguous()
        lm = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)
        b_logit, e_pred = self.aux(pooled)
        bce = F.binary_cross_entropy_with_logits(b_logit, b_label)
        e_t = F.normalize(e_t, dim=-1); e_pred_n = F.normalize(e_pred, dim=-1)
        cos = 1 - (e_pred_n*e_t).sum(dim=-1).mean()
        mse = F.mse_loss(e_pred, e_t)
        nce = F.cross_entropy(e_pred_n @ e_t.t(), torch.arange(e_t.size(0), device=e_t.device))
        loss = lm + bce + 0.5*(cos + mse) + 0.2*nce
        return loss, {'lm':lm.item(),'bce':bce.item(),'cos':cos.item(),'mse':mse.item(),'nce':nce.item()}

    def train(self):
        self.model.train(); self.aux.train(); self.mem.train()
        step=0
        for ep in range(self.cfg.epochs):
            for batch in self.loader:
                ids = batch['input_ids'].to(self.acc.device)
                labels = batch['labels'].to(self.acc.device)
                b_label = batch['b_label'].to(self.acc.device)
                seg_summaries = batch['seg_summaries']  # list of list[str]
                B, L = ids.size(); H = self.model.config.hidden_size

                # --- 동적 Proj_mem 주입: 각 샘플의 마지막 비어있지 않은 세그 요약을 사용해 m_t 생성 ---
                e_list=[]
                with torch.no_grad():
                    for segs in seg_summaries:
                        last = ""
                        for s in segs:
                            if s: last = s
                        if not last: last = "최근까지의 대화를 간결히 요약하세요."  # fallback
                        e = self.mem.encode(last)
                        e_list.append(e)
                e_t = torch.stack(e_list)  # [B,768]
                # m_t 갱신 후 soft prefix 생성
                self.mem.m_t = self.mem.mlp(torch.cat([torch.zeros_like(e_t), e_t], dim=-1)).mean(dim=0)  # simple init
                prefix = []
                for _ in range(B):
                    p = self.mem.proj_prefix(H)  # [r,H]
                    prefix.append(p)
                prefix = torch.stack(prefix, dim=0)  # [B,r,H]

                inp_emb = self.model.get_input_embeddings()(ids)
                inp_emb = torch.cat([prefix, inp_emb], dim=1)
                prefix_labels = torch.full((B, prefix.size(1)), -100, device=labels.device)
                labels2 = torch.cat([prefix_labels, labels], dim=1)

                out = self.model(inputs_embeds=inp_emb)
                logits = out.logits
                pooled = inp_emb[:, -64:, :].mean(dim=1)
                loss, logs = self._compute_losses(logits, labels2, pooled, b_label, e_t)

                self.acc.backward(loss)
                if (step+1)%self.cfg.grad_accum==0:
                    self.optimizer.step(); self.sched.step(); self.optimizer.zero_grad()
                if self.acc.is_main_process and step%20==0:
                    print({'step':step, **logs})
                step+=1
        if self.acc.is_main_process:
            os.makedirs(self.cfg.output_dir, exist_ok=True)
            self.model.save_pretrained(self.cfg.output_dir)
            torch.save({'aux': self.aux.state_dict(), 'mem': self.mem.state_dict()}, os.path.join(self.cfg.output_dir,'aux_mem.pt'))

if __name__ == '__main__':
    cfg = TrainConfig()
    Trainer(cfg).train()