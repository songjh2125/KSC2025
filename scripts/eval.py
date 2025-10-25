#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, argparse, math, random, sys, datetime
from typing import List, Dict, Any, Tuple

import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import BitsAndBytesConfig  # [추가] 4bit 지원
from peft import PeftModel
from statistics import mean
import importlib

# ====== AuxHeads (학습과 동일) ======
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
    def forward(self, pooled: torch.Tensor):
        b_logit = self.boundary(pooled).squeeze(-1)
        e_pred = self.summary_head(pooled)
        return b_logit, e_pred

# ====== 로딩 유틸 ======
def _maybe_read_base_from_adapter(ckpt_dir: str):
    path = os.path.join(ckpt_dir, "adapter_config.json")
    if not os.path.exists(path): return None
    try:
        cfg = json.load(open(path, "r", encoding="utf-8"))
        for k in ["base_model_name_or_path", "base_model_name"]:
            if isinstance(cfg.get(k), str) and cfg[k]:
                return cfg[k]
    except Exception:
        pass
    return None

def load_lm_and_tok(ckpt_dir: str, load_4bit: bool = False):
    base_model_path = _maybe_read_base_from_adapter(ckpt_dir) or ckpt_dir

    quant_cfg = None
    if load_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16),
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16 if quant_cfg is None else None,
        quantization_config=quant_cfg,
    )
    tok = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    # LoRA 어댑터 로드(있으면)
    if os.path.exists(os.path.join(ckpt_dir, "adapter_config.json")):
        try:
            model = PeftModel.from_pretrained(model, ckpt_dir)
        except Exception as e:
            print(f"[warn] adapter load failed: {e}")
    model.eval()
    try:
        model.config.output_hidden_states = True
        model.config.use_cache = True
    except Exception:
        pass
    return model, tok

def load_aux_and_embed(ckpt_dir: str, hidden_size: int, device: torch.device, embed_model_name: str = "jhgan/ko-sroberta-multitask"):
    aux_path = os.path.join(ckpt_dir, "aux.pt")
    embed_tok = AutoTokenizer.from_pretrained(embed_model_name)
    embed_model = AutoModel.from_pretrained(embed_model_name).to(device)
    for p in embed_model.parameters(): p.requires_grad = False
    emb_dim = getattr(embed_model.config, "hidden_size", 768)
    aux = AuxHeads(hidden_size=hidden_size, embed_dim=emb_dim).to(device)
    if os.path.exists(aux_path):
        state = torch.load(aux_path, map_location=device)
        aux.load_state_dict(state["aux"])
    else:
        print(f"[warn] aux.pt not found in {ckpt_dir}. Random init.")
    aux.eval()
    return aux, embed_tok, embed_model, emb_dim

@torch.no_grad()
def encode_embed(embed_tok, embed_model, device, text: str, emb_dim: int):
    if not text or not text.strip():
        return torch.zeros(emb_dim, device=device, dtype=torch.float32)
    toks = embed_tok(text, return_tensors='pt', truncation=True, padding=True).to(device)
    out = embed_model(**toks)
    emb = out.last_hidden_state.mean(dim=1).squeeze(0)
    emb = F.normalize(emb, dim=-1)
    return emb.to(device, dtype=torch.float32)

# ====== 데이터 로드 ======
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

# ====== 평가 ======
def eval_boundary_and_summary(
    model, tok, aux, embed_tok, embed_model, emb_dim, data, boundary_thr: float = 0.5, max_turns: int = 0
):
    device = next(model.parameters()).device
    sys_prompt = "당신은 한국어 비서입니다. 최근 대화의 요지를 반영해 간결하고 정확하게 답하세요."

    # 누적 통계
    TP=FP=FN=TN=0
    cos_sims: List[float] = []

    for ex in data:
        turns: List[str] = ex.get("text", []) or []
        boundaries: List[int] = ex.get("boundaries", []) or [0]*len(turns)
        seg_summaries: List[str] = ex.get("seg_summaries", []) or [""]*len(turns)

        # 길이 정합
        L = len(turns)
        if len(boundaries) < L: boundaries += [0]*(L-len(boundaries))
        if len(seg_summaries) < L: seg_summaries += [""]*(L-len(seg_summaries))

        # 롤링 평가
        upto = L if max_turns<=0 else min(L, max_turns)
        ctx_list = []
        for i in range(upto):
            ctx_list.append(turns[i])
            context = "\n".join(ctx_list)
            prompt = f"<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n{context}\n[B:] "
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
            out = model(input_ids=enc["input_ids"], attention_mask=(enc["input_ids"]!=tok.pad_token_id).long(), output_hidden_states=True)
            pooled = out.hidden_states[-1][:, -64:, :].mean(dim=1)  # 학습과 동일
            b_logit, e_pred = aux(pooled)
            prob = torch.sigmoid(b_logit)[0].item()
            pred = 1 if prob >= boundary_thr else 0
            gt = int(boundaries[i])

            # Confusion counts
            if pred==1 and gt==1: TP+=1
            elif pred==1 and gt==0: FP+=1
            elif pred==0 and gt==1: FN+=1
            else: TN+=1

            # Summary Cosine (GT boundary==1인 지점만)
            if gt==1:
                gt_sum = seg_summaries[i] if i < len(seg_summaries) else ""
                if gt_sum and gt_sum.strip():
                    e_t = encode_embed(embed_tok, embed_model, device, gt_sum, emb_dim)
                    e_pred_n = F.normalize(e_pred[0], dim=-1)
                    cos = float((e_pred_n * e_t).sum().item())
                    cos_sims.append(cos)

    prec = TP / (TP+FP) if (TP+FP)>0 else 0.0
    rec  = TP / (TP+FN) if (TP+FN)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    acc  = (TP+TN) / max(1, (TP+TN+FP+FN))
    cos_avg = mean(cos_sims) if cos_sims else 0.0

    return {
        "boundary": {"precision":prec, "recall":rec, "f1":f1, "acc":acc, "counts":{"TP":TP,"FP":FP,"FN":FN,"TN":TN}},
        "summary_cosine": {"mean": cos_avg, "n": len(cos_sims)}
    }

# ====== LLM-as-Judge ======
JUDGE_PROMPT = """당신은 대화 품질 평가자입니다.
주어진 대화 맥락과 모델 응답을 보고 '일관성, 관련성, 사실성'을 1~5 점수로 평가하세요.
오직 숫자만 출력하세요.

[대화 맥락]
{context}

[모델 응답]
{response}
"""

def call_openai_score(prompt: str, model_name: str = "gpt-4o-mini"):
    try:
        from openai import OpenAI
    except Exception:
        return None
    if not os.environ.get("OPENAI_API_KEY"):
        return None
    client = OpenAI()
    out = client.chat.completions.create(
        model=model_name,
        messages=[{"role":"user", "content": prompt}],
        temperature=0.0,
        max_tokens=4,
    )
    txt = out.choices[0].message.content.strip()
    try:
        score = float("".join(ch for ch in txt if ch.isdigit() or ch=='.'))
        return max(1.0, min(5.0, score))
    except Exception:
        return None

@torch.no_grad()
def eval_judge(model, tok, data, sample_n: int = 50, gen_max_new: int = 128, temperature: float = 0.7, top_p: float = 0.95, judge_model: str = "gpt-4o-mini"):
    if not os.environ.get("OPENAI_API_KEY"):
        return {"judge_avg": None, "n": 0, "note": "OPENAI_API_KEY not set; skipped."}

    device = next(model.parameters()).device
    sys_prompt = "당신은 한국어 비서입니다. 최근 대화의 요지를 반영해 간결하고 정확하게 답하세요."
    pool = random.sample(data, min(sample_n, len(data)))
    scores = []

    for ex in pool:
        turns: List[str] = ex.get("text", []) or []
        if len(turns) < 2: continue
        # 마지막 턴 직전까지 맥락, 마지막 턴을 사용자 입력으로 간주
        context = "\n".join(turns[:-1])
        user = turns[-1]
        prompt = f"<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n{context}\n[A:] {user}\n[B:] "
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        out_ids = model.generate(
            input_ids=enc["input_ids"],
            max_new_tokens=gen_max_new,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id
        )
        resp = tok.decode(out_ids[0], skip_special_tokens=True)
        resp = resp.split("[B:]")[-1].strip() if "[B:]" in resp else resp.strip()

        jprompt = JUDGE_PROMPT.format(context=context, response=resp)
        sc = call_openai_score(jprompt, model_name=judge_model)
        if sc is not None:
            scores.append(sc)

    return {"judge_avg": (sum(scores)/len(scores) if scores else None), "n": len(scores)}

# ====== main ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="out/solar-mem-qlora")
    ap.add_argument("--data", type=str, required=True, help="라벨 JSONL (text/boundaries/seg_summaries)")
    ap.add_argument("--boundary_thr", type=float, default=0.50)
    ap.add_argument("--max_turns", type=int, default=0, help="0은 전체 턴 평가")
    ap.add_argument("--judge", action="store_true", help="LLM-as-Judge 실행 (OPENAI_API_KEY 필요)")
    ap.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--sample_n", type=int, default=50)
    ap.add_argument("--mode", type=str, choices=["mem","base"], default="mem", help="mem: 경계/요약+Judge, base: Judge만")  # [추가]
    ap.add_argument("--load_4bit", action="store_true", help="본 모델 4bit QLoRA 로드")  # [추가]
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_lm_and_tok(args.ckpt, load_4bit=args.load_4bit)
    hidden = model.config.hidden_size

    data = load_jsonl(args.data)

    res12 = None
    res3 = None

    if args.mode == "mem":
        # Boundary/Summary + Judge
        aux, embed_tok, embed_model, emb_dim = load_aux_and_embed(args.ckpt, hidden, device)
        res12 = eval_boundary_and_summary(model, tok, aux, embed_tok, embed_model, emb_dim, data, boundary_thr=args.boundary_thr, max_turns=args.max_turns)
        print("[Boundary/Summary] ", json.dumps(res12, ensure_ascii=False, indent=2))
        if args.judge:
            res3 = eval_judge(model, tok, data, sample_n=args.sample_n, judge_model=args.judge_model)
            print("[LLM-as-Judge] ", json.dumps(res3, ensure_ascii=False, indent=2))
        else:
            print("[LLM-as-Judge] skipped (use --judge to enable)")
        subdir = "mem"
    else:
        # base: Judge only
        if args.judge:
            res3 = eval_judge(model, tok, data, sample_n=args.sample_n, judge_model=args.judge_model)
            print("[LLM-as-Judge] ", json.dumps(res3, ensure_ascii=False, indent=2))
        else:
            print("[LLM-as-Judge] skipped (use --judge to enable)")
        subdir = "base"

    # ---- save to log/base or log/mem ----
    os.makedirs(os.path.join("log", subdir), exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.basename(args.data).replace(".jsonl","")
    out_path = os.path.join("log", subdir, f"eval_{base}_{ts}.json")
    payload = {
        "ckpt": args.ckpt,
        "data": args.data,
        "mode": args.mode,
        "boundary_thr": args.boundary_thr,
        "max_turns": args.max_turns,
        "result_boundary_summary": res12,
        "result_judge": res3,
        "timestamp": ts,
        "load_4bit": args.load_4bit,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()
