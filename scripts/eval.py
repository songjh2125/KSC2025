# eval.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Evaluation script for baseline and memory-augmented models.
- Ensures the *same* embedding model used during training is used at eval time.
- Reads embedding metadata from aux.pt / embedding_meta.json if available.
- Provides boundary/summary evaluation, retrieval stats, (optional) LLM-as-Judge.
"""

import os, json, argparse, random, datetime, csv
from typing import List, Dict, Any, Tuple

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import BitsAndBytesConfig
from peft import PeftModel
from statistics import mean

# ====== AuxHeads (must match training) ======
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

# ====== Loading Utilities ======

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
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Load LoRA adapter if present
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

def load_aux_and_embed(
    ckpt_dir: str,
    hidden_size: int,
    device: torch.device,
    embed_model_name: str | None = None,
):
    """
    Resolve embedding model for evaluation with the following priority:
      1) aux.pt: {embed_model_name, emb_dim}
      2) embedding_meta.json: {embedding_model, emb_dim}
      3) user override: embed_model_name (CLI)
      4) training default: 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    Returns: aux, embed_tok, embed_model, emb_dim, final_embed_model
    """
    aux_path = os.path.join(ckpt_dir, "aux.pt")
    meta_model = None; meta_dim = None; state = None

    if os.path.exists(aux_path):
        try:
            state = torch.load(aux_path, map_location=device)
            meta_model = state.get("embed_model_name")
            meta_dim   = state.get("emb_dim")
        except Exception as e:
            print(f"[warn] failed to read aux meta: {e}")

    meta_json = os.path.join(ckpt_dir, "embedding_meta.json")
    if (meta_model is None or meta_dim is None) and os.path.exists(meta_json):
        try:
            j = json.load(open(meta_json, "r", encoding="utf-8"))
            meta_model = meta_model or j.get("embedding_model")
            meta_dim   = meta_dim   or j.get("emb_dim")
        except Exception:
            pass

    final_embed_model = meta_model or embed_model_name \
        or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    embed_tok = AutoTokenizer.from_pretrained(final_embed_model)
    embed_model = AutoModel.from_pretrained(final_embed_model).to(device)
    for p in embed_model.parameters():
        p.requires_grad = False

    # Infer dimension (prefer meta if present)
    emb_dim = int(meta_dim or getattr(embed_model.config, "hidden_size", 768))

    aux = AuxHeads(hidden_size=hidden_size, embed_dim=emb_dim).to(device)
    if state is not None and "aux" in state:
        try:
            aux.load_state_dict(state["aux"])  # type: ignore[arg-type]
        except RuntimeError as e:
            raise RuntimeError(
                "[shape-mismatch] AuxHeads and embedding dimension mismatch. "
                "Use the SAME embedding model as training. "
                f"final_embed_model={final_embed_model}, emb_dim={emb_dim}. Cause: {e}"
            )
    else:
        print(f"[warn] aux.pt not found or missing 'aux' in {ckpt_dir}. Random init.")

    aux.eval()
    return aux, embed_tok, embed_model, emb_dim, final_embed_model

@torch.no_grad()
def encode_embed(embed_tok, embed_model, device, text: str, emb_dim: int):
    if not text or not text.strip():
        return torch.zeros(emb_dim, device=device, dtype=torch.float32)
    toks = embed_tok(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    out = embed_model(**toks)
    attn = toks["attention_mask"].unsqueeze(-1).to(out.last_hidden_state.dtype)  # [B,L,1]
    # masked mean pooling (sentence-transformers 방식)
    token_sum = (out.last_hidden_state * attn).sum(dim=1)              # [B,D]
    lengths  = attn.sum(dim=1).clamp(min=1.0)                          # [B,1]
    emb = (token_sum / lengths).squeeze(0)                              # [D]
    emb = F.normalize(emb, dim=-1)
    return emb.to(device, dtype=torch.float32)

# ====== Data I/O ======

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

# ====== Sweep / Metrics ======

def parse_sweep(spec: str) -> List[float]:
    # "a:b:s" or "a,b,c"
    vals: List[float] = []
    if ":" in spec:
        a, b, s = spec.split(":")
        start = float(a); end = float(b); step = float(s)
        if step <= 0:
            return []
        n = int(round((end - start) / step)) + 1
        vals = [round(start + i*step, 10) for i in range(n)]
    else:
        vals = [float(x) for x in spec.split(",") if x.strip()!=""]
    # Clamp to [0,1], dedup, sort
    vals = sorted(set(max(0.0, min(1.0, v)) for v in vals))
    return vals

def prf1_from_probs_labels(probs: List[float], labels: List[int], thr: float):
    TP=FP=FN=TN=0
    for p, y in zip(probs, labels):
        pred = 1 if p >= thr else 0
        if pred==1 and y==1: TP+=1
        elif pred==1 and y==0: FP+=1
        elif pred==0 and y==1: FN+=1
        else: TN+=1
    prec = TP/(TP+FP) if (TP+FP)>0 else 0.0
    rec  = TP/(TP+FN) if (TP+FN)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    acc  = (TP+TN)/max(1, (TP+TN+FP+FN))
    return prec, rec, f1, acc, {"TP":TP,"FP":FP,"FN":FN,"TN":TN}

# ====== Core Eval (boundary/summary + retrieval) ======

@torch.no_grad()
def eval_boundary_and_summary(
    model, tok, aux, embed_tok, embed_model, emb_dim, data,
    boundary_thr: float = 0.5, max_turns: int = 0, thr: float = 0.60,
    eval_negatives_per_pos: int = 1, eval_include_last: bool = True
):
    device = next(model.parameters()).device
    sys_prompt = "당신은 한국어 비서입니다. 최근 대화의 요지를 반영해 간결하고 정확하게 답하세요."

    # Global counters
    TP=FP=FN=TN=0
    cos_sims: List[float] = []

    mem_queries = 0
    mem_hits = 0
    mem_max_sims: List[float] = []
    per_dialog_stats: List[Dict[str, Any]] = []

    boundary_probs: List[float] = []
    boundary_labels: List[int]  = []

    for ex in data:
        turns: List[str] = ex.get("text", []) or []
        boundaries: List[int] = ex.get("boundaries", []) or [0]*len(turns)
        seg_summaries: List[str] = ex.get("seg_summaries", []) or [""]*len(turns)

        L = len(turns)
        if len(boundaries) < L: boundaries += [0]*(L-len(boundaries))
        if len(seg_summaries) < L: seg_summaries += [""]*(L-len(seg_summaries))

        ltm_bank: List[torch.Tensor] = []
        dlg_q = 0
        dlg_hits = 0
        dlg_max_sims: List[float] = []

        upto = L if max_turns<=0 else min(L, max_turns)

        pos_idx = [i for i in range(upto) if int(boundaries[i]) == 1]
        neg_idx = [i for i in range(upto) if int(boundaries[i]) == 0]

        selected: List[Tuple[int, int]] = []  # (i, count_for_boundary=1|0)
        # 우선 모든 양성은 반드시 평가
        selected.extend((i, 1) for i in pos_idx)
        # 하드 네거티브 수집(±1)
        hard_negs: List[int] = []
        for p in pos_idx:
            if p-1 >= 0 and boundaries[p-1] == 0: hard_negs.append(p-1)
            if p+1 < upto and boundaries[p+1] == 0: hard_negs.append(p+1)
        # unique
        hard_negs = list(dict.fromkeys(hard_negs))
        K_total = eval_negatives_per_pos * max(1, len(pos_idx))
        chosen_negs = hard_negs[:K_total]
        remain = K_total - len(chosen_negs)
        if remain > 0:
            pool = [i for i in neg_idx if i not in set(hard_negs)]
            if len(pool) > 0:
                chosen_negs += random.sample(pool, min(remain, len(pool)))
        selected.extend((i, 1) for i in chosen_negs)

        # 마지막 턴 포함(컨텍스트/요약 임베딩용), 경계지표에는 포함하지 않음
        if eval_include_last and upto > 0:
            i_last = upto - 1
            if all(i_last != x for x, _ in selected):
                selected.append((i_last, 0))

        # 정렬/중복 제거
        seen = set()
        selected = [(i, c) for (i, c) in sorted(selected, key=lambda x: x[0]) if not (i in seen or seen.add(i))]

    ctx_list = []
    for i, count_for_boundary in selected:
        ctx_list.append(turns[i])
        context = "\n".join(ctx_list)

        # (학습과 동일한 인코딩)
        enc = tok(context, return_tensors="pt", truncation=True).to(device)
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc.get("attention_mask", None),
            output_hidden_states=True
        )

        # (pool_last_n_tokens=0 → 전체 평균 풀링)
        pooled = out.hidden_states[-1].mean(dim=1)

        aux_dtype = next(aux.parameters()).dtype
        if pooled.dtype != aux_dtype:
            pooled = pooled.to(dtype=aux_dtype)

        b_logit, e_pred = aux(pooled)
        prob = torch.sigmoid(b_logit)[0].item()
        pred = 1 if prob >= boundary_thr else 0
        gt = int(boundaries[i])

        if count_for_boundary == 1:
            if pred==1 and gt==1: TP+=1
            elif pred==1 and gt==0: FP+=1
            elif pred==0 and gt==1: FN+=1
            else: TN+=1
            boundary_probs.append(prob)
            boundary_labels.append(gt)

        # Retrieval
        if len(ltm_bank) > 0:
            q = F.normalize(e_pred[0].float(), dim=-1)
            bank = torch.stack(ltm_bank, dim=0).float()
            sims = torch.matmul(bank, q)
            max_sim = float(sims.max().item())
            mem_queries += 1
            dlg_q += 1
            mem_max_sims.append(max_sim)
            dlg_max_sims.append(max_sim)
            if max_sim >= thr:
                mem_hits += 1
                dlg_hits += 1

        # Commit GT summary when boundary==1
        if gt==1:
            gt_sum = seg_summaries[i] if i < len(seg_summaries) else ""
            if gt_sum and gt_sum.strip():
                e_t = encode_embed(embed_tok, embed_model, device, gt_sum, emb_dim)
                e_pred_n = F.normalize(e_pred[0].float(), dim=-1)
                cos = float((e_pred_n * e_t.float()).sum().item())
                cos_sims.append(cos)
                ltm_bank.append(e_t.float())


        per_dialog_stats.append({
            "hit_rate": (dlg_hits / dlg_q) if dlg_q > 0 else 0.0,
            "avg_max_sim": (mean(dlg_max_sims) if dlg_max_sims else 0.0),
            "queries": dlg_q,
            "hits": dlg_hits,
        })

    # Boundary metrics (for current boundary_thr)
    prec = TP / (TP+FP) if (TP+FP)>0 else 0.0
    rec  = TP / (TP+FN) if (TP+FN)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    acc  = (TP+TN) / max(1, (TP+TN+FP+FN))
    cos_avg = mean(cos_sims) if cos_sims else 0.0

    mem_hit_rate = (mem_hits / mem_queries) if mem_queries>0 else 0.0
    mem_avg_max_sim = mean(mem_max_sims) if mem_max_sims else 0.0

    return {
        "boundary": {"precision":prec, "recall":rec, "f1":f1, "acc":acc, "counts":{"TP":TP,"FP":FP,"FN":FN,"TN":TN}},
        "summary_cosine": {"mean": cos_avg, "n": len(cos_sims)},
        "memory_retrieval": {
            "thr": thr,
            "hit_rate": mem_hit_rate,
            "queries": mem_queries,
            "hits": mem_hits,
            "avg_max_sim": mem_avg_max_sim,
            "per_dialog": per_dialog_stats,
            "all_max_sims": mem_max_sims,
        },
        "boundary_raw": {
            "probs": boundary_probs,
            "labels": boundary_labels
        }
    }

# ====== LLM-as-Judge (optional) ======

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
def eval_judge(model, tok, data, sample_n: int = 50, gen_max_new: int = 128, temperature: float = 0.7, top_p: float = 0.95, judge_model: str = "gpt-4o-mini", use_all: bool=False):
    if not os.environ.get("OPENAI_API_KEY"):
        return {"judge_avg": None, "n": 0, "note": "OPENAI_API_KEY not set; skipped."}

    device = next(model.parameters()).device
    sys_prompt = "당신은 한국어 비서입니다. 최근 대화의 요지를 반영해 간결하고 정확하게 답하세요."

    pool = list(data) if (use_all or sample_n <= 0) else random.sample(data, min(sample_n, len(data)))

    scores = []
    for ex in pool:
        turns: List[str] = ex.get("text", []) or []
        if len(turns) < 2:
            continue
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

# ====== Plotting / CSV ======

def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def save_csv_per_dialog(per: List[Dict[str, Any]], path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dialog_index","queries","hits","hit_rate","avg_max_sim"])
        for i, d in enumerate(per):
            w.writerow([i, d.get("queries",0), d.get("hits",0), round(d.get("hit_rate",0.0),6), round(d.get("avg_max_sim",0.0),6)])

def plot_figures(per: List[Dict[str,Any]], res12: Dict[str,Any], gate_sweep: List[float], kept_fracs: List[float], judge_curve: List[Tuple[float,float]], outdir: str,
                 bthr_vals: List[float] | None = None, bthr_f1s: List[float] | None = None):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable: {e}")
        return

    # 1) hit_rate histogram
    rates = [d.get("hit_rate",0.0) for d in per]
    plt.figure()
    plt.hist(rates, bins=20, range=(0,1))
    plt.xlabel("hit_rate"); plt.ylabel("count"); plt.title("Per-dialog hit_rate histogram")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"hit_rate_hist.png")); plt.savefig(os.path.join(outdir,"hit_rate_hist.svg"))
    plt.close()

    # 2) hit_rate CDF
    xs = sorted(rates)
    ys = [i/len(xs) for i in range(1, len(xs)+1)] if xs else []
    plt.figure()
    if xs:
        plt.plot(xs, ys)
    plt.xlabel("hit_rate"); plt.ylabel("CDF"); plt.title("Per-dialog hit_rate CDF")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"hit_rate_cdf.png")); plt.savefig(os.path.join(outdir,"hit_rate_cdf.svg"))
    plt.close()

    # 3) scatter: queries vs hit_rate (size ~ hits)
    qs  = [d.get("queries",0) for d in per]
    hs  = [d.get("hits",0) for d in per]
    plt.figure()
    sizes = [max(10, h*20) for h in hs]
    plt.scatter(qs, rates, s=sizes, alpha=0.7)
    plt.xlabel("queries per dialog"); plt.ylabel("hit_rate"); plt.title("hit_rate vs queries (size~hits)")
    plt.tight_layout(); plt.savefig(os.path.join(outdir,"hit_rate_scatter.png")); plt.savefig(os.path.join(outdir,"hit_rate_scatter.svg"))
    plt.close()

    # 4) bthr sweep: F1 vs bthr
    if bthr_vals and bthr_f1s:
        plt.figure()
        plt.plot(bthr_vals, bthr_f1s)
        plt.xlabel("boundary threshold (bthr)")
        plt.ylabel("F1")
        plt.title("Boundary detection: F1 vs bthr")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir,"bthr_sweep_f1.png"))
        plt.savefig(os.path.join(outdir,"bthr_sweep_f1.svg"))
        plt.close()

    # 5) summary cosine mean (record to file)
    sc = res12.get("summary_cosine", {})
    with open(os.path.join(outdir, "summary_cosine.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(sc, ensure_ascii=False, indent=2))

# ====== MEM evaluation (sweep/report/save) ======

def run_mem_eval(args, ckpt: str, data_path: str):
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load_lm_and_tok(ckpt, load_4bit=args.load_4bit)
    hidden = model.config.hidden_size
    data = load_jsonl(data_path)

    aux, embed_tok, embed_model, emb_dim, final_embed_model = load_aux_and_embed(
        ckpt, hidden, device, embed_model_name=args.embed_model_name
    )
    print(f"[embed] using embedding model: {final_embed_model} (dim={emb_dim})")

    res12 = eval_boundary_and_summary(
        model, tok, aux, embed_tok, embed_model, emb_dim, data,
        boundary_thr=args.boundary_thr, max_turns=args.max_turns, thr=args.thr,
        eval_negatives_per_pos=args.eval_negatives_per_pos,
        eval_include_last=args.eval_include_last
    )
    print("[MEM Boundary/Summary] ", json.dumps(res12, ensure_ascii=False, indent=2))

    # Judge (optional) — no gating by default (left hooks kept for future use)
    res3 = None
    if args.judge:
        data_for_judge = data
        if args.judge_gate_hit_rate is not None:
            per = res12.get("memory_retrieval", {}).get("per_dialog", [])
            if per and len(per) == len(data):
                mask = [ (d.get("hit_rate", 0.0) >= args.judge_gate_hit_rate) for d in per ]
                data_for_judge = [ex for ex, ok in zip(data, mask) if ok]
                print(f"[gate] judge_gate_hit_rate={args.judge_gate_hit_rate} → {len(data_for_judge)}/{len(data)} dialogs kept")
            else:
                print("[gate] per_dialog stats missing or length mismatch; gating skipped.")

        res3 = eval_judge(
            model, tok, data_for_judge,
            sample_n=args.sample_n,
            gen_max_new=args.gen_max_new,
            temperature=args.temperature,
            top_p=args.top_p,
            judge_model=args.judge_model,
            use_all=args.judge_use_all
        )
        print("[MEM LLM-as-Judge] ", json.dumps(res3, ensure_ascii=False, indent=2))
    else:
        print("[MEM LLM-as-Judge] skipped (use --judge)")

    # bthr sweep (F1)
    bthr_vals: List[float] = []; bthr_f1s: List[float] = []
    best_bthr = None
    if args.bthr_sweep:
        raw = res12.get("boundary_raw", {})
        probs = raw.get("probs", []); labels= raw.get("labels", [])
        bthr_vals = parse_sweep(args.bthr_sweep)
        for t in bthr_vals:
            _, _, f1, _, _ = prf1_from_probs_labels(probs, labels, t)
            bthr_f1s.append(f1)
        if bthr_vals and bthr_f1s:
            i = int(np.argmax(bthr_f1s))
            best_bthr = float(bthr_vals[i])
            print(f"[bthr] best F1 at {best_bthr:.4f} (F1={bthr_f1s[i]:.4f})")

    # thr sweep (hit_rate)
    thr_vals: List[float] = []; thr_hit_rates: List[float] = []
    best_thr = None
    if args.thr_sweep:
        thr_vals = parse_sweep(args.thr_sweep)
        base_mr = res12.get("memory_retrieval", {})
        sims = base_mr.get("all_max_sims", None)
        if sims is None:
            print("[thr_sweep] all_max_sims missing; falling back to slow recompute.")
            for t in thr_vals:
                tmp = eval_boundary_and_summary(
                    model, tok, aux, embed_tok, embed_model, emb_dim, data,
                    boundary_thr=(best_bthr if best_bthr is not None else args.boundary_thr),
                    max_turns=args.max_turns, thr=t,
                    eval_negatives_per_pos=args.eval_negatives_per_pos,
                    eval_include_last=args.eval_include_last
                )
                thr_hit_rates.append(tmp.get("memory_retrieval", {}).get("hit_rate", 0.0))
        else:
            n = len(sims)
            for t in thr_vals:
                hits = sum(1 for v in sims if v >= t)
                thr_hit_rates.append( (hits / n) if n > 0 else 0.0 )
        if thr_vals and thr_hit_rates:
            j = int(np.argmax(thr_hit_rates))
            best_thr = float(thr_vals[j])
            print(f"[thr] best hit_rate at {best_thr:.4f} (hit_rate={thr_hit_rates[j]:.4f})")

    # Plots/CSV
    if args.plot_dir:
        outdir = os.path.join(args.plot_dir, "mem")
        ensure_dir(outdir)
        per = res12.get("memory_retrieval", {}).get("per_dialog", [])
        plot_figures(per, res12, [], [], [], outdir, bthr_vals=bthr_vals, bthr_f1s=bthr_f1s)
    if args.export_csv:
        path = args.export_csv
        if os.path.isdir(path):
            path = os.path.join(path, "per_dialog_mem.csv")
        per = res12.get("memory_retrieval", {}).get("per_dialog", [])
        save_csv_per_dialog(per, path)
        print(f"[csv] mem per-dialog stats saved to {path}")

    # Save best thresholds (optional)
    if args.save_best_thresholds:
        out_cfg = {
            "bthr": (best_bthr if best_bthr is not None else args.boundary_thr),
            "thr":  (best_thr  if best_thr  is not None else args.thr),
            "source": "eval.py(mem)",
            "note": "bthr from F1-max; thr from hit_rate-max (or provided args if sweep missing)."
        }
        try:
            ensure_dir(os.path.dirname(args.save_best_thresholds))
        except Exception:
            pass
        try:
            with open(args.save_best_thresholds, "w", encoding="utf-8") as f:
                json.dump(out_cfg, f, ensure_ascii=False, indent=2)
            print(f"[saved] best thresholds -> {args.save_best_thresholds}: {out_cfg}")
        except Exception as e:
            print(f"[save_best_thresholds] failed: {e}")

    # Persist log
    os.makedirs(os.path.join("log", "mem"), exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.basename(data_path).replace(".jsonl","")
    out_path = os.path.join("log", "mem", f"eval_{base}_{ts}.json")
    payload = {
        "ckpt": ckpt,
        "data": data_path,
        "mode": "mem",
        "boundary_thr": args.boundary_thr,
        "thr": args.thr,
        "max_turns": args.max_turns,
        "result_boundary_summary": res12,
        "result_judge": res3,
        "timestamp": ts,
        "load_4bit": args.load_4bit,
        "gen_max_new": args.gen_max_new,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "judge_gate_hit_rate": args.judge_gate_hit_rate,
        "plot_dir": args.plot_dir,
        "export_csv": args.export_csv,
        "bthr_sweep": args.bthr_sweep,
        "bthr_sweep_vals": bthr_vals if args.bthr_sweep else None,
        "bthr_sweep_f1s": bthr_f1s if args.bthr_sweep else None,
        "thr_sweep": args.thr_sweep,
        "thr_sweep_vals": thr_vals if args.thr_sweep else None,
        "thr_sweep_hit_rates": thr_hit_rates if args.thr_sweep else None,
        "best_bthr_from_sweep": best_bthr,
        "best_thr_from_sweep": best_thr,
        "embedding_model_used": final_embed_model,
        "embedding_dim": emb_dim,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out_path}")

# ====== Baseline-only evaluation (Judge only) ======

def run_base_eval(args, ckpt: str, data_path: str):
    random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    model, tok = load_lm_and_tok(ckpt, load_4bit=args.load_4bit)
    data = load_jsonl(data_path)

    res3 = None
    if args.judge:
        res3 = eval_judge(
            model, tok, data,
            sample_n=args.sample_n,
            gen_max_new=args.gen_max_new,
            temperature=args.temperature,
            top_p=args.top_p,
            judge_model=args.judge_model,
            use_all=args.judge_use_all
        )
        print("[BASE LLM-as-Judge] ", json.dumps(res3, ensure_ascii=False, indent=2))
    else:
        print("[BASE LLM-as-Judge] skipped (use --judge)")

    # Persist log
    os.makedirs(os.path.join("log", "base"), exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.basename(data_path).replace(".jsonl","")
    out_path = os.path.join("log", "base", f"eval_{base}_{ts}.json")
    payload = {
        "ckpt": ckpt,
        "data": data_path,
        "mode": "base",
        "result_judge": res3,
        "timestamp": ts,
        "load_4bit": args.load_4bit,
        "gen_max_new": args.gen_max_new,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[saved] {out_path}")

# ====== CLI ======

def main():
    ap = argparse.ArgumentParser()
    # Modes/paths
    ap.add_argument("--mode", type=str, choices=["base","mem","both"], default="both",
                    help="평가 모드: base(베이스라인), mem(메모리), both(둘 다 순차 실행)")
    ap.add_argument("--ckpt", type=str, default=None, help="공용 ckpt (개별 미지정 시 사용)")
    ap.add_argument("--ckpt_base", type=str, default=None, help="베이스라인 ckpt")
    ap.add_argument("--ckpt_mem", type=str, default=None, help="메모리 ckpt")
    ap.add_argument("--data", type=str, default=None, help="공용 데이터(jsonl)")
    ap.add_argument("--data_base", type=str, default=None, help="베이스라인 데이터(jsonl)")
    ap.add_argument("--data_mem", type=str, default=None, help="메모리 데이터(jsonl)")

    # Common
    ap.add_argument("--load_4bit", action="store_true", help="모델 4bit 로드")
    ap.add_argument("--gen_max_new", type=int, default=128, help="Judge 생성 토큰 수")
    ap.add_argument("--temperature", type=float, default=0.7, help="Judge 생성 temperature")
    ap.add_argument("--top_p", type=float, default=0.95, help="Judge 생성 top-p")
    ap.add_argument("--seed", type=int, default=42, help="난수 시드 고정")
    ap.add_argument("--judge", action="store_true", help="LLM-as-Judge 실행")
    ap.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--sample_n", type=int, default=50)
    ap.add_argument("--judge_use_all", action="store_true", help="Judge 전체 사용")

    # Embedding override (optional)
    ap.add_argument("--embed_model_name", type=str, default=None,
                    help="평가 시 강제로 사용할 임베딩 모델 이름(권장: 학습과 동일)")

    # MEM-only params
    ap.add_argument("--boundary_thr", type=float, default=0.50, help="경계 임계치 (bthr)")
    ap.add_argument("--thr", type=float, default=0.60, help="LTM retrieval threshold (cosine)")
    ap.add_argument("--max_turns", type=int, default=0, help="0은 전체 턴 평가")
    ap.add_argument("--eval-negatives-per-pos", type=int, default=1, help="평가 시 각 양성당 네거티브 후보 수")
    ap.add_argument("--eval-include-last", action="store_true", default=True, help="평가 시 마지막 턴을 컨텍스트/요약용으로 포함(경계지표 제외)")
    ap.add_argument("--judge_gate_hit_rate", type=float, default=None,
                    help="메모리 hit_rate가 이 값 이상인 대화만 Judge 집계")
    ap.add_argument("--thr_sweep", type=str, default=None,
                    help="LTM retrieval thr sweep (예: '0.40:0.80:0.02' 또는 '0.55,0.60,0.65')")
    ap.add_argument("--bthr_sweep", type=str, default=None,
                    help="경계 임계치 스윕 (예: '0.1:0.9:0.02' 또는 '0.3,0.4,0.5')")
    ap.add_argument("--save_best_thresholds", type=str, default=None,
                    help="최적 {bthr,thr}를 JSON으로 저장")
    ap.add_argument("--plot_dir", type=str, default=None, help="그림 저장 폴더 (없으면 그림 생성 안함)")
    ap.add_argument("--export_csv", type=str, default=None, help="per-dialog CSV 저장 경로")
    args = ap.parse_args()

    # Resolve paths
    ckpt_base = args.ckpt_base or args.ckpt
    ckpt_mem  = args.ckpt_mem  or args.ckpt
    data_base = args.data_base or args.data
    data_mem  = args.data_mem  or args.data

    if args.mode in ("base","both"):
        if not ckpt_base or not data_base:
            raise ValueError("base 평가에 필요한 --ckpt_base/--data_base (또는 공용 --ckpt/--data)가 필요합니다.")
        print("\n=== BASELINE EVAL ===")
        run_base_eval(args, ckpt=ckpt_base, data_path=data_base)

    if args.mode in ("mem","both"):
        if not ckpt_mem or not data_mem:
            raise ValueError("mem 평가에 필요한 --ckpt_mem/--data_mem (또는 공용 --ckpt/--data)가 필요합니다.")
        print("\n=== MEMORY EVAL ===")
        run_mem_eval(args, ckpt=ckpt_mem, data_path=data_mem)

if __name__ == "__main__":
    main()
