#!/usr/bin/env python3
import os, json, argparse
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

from src.data_utils import basic_normalize

def load_cache(path):
    arr=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            arr.append(json.loads(line))
    return arr

def boundary_f1(gt, pred):
    y_true = np.array(gt); y_pred = np.array(pred)
    p,r,f,_ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    return p,r,f

def cosine(a,b):
    a = a/np.linalg.norm(a); b=b/np.linalg.norm(b)
    return float(np.dot(a,b))

def embed_texts(texts, model_name='jhgan/ko-sroberta-multitask'):
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name)
    embs=[]
    for t in texts:
        out = enc(**tok(t, return_tensors='pt', truncation=True, padding=True))
        embs.append(out.last_hidden_state.mean(dim=1).squeeze(0).detach().numpy())
    return embs

def quality_heuristic(dialogue: str) -> float:
    # 아주 단순한 휴리스틱: 발화 길이 다양성 + 교차 질문 비율
    ut = [u for u in dialogue.split('\n') if u.strip()]
    lens = [len(u) for u in ut]
    if not lens: return 0.0
    var = np.var(lens)
    qrate = sum(1 for u in ut if u.strip().endswith('?'))/len(ut)
    return 0.5*np.tanh(var/100)+0.5*qrate

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels_cache', type=str, default='cache/labels_ko_samsum.jsonl')
    ap.add_argument('--max_samples', type=int, default=1000)
    args = ap.parse_args()

    cache = load_cache(args.labels_cache)
    cache_by_id = {c['id']:c for c in cache if 'id' in c}
    ds = load_dataset('heegyu/ko-samsum', split='test')
    ds = ds.select(range(min(args.max_samples, len(ds))))

    # Boundary F1: GT는 cache, 예측은 간단히 '스피커 전환 + 길이>=4' 규칙(베이스라인)
    P=R=F=0; N=0
    for idx, ex in enumerate(ds):
        dialog = ex.get('dialogue') or ex.get('conversation') or ''
        dialog = '\n'.join(basic_normalize(u) for u in dialog.split('\n'))
        gt = cache_by_id.get(idx, {}).get('boundaries', [])
        utts = [u for u in dialog.split('\n') if u.strip()]
        pred=[0]*len(utts)
        cur=[]
        for i,u in enumerate(utts):
            prev = utts[i-1][:3] if i>0 else None
            trig = (i>0 and u[:3]!=prev and len(cur)>=4)
            if trig: pred[i]=1; cur=[]
            cur.append(u)
        if gt and len(gt)==len(pred):
            p,r,f = boundary_f1(gt, pred)
            P+=p; R+=r; F+=f; N+=1
    if N>0:
        print({"BoundaryF1": F/N, "P": P/N, "R": R/N, "N": N})

    # 요약-임베딩 코사인: 세그먼트 요약(캐시) vs 전체 요약 텍스트 임베딩 비교
    test = ds.select(range(min(100, len(ds))))
    gt_summ=[]; ref_summ=[]
    for idx, ex in enumerate(test):
        dialog = ex.get('dialogue') or ex.get('conversation') or ''
        cache_rec = cache_by_id.get(idx, {})
        last = ''
        for s in cache_rec.get('seg_summaries', []):
            if s: last = s
        if last:
            gt_summ.append(last)
            ref_summ.append(ex.get('summary') or '')
    if gt_summ:
        gt_emb = embed_texts(gt_summ)
        ref_emb = embed_texts(ref_summ)
        cs = [cosine(a,b) for a,b in zip(gt_emb, ref_emb)]
        print({"SummaryCosine(mean)": float(np.mean(cs)), "n": len(cs)})

    # 대화 품질 휴리스틱(샘플 평균)
    qs=[]
    for _, ex in enumerate(test):
        dialog = ex.get('dialogue') or ex.get('conversation') or ''
        qs.append(quality_heuristic(dialog))
    if qs:
        print({"QualityHeuristic(mean)": float(np.mean(qs)), "n": len(qs)})