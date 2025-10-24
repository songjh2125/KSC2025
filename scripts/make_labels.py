#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_labels.py
- heegyu/ko-samsum 을 읽어 발화 단위 토픽 경계(0/1)와 세그먼트 요약을 생성.
- OpenAI gpt-4o-mini 사용(환경변수 OPENAI_API_KEY 필요). 실패/비용 제한 시 규칙기반 백업.
- 결과는 JSONL 캐시("cache/labels_ko_samsum.jsonl").
"""
import os, json, time, argparse
from datasets import load_dataset

# OpenAI SDK v1.x 기준
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

PROMPT = (
    "대화를 줄 단위로 보고, 각 줄 i에 대해 'i\tboundary\tsummary'를 출력하세요.\n"
    "boundary는 0 또는 1. 1은 새로운 토픽으로 넘어가는 분기점입니다.\n"
    "summary는 boundary=1인 줄에 한해, 직전 세그먼트를 한국어로 100자 이내 요약하세요.\n"
    "boundary=0인 줄은 summary를 빈 문자열로 두세요.\n"
)

def rule_fallback(dialogue: str):
    utts = [u.strip() for u in dialogue.split('\n') if u.strip()]
    boundaries = [0]*len(utts)
    seg_summaries = [""]*len(utts)
    cur = []
    for i,u in enumerate(utts):
        prev = utts[i-1][:3] if i>0 else None
        cursp = u[:3]
        trigger = (i>0 and cursp!=prev and len(cur)>=4)
        if trigger:
            boundaries[i] = 1
            seg_summaries[i] = (cur[0] + (" … " + cur[-1] if len(cur)>1 else ""))[:100]
            cur = []
        cur.append(u)
    if cur:
        seg_summaries[-1] = (cur[0] + (" … " + cur[-1] if len(cur)>1 else ""))[:100]
    return boundaries, seg_summaries


def call_gpt4o(dialogue: str):
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return rule_fallback(dialogue)
    client = OpenAI()
    msgs = [
        {"role":"system","content":PROMPT},
        {"role":"user","content":f"대화:\n{dialogue}"}
    ]
    try:
        resp = client.chat.completions.create(model="gpt-4o-mini", messages=msgs, temperature=0)
        text = resp.choices[0].message.content
        lines = [l for l in text.splitlines() if l.strip()]
        utts = [u for u in dialogue.split('\n') if u.strip()]
        boundaries=[0]*len(utts); seg_summaries=[""]*len(utts)
        for l in lines:
            parts=l.split('\t')
            if len(parts)>=2 and parts[0].isdigit():
                i=int(parts[0]);
                if 0<=i<len(utts):
                    boundaries[i]=1 if parts[1].strip()=="1" else 0
                    if len(parts)>=3:
                        seg_summaries[i]=parts[2].strip()
        return boundaries, seg_summaries
    except Exception:
        return rule_fallback(dialogue)


def main(out_path: str, max_samples: int=0, sleep_s: float=0.0):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds = load_dataset("heegyu/ko-samsum", split="train")
    if max_samples and max_samples>0:
        ds = ds.select(range(min(max_samples, len(ds))))
    with open(out_path, 'w', encoding='utf-8') as f:
        for idx, ex in enumerate(ds):
            dialog = ex.get("dialogue") or ex.get("conversation") or ""
            summary = ex.get("summary") or ""
            if len(summary) < 100:
                continue
            boundaries, seg_summaries = call_gpt4o(dialog)
            rec = {
                "id": idx,
                "boundaries": boundaries,
                "seg_summaries": seg_summaries,
            }
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")
            if sleep_s>0:
                time.sleep(sleep_s)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default='cache/labels_ko_samsum.jsonl')
    ap.add_argument('--max_samples', type=int, default=0)
    ap.add_argument('--sleep', type=float, default=0.0)
    args = ap.parse_args()
    main(args.out, args.max_samples, args.sleep)