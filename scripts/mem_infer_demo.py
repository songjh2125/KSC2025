#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mem_infer_demo.py
- 훈련된 SOLAR(QLoRA) + 메모리 모듈로 대화 추론 데모
- 기능: STM 버퍼, 경계 탐지(보조헤드/룰 백업), 경계 시 LTM 전이, Proj_mem 동적 주입, 로컬 요약기
- 외부 API 불필요. 학습 산출물(out/...)만 있으면 동작.

실행 예시:
$ python mem_infer_demo.py \
    --ckpt out/solar-mem-qlora \
    --max_new_tokens 256 --temperature 0.7

대화 종료: /exit
메모리 초기화: /reset
상태 보기: /state
"""
from __future__ import annotations
import os, argparse, sys
from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from transformers import AutoModel  # ko-sroberta
from mem_modules import MemConfig, MemoryManager, AuxHeads
from summarizer_local import LocalSummarizer

# ------------------------
# Utils
# ------------------------

def load_model(ckpt: str):
    base = AutoModelForCausalLM.from_pretrained(
        ckpt, device_map='auto', torch_dtype=torch.float16
    )
    # peft 어댑터가 이미 병합/저장된 형식일 수도 있으므로, 실패해도 계속 진행
    try:
        base = PeftModel.from_pretrained(base, ckpt)
    except Exception:
        pass
    tok = AutoTokenizer.from_pretrained(ckpt, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok, base


def load_aux_mem(ckpt: str, hidden_size: int, device: torch.device):
    path = os.path.join(ckpt, 'aux_mem.pt')
    if not os.path.exists(path):
        print('[경고] aux_mem.pt가 없어 보조헤드/메모리 파라미터를 초기값으로 사용합니다.', file=sys.stderr)
        aux = AuxHeads(hidden_size=hidden_size, embed_dim=768).to(device)
        embed_tok = None; embed_model = None
        return aux, None, None
    state = torch.load(path, map_location=device)
    aux = AuxHeads(hidden_size=hidden_size, embed_dim=768).to(device)
    aux.load_state_dict(state['aux'])
    # MemoryManager 파라미터는 학습 시점 저장 형식에 따라 다르므로, 로드 실패시 초기화
    embed_tok = None; embed_model = None
    try:
        from transformers import AutoTokenizer as _AT, AutoModel as _AM
        embed_tok = _AT.from_pretrained('jhgan/ko-sroberta-multitask')
        embed_model = _AM.from_pretrained('jhgan/ko-sroberta-multitask').to(device)
    except Exception:
        pass
    return aux, embed_tok, embed_model


# ------------------------
# Inference Loop
# ------------------------
class InferSession:
    def __init__(self, ckpt: str, max_new_tokens: int=256, temperature: float=0.7, top_p: float=0.95):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tok, self.model = load_model(ckpt)
        self.model.eval()
        self.hidden = self.model.config.hidden_size
        self.aux, embed_tok, embed_model = load_aux_mem(ckpt, self.hidden, self.device)
        self.aux.eval()
        self.mem = MemoryManager(MemConfig(), embed_model, embed_tok, self.hidden).to(self.device)
        self.summarizer = LocalSummarizer(model_name=ckpt)
        self.dialog_history: List[str] = []  # 원문 기록
        self.max_new = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def _pooled_last(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # 마지막 64 토큰 평균
        return inputs_embeds[:, -64:, :].mean(dim=1)

    def _predict_boundary(self, pooled: torch.Tensor) -> int:
        with torch.no_grad():
            logit, _ = self.aux(pooled)
            prob = torch.sigmoid(logit)[0].item()
        # 간단 임계치 0.5, 실패 시 룰 백업
        try:
            return 1 if prob >= 0.5 else 0
        except Exception:
            return 0

    @torch.no_grad()
    def _gen(self, prompt_ids: torch.Tensor, prefix: torch.Tensor):
        # prefix: [1, r, H]; prompt_ids -> inputs_embeds
        inp_emb = self.model.get_input_embeddings()(prompt_ids.to(self.device))
        inp = torch.cat([prefix, inp_emb], dim=1)
        out = self.model.generate(
            inputs_embeds=inp,
            max_new_tokens=self.max_new,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id,
        )
        text = self.tok.decode(out[0], skip_special_tokens=True)
        return text

    def step(self, user_text: str) -> str:
        # 1) 입력 정규화/버퍼 반영
        self.dialog_history.append(f"[A:] {user_text}")
        self.mem.stm_append(f"[A:] {user_text}")
        # 2) STM 길이 초과 시 중간 요약 → 버퍼 치환
        self.mem.stm_summarize_if_needed(self.tok, lambda mid: self.summarizer.summarize(mid))
        # 3) 현재 세그먼트 임시 요약(응답 질 개선용)
        seg_hint = self.summarizer.summarize('\n'.join(self.mem.stm_text[-8:])) if self.mem.stm_text else ""
        # 4) prefix 생성: m_t가 없으면 seg_hint로 초기화
        if (self.mem.m_t is None or self.mem.m_t.sum()==0) and seg_hint:
            try:
                e = self.mem.encode(seg_hint)
                self.mem.m_t = self.mem.mlp(torch.cat([torch.zeros_like(e), e], dim=-1))
            except Exception:
                pass
        prefix = self.mem.proj_prefix(self.hidden).unsqueeze(0)  # [1,r,H]

        # 5) 프롬프트 구성
        sys_prompt = (
            "당신은 한국어 비서입니다. 최근 대화의 요지를 반영해 간결하고 정확하게 답하세요."
        )
        context = '\n'.join(self.dialog_history[-20:])
        prompt = f"<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n{context}\n[B:] "
        ids = self.tok(prompt, return_tensors='pt')['input_ids']

        # boundary 예측을 위해 임시 forward
        with torch.no_grad():
            tmp_emb = self.model.get_input_embeddings()(ids.to(self.device))
            pooled = self._pooled_last(torch.cat([prefix, tmp_emb], dim=1))
        boundary = self._predict_boundary(pooled)

        # 6) 경계면 LTM 전이(세그 요약 생성 → m_t 갱신)
        if boundary == 1:
            seg_summary = self.summarizer.summarize('\n'.join(self.mem.stm_text))
            self.mem.ltm_transfer(1, seg_summary)
            prefix = self.mem.proj_prefix(self.hidden).unsqueeze(0)

        # 7) 생성
        out = self._gen(ids, prefix)
        # 모델 출력 후 사용자 프롬프트까지 포함될 수 있으므로, 마지막 [B:] 이후만 취함
        if '[B:]' in out:
            resp = out.split('[B:]')[-1].strip()
        else:
            resp = out.strip()
        # 8) 히스토리/STM에 반영
        self.dialog_history.append(f"[B:] {resp}")
        self.mem.stm_append(f"[B:] {resp}")
        return resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, default='out/solar-mem-qlora')
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top_p', type=float, default=0.95)
    args = ap.parse_args()

    sess = InferSession(args.ckpt, args.max_new_tokens, args.temperature, args.top_p)

    print('메모리 추론 데모 시작. /exit 로 종료, /reset 으로 메모리 초기화, /state 로 내부 상태 확인')
    while True:
        try:
            user = input('\n사용자 > ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\n종료.'); break
        if not user:
            continue
        if user == '/exit':
            print('종료.'); break
        if user == '/reset':
            sess.mem.m_t.zero_(); sess.mem.stm_text.clear(); sess.dialog_history.clear(); print('메모리 리셋 완료.'); continue
        if user == '/state':
            print(f"STM_len={len(sess.mem.stm_text)}, m_t_norm={(sess.mem.m_t.norm().item() if sess.mem.m_t is not None else 0):.4f}")
            print('최근 STM tail:')
            for t in sess.mem.stm_text[-6:]:
                print('  ', t)
            continue
        resp = sess.step(user)
        print(f"모델  > {resp}")

if __name__ == '__main__':
    main()
