# infer.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
mem 전용 추론 스크립트

- STM 버퍼: 길이-적응 요약(Chunk)로 관리
- boundary==1 감지되면 STM 전체 요약→LTM 전이(200자 지식카드)
- 응답 시 LTM에서 Top-1(임계치 이상)만 골라 KV-priming → 사용자 프롬프트 이어서 생성

런타임 명령:
  /thr [값]    : LTM 검색 임계치(0~1) 조회/설정
  /bthr [값]   : 토픽 경계 임계치(0~1) 조회/설정
  /state       : 현재 상태(임계치 포함) 보기
  /reset       : STM/히스토리 초기화
  /exit        : 종료
"""

import os
import json
import argparse
import datetime
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import BitsAndBytesConfig
from peft import PeftModel

from src.mem_modules import MemConfig, MemoryManager
from src.summarizer_local import LocalSummarizer


# ---- 유틸: LoRA 어댑터에서 base 경로 추출 ----
def _maybe_read_base_from_adapter(ckpt_dir: str) -> Optional[str]:
    """LoRA 어댑터 폴더의 adapter_config.json에서 base 모델 경로를 추출."""
    cfg_path = os.path.join(ckpt_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k in ["base_model_name_or_path", "base_model_name"]:
            if isinstance(cfg.get(k), str) and cfg[k]:
                return cfg[k]
    except Exception:
        pass
    return None


# ---- 보조헤드 (경계/요약임베딩) ----
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


# ---- 모델/토크나이저 로드 ----
def load_model_and_tokenizer(ckpt_dir: str, load_4bit: bool = False):
    """
    ckpt_dir 가 LoRA 어댑터 디렉터리일 수도, 전체 모델 디렉터리일 수도 있음.
    1) adapter_config.json이 있으면 → base 모델 경로를 읽어 base 로드 + 어댑터 merge
    2) 없으면 → 그 경로 자체를 base로 간주하고 로드(어댑터 없음)
    """
    base_model_path = _maybe_read_base_from_adapter(ckpt_dir) or ckpt_dir

    quant_cfg = None
    if load_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
                else torch.float16
            ),
        )

    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16 if quant_cfg is None else None,
        quantization_config=quant_cfg,
    )
    tok = AutoTokenizer.from_pretrained(base_model_path, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # LoRA 어댑터가 있으면 로드
    adapter_cfg = os.path.join(ckpt_dir, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        try:
            base = PeftModel.from_pretrained(base, ckpt_dir)
        except Exception as e:
            print(f"[warn] LoRA adapter load failed: {e} (continue with base model)")

    # 추론용 설정
    base.eval()
    try:
        base.config.output_hidden_states = True
        base.config.use_cache = True
    except Exception:
        pass

    return tok, base


def load_aux(ckpt_dir: str, hidden_size: int, device: torch.device):
    """
    학습 코드에서는 aux만 'aux.pt'로 저장했으므로 여기서 그것만 로드.
    """
    aux_path = os.path.join(ckpt_dir, "aux.pt")

    # 추론 임베딩 모델 (학습 시 사용한 것과 동일 권장)
    embed_model_name = "jhgan/ko-sroberta-multitask"
    embed_tok = AutoTokenizer.from_pretrained(embed_model_name)
    embed_model = AutoModel.from_pretrained(embed_model_name).to(device)
    for p in embed_model.parameters():
        p.requires_grad = False
    emb_dim = getattr(embed_model.config, "hidden_size", 768)

    aux = AuxHeads(hidden_size=hidden_size, embed_dim=emb_dim).to(device)
    if os.path.exists(aux_path):
        state = torch.load(aux_path, map_location=device)
        aux.load_state_dict(state["aux"])
    else:
        print(f"[warn] aux.pt not found in {ckpt_dir}. Proceeding with randomly initialized AuxHeads.")

    aux.eval()
    return aux, embed_tok, embed_model, emb_dim


# ---- 추론 세션 (mem-only) ----
class InferSession:
    def __init__(
        self,
        ckpt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        thr: float = 0.60,       # LTM 검색 임계치 (코사인 유사도)
        bthr: float = 0.50,      # 경계 임계치 (시그모이드 확률)
        summarizer_model: str = None,
        sum_8bit: bool = False,
        sum_4bit: bool = False,
        load_4bit: bool = False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tok, self.model = load_model_and_tokenizer(ckpt, load_4bit=load_4bit)
        self.hidden = self.model.config.hidden_size

        # mem 구성요소
        self.aux, embed_tok, embed_model, self.emb_dim = load_aux(ckpt, self.hidden, self.device)
        self.mem = MemoryManager(MemConfig(), embed_model, embed_tok)

        base_for_summarizer = summarizer_model or _maybe_read_base_from_adapter(ckpt) or ckpt
        self.summarizer = LocalSummarizer(
            model_name=base_for_summarizer,
            max_new_tokens=128,
            load_in_8bit=sum_8bit,
            load_in_4bit=sum_4bit,
            try_bfloat16=True,
        )

        self.dialog_history: List[str] = []
        self.max_new = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # 임계치(런타임 조정 가능)
        self._ltm_threshold = float(thr)
        self._boundary_threshold = float(bthr)

    # ----- 임계치 세터/게터 -----
    def set_ltm_threshold(self, thr: float) -> float:
        thr = max(0.0, min(1.0, float(thr)))
        self._ltm_threshold = thr
        return self._ltm_threshold

    def get_ltm_threshold(self) -> float:
        return float(self._ltm_threshold)

    def set_boundary_threshold(self, thr: float) -> float:
        thr = max(0.0, min(1.0, float(thr)))
        self._boundary_threshold = thr
        return self._boundary_threshold

    def get_boundary_threshold(self) -> float:
        return float(self._boundary_threshold)

    # ---------- STM→LTM ----------
    def _commit_stm_to_ltm(self):
        if self.mem is None or self.summarizer is None:
            return
        seg_text = "\n".join(self.mem.stm_text)
        if not seg_text.strip():
            return
        ku = self.summarizer.to_knowledge_unit(seg_text)  # 180~220자
        topic_key = self.summarizer.summarize(seg_text)   # 1~2문장
        self.mem.add_ltm(topic_key=topic_key, text=ku, meta={"source": "runtime"})

    def _retrieve_top1_card(self) -> Optional[str]:
        """
        STM tail 기반으로 LTM Top-1 검색 (임계치 미달 시 None).
        """
        if self.mem is None or not self.mem.stm_text:
            return None

        tail = "\n".join(self.mem.stm_text[-10:])
        topic_key = self.summarizer.summarize("\n".join(self.mem.stm_text)) if self.mem.stm_text else ""
        q_tail = self.summarizer.summarize(tail) if tail else ""
        q_key = topic_key

        best_entry, best_sim = None, -1.0
        for q in [q_tail, q_key]:
            if not q:
                continue
            z = self.mem.retrieve_top1(q, thr=self._ltm_threshold)
            if not z:
                continue
            # 유사도 재평가
            q_emb = self.mem.encode(q)
            if not isinstance(q_emb, torch.Tensor):
                q_emb = torch.tensor(q_emb)
            q_emb = q_emb.detach().cpu().numpy()
            qn = q_emb / (np.linalg.norm(q_emb) + 1e-9)
            zn = z.emb / (np.linalg.norm(z.emb) + 1e-9)
            sim = float(np.dot(qn, zn))
            if sim > best_sim:
                best_entry, best_sim = z, sim

        return best_entry.text if (best_entry and best_sim >= self._ltm_threshold) else None

    # ---------- KV-priming ----------
    @torch.no_grad()
    def _kv_prime_with_memory_and_prompt(self, memory_text: str, prompt_ids: torch.Tensor):
        if not memory_text:
            return None, None
        mem_prompt = (
            "<s>[INST] <<SYS>>\n"
            "다음 <MEM>은 참고지식이다. 모델 출력에 직접 인용/복사하지 말고 사실관계를 반영하라.\n"
            "<</SYS>>\n<MEM>\n" + memory_text + "\n</MEM>\n[/INST]"
        )
        mem_ids = self.tok(mem_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
        out_mem = self.model(input_ids=mem_ids, use_cache=True)
        pkv_mem = out_mem.past_key_values

        out_ctx = self.model(input_ids=prompt_ids.to(self.device), past_key_values=pkv_mem, use_cache=True)
        pkv_ctx = out_ctx.past_key_values
        last_tok = prompt_ids[:, -1:].to(self.device)
        return pkv_ctx, last_tok

    @torch.no_grad()
    def _gen_with_primed_kv(self, last_token_ids: torch.Tensor, pkv_ctx):
        out = self.model.generate(
            input_ids=last_token_ids,
            past_key_values=pkv_ctx,
            max_new_tokens=self.max_new,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id,
            use_cache=True,
        )
        return self.tok.decode(out[0], skip_special_tokens=True)

    # ---------- 단일 스텝 ----------
    def step(self, user_text: str) -> str:
        # 1) STM 누적
        self.dialog_history.append(f"[A:] {user_text}")
        self.mem.stm_append(f"[A:] {user_text}")

        # 2) 길이-적응 요약(Chunk)
        self.mem.stm_summarize_if_needed(self.tok, lambda mid: self.summarizer.summarize(mid))

        # 3) 프롬프트 구성
        sys_prompt = "당신은 한국어 비서입니다. 최근 대화의 요지를 반영해 간결하고 정확하게 답하세요."
        context = "\n".join(self.dialog_history[-20:])
        prompt = f"<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n{context}\n[B:] "
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(self.device)
        attn = (ids != self.tok.pad_token_id).long()

        # 4) boundary 예측
        with torch.no_grad():
            out_tmp = self.model(input_ids=ids, attention_mask=attn, output_hidden_states=True)
            pooled = out_tmp.hidden_states[-1][:, -64:, :].mean(dim=1)
            b_logit, _ = self.aux(pooled)
            prob = torch.sigmoid(b_logit)[0].item()
        boundary = 1 if prob >= self._boundary_threshold else 0

        # 5) 새 토픽이면 STM→LTM 전이
        if boundary == 1:
            self._commit_stm_to_ltm()

        # 6) LTM Top-1 검색
        mem_card = self._retrieve_top1_card()

        # 7) KV-priming (있을 때만)
        if mem_card:
            pkv_ctx, last_tok = self._kv_prime_with_memory_and_prompt(mem_card, ids)
            out_text = self._gen_with_primed_kv(last_tok, pkv_ctx)
        else:
            out_ids = self.model.generate(
                input_ids=ids,
                attention_mask=attn,
                max_new_tokens=self.max_new,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                eos_token_id=self.tok.eos_token_id,
                pad_token_id=self.tok.pad_token_id,
            )
            out_text = self.tok.decode(out_ids[0], skip_special_tokens=True)

        resp = out_text.split("[B:]")[-1].strip() if "[B:]" in out_text else out_text.strip()
        self.dialog_history.append(f"[B:] {resp}")
        self.mem.stm_append(f"[B:] {resp}")
        return resp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="out/solar-mem-qlora",
                    help="LoRA 어댑터 디렉터리 또는 전체 모델 디렉터리")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--thr", type=float, default=0.60, help="LTM retrieval 임계치 (코사인 유사도)")
    ap.add_argument("--bthr", type=float, default=0.50, help="경계(boundary) 임계치 (시그모이드 확률)")
    ap.add_argument("--summarizer_model", type=str,
                    default=os.environ.get("SUMMARIZER_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
                    help="요약용 경량 모델 (기본: TinyLlama)")
    ap.add_argument("--sum_8bit", action="store_true", help="요약기 8bit 로드")
    ap.add_argument("--sum_4bit", action="store_true", help="요약기 4bit 로드")
    ap.add_argument("--load_4bit", action="store_true", help="본 모델 4bit QLoRA 로드")
    ap.add_argument("--thresholds_json", type=str, default=None,
                    help="DEV에서 저장한 {\"bthr\": float, \"thr\": float} JSON 경로")
    args = ap.parse_args()

    # 로그 디렉터리 (항상 mem)
    subdir = "mem"
    os.makedirs(os.path.join("log", subdir), exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    infer_log_path = os.path.join("log", subdir, f"infer_{ts}.txt")
    with open(infer_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "event": "infer_start",
            "timestamp": ts,
            "ckpt": args.ckpt,
            "mode": "mem",
            "load_4bit": args.load_4bit
        }, ensure_ascii=False) + "\n")

    # thresholds_json 로드(있으면 CLI 기본값을 덮어씀)
    if args.thresholds_json:
        try:
            cfg = json.load(open(args.thresholds_json, "r", encoding="utf-8"))
            if "bthr" in cfg and cfg["bthr"] is not None:
                args.bthr = float(cfg["bthr"])
            if "thr" in cfg and cfg["thr"] is not None:
                args.thr = float(cfg["thr"])
            print(f"[thresholds_json] loaded: bthr={args.bthr:.4f}, thr={args.thr:.4f}")
        except Exception as e:
            print(f"[thresholds_json] load failed: {e} (ignored)")

    # 세션 생성
    sess = InferSession(
        ckpt=args.ckpt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        thr=args.thr,
        bthr=args.bthr,
        summarizer_model=args.summarizer_model,
        sum_8bit=args.sum_8bit,
        sum_4bit=args.sum_4bit,
        load_4bit=args.load_4bit,
    )

    print("메모리 추론 데모 시작.")
    print("명령: /exit 종료, /reset 초기화, /state 상태, /thr [값] LTM 임계치, /bthr [값] 경계 임계치")

    while True:
        try:
            user = input("\n사용자 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료.")
            break

        if not user:
            continue

        # 로그: user 입력
        with open(infer_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "user", "text": user}, ensure_ascii=False) + "\n")

        # ----- 런타임 임계치 조정 -----
        if user.startswith("/thr"):
            parts = user.split()
            if len(parts) == 1:
                print(f"현재 LTM 임계치(thr) = {sess.get_ltm_threshold():.2f}")
            else:
                try:
                    cur = sess.set_ltm_threshold(float(parts[1]))
                    print(f"LTM 임계치(thr)를 {cur:.2f} 로 설정했습니다.")
                except ValueError:
                    print("형식: /thr 0.60  (0.0~1.0 사이 실수)")
            with open(infer_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "cmd", "cmd": user}, ensure_ascii=False) + "\n")
            continue

        if user.startswith("/bthr"):
            parts = user.split()
            if len(parts) == 1:
                print(f"현재 경계 임계치(bthr) = {sess.get_boundary_threshold():.2f}")
            else:
                try:
                    cur = sess.set_boundary_threshold(float(parts[1]))
                    print(f"경계 임계치(bthr)를 {cur:.2f} 로 설정했습니다.")
                except ValueError:
                    print("형식: /bthr 0.50  (0.0~1.0 사이 실수)")
            with open(infer_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "cmd", "cmd": user}, ensure_ascii=False) + "\n")
            continue
        # --------------------------------

        if user == "/exit":
            print("\n종료.")
            with open(infer_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "infer_end"}, ensure_ascii=False) + "\n")
            break

        if user == "/reset":
            sess.mem.stm_text.clear()
            sess.dialog_history.clear()
            print("리셋 완료.")
            with open(infer_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "cmd", "cmd": user}, ensure_ascii=False) + "\n")
            continue

        if user == "/state":
            print(f"STM_len={len(sess.mem.stm_text)}")
            print("최근 STM tail:")
            for t in sess.mem.stm_text[-6:]:
                print("  ", t)
            print(f"LTM 임계치(thr): {sess.get_ltm_threshold():.2f}")
            print(f"경계 임계치(bthr): {sess.get_boundary_threshold():.2f}")
            with open(infer_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"event": "cmd", "cmd": user}, ensure_ascii=False) + "\n")
            continue

        # 실제 응답 생성
        resp = sess.step(user)
        print(f"모델  > {resp}")
        with open(infer_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "model", "text": resp}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
