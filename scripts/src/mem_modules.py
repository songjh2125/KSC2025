# mem_modules.py
import torch, torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional
import uuid, time
import numpy as np

@dataclass
class MemConfig:
    # 길이-적응 요약(Chunk 요약) 파라미터
    T_chunk = 2048              # 총 토큰 수가 이 값을 초과하면 요약 수행
    overlap = 128               # 머리 부분은 원문 유지
    RAW_BUDGET = 256            # 꼬리 원문 유지 예산 (최근 원문은 보존)

@dataclass
class LTMEntry:
    id: str
    topic_key: str
    text: str        # 180~220자 지식 단위
    emb: np.ndarray  # 768-d float (numpy)
    ts: float
    meta: dict = field(default_factory=dict)

class MemoryManager(nn.Module):
    """
    - STM: 토픽 내 발화 누적 버퍼 + 길이-적응 요약
    - LTM: 200자 지식카드 저장 및 Top-1 검색
    """
    def __init__(self, mem_cfg: MemConfig, embed_model, embed_tokenizer):
        super().__init__()
        self.cfg = mem_cfg
        self.embed_model = embed_model
        self.embed_tok = embed_tokenizer
        self.stm_text: List[str] = []
        self.ltm: List[LTMEntry] = []
        self._stm_cache = {}  # mid 텍스트→요약 결과 캐시

    # -------- Embedding --------
    @torch.no_grad()
    def encode(self, text: str):
        toks = self.embed_tok(text, return_tensors='pt', truncation=True, padding=True).to(next(self.embed_model.parameters()).device)
        out = self.embed_model(**toks)
        return out.last_hidden_state.mean(dim=1).squeeze(0)

    # -------- STM --------
    def stm_append(self, utter: str):
        self.stm_text.append(utter)

    def _stm_token_ids(self, tokenizer):
        return tokenizer('\n'.join(self.stm_text))['input_ids']

    def stm_summarize_if_needed(self, tokenizer, summarize_fn) -> Optional[str]:
        """
        총 길이가 T_chunk 초과 시:
          - head=overlap, tail=RAW_BUDGET 를 원문 유지
          - mid만 요약하여 치환
          - 동일 mid에 대한 재요약은 캐시로 스킵
        """
        tok_ids = self._stm_token_ids(tokenizer)
        if len(tok_ids) <= self.cfg.T_chunk:
            return None

        head = tok_ids[: self.cfg.overlap]
        tail = tok_ids[-min(len(tok_ids), self.cfg.RAW_BUDGET):]
        mid  = tok_ids[len(head): len(tok_ids)-len(tail)]
        if len(mid) == 0:
            return None

        mid_text = tokenizer.decode(mid)
        summary = self._stm_cache.get(mid_text)
        if summary is None:
            summary = summarize_fn(mid_text)
            self._stm_cache[mid_text] = summary

        summary = "\n" + summary.strip().replace("\n", " ") + "\n"

        summ_ids = tokenizer(summary, add_special_tokens=False)['input_ids']
        room = max(0, self.cfg.T_chunk - len(head) - len(tail))
        summ_ids = summ_ids[:room]

        new_ids = head + summ_ids + tail
        self.stm_text = tokenizer.decode(new_ids).split('\n')
        return summary.strip()

    # -------- LTM --------
    @torch.no_grad()
    def add_ltm(self, topic_key: str, text: str, meta=None) -> str:
        e = self.encode(text).detach().cpu().numpy()
        ent = LTMEntry(str(uuid.uuid4()), topic_key, text, e, time.time(), meta or {})
        self.ltm.append(ent)
        return ent.id

    def retrieve_top1(self, query_text: str, thr: float=0.40) -> Optional[LTMEntry]:
        if not self.ltm:
            return None
        q = self.encode(query_text).detach().cpu().numpy()
        qn = q / (np.linalg.norm(q) + 1e-9)
        best, best_sim = None, -1.0
        for z in self.ltm:
            zn = z.emb / (np.linalg.norm(z.emb) + 1e-9)
            sim = float(np.dot(qn, zn))
            if sim > best_sim:
                best, best_sim = z, sim
        return best if best_sim >= thr else None
