# mem_modules.py
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class MemConfig:
    T_chunk: int = 256
    overlap: int = 64
    T_summary_trigger: int = 220
    RAW_BUDGET: int = 512
    r_prefix: int = 4
    mem_hidden: int = 768  # ko-sroberta hidden size

class MemoryManager(nn.Module):
    def __init__(self, mem_cfg: MemConfig, embed_model, embed_tokenizer, hidden_size: int):
        super().__init__()
        self.cfg = mem_cfg
        self.embed_model = embed_model
        self.embed_tok = embed_tokenizer
        self.mlp = nn.Sequential(nn.Linear(self.cfg.mem_hidden*2, self.cfg.mem_hidden), nn.Tanh())
        self.W = nn.Linear(self.cfg.mem_hidden, self.cfg.r_prefix*hidden_size, bias=False)
        self.register_buffer('m_t', torch.zeros(self.cfg.mem_hidden))
        self.stm_text: List[str] = []

    @torch.no_grad()
    def encode(self, text: str):
        toks = self.embed_tok(text, return_tensors='pt', truncation=True, padding=True).to(next(self.embed_model.parameters()).device)
        out = self.embed_model(**toks)
        return out.last_hidden_state.mean(dim=1).squeeze(0)

    def stm_append(self, utter: str):
        self.stm_text.append(utter)

    def stm_token_length(self, tokenizer) -> int:
        return len(tokenizer('\n'.join(self.stm_text))['input_ids'])

    def stm_summarize_if_needed(self, tokenizer, summarize_fn) -> Optional[str]:
        if self.stm_token_length(tokenizer) <= self.cfg.T_summary_trigger:
            return None
        ids = tokenizer('\n'.join(self.stm_text))['input_ids']
        head = ids[: self.cfg.overlap]
        tail = ids[-min(len(ids), self.cfg.RAW_BUDGET):]
        mid = ids[len(head): len(ids)-len(tail)]
        mid_text = tokenizer.decode(mid)
        summary = summarize_fn(mid_text)
        new_ids = head + tokenizer(summary)['input_ids'] + tail
        self.stm_text = tokenizer.decode(new_ids).split('\n')
        return summary

    def ltm_transfer(self, boundary: int, segment_summary: Optional[str]):
        if boundary!=1 or not segment_summary:
            return None
        with torch.no_grad():
            e_t = self.encode(segment_summary)
        m_prev = self.m_t if self.m_t is not None else torch.zeros_like(e_t)
        self.m_t = self.mlp(torch.cat([m_prev, e_t], dim=-1))
        return self.m_t

    def proj_prefix(self, hidden_size: int) -> torch.Tensor:
        m = self.m_t
        if m is None or m.numel()==0:
            return torch.zeros(self.cfg.r_prefix, hidden_size, device=self.W.weight.device)
        proj = self.W(m)  # [r*H]
        return proj.view(self.cfg.r_prefix, hidden_size)

class AuxHeads(nn.Module):
    def __init__(self, hidden_size: int, embed_dim: int=768):
        super().__init__()
        self.boundary = nn.Sequential(nn.Linear(hidden_size, hidden_size//2), nn.Tanh(), nn.Linear(hidden_size//2, 1))
        self.summary_head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, embed_dim))
    def forward(self, pooled: torch.Tensor):
        b_logit = self.boundary(pooled).squeeze(-1)
        e_pred = self.summary_head(pooled)
        return b_logit, e_pred