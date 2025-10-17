# txl_hf/mem_baseline.py
import torch.nn.functional as F
from txl.mem_transformer import MemTransformerLM
from typing import Optional

class HFMemBaseline(MemTransformerLM):
    """
    Hugging Face Trainer 인터페이스에 맞춘 원본 TXL 래퍼
    """
    def forward(self, input_ids, labels: Optional=None, topic_boundary: Optional=None, **kwargs):
        out = super().forward(input_ids=input_ids)  # 원본은 topic_boundary 없음
        logits = out["logits"] if isinstance(out, dict) else out
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        return {"loss": loss, "logits": logits}