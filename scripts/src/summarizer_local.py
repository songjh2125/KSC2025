# summarizer_local.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

SYSTEM = (
    "당신은 한국어 대화 요약기입니다. 핵심만 2~3문장, 최대 120자 내로 요약하세요. 불필요한 감탄사는 제거합니다."
)

class LocalSummarizer:
    def __init__(self, model_name: str="Upstage/SOLAR-10.7B-Instruct-v1.0", max_new_tokens: int=64, device='cuda'):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map='auto')
        self.max_new = max_new_tokens
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
    @torch.no_grad()
    def summarize(self, text: str) -> str:
        prompt = f"<s>[INST] <<SYS>>\n{SYSTEM}\n<</SYS>>\n다음 내용을 요약:\n{text}\n[/INST]"
        ids = self.tok(prompt, return_tensors='pt').to(self.model.device)
        out = self.model.generate(**ids, max_new_tokens=self.max_new, do_sample=False)
        ans = self.tok.decode(out[0], skip_special_tokens=True)
        # post-trim to ~120 chars
        ans = ans.split('[/INST]')[-1].strip()
        return ans[:120]