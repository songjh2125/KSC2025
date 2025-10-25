# summarizer_local.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

OUT_MAX_CHARS_SUM = 120
OUT_MAX_CHARS_KU  = 220

SYSTEM_SUM = "당신은 한국어 대화 요약기입니다. 핵심만 2~3문장, 최대 120자 내로 요약하세요. 불필요한 감탄사는 제거합니다."
SYSTEM_KU  = (
    "당신은 대화 속 핵심 사실을 '지식 단위'로 정제하는 한국어 편집자입니다. "
    "사실관계만 유지하고 군더더기 없이 180~220자로 쓰세요. "
    "고유명사, 정의, 배경 1~2문장, 관습/사례는 간략히."
)

class LocalSummarizer:
    def __init__(
        self,
        model_name: str = "Upstage/SOLAR-10.7B-Instruct-v1.0",
        max_new_tokens: int = 128,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        try_bfloat16: bool = True,
    ):
        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        qconf = None
        torch_dtype = torch.bfloat16 if try_bfloat16 and torch.cuda.is_available() else torch.float16

        if load_in_8bit or load_in_4bit:
            qconf = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
            quantization_config=qconf,
            low_cpu_mem_usage=True,
        )
        self.max_new = max_new_tokens

    @torch.no_grad()
    def summarize(self, text: str) -> str:
        prompt = f"<s>[INST] <<SYS>>\n{SYSTEM_SUM}\n<</SYS>>\n아래 내용을 '문자 수 기준' 최대 {OUT_MAX_CHARS_SUM}자로 요약:\n{text}\n[/INST]"
        ids = self.tok(prompt, return_tensors='pt').to(self.model.device)
        out = self.model.generate(**ids, max_new_tokens=256, do_sample=False, repetition_penalty=1.07, eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.pad_token_id)
        ans = self.tok.decode(out[0], skip_special_tokens=True).split('[/INST]')[-1].strip()
        return ans[:OUT_MAX_CHARS_SUM]

    @torch.no_grad()
    def to_knowledge_unit(self, text: str, target_min=180, target_max=220) -> str:
        prompt = (f"<s>[INST] <<SYS>>\n{SYSTEM_KU}\n<</SYS>>\n"
                  f"아래 내용을 '문자 수 기준' {target_min}~{target_max}자로 1개 지식 단위로 정리:\n{text}\n[/INST]")
        ids = self.tok(prompt, return_tensors='pt').to(self.model.device)
        out = self.model.generate(**ids, max_new_tokens=512, do_sample=False, repetition_penalty=1.07, eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.pad_token_id)
        ans = self.tok.decode(out[0], skip_special_tokens=True).split('[/INST]')[-1].strip()
        if len(ans) < target_min:
            aux = self.summarize(text)
            ans = (ans + " " + aux).strip()
        return ans[:target_max]
