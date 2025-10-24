# data_utils.py
import re

# 가벼운 정규화(외부 의존 최소화). 고급 교정은 hanspell/ekorespell 등을 연결해도 됨.

_JOSA_RULES = [
    (r"(으로|로)\s+가?", r"\1 "),
]

def basic_normalize(text: str) -> str:
    t = text
    t = re.sub(r"\s+", " ", t)
    t = t.replace(' ,', ',').replace(' .', '.')
    for pat, rep in _JOSA_RULES:
        t = re.sub(pat, rep, t)
    return t.strip()

try:
    from hanspell import spell_checker
    def spell_normalize(text: str) -> str:
        try:
            res = spell_checker.check(text)
            return res.checked
        except Exception:
            return basic_normalize(text)
except Exception:
    def spell_normalize(text: str) -> str:
        return basic_normalize(text)