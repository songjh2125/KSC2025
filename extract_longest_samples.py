#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
가장 긴 토큰 길이를 가진 8개 샘플을 추출하는 스크립트
학습 코드와 동일한 토크나이저를 사용하여 정확한 토큰 길이를 측정합니다.
"""

import json
import os
import re
from pathlib import Path
from transformers import AutoTokenizer

# 정규화 함수 (src/data_utils.py에서 가져옴)
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

def extract_longest_samples():
    """가장 긴 토큰 길이를 가진 8개 샘플을 추출합니다."""
    
    # 학습 코드와 동일한 토크나이저 로드
    print("토크나이저를 로딩 중...")
    base_model = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("토크나이저 로딩 완료!")
    
    # 설정
    max_length = 1024  # 학습 코드의 기본값
    train_path = "data/train_data.jsonl"
    output_path = "data/longest_8_samples.jsonl"
    
    # 샘플들을 저장할 리스트 (토큰 길이와 함께)
    samples_with_length = []
    
    print(f"데이터 파일을 읽는 중: {train_path}")
    
    try:
        with open(train_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    ex = json.loads(line)
                    
                    # 학습 코드와 동일한 전처리
                    turns = ex.get("text", []) or []
                    turns = [basic_normalize(t) for t in turns]  # basic normalizer 사용
                    dialogue = "\n".join(turns)
                    
                    # 토큰화 (학습 코드와 동일한 방식)
                    enc = tokenizer(
                        dialogue,
                        truncation=True,
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    ids = enc["input_ids"].squeeze(0)
                    token_length = len(ids)
                    
                    # 원본 데이터와 토큰 길이를 함께 저장
                    samples_with_length.append({
                        'original_data': ex,
                        'token_length': token_length,
                        'dialogue': dialogue
                    })
                    
                    if line_num % 1000 == 0:
                        print(f"  처리된 라인: {line_num:,}")
                        
                except json.JSONDecodeError as e:
                    print(f"  경고: {line_num}번째 줄 JSON 파싱 오류: {e}")
                    continue
                except Exception as e:
                    print(f"  경고: {line_num}번째 줄 처리 오류: {e}")
                    continue
    
    except FileNotFoundError:
        print(f"오류: {train_path} 파일을 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"오류: 파일 읽기 중 오류 발생: {e}")
        return
    
    print(f"총 {len(samples_with_length)}개 샘플을 처리했습니다.")
    
    # 토큰 길이 기준으로 정렬 (내림차순)
    samples_with_length.sort(key=lambda x: x['token_length'], reverse=True)
    
    # 가장 긴 8개 선택
    longest_8 = samples_with_length[:8]
    
    print("\n가장 긴 8개 샘플의 토큰 길이:")
    for i, sample in enumerate(longest_8, 1):
        print(f"  {i}. 토큰 길이: {sample['token_length']}")
        print(f"     대화 길이: {len(sample['dialogue'])} 문자")
        print(f"     ID: {sample['original_data'].get('id', 'N/A')}")
        print()
    
    # 결과를 JSONL 파일로 저장
    print(f"결과를 {output_path}에 저장 중...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in longest_8:
            # 원본 데이터를 그대로 저장
            f.write(json.dumps(sample['original_data'], ensure_ascii=False) + '\n')
    
    print(f"저장 완료: {output_path}")
    
    # 통계 정보 출력
    print("\n" + "="*60)
    print("통계 정보")
    print("="*60)
    print(f"전체 샘플 수: {len(samples_with_length):,}")
    print(f"최대 토큰 길이: {max(s['token_length'] for s in samples_with_length)}")
    print(f"최소 토큰 길이: {min(s['token_length'] for s in samples_with_length)}")
    print(f"평균 토큰 길이: {sum(s['token_length'] for s in samples_with_length) / len(samples_with_length):.2f}")
    
    # 1024 토큰 초과 샘플 수
    over_1024 = sum(1 for s in samples_with_length if s['token_length'] > 1024)
    print(f"1024 토큰 초과 샘플: {over_1024:,} ({over_1024/len(samples_with_length)*100:.1f}%)")
    
    print(f"\n선택된 8개 샘플의 토큰 길이 범위: {longest_8[-1]['token_length']} ~ {longest_8[0]['token_length']}")
    
    return longest_8

if __name__ == "__main__":
    print("가장 긴 토큰 길이를 가진 8개 샘플을 추출합니다...")
    samples = extract_longest_samples()
    print("\n추출이 완료되었습니다!")
