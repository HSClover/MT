# -*- coding: utf-8 -*-
"""kiwipiepy가 refiner(문장 생성/보정)에 실제로 쓸 수 있는지 검증."""
from kiwipiepy import Kiwi

kiwi = Kiwi()

print("=== 1) 조사 자동 판별 (손으로 받침 규칙 짤 필요 있나?) ===")
for probe in ["돈를", "돈을", "아이가", "아이을", "학교가", "학교을", "친구를", "친구을"]:
    toks = kiwi.tokenize(probe)
    tags = [(t.form, t.tag) for t in toks]
    print(f"  {probe:8s} -> {tags}")

print()
print("=== 2) 불규칙 활용 동사의 어근 추출 (사전 없이 가능한가?) ===")
for verb in ["알다", "듣다", "뛰다", "사다", "울다", "주다", "마시다", "기다리다"]:
    toks = kiwi.tokenize(verb)
    print(f"  {verb:8s} -> {[(t.form, t.tag) for t in toks]}")

print()
print("=== 3) 이미 완성된 문장 재분석 (보정 검증용) ===")
for sent in ["아이가 돈을 줍니다", "학교에 가서 친구를 만납니다", "나 물를 마십니다"]:
    toks = kiwi.tokenize(sent)
    print(f"  {sent}")
    print(f"     -> {[(t.form, t.tag) for t in toks]}")

print()
print("=== 4) spellcheck 가능 클래스 존재 여부 ===")
import kiwipiepy
print("  KiwiSgiga:", hasattr(kiwipiepy, "KiwiSgiga"))
print("  KiwiBpe:", hasattr(kiwipiepy, "KiwiBpe"))
print("  KiwiPl:", hasattr(kiwipiepy, "KiwiPl"))
