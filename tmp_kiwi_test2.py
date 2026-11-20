# -*- coding: utf-8 -*-
"""kiwipiepy로 refiner의 '활용' 단계까지 대체 가능한지 검증."""
import json
from kiwipiepy import Kiwi, KoreanStem

kiwi = Kiwi()
stem = KoreanStem()

print("=== A) KoreanStem.stem_to_surface 로 활용 가능한가? ===")
cases = [
    ("주", "ㅂ니다"), ("알", "ㅂ니다"), ("듣", "ㅂ니다"), ("뛰", "ㅂ니다"),
    ("사", "ㅂ니다"), ("울", "ㅂ니다"), ("마시", "ㅂ니다"), ("가", "ㅂ니다"),
    ("좋", "ㅂ니다"), ("있", "ㅂ니다"), ("없", "ㅂ니다"), ("만나", "ㅂ니다"),
    ("가", "아"), ("주", "아"), ("먹", "아"), ("서", "아"),
]
for s, e in cases:
    try:
        print(f"  {s}+{e}  ->  {stem.stem_to_surface(s, e)}")
    except Exception as ex:
        print(f"  {s}+{e}  ->  ERROR {ex}")

print()
print("=== B) 실제 어휘 100개 자동 품사 태깅 (사전 대체 가능한가?) ===")
raw = list(json.load(open("models/label_map.json", encoding="utf-8")).values())
vocab = sorted({w.rstrip("123456789#") for w in raw})
buckets = {}
for w in vocab:
    toks = kiwi.tokenize(w)
    if not toks:
        tag = "NONE"
        form = w
    else:
        tag = toks[-1].tag
        form = toks[0].form
    buckets.setdefault(tag, []).append((w, form))
for tag, items in sorted(buckets.items()):
    print(f"  [{tag}] {len(items)}: {[i[0] for i in items]}")

print()
print("=== C) 어휘 어근 + VV/VA 판별 결과 ===")
for w in vocab:
    toks = kiwi.tokenize(w)
    tags = [t.tag for t in toks]
    if any(t in ("VV", "VA", "VX") for t in tags):
        base = next(t.form for t in toks if t.tag in ("VV", "VA", "VX"))
        print(f"  {w:12s} -> stem={base:8s} tags={tags}")
