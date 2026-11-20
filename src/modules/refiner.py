"""수어 단어 수열 → 자연스러운 한국어 문장 변환 (Refiner).

기본 동작은 한국어 문법 규칙 기반(조사 삽입 + 서술어 활용)이며,
transformers 라이브러리와 파인튜닝된 KoBART/T5 체크포인트가 준비된 경우
MT_USE_KOBART=1 로 신경망 기반 변환을 사용할 수 있다.
"""
import os
import sys
import time
import queue
import threading
from pathlib import Path
from typing import List, Optional

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import LLM_MODEL_NAME
except ImportError:
    from config import LLM_MODEL_NAME

# 신경망(KoBART) 경로 사용 여부 (기본: 규칙 기반)
USE_KOBART = os.environ.get("MT_USE_KOBART", "0") == "1"
REFINER_MODEL_NAME = os.environ.get("MT_REFINER_MODEL", LLM_MODEL_NAME)


# ==========================================
# 1. 한국어 규칙 기반 후처리 엔진
# ==========================================
# 자주 나오는 서술어 사전 (활용형)
_VERB_POLITE = {
    "주다": "줍니다", "받다": "받습니다", "가다": "갑니다", "오다": "옵니다",
    "보다": "봅니다", "먹다": "먹습니다", "마시다": "마십니다", "하다": "합니다",
    "만나다": "만납니다", "알다": "압니다", "모르다": "모릅니다", "좋아하다": "좋아합니다",
    "싫어하다": "싫어합니다", "말하다": "말합니다", "듣다": "듣습니다",
    "열다": "열습니다", "닫다": "닫습니다", "앉다": "앉습니다", "서다": "섭니다",
    "자다": "잡니다", "살다": "삽니다", "죽다": "죽습니다", "울다": "웁니다",
    "웃다": "웃습니다", "걸다": "걸습니다", "뛰다": "뜁니다", "노다": "놉니다",
    "기다리다": "기다립니다", "도와주다": "도와줍니다", "도와": "돕니다",
    "가져가다": "가져갑니다", "가져오다": "가져옵니다", "만들다": "만듭니다",
    "배우다": "배웁니다", "가르치다": "가르칩니다", "알려주다": "알려줍니다",
    "있다": "있습니다", "없다": "없습니다",
}

# 조사 삽입 시 주격으로 취급하지 않는 기능어/부사
_SKIP_AS_SUBJECT = {"아니요", "아니", "너무", "매우", "지금", "여기", "거기", "저기",
                    "오늘", "어제", "내일", "나", "너", "우리", "저", "그", "이"}


def _is_verb(word: str) -> bool:
    return len(word) >= 2 and word.endswith("다")


def _conjugate(verb: str) -> str:
    """사전형 서술어를 합쇼체로 변환."""
    if verb in _VERB_POLITE:
        return _VERB_POLITE[verb]
    stem = verb[:-1]  # '다' 제거
    last = stem[-1] if stem else ""
    # 모음으로 끝나면 ㅂ니다, 자음이면 습니다 (단순 규칙)
    if last in "아야어여오요으이":
        return stem + "ㅂ니다"
    return stem + "습니다"


def _has_batchim(noun: str) -> bool:
    """마지막 한글 글자에 받침이 있는지 판별."""
    if not noun:
        return False
    code = ord(noun[-1]) - ord("가")
    # 완성형 한글(0xAC00~0xD7A3, 총 11172자)만 대상.
    # code % 28 이 받침 인덱스 (0 = 받침 없음)
    return 0 <= code < 11172 and (code % 28) >= 1


def _pick_particle(noun: str, role: str) -> str:
    """명사 뒤 조사 선택 (받침 여부 기반)."""
    batchim = _has_batchim(noun)
    if role == "subject":
        return "이" if batchim else "가"
    if role == "object":
        return "을" if batchim else "를"
    if role == "dative":
        return "에게" if batchim else "에"
    return ""


def refine_words_rule_based(words: List[str]) -> str:
    """단어 수열을 SOV 한국어 문장으로 변환."""
    if not words:
        return ""
    if len(words) == 1:
        w = words[0]
        return _conjugate(w) + "." if _is_verb(w) else w + "."

    # 서술어는 마지막 단어(사전형)로 가정
    verb = words[-1] if _is_verb(words[-1]) else None
    nouns = words[:-1] if verb else words[:]

    parts: List[str] = []
    for i, noun in enumerate(nouns):
        if noun in _SKIP_AS_SUBJECT:
            parts.append(noun)
            continue
        # 첫 명사는 주격, 그 뒤 명사는 목적격으로 단순 배치
        role = "subject" if i == 0 else "object"
        parts.append(noun + _pick_particle(noun, role))

    if verb:
        parts.append(_conjugate(verb))

    return " ".join(parts) + "."


# ==========================================
# 2. (선택) KoBART / T5 신경망 변환 엔진
# ==========================================
class KobartRefiner:
    """transformers 기반 seq2seq 변환. 파인튜닝된 체크포인트가 있을 때만 유용."""

    def __init__(self, model_name: str = REFINER_MODEL_NAME):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # lazy import
        import torch
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def refine(self, words: List[str]) -> str:
        raw = " ".join(words)
        inputs = self.tokenizer(raw, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            gen = self.model.generate(**inputs, max_new_tokens=48)
        return self.tokenizer.decode(gen[0], skip_special_tokens=True)


# ==========================================
# 3. 실시간 비동기 Refiner (큐 기반)
# ==========================================
class AsyncSignRefiner:
    def __init__(self, use_kobart: bool = USE_KOBART):
        self.word_queue: queue.Queue = queue.Queue()
        self.current_sentence: str = "수어 동작을 대기 중입니다..."
        self.is_running: bool = True

        self._kobart: Optional[KobartRefiner] = None
        if use_kobart:
            try:
                self._kobart = KobartRefiner()
                print(f"[*] Refiner: KoBART 모드 ({REFINER_MODEL_NAME})")
            except Exception as e:
                print(f"[!] KoBART 로드 실패, 규칙 기반으로 대체: {e}")
                self._kobart = None
        if self._kobart is None:
            print("[*] Refiner: 한국어 규칙 기반 모드")

        self.worker_thread = threading.Thread(target=self._refine_worker, daemon=True)
        self.worker_thread.start()

    def push_word(self, word: str):
        self.word_queue.put(word)

    def stop(self):
        self.is_running = False

    def _refine_worker(self):
        collected: List[str] = []
        last_input_time = time.time()

        while self.is_running:
            try:
                word = self.word_queue.get(timeout=0.4)
                if not collected or collected[-1] != word:
                    collected.append(word)
                last_input_time = time.time()
            except queue.Empty:
                # 1.2초간 새 단어가 없으면 현재 수열을 문장으로 확정
                if collected and (time.time() - last_input_time > 1.2):
                    self.current_sentence = self._refine(collected)
                    print(f"\n[실시간 문맥 보정 완료]: {self.current_sentence}")
                    collected.clear()

    def _refine(self, words: List[str]) -> str:
        if self._kobart is not None:
            try:
                return self._kobart.refine(words)
            except Exception as e:
                print(f"[!] KoBART 추론 실패, 규칙 기반 대체: {e}")
        return refine_words_rule_based(words)


if __name__ == "__main__":
    # 간단한 자체 테스트
    tests = [
        ["아이", "돈", "주다"],
        ["나", "물", "마시다"],
        ["학교", "가다"],
        ["그", "친구", "만나다"],
    ]
    for t in tests:
        print(f"{t}  ->  {refine_words_rule_based(t)}")
