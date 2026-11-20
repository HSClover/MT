import sys
import time
import queue
import threading
from pathlib import Path
from typing import List, Optional

import torch

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LLM_MODEL_NAME, MODELS_DIR


class AsyncSignRefiner:
    def __init__(self, model_name: str = LLM_MODEL_NAME, use_llm: bool = True):
        self.word_queue: queue.Queue = queue.Queue()
        self.current_sentence: str = "수어 동작을 대기 중입니다..."
        self.is_running: bool = True
        self._lock = threading.Lock()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None

        if use_llm:
            self._load_kobart(model_name)

        self.worker_thread = threading.Thread(target=self._refine_worker, daemon=True)
        self.worker_thread.start()

    def _load_kobart(self, model_name: str):
        """로컬 파인튜닝 KoBART 또는 HuggingFace KoBART 로드 (실패 시 규칙 폴백)"""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            local_finetuned = MODELS_DIR / "kobart_sign_refiner"
            target_model = str(local_finetuned) if local_finetuned.exists() else model_name

            self.tokenizer = AutoTokenizer.from_pretrained(target_model)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(target_model).to(self.device)
            self.model.eval()
            print(f"[*] KoBART 문장 교정 모델 로드 완료: {target_model}")
        except Exception as e:
            print(f"[!] KoBART 로드 건너뜀 (규칙 기반 폴백 모드 동작): {e}")
            self.tokenizer = None
            self.model = None

    def push_word(self, word: str):
        self.word_queue.put(word)

    def get_sentence(self) -> str:
        with self._lock:
            return self.current_sentence

    def stop(self):
        self.is_running = False

    def _refine_worker(self):
        collected_words: List[str] = []
        last_input_time = time.time()

        while self.is_running:
            try:
                word = self.word_queue.get(timeout=0.4)
                if not collected_words or collected_words[-1] != word:
                    collected_words.append(word)
                last_input_time = time.time()
            except queue.Empty:
                if collected_words and (time.time() - last_input_time >= 1.0):
                    refined = self._seq2seq_refine(collected_words)
                    with self._lock:
                        self.current_sentence = refined
                    collected_words.clear()

    def _seq2seq_refine(self, words: List[str]) -> str:
        raw_text = " ".join(words)

        if self.model is not None and self.tokenizer is not None:
            try:
                inputs = self.tokenizer(
                    raw_text,
                    return_tensors="pt",
                    max_length=64,
                    truncation=True,
                ).to(self.device)

                with torch.no_grad():
                    output_ids = self.model.generate(
                        inputs["input_ids"],
                        max_new_tokens=48,
                        num_beams=3,
                        early_stopping=True,
                    )
                decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                if decoded:
                    return decoded if decoded.endswith((".", "?", "!")) else decoded + "."
            except Exception:
                pass

        return raw_text + "."