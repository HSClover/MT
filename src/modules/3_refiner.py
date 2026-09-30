import time
import queue
import threading
from typing import List


class AsyncSignRefiner:
    def __init__(self):
        self.word_queue = queue.Queue()
        self.current_sentence = ""
        self.is_running = True
        
        # 백그라운드 문맥 정제 스레드 시작
        self.worker_thread = threading.Thread(target=self._refine_worker, daemon=True)
        self.worker_thread.start()

    def push_word(self, word: str):
        """실시간 단어 추론 모듈에서 단어가 들어올 때 호출 (0ms 지연)"""
        self.word_queue.put(word)

    def _refine_worker(self):
        """백그라운드에서 단어 수열을 수집하여 자연스러운 문장으로 변환"""
        collected_words: List[str] = []
        
        while self.is_running:
            try:
                # 0.5초 동안 새로운 단어가 들어오는지 대기
                word = self.word_queue.get(timeout=0.5)
                if not collected_words or collected_words[-1] != word:
                    collected_words.append(word)
            except queue.Empty:
                # 단어 입력이 잠시 멈췄을 때 문맥 정제 실행
                if collected_words:
                    self.current_sentence = self._lightweight_seq2seq(collected_words)
                    print(f"\n[실시간 보정 완료]: {self.current_sentence}")
                    collected_words.clear()

    def _lightweight_seq2seq(self, words: List[str]) -> str:
        """
        경량 KoBART / T5 추론 위치 (30~50ms 소요)
        예: ['아이', '돈', '주다'] -> '아이에게 돈을 줍니다.'
        """
        raw_text = " ".join(words)
        
        # 간단한 규칙 예시 (추후 KoBART model.generate()로 교체)
        if "돈" in words and "주다" in words:
            return raw_text.replace("아이 돈 주다", "아이에게 돈을 줍니다.")
        
        return raw_text


if __name__ == "__main__":
    refiner = AsyncSignRefiner()
    
    # 실시간 단어 입력 시뮬레이션
    print("[*] 실시간 수어 입력 시작...")
    refiner.push_word("아이")
    time.sleep(0.2)
    refiner.push_word("돈")
    time.sleep(0.2)
    refiner.push_word("주다")
    
    # 보정 스레드 동작 대기
    time.sleep(1.5)