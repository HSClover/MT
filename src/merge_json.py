import json
from pathlib import Path
from tqdm import tqdm

# 1. 경로 설정 (src 폴더 위치 고려)
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "merged_nikl_data.json"


def merge_all_json_files():
    print(f"[*] 프로젝트 루트: {PROJECT_ROOT}")
    print(f"[*] 탐색 대상 경로: {RAW_DIR}")

    if not RAW_DIR.exists():
        print(f"[!] {RAW_DIR} 경로가 존재하지 않습니다. 경로를 확인해 주세요.")
        return

    # 2. 하위 모든 .json 파일 검색
    print("[*] JSON 파일 목록을 수집 중입니다...")
    json_files = list(RAW_DIR.rglob("*.json"))
    total_count = len(json_files)

    print(f"[*] 총 탐색된 원본 JSON 파일 수: {total_count}개")

    if total_count == 0:
        print("[!] data/raw/ 하위에서 json 파일을 찾지 못했습니다.")
        return

    merged_data = {}
    success_count = 0

    # 3. tqdm 진행바 적용
    for file_path in tqdm(json_files, desc="[JSON 병합 진행률]", unit="file"):
        file_stem = file_path.stem

        try:
            rel_path = str(file_path.relative_to(RAW_DIR))
        except ValueError:
            rel_path = str(file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                json_content = json.load(f)

                merged_data[file_stem] = {
                    "path": rel_path,
                    "content": json_content,
                }
                success_count += 1
        except Exception as e:
            # tqdm 중 출력 깨짐 방지
            tqdm.write(f"[!] 파일 읽기 에러 ({file_path.name}): {e}")

    # 4. 저장 처리
    print(f"\n[*] 병합 결과를 {OUTPUT_FILE} 에 저장하는 중입니다...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False)

    print(f"[v] 병합 완료! 총 {success_count}개 JSON 저장 -> {OUTPUT_FILE}")


if __name__ == "__main__":
    merge_all_json_files()