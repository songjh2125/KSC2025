# preprocess.py
import json
import sys
from pathlib import Path
from glob import glob

# config 경로
DEFAULT_CONFIG = "../configs/preprocess.yaml"

# ==========================
# Config 로드 함수
# ==========================
def load_config(cfg_path: str):
    p = Path(cfg_path)
    text = p.read_text(encoding="utf-8")

    if p.suffix.lower() in [".yml", ".yaml"]:
        try:
            import yaml  # type: ignore
        except Exception:
            sys.stderr.write(
                "[ERR] PyYAML이 없습니다. `pip install pyyaml` 후 다시 시도하거나 JSON 설정을 사용하세요.\n"
            )
            sys.exit(1)
        return yaml.safe_load(text)
    else:
        return json.loads(text)

# ---------------------------------------------------
# 안전 UTF-8 리더 (LS/PS/BOM 제거)
# ---------------------------------------------------
def safe_read_utf8(path: Path) -> str:
    s = path.read_text(encoding="utf-8", errors="replace")
    return s.replace("\u2028", "").replace("\u2029", "").replace("\ufeff", "")

# ---------------------------------------------------
# 화자 라벨링 (파일 전체 기준 A,B,C,...)
# ---------------------------------------------------
def label_all_sessions(sessions):
    """
    sessions: list of { dialog: [{speaker, utterance}, ...], sessionSummary: {dialogSummary} }
    변경점:
      - 두 번째 세션(i>=1)부터는 앞의 2개 발화(보통 A1, B1)를 제거하고 이어붙임
      - boundaries/seg_summaries는 '실제로 남은(turns) 마지막 발화' 자리에만 기록
    """
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    spk2lab = {}
    next_idx = 0

    text, boundaries, seg_summaries = [], [], []

    for i, sess in enumerate(sessions):
        dialog = sess.get("dialog", []) or []
        ss = sess.get("sessionSummary", {}) or {}
        sum_text = str(ss.get("dialogSummary", "") or "").strip()

        # 추가된 부분: 두 번째 세션부터 앞 2개 발화 제거
        turns = dialog if i == 0 else dialog[2:]

        # 남은 턴만 이어붙임
        for turn in turns:
            spk = str(turn.get("speaker", "")).strip()
            utt = str(turn.get("utterance", "")).strip()
            if not utt:
                continue
            if spk not in spk2lab:
                spk2lab[spk] = labels[next_idx] if next_idx < len(labels) else f"SPK{next_idx}"
                next_idx += 1
            text.append(f"{spk2lab[spk]}: {utt}")
            boundaries.append(0)
            seg_summaries.append("")

        # 남은 턴이 있을 때만 boundary/summary 표기
        if turns:
            last_idx = len(text) - 1
            boundaries[last_idx] = 1
            seg_summaries[last_idx] = sum_text

    return text, boundaries, seg_summaries

# ---------------------------------------------------
# 단일 파일 -> 1 레코드 (세션 합치기)
# ---------------------------------------------------
def convert_file_to_single_record(in_path: Path):
    """
    변경 규칙:
      - 파일 안의 모든 세션을 하나의 레코드로 합침
      - id: 파일 아이디 (FileInfo.filename 있으면 그 stem, 없으면 파일명 stem)
      - text: 모든 세션 dialog를 순서대로 이어붙인 리스트
      - boundaries: 각 세션의 마지막 턴 인덱스만 1
      - seg_summaries: boundary 위치에만 해당 세션 요약 텍스트
    """
    try:
        data = json.loads(safe_read_utf8(in_path))
    except Exception as e:
        sys.stderr.write(f"[WARN] 읽기 실패 {in_path}: {e}\n")
        return None

    file_info = data.get("FileInfo", {}) or {}
    filename = str(file_info.get("filename", "")).strip()
    file_id = Path(filename).stem if filename else in_path.stem

    sessions = data.get("sessionInfo", []) or []
    if not sessions:
        return None

    text, boundaries, seg_summaries = label_all_sessions(sessions)
    if not text:
        return None

    return {
        "id": file_id,
        "text": text,
        "boundaries": boundaries,
        "seg_summaries": seg_summaries,
    }

# ---------------------------------------------------
# 파일 수집/쓰기
# ---------------------------------------------------
def collect_files(input_path: Path, pattern: str = "**/*.json"):
    if input_path.is_dir():
        return [Path(p) for p in glob(str(input_path / pattern), recursive=True)]
    return [input_path]


def write_jsonl(records, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as w:
        for r in records:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")


# ==========================
# 메인 실행부
# ==========================
def main():
    cfg_path = Path(DEFAULT_CONFIG)
    if not cfg_path.exists():
        sys.stderr.write(f"[ERR] 기본 config 파일({cfg_path})을 찾을 수 없습니다.\n")
        sys.exit(1)

    cfg = load_config(cfg_path)
    targets = cfg.get("targets", [])
    if not isinstance(targets, list) or not targets:
        sys.stderr.write("[ERR] config의 'targets'가 비었거나 형식이 잘못되었습니다.\n")
        sys.exit(1)

    merged = []

    for t in targets:
        input_spec = t.get("input")
        output = t.get("output")
        pattern = t.get("pattern", "**/*.json")

        if not input_spec or not output:
            sys.stderr.write("[WARN] target에 'input' 혹은 'output'이 없습니다. 건너뜁니다.\n")
            continue

        in_path = Path(input_spec)
        files = collect_files(in_path, pattern)

        out_records = []
        for fp in sorted(files):
            rec = convert_file_to_single_record(fp)
            if rec:
                out_records.append(rec)

        write_jsonl(out_records, Path(output))
        merged.extend(out_records)
        print(f"[INFO] wrote {len(out_records)} records → {output}")

    # 여러 파일을 하나로 합치기 (옵션)
    merge_output = cfg.get("merge_output")
    if merge_output:
        write_jsonl(merged, Path(merge_output))
        print(f"[INFO] merged {len(merged)} records → {merge_output}")

    print("[DONE] 모든 변환이 완료되었습니다")


if __name__ == "__main__":
    main()
