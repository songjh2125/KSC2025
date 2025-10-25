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
# 2세션 파일 파싱: (file_id, [sess1, sess2]) 반환
#  - sess = {"dialog": [...], "summary": "..."}
#  - 세션이 정확히 2개가 아니면 None
# ---------------------------------------------------
def read_two_session_file(in_path: Path):
    try:
        data = json.loads(safe_read_utf8(in_path))
    except Exception as e:
        sys.stderr.write(f"[WARN] 읽기 실패 {in_path}: {e}\n")
        return None

    file_info = data.get("FileInfo", {}) or {}
    filename = str(file_info.get("filename", "")).strip()
    file_id = Path(filename).stem if filename else in_path.stem

    sessions = data.get("sessionInfo", []) or []
    if len(sessions) != 2:
        return None

    out = []
    for s in sessions:
        dialog = s.get("dialog", []) or []
        sum_text = str((s.get("sessionSummary", {}) or {}).get("dialogSummary", "") or "").strip()
        out.append({"dialog": dialog, "summary": sum_text})
    return file_id, out

# ---------------------------------------------------
# 파일 두 개를 짝지어 1레코드 생성:
#   text: A-1, B-1, A-2', B-2'
#   (A-2', B-2'는 앞 2발화 제거)
#   boundaries/seg_summaries는 각 세그먼트의 "남은 마지막 턴"에만 기록
#   id: "<file_id_A>__<file_id_B>"
# ---------------------------------------------------
def build_paired_record(fileA, fileB):
    (idA, [A1, A2]) = fileA
    (idB, [B1, B2]) = fileB

    segs = [
        {"turns": A1["dialog"],     "summary": A1["summary"]},
        {"turns": B1["dialog"],     "summary": B1["summary"]},
        {"turns": A2["dialog"][2:], "summary": A2["summary"]},  # 앞 2발화 제거
        {"turns": B2["dialog"][2:], "summary": B2["summary"]},  # 앞 2발화 제거
    ]

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    spk2lab, next_idx = {}, 0

    text, boundaries, seg_summaries = [], [], []

    for seg in segs:
        turns = seg["turns"]
        for t in turns:
            spk = str(t.get("speaker", "")).strip()
            utt = str(t.get("utterance", "")).strip()
            if not utt:
                continue
            if spk not in spk2lab:
                spk2lab[spk] = labels[next_idx] if next_idx < len(labels) else f"SPK{next_idx}"
                next_idx += 1
            text.append(f"{spk2lab[spk]}: {utt}")
            boundaries.append(0)
            seg_summaries.append("")

        if turns:
            last_idx = len(text) - 1
            boundaries[last_idx] = 1
            seg_summaries[last_idx] = seg["summary"]

    rec_id = f"{idA}__{idB}"
    return {
        "id": rec_id,
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

        # 2세션 파일만 파싱
        parsed = []
        for fp in sorted(files):
            parsed_file = read_two_session_file(fp)
            if parsed_file is not None:
                parsed.append(parsed_file)

        # 짝짓기 (A,B), (C,D), ...  홀수면 마지막 하나 드롭
        out_records = []
        if len(parsed) % 2 == 1:
            sys.stderr.write(f"[WARN] 2세션 파일의 개수가 홀수({len(parsed)})입니다. 마지막 하나는 건너뜁니다.\n")

        limit = len(parsed) - (len(parsed) % 2)
        for i in range(0, limit, 2):
            rec = build_paired_record(parsed[i], parsed[i+1])
            if rec["text"]:  # 빈 레코드 방지
                out_records.append(rec)

        write_jsonl(out_records, Path(output))
        merged.extend(out_records)
        print(f"[INFO] wrote {len(out_records)} paired records → {output}")

    # 여러 파일을 하나로 합치기 (옵션)
    merge_output = cfg.get("merge_output")
    if merge_output:
        write_jsonl(merged, Path(merge_output))
        print(f"[INFO] merged {len(merged)} records → {merge_output}")

    print("[DONE] 모든 변환이 완료되었습니다")


if __name__ == "__main__":
    main()
