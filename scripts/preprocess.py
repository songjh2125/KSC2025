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

# ==========================
# 변환 로직 (단일 파일 → SAMSum형 레코드)
# ==========================
def map_speakers_to_labels(turns):
    """speaker 순서대로 A/B/C... 라벨링"""
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label_map = {}
    next_idx = 0
    out = []
    for t in turns:
        spk = str(t.get("speaker", "")).strip()
        utt = str(t.get("utterance", "")).strip()
        if not utt:
            continue
        if spk not in label_map:
            if next_idx < len(labels):
                label_map[spk] = labels[next_idx]
                next_idx += 1
            else:
                label_map[spk] = f"SPK{next_idx}"
                next_idx += 1
        out.append(f"{label_map[spk]}: {utt}")
    return out


def convert_aihub_json_to_records(in_path: Path):
    """
    요구사항:
      - id = sessionInfo[].sessionID
      - text = sessionInfo[].dialog -> "A: ..." 리스트
      - summary = sessionInfo[].sessionSummary.dialogSummary
      - 세션이 여러 개면 레코드 여러 개 생성
    """
    try:
        data = json.loads(in_path.read_text(encoding="utf-8"))
    except Exception as e:
        sys.stderr.write(f"[WARN] 읽기 실패 {in_path}: {e}\n")
        return []

    sessions = data.get("sessionInfo", []) or []
    out = []

    for s in sessions:
        sid = str(s.get("sessionID", "")).strip()
        dialog = s.get("dialog", []) or []
        ss = s.get("sessionSummary", {}) or {}
        summary = str(ss.get("dialogSummary", "") or "").strip()

        text_list = map_speakers_to_labels(dialog)
        if not sid or not text_list:
            continue

        out.append({
            "id": sid,
            "text": text_list,
            "summary": summary
        })
    return out


# ==========================
# 파일 수집 및 병합
# ==========================
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

        total = 0
        out_records = []
        for fp in sorted(files):
            recs = convert_aihub_json_to_records(fp)
            if recs:
                out_records.extend(recs)
                total += len(recs)

        write_jsonl(out_records, Path(output))
        merged.extend(out_records)
        print(f"[INFO] wrote {total} records → {output}")

    # 여러 파일을 하나로 합치기 (옵션)
    merge_output = cfg.get("merge_output")
    if merge_output:
        write_jsonl(merged, Path(merge_output))
        print(f"[INFO] merged {len(merged)} records → {merge_output}")

    print("[DONE] 모든 변환이 완료되었습니다")


if __name__ == "__main__":
    main()
