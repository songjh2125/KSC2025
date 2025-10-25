#!/usr/bin/env bash
set -euo pipefail

# ------------------------------
# aihubshell 자동 설치 + 다운로드 실행 스크립트
# - ./.env 에서 AIHUB_APIKEY 읽음
# - ./config.yml 에서 dataset_key, output_dir 읽음
# - macOS: /usr/local/bin (권장), Linux: /usr/bin 또는 /usr/local/bin
# ------------------------------

AIHUB_URL="https://api.aihub.or.kr/api/aihubshell.do"
BIN_NAME="aihubshell"

# --- 작은 유틸들 ---
log()   { printf "\033[1;36m[INFO]\033[0m %s\n" "$*"; }
warn()  { printf "\033[1;33m[WARN]\033[0m %s\n" "$*"; }
error() { printf "\033[1;31m[ERR ]\033[0m %s\n" "$*" >&2; exit 1; }

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# --- OS/설치 경로 결정 ---
detect_install_dir() {
  local try_dirs=()

  # macOS는 SIP 때문에 /usr/bin 쓰기 제한. /usr/local/bin 우선.
  if [[ "$(uname -s)" == "Darwin" ]]; then
    try_dirs=(/usr/local/bin /opt/homebrew/bin /usr/bin)
  else
    try_dirs=(/usr/local/bin /usr/bin)
  fi

  for d in "${try_dirs[@]}"; do
    if [[ -d "$d" && -w "$d" ]]; then
      echo "$d"
      return
    fi
  done

  # 쓰기 권한이 없다면 기본은 /usr/local/bin 로 두고 sudo 사용
  echo "/usr/local/bin"
}

INSTALL_DIR="$(detect_install_dir)"

# --- .env 로드 ---
load_env() {
  local env_file="${1:-.env}"
  if [[ ! -f "$env_file" ]]; then
    error ".env 파일을 찾을 수 없습니다. 현재 경로에 .env를 생성해 주세요."
  fi

  # shellcheck disable=SC1090
  set -a
  source "$env_file"
  set +a

  if [[ -z "${AIHUB_APIKEY:-}" ]]; then
    error ".env 내 AIHUB_APIKEY 가 비어 있습니다."
  fi
}

# --- config.yml 파싱 (간단 YAML: key: value) ---
# yq 미사용. dataset_key / output_dir 만 읽음.
parse_config() {
  local cfg="${1:-config.yml}"
  if [[ ! -f "$cfg" ]]; then
    error "config.yml 파일을 찾을 수 없습니다."
  fi

  # 따옴표/공백 제거하면서 값만 추출
  DATASET_KEY="$(awk -F':' '/^[[:space:]]*dataset_key[[:space:]]*:/ {sub(/^[^:]*:[[:space:]]*/,""); gsub(/["'\'']/,""); print}' "$cfg" | head -n1)"
  OUTPUT_DIR="$(awk -F':' '/^[[:space:]]*output_dir[[:space:]]*:/ {sub(/^[^:]*:[[:space:]]*/,""); gsub(/["'\'']/,""); print}' "$cfg" | head -n1)"

  if [[ -z "${DATASET_KEY:-}" ]]; then
    error "config.yml 내 dataset_key 가 비어 있습니다."
  fi

  if [[ -z "${OUTPUT_DIR:-}" ]]; then
    OUTPUT_DIR="."
  fi
}

# --- aihubshell 설치 ---
install_aihubshell() {
  log "aihubshell 다운로드: $AIHUB_URL"
  curl -fsSL -o "$BIN_NAME" "$AIHUB_URL" || error "aihubshell 다운로드 실패"
  chmod +x "$BIN_NAME"

  # 설치 경로에 복사 (권한 없으면 sudo 시도)
  local target="$INSTALL_DIR/$BIN_NAME"

  if cp "$BIN_NAME" "$target" 2>/dev/null; then
    :
  else
    warn "쓰기 권한이 없어 sudo로 설치를 시도합니다: $target"
    sudo cp "$BIN_NAME" "$target"
  fi

  rm -f "$BIN_NAME"

  if ! have_cmd "$BIN_NAME"; then
    # PATH에 없을 수 있으니 메시지 안내
    warn "'$target' 로 설치했지만 PATH에 없을 수 있습니다. 쉘에서 인식되는지 확인하세요."
  fi

  log "설치 완료: $(command -v "$BIN_NAME" || echo "$target")"
}

# --- 설치 필요 여부 점검 ---
ensure_installed() {
  if have_cmd "$BIN_NAME"; then
    log "이미 설치되어 있습니다: $(command -v "$BIN_NAME")"
    return
  fi
  install_aihubshell
}

# --- 실행 파일 형식 점검 (ELF 여부 경고) ---
check_binary_format() {
  local path
  path="$(command -v "$BIN_NAME" || true)"
  [[ -z "$path" ]] && return

  if have_cmd file; then
    local f; f="$(file "$path" 2>/dev/null || true)"
    # macOS에서 ELF라면 실행 불가 → 경고
    if [[ "$(uname -s)" == "Darwin" ]] && echo "$f" | grep -q "ELF .* executable"; then
      warn "현재 aihubshell이 Linux ELF 바이너리로 보입니다(file: $f). macOS에서는 실행되지 않을 수 있습니다."
      warn "필요 시 Docker/VM(리눅스)에서 실행하세요."
    fi
  fi
}

# --- 다운로드 실행 ---
do_download() {
  mkdir -p "$OUTPUT_DIR"
  log "작업 디렉터리: $OUTPUT_DIR"
  ( cd "$OUTPUT_DIR"
    # 중요한 부분: API key는 큰따옴표로 안전하게 전달
    log "다운로드 시작 (dataset_key=${DATASET_KEY})"
    "$BIN_NAME" -mode d -datasetkey "$DATASET_KEY" -aihubapikey "$AIHUB_APIKEY"
  )
  log "다운로드 작업이 완료되었습니다."
}

# ------------------------------
# 메인
# ------------------------------
load_env ".env"
parse_config "configs/get_data.yaml"
ensure_installed
check_binary_format
do_download
