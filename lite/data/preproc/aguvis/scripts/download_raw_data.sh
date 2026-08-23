#!/usr/bin/env bash
set -e

usage() {
    echo "Usage: $0 [--stage 1|2|all] [--dry-run] [-h|--help]"
    echo ""
    echo "Download Aguvis raw data from Hugging Face."
    echo ""
    echo "Options:"
    echo "  --stage    Which stage to download: 1, 2, or all (default: all)"
    echo "  --dry-run  Print commands without executing"
    echo "  -h, --help Show this help message"
    exit 0
}

STAGE="all"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            [[ $# -ge 2 ]] || { echo "--stage requires 1, 2, or all" >&2; exit 2; }
            STAGE="$2"
            shift 2
            ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

case "$STAGE" in
    1|2|all) ;;
    *) echo "Invalid --stage: $STAGE (expected 1, 2, or all)" >&2; exit 2 ;;
esac

if [[ -z "${CUA_LITE_RAW_DATASETS_ROOT:-}" ]]; then
    echo "ERROR: CUA_LITE_RAW_DATASETS_ROOT is not set." >&2
    exit 1
fi

if command -v hf &>/dev/null; then
    HF_CMD=hf
elif command -v huggingface-cli &>/dev/null; then
    HF_CMD=huggingface-cli
else
    echo "ERROR: Neither 'hf' nor 'huggingface-cli' found on PATH." >&2
    exit 1
fi

run_cmd() {
    if $DRY_RUN; then
        echo "[dry-run] $*"
    else
        echo ">>> $*"
        "$@"
    fi
}

if [[ "$STAGE" == "1" || "$STAGE" == "all" ]]; then
    echo "=== Downloading xlangai/aguvis-stage1 ==="
    run_cmd "$HF_CMD" download xlangai/aguvis-stage1 \
        --repo-type dataset \
        --local-dir "${CUA_LITE_RAW_DATASETS_ROOT}/xlangai/aguvis-stage1"
fi

if [[ "$STAGE" == "2" || "$STAGE" == "all" ]]; then
    echo "=== Downloading xlangai/aguvis-stage2 ==="
    run_cmd "$HF_CMD" download xlangai/aguvis-stage2 \
        --repo-type dataset \
        --local-dir "${CUA_LITE_RAW_DATASETS_ROOT}/xlangai/aguvis-stage2"
fi

echo "Done."
