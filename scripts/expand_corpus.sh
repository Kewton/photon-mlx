#!/usr/bin/env bash
# expand_corpus.sh  –  Clone 5 repos, ingest, generate per-repo corpus, merge into multi.
#
# Usage:
#     bash scripts/expand_corpus.sh
#
# Prerequisites:
#     - Python environment with baseline_reporag + transformers installed
#     - configs/baseline.yaml present
#
# Output:
#     data/processed/train_multi.jsonl
#     data/processed/val_multi.jsonl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="${PROJECT_ROOT}/data/raw"
PROCESSED_DIR="${PROJECT_ROOT}/data/processed"
CONFIG="${PROJECT_ROOT}/configs/baseline.yaml"
PHOTON_CONFIG="${PROJECT_ROOT}/configs/photon_small.yaml"

# ── Target repositories ──────────────────────────────────────────────
declare -a REPOS=(
    "tiangolo/fastapi"
    "pallets/flask"
    "encode/starlette"
    "encode/httpx"
    "pydantic/pydantic"
)

mkdir -p "$RAW_DIR" "$PROCESSED_DIR"

# ── Helper: derive repo_id from owner/name (e.g. tiangolo/fastapi -> tiangolo_fastapi) ──
repo_id_from() {
    echo "$1" | tr '/' '_'
}

# ── Phase 1: Clone (skip if already present) ─────────────────────────
echo "=== Phase 1: Clone repositories ==="
for repo in "${REPOS[@]}"; do
    local_name="${repo##*/}"          # e.g. "fastapi"
    clone_dir="${RAW_DIR}/${local_name}"
    if [ -d "$clone_dir/.git" ]; then
        echo "  [skip] ${repo} already cloned at ${clone_dir}"
    else
        echo "  [clone] ${repo} -> ${clone_dir}"
        git clone --depth 1 "https://github.com/${repo}.git" "$clone_dir"
    fi
done

# ── Phase 2: Ingest + generate per-repo corpus ──────────────────────
echo ""
echo "=== Phase 2: Ingest & generate per-repo corpus ==="
for repo in "${REPOS[@]}"; do
    local_name="${repo##*/}"
    clone_dir="${RAW_DIR}/${local_name}"
    rid="$(repo_id_from "$repo")"

    echo ""
    echo "--- ${repo} (repo_id=${rid}) ---"

    # 2a. Ingest into chunk store
    repo_sha="$(git -C "$clone_dir" rev-parse HEAD)"
    echo "  [ingest] ${clone_dir} (commit=${repo_sha:0:7})"
    python "${SCRIPT_DIR}/ingest_repo.py" \
        --repo "$clone_dir" \
        --repo-id "$rid" \
        --commit "$repo_sha" \
        --config "$CONFIG"

    # 2b. Generate training corpus (per-repo subdirectory)
    repo_out_dir="${PROCESSED_DIR}/multi_repo/${rid}"
    echo "  [corpus] generating train/val for ${rid} (commit=${repo_sha:0:7}) -> ${repo_out_dir}"
    python "${SCRIPT_DIR}/generate_training_corpus.py" \
        --repo-id "$rid" \
        --config "$CONFIG" \
        --photon-config "$PHOTON_CONFIG" \
        --output-dir "$repo_out_dir" \
        --commit "$repo_sha" \
        --val-ratio 0.1
done

# ── Phase 3: Merge into multi-repo corpus ────────────────────────────
echo ""
echo "=== Phase 3: Merge into train_multi.jsonl / val_multi.jsonl ==="

TRAIN_MULTI="${PROCESSED_DIR}/train_multi.jsonl"
VAL_MULTI="${PROCESSED_DIR}/val_multi.jsonl"

# Clear previous output
: > "$TRAIN_MULTI"
: > "$VAL_MULTI"

for repo in "${REPOS[@]}"; do
    rid="$(repo_id_from "$repo")"
    repo_out_dir="${PROCESSED_DIR}/multi_repo/${rid}"
    train_file="${repo_out_dir}/train_tiny.jsonl"
    val_file="${repo_out_dir}/val_tiny.jsonl"

    if [ -f "$train_file" ]; then
        cat "$train_file" >> "$TRAIN_MULTI"
        echo "  [merge] ${rid} train: $(wc -l < "$train_file") docs"
    else
        echo "  [warn] ${train_file} not found for ${rid}"
    fi

    if [ -f "$val_file" ]; then
        cat "$val_file" >> "$VAL_MULTI"
        echo "  [merge] ${rid} val:   $(wc -l < "$val_file") docs"
    else
        echo "  [warn] ${val_file} not found for ${rid}"
    fi
done

TRAIN_TOTAL="$(wc -l < "$TRAIN_MULTI")"
VAL_TOTAL="$(wc -l < "$VAL_MULTI")"

echo ""
echo "=== Done ==="
echo "  train_multi.jsonl : ${TRAIN_TOTAL} documents"
echo "  val_multi.jsonl   : ${VAL_TOTAL} documents"
echo "  Location          : ${PROCESSED_DIR}/"
