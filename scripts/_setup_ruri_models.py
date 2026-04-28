"""ruri 系モデル (spiece.model) を明示的に取得する setup helper。

sentence-transformers / transformers の version 違いに依存しない。

スコープ: 本 Issue (#144) では ruri-small-v2 のみを対象とする。
将来拡張: #135 で ruri を tokenizer 候補にする場合は RURI_MODELS list を拡張する。

Security/Log hygiene (Issue #144 設計レビュー DR4-002):
- 成功 log: filename と file size のみ。HF cache full path は出さない。
- 失敗時 traceback: local 判別に留め、Issue コメントには exception type と
  sanitized summary のみ記録する方針 (caller の責任)。HF token / PAT / full
  cache path / raw traceback を Issue コメントに貼ってはいけない。
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError
except ImportError as e:
    raise ImportError(
        "huggingface_hub が必要です。`pip install huggingface_hub` または "
        "requirements.txt 経由で install してください。"
    ) from e

RURI_MODELS: tuple[str, ...] = ("cl-nagoya/ruri-small-v2",)
RURI_FILES: tuple[str, ...] = ("spiece.model",)

# Fail-loud at import time (CB-002): 空 tuple は silent success を招くため早期検出する。
if not RURI_MODELS:
    raise RuntimeError("RURI_MODELS must not be empty")
if not RURI_FILES:
    raise RuntimeError("RURI_FILES must not be empty")


def fetch_ruri_files() -> None:
    """RURI_MODELS × RURI_FILES の cross product を hf_hub_download で取得する。

    cache hit 時は network 不要 (no-op)。失敗は fail-fast で raise する。
    成功 log は filename + file size のみで、HF cache full path は出力しない。
    """
    for model_id in RURI_MODELS:
        for filename in RURI_FILES:
            try:
                path = hf_hub_download(model_id, filename)
            except (HfHubHTTPError, OSError) as e:
                print(
                    f"ERROR: failed to fetch {filename} for {model_id}: "
                    f"{type(e).__name__}",
                    file=sys.stderr,
                )
                raise
            size_bytes = Path(path).stat().st_size
            print(f"[OK] fetch: {model_id}:{filename} ({size_bytes} bytes)")


def verify_load() -> None:
    """各 model_id について SentenceTransformer 単独 load を検証する。

    tokenizer / model load が例外なく完了することを fail-fast で検証する。
    失敗時は SentenceTransformer の例外を catch せず raise する (caller が
    sanitized summary のみを記録する責任)。
    """
    from sentence_transformers import SentenceTransformer

    for model_id in RURI_MODELS:
        SentenceTransformer(model_id)
        print(f"[OK] verify_load: {model_id}")


def main() -> None:
    """Entry point with sanitized error wrapping (CB-001 / DR4-002).

    raw traceback は RURI_SETUP_DEBUG=1 が明示的にセットされた時のみ出力する。
    """
    try:
        fetch_ruri_files()
        verify_load()
    except Exception as exc:
        print(f"ERROR: setup failed: {type(exc).__name__}", file=sys.stderr)
        if os.environ.get("RURI_SETUP_DEBUG") == "1":
            traceback.print_exc(file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
