"""
generate_photon.py  –  Greedy decode from a trained PHOTON checkpoint.

Usage (manual CLI training – final/ is the stop-time weights unless
restore_best=true):
    python scripts/generate_photon.py \
        --config configs/photon_tiny.yaml \
        --checkpoint checkpoints/final \
        --prompt "def get_current_user(" \
        --max-new-tokens 96

Usage (Streamlit-app-launched run – per-run namespace; best/ exists only
when early_stopping is enabled, otherwise point at final/):
    python scripts/generate_photon.py \
        --config configs/photon_<repo_id>.yaml \
        --checkpoint checkpoints/<repo_id>/<job_id>/best \
        --prompt "def get_current_user(" \
        --max-new-tokens 96
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx

from photon_mlx.trainer import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="PHOTON greedy decode")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Max tokens to generate (default: from YAML or 96)",
    )
    args = parser.parse_args()

    # --- Validate paths ---
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if config_path.suffix not in {".yaml", ".yml"}:
        raise ValueError("Config must be a YAML file (.yaml / .yml)")
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_path}")
    if not (checkpoint_path / "weights.npz").exists():
        raise FileNotFoundError("Checkpoint dir must contain weights.npz")
    if not (checkpoint_path / "state.json").exists():
        raise FileNotFoundError("Checkpoint dir must contain state.json")

    # --- Resolve max_new_tokens ---
    import yaml

    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    inf = raw.get("inference", {})
    max_new = args.max_new_tokens
    if max_new is None:
        max_new = inf.get("answer_max_new_tokens", 96)
        # Clamp for verification script
        max_new = min(max_new, 256)

    if not (1 <= max_new <= 256):
        raise ValueError("--max-new-tokens must be in [1, 256]")

    # --- Load model ---
    print("Loading model...")
    model = load_model(config_path, checkpoint_path)
    print("Model loaded.")

    # --- Load tokenizer (script layer only) ---
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers package required. Install: pip install transformers"
        )

    tok_cfg = raw.get("tokenizer", {})
    tokenizer_id = tok_cfg.get("tokenizer_id", "meta-llama/Llama-2-7b-hf")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer '{tokenizer_id}'. "
            f"If this is a gated model, run: huggingface-cli login\n"
            f"Error: {e}"
        ) from e

    # --- Tokenize prompt ---
    prompt_ids = tokenizer.encode(args.prompt, return_tensors=None)
    if len(prompt_ids) == 0:
        raise ValueError("Prompt must not be empty after tokenization")
    if len(prompt_ids) > 512:
        raise ValueError(f"Prompt too long: {len(prompt_ids)} tokens (max 512)")

    input_ids = mx.array([prompt_ids])

    # --- Generate ---
    print(f"Generating {max_new} tokens...")
    t0 = time.time()
    generated_ids, _step_logits = model.generate(input_ids, max_new_tokens=max_new)
    elapsed = time.time() - t0

    # --- Output (generated suffix only) ---
    output_ids = generated_ids[0].tolist()[len(prompt_ids) :]
    output_text = tokenizer.decode(output_ids)
    tokens_per_sec = max_new / elapsed if elapsed > 0 else 0

    print("\n=== Generated Text (suffix only) ===")
    print(output_text)
    print("\n=== Stats ===")
    print(f"Prompt tokens: {len(prompt_ids)}")
    print(f"Generated tokens: {max_new}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Speed: {tokens_per_sec:.1f} tok/s")
    print("KV cache enabled (Phase 1, top-level only)")


if __name__ == "__main__":
    main()
