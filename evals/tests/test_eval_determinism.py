"""Determinism integration test for ``Generator.generate(*, seed=...)``.

Issue #143 / Step 4 acceptance criterion 1: with the seed pinned, two
back-to-back invocations of the real ``Generator`` against the same
prompt MUST produce byte-identical output. We exercise the production
code path (no mocks) so that any silent truncation of the seed kwarg or
nondeterminism inside ``mlx_lm.generate`` surfaces here rather than
quietly drifting the institutional eval.

The fixture below skips the test when MLX is not installed (CI hosted
environments without Apple Silicon) so the suite remains green on
Linux runners; the test is meaningful only on the same self-hosted
runner that actually executes the institutional eval.
"""

from __future__ import annotations

import importlib.util

import pytest

# Skip cleanly when MLX is missing (Linux CI / pure-baseline environments).
_HAS_MLX = (
    importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm") is not None
)


pytestmark = pytest.mark.skipif(
    not _HAS_MLX,
    reason="requires mlx + mlx_lm (Apple Silicon self-hosted runner only)",
)


# ---------------------------------------------------------------------------
# Module-scoped Generator: model load is the expensive step (~10s + memory),
# so we share one instance across the two-run determinism check.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def generator():
    """Initialise a Qwen-14B-Instruct-4bit Generator once for the module.

    We use the same model id the institutional eval pins so the test is a
    direct proxy for the production sampling path. ``mlx_lm.load`` is
    skipped if the local model cache is missing — we surface the absence
    via ``pytest.skip`` so a fresh checkout doesn't fail CI just because
    the heavy weights haven't been pulled.
    """
    from baseline_reporag.generation.generator import Generator

    gen = Generator(
        model_id="mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
        max_new_tokens=64,
        temperature=0.2,
        top_p=0.9,
    )
    try:
        gen._load()
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"MLX model not available locally: {exc!r}")
    return gen


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


def test_generator_seeded_two_runs_produce_identical_output(generator) -> None:
    """``Generator.generate(*, seed=42)`` × 2 → byte-identical output.

    Hard equality: any drift here means the seed kwarg was silently
    dropped (DR3-002) or mlx-lm has nondeterminism we did not mitigate
    (work-plan §11). If this becomes flaky in production, downgrade to a
    token-edit-distance soft assert and capture the noise floor in
    ``reports/institutional_eval_noise_floor.md``.
    """
    messages = [
        {
            "role": "user",
            "content": (
                "Summarise the contract: Generator.generate must be "
                "deterministic when seed is pinned."
            ),
        }
    ]
    out1 = generator.generate(messages, max_new_tokens=64, seed=42)
    out2 = generator.generate(messages, max_new_tokens=64, seed=42)
    assert out1 == out2, (
        "Generator.generate(seed=42) produced divergent outputs across two "
        "back-to-back runs — eval reproducibility broken. "
        f"out1[:80]={out1[:80]!r} out2[:80]={out2[:80]!r}"
    )
