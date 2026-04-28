"""Shared config loader for torch_ref and photon_mlx."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_logger = logging.getLogger(__name__)


# v1 scope: only "none" (vanilla RoPE) and "ntk" (NTK-aware interpolated RoPE).
# Extend this set when adding new scaling methods (e.g. "linear", "yarn").
ROPE_SCALING_CHOICES: frozenset[str] = frozenset({"none", "ntk"})


@dataclass
class ModelConfig:
    architecture: str = "photon_decoder"
    base_embed_dim: int = 160
    hidden_size: int = 640
    intermediate_size: int = 1664
    num_attention_heads: int = 10
    num_key_value_heads: int = 10
    head_dim: int = 64
    max_position_embeddings: int = 2048
    rope_theta: float = 1_000_000.0
    rope_scaling: str = "none"
    rope_scale_factor: float = 1.0
    norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    dropout: float = 0.0
    bias: bool = False
    # Issue #140 / S7-001: σ threshold for the start-up embedding-norm sanity
    # check inside ``PhotonInference.__init__``. When ``std(token_embed.weight)``
    # exceeds this value we emit a WARNING (random-init suspect; see Issue
    # #135). Default 0.3 is a placeholder calibrated against random init; a
    # follow-up Issue will re-tune once a trained checkpoint is available.
    embedding_random_init_threshold: float = 0.3

    def __post_init__(self) -> None:
        if self.rope_scaling not in ROPE_SCALING_CHOICES:
            raise ValueError(
                f"invalid rope_scaling: {self.rope_scaling!r} "
                f"(expected one of {sorted(ROPE_SCALING_CHOICES)})"
            )
        if self.rope_scale_factor < 1.0:
            raise ValueError(
                f"rope_scale_factor must be >= 1.0 (got {self.rope_scale_factor})"
            )
        if self.rope_scaling == "none" and self.rope_scale_factor != 1.0:
            _logger.warning(
                "rope_scale_factor=%s is ignored because rope_scaling='none'; "
                "set rope_scaling='ntk' to apply the scale factor.",
                self.rope_scale_factor,
            )
        # Issue #140 DR4-002: validate embedding_random_init_threshold.
        # ``bool`` is rejected before ``int``/``float`` because Python's
        # ``bool`` is an ``int`` subclass and silent True/False would mask
        # configuration mistakes.
        threshold = self.embedding_random_init_threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(
                "embedding_random_init_threshold must be a non-negative "
                f"finite number, got type {type(threshold).__name__}"
            )
        threshold_f = float(threshold)
        if not math.isfinite(threshold_f) or threshold_f < 0.0:
            raise ValueError(
                "embedding_random_init_threshold must be a non-negative "
                f"finite number, got {threshold!r}"
            )
        self.embedding_random_init_threshold = threshold_f

    @classmethod
    def rope_scaling_from(cls, m: Any) -> tuple[str, float]:
        """Single source of defaults for ``rope_scaling`` / ``rope_scale_factor``.

        Handles MagicMock-based configs in tests (unknown attrs return
        ``MagicMock`` instances), so we use ``getattr`` with concrete
        fallbacks.  ``scaling="ntk"`` with ``scale_factor=1.0`` is
        mathematically equivalent to ``scaling="none"`` (the scale factor
        term collapses to ``theta * 1.0``).
        """
        return (
            getattr(m, "rope_scaling", "none"),
            float(getattr(m, "rope_scale_factor", 1.0)),
        )


@dataclass
class HierarchyConfig:
    levels: int = 2
    chunk_sizes: list[int] = field(default_factory=lambda: [4, 4])
    converter_prefix_lengths: list[int] = field(default_factory=lambda: [2, 2])
    encoder_layers_per_level: list[int] = field(default_factory=lambda: [2, 2])
    decoder_layers_per_level: list[int] = field(default_factory=lambda: [2, 2])
    context_encoder_arch: str = "llama_decoder_style"
    context_decoder_arch: str = "llama_decoder_style"
    context_converter_type: str = "conv1d"
    recursive_loss_weight: float = 0.0


@dataclass
class TokenizerConfig:
    tokenizer_id: str = "meta-llama/Llama-2-7b-hf"
    vocab_size: int = 32_000


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration (opt-in; Issue #60).

    Fields:
      - enabled: master on/off switch. Default False for backward compat.
      - patience: number of eval rounds without improvement before stopping.
      - min_delta: minimum decrease in val_loss to count as improvement.
      - restore_best: when True, the final/ checkpoint at end of train() is
                      populated from the best checkpoint rather than from the
                      stop-time weights.
    """

    enabled: bool = False
    patience: int = 3
    min_delta: float = 0.0
    restore_best: bool = True

    def __post_init__(self) -> None:  # pragma: no cover - trivial validation
        if not isinstance(self.enabled, bool):
            raise TypeError(
                f"EarlyStoppingConfig.enabled must be bool, got {type(self.enabled)}"
            )
        if not isinstance(self.restore_best, bool):
            raise TypeError(
                "EarlyStoppingConfig.restore_best must be bool, "
                f"got {type(self.restore_best)}"
            )
        if not isinstance(self.patience, int) or isinstance(self.patience, bool):
            raise TypeError(
                f"EarlyStoppingConfig.patience must be int, got {type(self.patience)}"
            )
        if self.enabled and self.patience < 1:
            raise ValueError(
                f"EarlyStoppingConfig.patience must be >= 1 when enabled, "
                f"got {self.patience}"
            )
        if not isinstance(self.min_delta, (int, float)) or isinstance(
            self.min_delta, bool
        ):
            raise TypeError(
                f"EarlyStoppingConfig.min_delta must be float, "
                f"got {type(self.min_delta)}"
            )
        self.min_delta = float(self.min_delta)
        if not math.isfinite(self.min_delta) or self.min_delta < 0.0:
            raise ValueError(
                f"EarlyStoppingConfig.min_delta must be finite and >= 0.0, "
                f"got {self.min_delta}"
            )


@dataclass
class TrainingConfig:
    learning_rate: float = 2e-4
    min_learning_rate: float = 0.0
    warmup_ratio: float = 0.0
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    context_length: int = 2048
    max_steps: int = 5000
    eval_every_steps: int = 200
    save_every_steps: int = 500
    log_every_steps: int = 20
    train_corpus: str = ""
    val_corpus: str = ""
    # Issue #135 / Phase 4-1: optional mixed-corpus training.
    # When ``train_corpora_mix`` is set, the trainer uses
    # ``photon_mlx.data.iterate_mixed_batches`` instead of the legacy
    # single ``train_corpus`` path. ``val_split`` carves a held-out
    # fraction from the train mixture (DR1-005: simpler than a separate
    # ``val_corpora_mix`` dict — train and val share the same ratio).
    train_corpora_mix: dict[str, float] | None = None
    val_split: float = 0.0
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)

    def __post_init__(self) -> None:
        # DR1-003 strict validation: invalid mixes must fail at config
        # construction so the trainer is never built around a broken spec.
        if self.train_corpora_mix is not None:
            mix = self.train_corpora_mix
            if not isinstance(mix, dict) or not mix:
                raise ValueError(
                    "train_corpora_mix must be a non-empty dict, got "
                    f"{type(mix).__name__}"
                )
            total = 0.0
            for path, weight in mix.items():
                if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                    raise TypeError(
                        f"train_corpora_mix weight must be a real number, "
                        f"got {type(weight).__name__} for {path!r}"
                    )
                if not math.isfinite(float(weight)):
                    raise ValueError(
                        f"train_corpora_mix weight must be finite, "
                        f"got {weight!r} for {path!r}"
                    )
                if weight <= 0.0:
                    raise ValueError(
                        f"train_corpora_mix weight must be > 0, "
                        f"got {weight} for {path!r}"
                    )
                total += float(weight)
            if abs(total - 1.0) > 1e-6:
                raise ValueError(
                    f"train_corpora_mix weights must sum to 1.0 (±1e-6), "
                    f"got {total} (DR1-003)"
                )

        if not isinstance(self.val_split, (int, float)) or isinstance(
            self.val_split, bool
        ):
            raise TypeError(
                f"val_split must be a real number, got {type(self.val_split).__name__}"
            )
        if self.val_split < 0.0 or self.val_split >= 1.0:
            raise ValueError(
                f"val_split must be in [0, 1), got {self.val_split} (DR1-005)"
            )


@dataclass
class PhotonConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    training: TrainingConfig | None = None


def _set_fields(dc: Any, raw: dict) -> None:
    for k, v in raw.items():
        if hasattr(dc, k):
            setattr(dc, k, v)
        else:
            _logger.warning(
                "unknown config key ignored: %s (in %s)",
                k,
                type(dc).__name__,
            )


def _prod(xs: list[int]) -> int:
    result = 1
    for x in xs:
        result *= int(x)
    return result


def _validate_cross_config(
    model: ModelConfig,
    hierarchy: HierarchyConfig,
    training: TrainingConfig,
) -> None:
    """Enforce cross-dataclass invariants that can only be checked at load time.

    - ``training.context_length`` must be a multiple of ``prod(hierarchy.chunk_sizes)``
      so PHOTON can chunk the sequence cleanly into hierarchical tiles.
    - ``training.context_length`` must not exceed ``model.max_position_embeddings``
      since the RoPE table is precomputed to that length.
    """
    cl = int(training.context_length)
    cl_mult = _prod(hierarchy.chunk_sizes)
    if cl_mult > 0 and cl % cl_mult != 0:
        raise ValueError(
            f"training.context_length ({cl}) must be a multiple of "
            f"prod(hierarchy.chunk_sizes)={cl_mult}"
        )
    if cl > int(model.max_position_embeddings):
        raise ValueError(
            f"training.context_length ({cl}) must be <= "
            f"model.max_position_embeddings ({model.max_position_embeddings})"
        )


def load_photon_config(path: str | Path) -> PhotonConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = PhotonConfig()
    _set_fields(cfg.model, raw.get("model", {}))
    # Re-run ModelConfig.__post_init__ to validate fields set via _set_fields
    # (setattr does not trigger __post_init__).
    cfg.model.__post_init__()
    _set_fields(cfg.hierarchy, raw.get("hierarchy", {}))
    _set_fields(cfg.tokenizer, raw.get("tokenizer", {}))
    if "training" in raw:
        cfg.training = TrainingConfig()
        training_raw = dict(raw["training"] or {})
        # Nested early_stopping dataclass: pop first so _set_fields doesn't
        # assign a raw dict to the field (which would break typing).
        # Explicit kwargs expansion here lets typos surface as TypeError.
        es_raw = training_raw.pop("early_stopping", None)
        _set_fields(cfg.training, training_raw)
        if es_raw is not None:
            if not isinstance(es_raw, dict):
                raise TypeError(
                    "training.early_stopping must be a mapping, "
                    f"got {type(es_raw).__name__}"
                )
            cfg.training.early_stopping = EarlyStoppingConfig(**es_raw)
        # Soft check: warn when patience * eval_every_steps would prevent any
        # stop from ever firing within max_steps.
        es = cfg.training.early_stopping
        if (
            es.enabled
            and cfg.training.eval_every_steps > 0
            and es.patience * cfg.training.eval_every_steps >= cfg.training.max_steps
        ):
            _logger.warning(
                "Early stopping may not trigger within max_steps: "
                "patience=%d * eval_every_steps=%d >= max_steps=%d",
                es.patience,
                cfg.training.eval_every_steps,
                cfg.training.max_steps,
            )
    if cfg.training is not None:
        _validate_cross_config(cfg.model, cfg.hierarchy, cfg.training)
    return cfg
