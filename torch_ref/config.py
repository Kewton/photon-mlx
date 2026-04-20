"""Shared config loader for torch_ref and photon_mlx."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_logger = logging.getLogger(__name__)


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
    norm_eps: float = 1e-5
    tie_word_embeddings: bool = False
    dropout: float = 0.0
    bias: bool = False


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
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)


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


def load_photon_config(path: str | Path) -> PhotonConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = PhotonConfig()
    _set_fields(cfg.model, raw.get("model", {}))
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
    return cfg
