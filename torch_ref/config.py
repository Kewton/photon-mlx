"""Shared config loader for torch_ref and photon_mlx."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


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
class PhotonConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    hierarchy: HierarchyConfig = field(default_factory=HierarchyConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)


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
    return cfg
