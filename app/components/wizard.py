"""Project-wizard helpers for the Streamlit app (Issue #82 Wave 2 / 5).

Never uses :func:`yaml.load`; only :func:`yaml.safe_load` is permitted
(D4-005). This module is streamlit-free — the wizard *UI* lives in
``app/photon_app.py`` and only depends on the pure helpers exported here.

Wave 2 provides the YAML safety allowlist (:func:`_assert_safe_yaml`).
Wave 5 adds :func:`apply_best_practice` (5-key merge) and
:func:`generate_yaml_from_wizard` (toggle → YAML).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # PyYAML — safe_load-only, see module docstring

# Only these scalar / container types are permitted in a loaded YAML tree
# (D4-005 reflected). Anything else — including custom Python objects
# produced by ``!!python/object`` tags — causes ``_assert_safe_yaml`` to
# raise :class:`ValueError`.
ALLOWED_YAML_TYPES: tuple[type, ...] = (
    str,
    int,
    float,
    bool,
    type(None),
    list,
    dict,
)


# Issue #82 Wave 5 (W5-T1): the 5 best-practice keys we merge into a
# profile YAML when the user opts in to ``[Apply best-practice]``.  The
# key is a tuple path into the nested YAML dict; the value is the target
# state expected by the current operational guidance (design §7.3).
BEST_PRACTICE_KEYS: dict[tuple[str, ...], Any] = {
    ("safe_recgen", "enabled"): True,
    ("generation", "evidence_pruning_enabled"): True,
    ("session_memory", "working_memory", "enabled"): True,
    ("inference", "photon_generation_enabled"): False,
    ("retrieval", "two_pass_search", "enabled"): False,
}


# Allowlist for ``inference.generation_fallback_policy`` — the wizard
# rejects any other value early so the saved YAML can never reference an
# unsupported policy string.
ALLOWED_FALLBACK_POLICIES: tuple[str, ...] = ("qwen", "abort")


# Profiles where a pre-existing non-best-practice value is *intentional*
# (e.g. ``photon_tiny_recgen`` ships RecGen ON for experiments, and
# ``photon_tiny`` / ``photon_600m_paper`` omit working_memory on purpose).
# For these profiles we still apply the best-practice override when the
# user asks, but we surface a warning so the operator notices the drift.
_INTENTIONAL_CONFLICT_PROFILES: frozenset[str] = frozenset(
    {"photon_tiny_recgen", "photon_tiny", "photon_600m_paper"}
)


def _assert_safe_yaml(obj: Any, path: str = "<root>") -> None:
    """Recursively verify ``obj`` contains only :data:`ALLOWED_YAML_TYPES`.

    ``yaml.safe_load`` already rejects most dangerous tags (e.g.
    ``!!python/object/apply:os.system``) with a
    :class:`yaml.constructor.ConstructorError`, so in practice this
    function is the second line of defense for values that *do* load
    cleanly but whose types still fall outside our allowlist (e.g. a
    :class:`tuple` sneaking in via some custom loader variant).

    Args:
        obj: The parsed YAML tree (or any nested value).
        path: Dotted path used for error messages, e.g. ``"<root>.foo[2]"``.

    Raises:
        ValueError: on first encountered non-allowlisted type, mentioning
            the offending type and its dotted path.
    """

    # ``bool`` is a subclass of ``int`` in Python; isinstance handles it
    # correctly because ``bool`` is listed explicitly.
    if not isinstance(obj, ALLOWED_YAML_TYPES):
        raise ValueError(f"Unsafe YAML type {type(obj).__name__!r} at {path}")

    if isinstance(obj, dict):
        for k, v in obj.items():
            # Keys must also be primitives (never a custom object).
            if not isinstance(k, (str, int, float, bool, type(None))):
                raise ValueError(f"Unsafe YAML key type {type(k).__name__!r} at {path}")
            _assert_safe_yaml(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_safe_yaml(v, f"{path}[{i}]")


def _deep_get(d: dict, path: tuple[str, ...]) -> Any:
    """Return ``d[path[0]][path[1]]…`` or raise :class:`KeyError`.

    Raises:
        KeyError: if any intermediate key is missing or a non-mapping
            node is encountered on the traversal.
    """

    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            raise KeyError(path)
        cur = cur[k]
    return cur


def _deep_set(d: dict, path: tuple[str, ...], value: Any) -> None:
    """Create nested dicts as needed, then set ``d[path[0]]…[path[-1]] = value``.

    Any non-dict node encountered along ``path[:-1]`` is overwritten with
    an empty dict — the caller is expected to have validated ``d`` via
    :func:`_assert_safe_yaml` before invoking this helper.
    """

    cur = d
    for k in path[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value


def apply_best_practice(
    yaml_text: str,
    profile: str,
    overrides: dict[tuple[str, ...], Any] | None = None,
) -> tuple[str, list[str]]:
    """Merge best-practice keys into a YAML document (Issue #82 Wave 5).

    Loads ``yaml_text`` with :func:`yaml.safe_load` only, validates it
    with :func:`_assert_safe_yaml`, then for each ``(path, value)`` in
    ``overrides``:

    - if ``path`` is missing entirely → create the nested keys and add a
      warning ``"Added new path … to profile …"``;
    - if the existing value differs from the target AND ``profile`` is one
      of :data:`_INTENTIONAL_CONFLICT_PROFILES` → add a warning noting the
      profile's intentional setting; in all differ cases the override is
      still applied;
    - if the existing value equals the target → no-op, no warning.

    Then dump back with ``yaml.safe_dump(sort_keys=False, allow_unicode=True)``
    so the resulting text keeps key ordering stable across round-trips.

    Args:
        yaml_text: Source YAML document (typically a profile file).
        profile: Profile name (e.g. ``"photon_small"``) used in warnings
            and to decide whether conflicts are intentional.
        overrides: Optional custom override mapping; defaults to
            :data:`BEST_PRACTICE_KEYS`.

    Returns:
        Tuple of ``(new_yaml_text, warnings)``.

    Raises:
        ValueError: if ``yaml_text`` contains non-allowlisted YAML types
            (see :func:`_assert_safe_yaml`), or the top-level node is not
            a mapping.
    """

    if overrides is None:
        overrides = BEST_PRACTICE_KEYS
    doc = yaml.safe_load(yaml_text) or {}
    if not isinstance(doc, dict):
        raise ValueError("Top-level YAML must be a mapping")
    _assert_safe_yaml(doc)

    warnings: list[str] = []
    for path, target in overrides.items():
        try:
            current = _deep_get(doc, path)
        except KeyError:
            _deep_set(doc, path, target)
            warnings.append(
                f"Added new path {'.'.join(path)} = {target} to profile {profile}"
            )
            continue

        if current == target:
            continue

        if profile in _INTENTIONAL_CONFLICT_PROFILES:
            warnings.append(
                f"Overriding {'.'.join(path)}: {current} -> {target} "
                f"(profile {profile} has intentional setting)"
            )
        _deep_set(doc, path, target)

    new_text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
    return new_text, warnings


# Wizard-form toggle key → nested YAML dotted path.  Centralised so the
# UI layer can iterate ``user_toggles`` without duplicating the mapping.
_WIZARD_TOGGLE_MAPPING: dict[str, tuple[str, ...]] = {
    "recgen_enabled": ("inference", "photon_generation_enabled"),
    "fallback_policy": ("inference", "generation_fallback_policy"),
    "two_pass_search_enabled": ("retrieval", "two_pass_search", "enabled"),
    "two_pass_pass1_top_k": ("retrieval", "two_pass_search", "pass1_top_k"),
    "two_pass_pass2_top_k": ("retrieval", "two_pass_search", "pass2_top_k"),
    "working_memory_enabled": ("session_memory", "working_memory", "enabled"),
    "working_memory_max_turns": ("session_memory", "working_memory", "max_turns"),
    "working_memory_aggregation": ("session_memory", "working_memory", "aggregation"),
    "working_memory_storage_mode": ("session_memory", "working_memory", "storage_mode"),
    "past_turn_pinning_enabled": (
        "session_memory",
        "working_memory",
        "past_turn_pinning_enabled",
    ),
}


def generate_yaml_from_wizard(
    base_profile: str,
    user_toggles: dict[str, Any],
    base_yaml_text: str | None = None,
) -> str:
    """Generate a YAML document from a base profile + wizard form toggles.

    ``user_toggles`` supports the following keys (extra keys are
    silently ignored so the UI can feed in opaque form-state dicts):

    - ``recgen_enabled`` (bool) → ``inference.photon_generation_enabled``
    - ``fallback_policy`` (``"qwen"`` | ``"abort"``) →
      ``inference.generation_fallback_policy``
    - ``two_pass_search_enabled`` (bool) →
      ``retrieval.two_pass_search.enabled``
    - ``two_pass_pass1_top_k`` (int) → ``retrieval.two_pass_search.pass1_top_k``
    - ``two_pass_pass2_top_k`` (int) → ``retrieval.two_pass_search.pass2_top_k``
    - ``working_memory_enabled`` (bool) →
      ``session_memory.working_memory.enabled``
    - ``working_memory_max_turns`` (int) →
      ``session_memory.working_memory.max_turns``
    - ``working_memory_aggregation``
      (``"weighted"`` | ``"attention"`` | ``"last"``) →
      ``session_memory.working_memory.aggregation``
    - ``working_memory_storage_mode`` (``"full"`` | ``"top_level_only"``) →
      ``session_memory.working_memory.storage_mode``
    - ``past_turn_pinning_enabled`` (bool) →
      ``session_memory.working_memory.past_turn_pinning_enabled``

    Args:
        base_profile: Name of the profile to start from (e.g.
            ``"photon_small"``).  When ``base_yaml_text`` is ``None``,
            the file ``configs/<base_profile>.yaml`` is read from disk.
        user_toggles: Mapping of wizard form values (see above).
        base_yaml_text: Optional pre-loaded profile YAML (used by tests
            to avoid filesystem dependencies).

    Returns:
        YAML text with the toggles applied.

    Raises:
        ValueError: on an invalid ``fallback_policy`` value or an unsafe
            base YAML document.
    """

    fp = user_toggles.get("fallback_policy")
    if fp is not None and fp not in ALLOWED_FALLBACK_POLICIES:
        raise ValueError(
            f"Invalid fallback_policy: {fp!r} "
            f"(must be one of {ALLOWED_FALLBACK_POLICIES})"
        )

    if base_yaml_text is None:
        cfg_path = (
            Path(__file__).resolve().parents[2] / "configs" / f"{base_profile}.yaml"
        )
        base_yaml_text = cfg_path.read_text(encoding="utf-8")

    doc = yaml.safe_load(base_yaml_text) or {}
    if not isinstance(doc, dict):
        raise ValueError("Top-level YAML must be a mapping")
    _assert_safe_yaml(doc)

    for toggle_key, path in _WIZARD_TOGGLE_MAPPING.items():
        if toggle_key in user_toggles:
            _deep_set(doc, path, user_toggles[toggle_key])

    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
