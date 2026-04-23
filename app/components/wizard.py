"""Project-wizard helpers for the Streamlit app (Issue #82 Wave 2 / 5).

Never uses :func:`yaml.load`; only :func:`yaml.safe_load` is permitted
(D4-005). This module is streamlit-free — the wizard *UI* lives in
``app/photon_app.py`` and only depends on the pure helpers exported here.

Wave 2 provides the YAML safety allowlist (:func:`_assert_safe_yaml`).
``apply_best_practice`` and ``generate_yaml_from_wizard`` land in Wave 5.
"""

from __future__ import annotations

from typing import Any

import yaml  # noqa: F401  # PyYAML — safe_load-only, see module docstring

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
