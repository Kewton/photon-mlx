"""CI gate: production code paths must not contain test/dev scaffolding symbols.

Issue #139 / Task 1: ``_StubTokenizer`` / ``_get_stub_tokenizer`` were removed
from ``baseline_reporag/photon_pipeline.py`` to close the structural gap that
let S7-001 (random-init weights in production eval) go undetected. This test
locks the door behind us — any future ``_Stub`` / ``_Mock`` / ``_Dummy`` /
``_Placeholder`` named symbol that lands in a production module fails CI.

The test is anchored to the repository root via ``__file__`` so it is
cwd-independent (Issue #139 / DR Stage 7 S7-001 / DR4-003: pytest invoked
from a sub-directory must not silently scan zero files and produce a false
pass).
"""

from __future__ import annotations

import re
from pathlib import Path

# Repo-root anchor (Issue #139 / S7-001): never use cwd-relative ``Path('...')``
# here. Going from ``tests/test_no_scaffolding_in_prod.py`` up one level
# ``parents[1]`` lands on the repository root regardless of where pytest is
# invoked. If this file ever moves, update ``parents[N]`` to match.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Production code roots that must stay free of test/dev scaffolding naming.
# bench/, scripts/, demo/ are excluded by design — they are research harnesses
# and CLI utilities that don't run in the production request path (see
# Issue #139 設計判断 #1 / scope section).
PROD_ROOTS = (
    REPO_ROOT / "baseline_reporag",
    REPO_ROOT / "photon_mlx",
    REPO_ROOT / "torch_ref",
)

# Path components that flag a file as test fixture / generated artifact.
# ``tests`` is **plural** (matches actual repo layout); ``test`` (singular)
# would NOT match ``('baseline_reporag', 'tests', 'foo.py')`` and would let
# fixtures slip through (Issue #139 / DR Stage 3 S3-003).
EXCLUDED_DIR_PARTS = frozenset({"tests", "__pycache__"})

# Identifiers that indicate test/dev scaffolding. The pattern is
# ``\b_Stub`` / ``\b_Mock`` / ``\b_Dummy`` / ``\b_Placeholder`` followed by
# any identifier characters (``\w*``). Earlier draft used ``r'_Stub\b'``,
# which fails: ``\b`` does not match between underscore-prefixed identifiers
# and following word characters (DR Stage 3 S3-007 / DR Stage 1 DR1-005).
FORBIDDEN_PATTERN = re.compile(r"\b_(?:Stub|Mock|Dummy|Placeholder)\w*")

# Hardening (Issue #139 / DR4-003): cap individual file size to detect
# anomalous files (e.g. accidentally committed binaries or compressed data
# that ``read_text`` would still try to decode). 1 MiB is well above typical
# Python source size in this repo.
_MAX_FILE_BYTES = 1 * 1024 * 1024


def _is_excluded(path: Path) -> bool:
    return bool(EXCLUDED_DIR_PARTS.intersection(path.parts))


def test_no_scaffolding_naming_in_production() -> None:
    """No production module under PROD_ROOTS may contain ``_Stub`` / ``_Mock`` /
    ``_Dummy`` / ``_Placeholder`` -prefixed identifiers.

    Hardening rules (DR4-003):
    - Symbolic links inside production roots are flagged as violations to
      prevent root-escape via a future ``ln -s /etc/passwd
      baseline_reporag/leak.py``.
    - Files larger than 1 MiB are flagged (anomalous source size).
    - ``UnicodeDecodeError`` during read is flagged (binary or
      non-UTF-8 file under a production source tree is itself a violation).
    - Missing root directory is a hard failure (no silent zero-file pass).
    """

    violations: list[tuple[str, str]] = []

    for root in PROD_ROOTS:
        # If a production root has gone missing the test must fail loudly,
        # not silently scan zero files (DR4-003).
        assert root.is_dir(), f"production root missing: {root}"

        for f in root.rglob("*.py"):
            if _is_excluded(f):
                continue

            # Symlink leaving the root would pull in arbitrary content.
            if f.is_symlink():
                violations.append((str(f), "<symlink>"))
                continue

            # Make sure resolution stays inside the repo root after symlink
            # canonicalization. Defense-in-depth alongside the is_symlink
            # check above.
            try:
                resolved = f.resolve()
            except OSError as exc:  # pragma: no cover - filesystem race
                violations.append((str(f), f"<unresolvable: {type(exc).__name__}>"))
                continue
            if not _is_within(resolved, REPO_ROOT):
                violations.append((str(f), "<resolves outside repo root>"))
                continue

            try:
                size = f.stat().st_size
            except OSError as exc:  # pragma: no cover - filesystem race
                violations.append((str(f), f"<unstatable: {type(exc).__name__}>"))
                continue
            if size > _MAX_FILE_BYTES:
                violations.append((str(f), f"<oversize:{size}B>"))
                continue

            try:
                content = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # A non-UTF-8 file under production source tree is itself a
                # violation (production code is expected to be UTF-8 .py).
                violations.append((str(f), "<non-utf-8>"))
                continue

            for match in FORBIDDEN_PATTERN.finditer(content):
                violations.append((str(f), match.group(0)))

    assert not violations, (
        "Scaffolding naming found in production code paths. "
        "Move test fixtures under */tests/ or rename. "
        f"Violations: {violations}"
    )


def _is_within(child: Path, parent: Path) -> bool:
    """Backport of ``Path.is_relative_to`` for clarity (Python 3.12+ has it)."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False
