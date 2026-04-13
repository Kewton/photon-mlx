from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

LANGUAGE_MAP: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".sh": "shell",
}


@dataclass
class FileRecord:
    rel_path: str
    abs_path: str
    language: str
    content: str
    size_bytes: int
    line_count: int


def _matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def extract_files(
    repo_path: str | Path,
    include: list[str],
    exclude: list[str],
) -> Iterator[FileRecord]:
    repo_path = Path(repo_path)
    for root, dirs, files in os.walk(repo_path):
        rel_root = Path(root).relative_to(repo_path)
        # Prune excluded directories early
        dirs[:] = [
            d for d in dirs if not _matches_any(str(rel_root / d) + "/", exclude)
        ]
        for fname in sorted(files):
            rel_path = str(rel_root / fname)
            if not _matches_any(rel_path, include):
                continue
            if _matches_any(rel_path, exclude):
                continue
            abs_path = Path(root) / fname
            suffix = abs_path.suffix.lower()
            language = LANGUAGE_MAP.get(suffix, "text")
            try:
                content = abs_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            yield FileRecord(
                rel_path=rel_path,
                abs_path=str(abs_path),
                language=language,
                content=content,
                size_bytes=abs_path.stat().st_size,
                line_count=content.count("\n") + 1,
            )
