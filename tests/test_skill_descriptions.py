"""Skill description / docs / CLAUDE.md / PM command snippet tests (Issue #140).

These tests pin the wording added in Issue #140 (S7-001 follow-up) so that
future edits can't silently drop:
- "Codex 担当 Stage は必須" wording in the two multi-stage review skills
- reviewer="codex" verification snippets in pm-auto-issue2dev / pm-auto-design2dev
- the new docs/code_review_checklist.md
- the four target slash commands listed in CLAUDE.md

The reviewer snippet smoke test (parametrized over issue-review / design-review
× reviewer="codex" / "claude" / missing) extracts the snippet via marker
comments and runs it under bash to confirm WARNING / silent behavior.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILL_DIR = REPO_ROOT / ".claude" / "commands"
DOCS_DIR = REPO_ROOT / "docs"

# DR2-007 / DR3-003: snippet extraction by marker range (not by ```bash fence)
_SNIPPET_RE = re.compile(
    r"#\s*REVIEWER_VERIFICATION_SNIPPET_BEGIN\s*\((?P<kind>issue-review|design-review)\)\s*\n"
    r"(?P<body>.*?)\n"
    r"#\s*REVIEWER_VERIFICATION_SNIPPET_END",
    re.S,
)


def _extract_snippets(md_path: Path) -> dict[str, str]:
    """Return {kind: body} for each REVIEWER_VERIFICATION_SNIPPET marker block."""
    md = md_path.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for m in _SNIPPET_RE.finditer(md):
        out[m.group("kind")] = m.group("body")
    return out


# ─────────────────────────────────────────────────────────────────────
# String-existence tests (Task 1.7 / Issue #140 §7.1)
# ─────────────────────────────────────────────────────────────────────


def test_design_review_codex_required():
    """multi-stage-design-review.md には Codex 担当 Stage 必須 + WARNING + completion report の文言が必須 (DR2-009)."""
    body = (SKILL_DIR / "multi-stage-design-review.md").read_text(encoding="utf-8")
    assert "Codex 担当 Stage は必須" in body
    assert "WARNING" in body
    assert "completion report" in body


def test_issue_review_codex_required():
    """multi-stage-issue-review.md には Codex 担当 Stage 必須 + WARNING + completion report の文言が必須."""
    body = (SKILL_DIR / "multi-stage-issue-review.md").read_text(encoding="utf-8")
    assert "Codex 担当 Stage は必須" in body
    assert "WARNING" in body
    assert "completion report" in body


def test_pm_auto_issue2dev_reviewer_check_snippet():
    """pm-auto-issue2dev.md には issue-review と design-review の両 reviewer snippet が存在する."""
    snippets = _extract_snippets(SKILL_DIR / "pm-auto-issue2dev.md")
    assert "issue-review" in snippets, "issue-review snippet missing"
    assert "design-review" in snippets, "design-review snippet missing"
    assert (
        'reviewer="codex"' in snippets["issue-review"]
        or "reviewer" in snippets["issue-review"]
    )
    assert "stage" in snippets["issue-review"]
    assert "stage" in snippets["design-review"]


def test_pm_auto_design2dev_reviewer_check_snippet():
    """pm-auto-design2dev.md には design-review reviewer snippet が存在する."""
    snippets = _extract_snippets(SKILL_DIR / "pm-auto-design2dev.md")
    assert "design-review" in snippets, "design-review snippet missing"
    assert "stage" in snippets["design-review"]


def test_claude_md_lists_target_skills():
    """CLAUDE.md スラッシュコマンド表に Issue #140 で必須化した 4 skill が記載されている (S3-003)."""
    body = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    for skill in [
        "/multi-stage-issue-review",
        "/multi-stage-design-review",
        "/pm-auto-issue2dev",
        "/pm-auto-design2dev",
    ]:
        assert skill in body, f"{skill} not in CLAUDE.md"


def test_code_review_checklist_exists():
    """docs/code_review_checklist.md が存在し必須キーワードを含む."""
    path = DOCS_DIR / "code_review_checklist.md"
    assert path.exists(), "docs/code_review_checklist.md missing"
    body = path.read_text(encoding="utf-8")
    assert "_Stub" in body
    assert "_Mock" in body
    assert "*/tests/**" in body  # 除外パターン
    assert "S7-001" in body  # 由来の Issue を明記


def test_claude_md_links_code_review_checklist():
    """CLAUDE.md から docs/code_review_checklist.md へのリンクが存在する (Task 3.2)."""
    body = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert "docs/code_review_checklist.md" in body


def test_auto_skip_removed_from_issue_review():
    """multi-stage-issue-review.md の auto-skip 廃止が反映されている (S5-001 / DR1-008)."""
    body = (SKILL_DIR / "multi-stage-issue-review.md").read_text(encoding="utf-8")
    # auto-skip 判定セクション (見出し) が消えているか、廃止が明示されている
    if "2回目イテレーション自動スキップ判定" in body:
        # 残っている場合は「廃止」の記述が同じ近くに必要
        idx = body.index("2回目イテレーション自動スキップ判定")
        snippet = body[max(0, idx - 50) : idx + 600]
        assert "廃止" in snippet, "auto-skip section still present without 廃止 marker"
    # summary 表の「完了/スキップ」表記が「完了 (reviewer=codex 検証済)」へ変更されていること
    assert "完了 (reviewer=codex 検証済)" in body or "完了/スキップ" not in body


# ─────────────────────────────────────────────────────────────────────
# Reviewer snippet smoke test (Task 1.8 / Issue #140 §7.5 / DR4-001)
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kind,md_file",
    [
        ("issue-review", "pm-auto-issue2dev.md"),
        ("design-review", "pm-auto-issue2dev.md"),
        ("design-review", "pm-auto-design2dev.md"),
    ],
)
@pytest.mark.parametrize(
    "reviewer_value,expect_warning",
    [
        ("codex", False),  # 正常系: WARNING なし
        ("claude", True),  # reviewer 不一致: WARNING あり
        (None, True),  # ファイル欠落: WARNING あり
    ],
)
def test_reviewer_snippet_smoke(
    tmp_path, kind, md_file, reviewer_value, expect_warning
):
    """snippet を bash で実行し、reviewer 値に応じて WARNING が出るかを検証 (DR1-005 / DR3-003)."""
    snippets = _extract_snippets(SKILL_DIR / md_file)
    snippet = snippets.get(kind)
    if snippet is None:
        pytest.skip(f"{kind} snippet not found in {md_file}")

    # snippet 内の {issue_number} placeholder は実行時に ISSUE 環境変数で上書きできるよう
    # `${ISSUE:-{issue_number}}` の形を取っている。テストでは ISSUE=140 を明示的に渡す。
    issue_id = "140"

    # 擬似 workspace を tmp_path に構築
    if kind == "issue-review":
        sub = tmp_path / "workspace" / "issues" / issue_id / "issue-review"
        stages = (5, 7)
    else:
        sub = tmp_path / "workspace" / "issues" / issue_id / "multi-stage-design-review"
        stages = (3, 4)
    sub.mkdir(parents=True)

    # reviewer_value=None の場合はファイルを作らない (欠落ケース)
    if reviewer_value is not None:
        for stage in stages:
            (sub / f"stage{stage}-review-result.json").write_text(
                json.dumps({"stage": stage, "reviewer": reviewer_value}),
                encoding="utf-8",
            )

    # snippet 内の {issue_number} placeholder を実 issue 番号に置換
    runnable = snippet.replace("{issue_number}", issue_id)

    # 一時ディレクトリで bash 実行
    result = subprocess.run(
        ["bash", "-c", runnable],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
        env={"ISSUE": issue_id, "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )

    combined = (result.stdout or "") + (result.stderr or "")
    has_warning = "WARNING" in combined

    # CB-003: snippet 自体の syntax / runtime error を検出するため exit code を検証
    assert result.returncode == 0, (
        f"snippet exited with returncode={result.returncode}: stderr={result.stderr!r}"
    )

    if expect_warning:
        assert has_warning, (
            f"expected WARNING for {kind=} reviewer={reviewer_value!r} but got: {combined!r}"
        )
    else:
        assert not has_warning, (
            f"unexpected WARNING for {kind=} reviewer={reviewer_value!r}: {combined!r}"
        )


@pytest.mark.parametrize(
    "malicious_issue",
    [
        "../etc",  # path traversal
        "140;touch injected",  # command injection attempt
        "140 && touch injected",
        '140"injected',  # quote injection
    ],
)
def test_reviewer_snippet_rejects_invalid_issue(tmp_path, malicious_issue):
    """invalid ISSUE 値で path traversal / command injection が起きないことを確認 (DR4-001)."""
    snippets = _extract_snippets(SKILL_DIR / "pm-auto-issue2dev.md")
    snippet = snippets.get("issue-review")
    if snippet is None:
        pytest.skip("issue-review snippet missing")

    runnable = snippet.replace("{issue_number}", "140")

    # ISSUE 変数を悪意ある値に
    result = subprocess.run(
        ["bash", "-c", runnable],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
        env={"ISSUE": malicious_issue, "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    # ファイルを書き込んでいないこと (command injection 防止)
    assert not (tmp_path / "injected").exists()
    # CB-003: snippet 自体の syntax / runtime error を検出
    assert result.returncode == 0, (
        f"snippet exited with returncode={result.returncode}: stderr={result.stderr!r}"
    )
    # WARNING を出して安全に skip していること
    combined = (result.stdout or "") + (result.stderr or "")
    assert "WARNING" in combined
