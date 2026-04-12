"""
Demo scenarios for PHOTON-RepoRAG.

5 scenarios targeting fastapi/fastapi, each with multi-turn follow-up.
Designed to showcase: working memory, citation, graph expansion,
Safe RecGen fallback, and change comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DemoTurn:
    question: str
    notes: str = ""


@dataclass
class DemoScenario:
    id: str
    title: str
    axis: str
    description: str
    turns: list[DemoTurn] = field(default_factory=list)


SCENARIOS: list[DemoScenario] = [
    # ------------------------------------------------------------------
    # 1. Onboarding
    # ------------------------------------------------------------------
    DemoScenario(
        id="demo-01",
        title="Repo onboarding — dependency injection deep-dive",
        axis="onboarding",
        description=(
            "Start from a high-level repo overview, then progressively "
            "narrow into FastAPI's dependency injection mechanism. "
            "Shows working memory carrying context across turns."
        ),
        turns=[
            DemoTurn(
                "この repo の全体構成と主要モジュールを教えてください。",
                "repo 構造の概要 + citation が出ること",
            ),
            DemoTurn(
                "依存性注入の仕組みをコードで説明してください。",
                "Depends() の実装を引用すること",
            ),
            DemoTurn(
                "新しい機能を足すならどこから読み始めればいいですか？",
                "前ターンの文脈を前提に回答すること (session consistency)",
            ),
        ],
    ),
    # ------------------------------------------------------------------
    # 2. Impact analysis
    # ------------------------------------------------------------------
    DemoScenario(
        id="demo-02",
        title="Impact analysis — Depends() implementation change",
        axis="impact_analysis",
        description=(
            "Analyse the blast radius of changing Depends(). "
            "Shows graph expansion and citation precision."
        ),
        turns=[
            DemoTurn(
                "`Depends()` の実装を変えると波及先はどこですか？",
                "依存グラフに沿った回答 + citation",
            ),
            DemoTurn(
                "その中で壊れやすい箇所はどこですか？",
                "follow-up で絞り込み",
            ),
        ],
    ),
    # ------------------------------------------------------------------
    # 3. Bug localization
    # ------------------------------------------------------------------
    DemoScenario(
        id="demo-03",
        title="Bug localization — 401 in auth middleware",
        axis="bug_localization",
        description=(
            "Identify likely root causes for a 401 error in authentication. "
            "Shows multi-turn hypothesis refinement."
        ),
        turns=[
            DemoTurn(
                "認証ミドルウェアで 401 が返る原因候補を 3 つ出してください。",
                "3 候補 + それぞれ citation",
            ),
            DemoTurn(
                "一番怪しい箇所のコードを示してください。",
                "具体コードの引用 (local refresh が必要なケース)",
            ),
        ],
    ),
    # ------------------------------------------------------------------
    # 4. Safe RecGen trigger
    # ------------------------------------------------------------------
    DemoScenario(
        id="demo-04",
        title="Safe RecGen — exact quote of security handler",
        axis="safe_recgen",
        description=(
            "Request exact code quote from security.py. "
            "Shows Safe RecGen firing on exact_quote trigger "
            "and forcing local re-read."
        ),
        turns=[
            DemoTurn(
                "`security.py` の `get_current_user` を exact quote で出してください。",
                "Safe RecGen: exact_quote trigger → re_retrieve + strengthen_local_refresh",
            ),
            DemoTurn(
                "その関数の引数の型を確認してください。",
                "follow-up: 前ターンの exact quote を前提に型情報を追加",
            ),
        ],
    ),
    # ------------------------------------------------------------------
    # 5. Change planning comparison
    # ------------------------------------------------------------------
    DemoScenario(
        id="demo-05",
        title="Change planning — middleware vs decorator for auth",
        axis="change_planning",
        description=(
            "Compare two approaches for restructuring authorization logic. "
            "Shows drift detection and citation-backed comparison."
        ),
        turns=[
            DemoTurn(
                "認可ロジックを middleware に移す案と decorator に残す案を比較してください。",
                "比較表 + citation + メリット/リスク",
            ),
            DemoTurn(
                "最小変更で済む方法はどれですか？",
                "前ターンの比較を前提に結論",
            ),
        ],
    ),
]


def print_scenarios() -> None:
    """Print all demo scenarios in human-readable format."""
    for s in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"[{s.id}] {s.title}")
        print(f"Axis: {s.axis}")
        print(f"Description: {s.description}")
        for i, t in enumerate(s.turns, 1):
            print(f"  Turn {i}: {t.question}")
            if t.notes:
                print(f"          → {t.notes}")
    print()


if __name__ == "__main__":
    print_scenarios()
