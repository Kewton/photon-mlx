"""generate_commandmate_eval_set.py — MT eval set for commandmate (TS+Next.js).

Hand-crafted 15 sessions × 6 turns covering:
- Worktree management, tmux/Claude CLI integration, SQLite session persistence,
- Next.js frontend, CLI tools, i18n, recent fixes (sidebar tooltip, PDF preview),
- issue management, auto-yes, multi-agent orchestration, tests, deployment.

Output: data/eval_sets/commandmate_multi_turn_eval.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

SESSIONS = [
    {
        "session_id": "CM-MT-001",
        "category": "onboarding",
        "scenario": "プロジェクト全体構造の段階的把握",
        "session_tags": ["topic_narrowing", "commandmate"],
        "questions": [
            "CommandMate プロジェクトの全体構造とディレクトリ役割を教えてください。",
            "Next.js の app router 配下のページ構成はどうなっていますか?",
            "API ルートはどのファイルで定義されていますか?",
            "src/lib 配下のユーティリティはどんな種類がありますか?",
            "CLI ツール (bin/) と Web UI の連携はどう実装されていますか?",
            "TypeScript の型定義は src/types でどう整理されていますか?",
        ],
    },
    {
        "session_id": "CM-MT-002",
        "category": "feature_implementation",
        "scenario": "Worktree 機能の実装深掘り",
        "session_tags": ["topic_narrowing", "commandmate", "worktree"],
        "questions": [
            "CommandMate の Worktree 管理機能はどう実装されていますか?",
            "Worktree の作成 (commandmate worktree add) はどのコードで行われますか?",
            "Worktree のメタデータ (branch, status) はどこに保存されますか?",
            "Worktree 一覧表示の React コンポーネントはどこにありますか?",
            "Worktree 削除時のクリーンアップ処理は何をしていますか?",
            "Worktree 機能のテストはどのファイルで定義されていますか?",
        ],
    },
    {
        "session_id": "CM-MT-003",
        "category": "feature_implementation",
        "scenario": "tmux + Claude CLI 統合",
        "session_tags": ["topic_narrowing", "commandmate", "tmux"],
        "questions": [
            "CommandMate と tmux の統合はどのように実装されていますか?",
            "tmux セッションの起動はどのコードで行われますか?",
            "Claude CLI agent の選択 (claude / codex / gemini) はどう切り替えますか?",
            "agent 切り替え時の tmux 操作の中身は何ですか?",
            "tmux セッションへのメッセージ送信 (commandmate send) はどう動作しますか?",
            "auto-yes 機能はどう実装されていますか?",
        ],
    },
    {
        "session_id": "CM-MT-004",
        "category": "feature_implementation",
        "scenario": "SQLite による状態永続化",
        "session_tags": ["drill_down", "commandmate", "database"],
        "questions": [
            "CommandMate のデータ永続化はどうなっていますか?",
            "SQLite のスキーマ定義はどのファイルにありますか?",
            "セッション状態の保存と読み込みのコードは?",
            "DB マイグレーションの仕組みは実装されていますか?",
            "better-sqlite3 のラッパーはどう抽象化されていますか?",
            "DB の単体テストはどう実装されていますか?",
        ],
    },
    {
        "session_id": "CM-MT-005",
        "category": "frontend",
        "scenario": "Next.js + React UI 構成",
        "session_tags": ["drill_down", "commandmate", "frontend"],
        "questions": [
            "CommandMate の React コンポーネントはどう構成されていますか?",
            "サイドバー UI の実装は src/components のどこにありますか?",
            "ブランチツールチップのバグ修正 (#676) は何を変更しましたか?",
            "状態管理 (context / hooks) はどう設計されていますか?",
            "Tailwind の設定とテーマカスタマイズはどこで定義していますか?",
            "PDF プレビュー機能 (#673 fix) はどのコンポーネントで実装されていますか?",
        ],
    },
    {
        "session_id": "CM-MT-006",
        "category": "feature_implementation",
        "scenario": "GitHub Issue 管理機能",
        "session_tags": ["topic_narrowing", "commandmate", "github"],
        "questions": [
            "CommandMate の Issue 管理機能はどう実装されていますか?",
            "GitHub API への認証はどう行っていますか?",
            "Issue の作成・更新フローの実装ファイルはどれですか?",
            "GitHub App の token 取得スクリプトはどこにありますか?",
            "Issue templating (bug / feature) の仕組みは?",
            "Issue 関連の単体テストは何を検証していますか?",
        ],
    },
    {
        "session_id": "CM-MT-007",
        "category": "cli",
        "scenario": "CLI tool の構成",
        "session_tags": ["topic_narrowing", "commandmate", "cli"],
        "questions": [
            "CommandMate の CLI バイナリ (bin/) はどう構成されていますか?",
            "commandmate コマンドのエントリポイントはどのファイルですか?",
            "サブコマンド (worktree / send / wait / capture) の dispatch はどう実装されていますか?",
            "auto-yes サブコマンドの動作仕様は?",
            "wait コマンドはどうやってエージェントの完了を検知していますか?",
            "respond コマンドはどう agent の prompt に応答しますか?",
        ],
    },
    {
        "session_id": "CM-MT-008",
        "category": "feature_implementation",
        "scenario": "i18n 実装",
        "session_tags": ["drill_down", "commandmate", "i18n"],
        "questions": [
            "CommandMate の多言語対応はどう実装されていますか?",
            "翻訳ファイルはどこに置かれていますか?",
            "言語切り替え UI のコンポーネントはどれですか?",
            "i18n.ts のセットアップコードは何をしていますか?",
            "新しい翻訳キー追加の手順は?",
            "i18n 関連のテストはありますか?",
        ],
    },
    {
        "session_id": "CM-MT-009",
        "category": "tests",
        "scenario": "テスト構成と CI",
        "session_tags": ["topic_narrowing", "commandmate", "tests"],
        "questions": [
            "CommandMate のテスト戦略を教えてください。",
            "Vitest による unit test の設定はどうなっていますか?",
            "Playwright による e2e テストの設定はどこにありますか?",
            "CI ワークフロー (.github/workflows) で何を実行していますか?",
            "Snapshot tests は何を保証していますか?",
            "テスト用の fixtures / mocks はどう用意されていますか?",
        ],
    },
    {
        "session_id": "CM-MT-010",
        "category": "feature_implementation",
        "scenario": "Multi-agent orchestration",
        "session_tags": ["drill_down", "commandmate", "multi-agent"],
        "questions": [
            "CommandMate のマルチエージェント機能とは何ですか?",
            "複数の worktree を並列で動かす仕組みはどう実装されていますか?",
            "エージェント間の状態共有や同期はどう設計されていますか?",
            "Issue から自動で worktree + agent を立ち上げる流れは?",
            "report (daily / weekly) の生成ロジックはどこにありますか?",
            "エージェントのリソース監視 (CPU / RAM) はしていますか?",
        ],
    },
    {
        "session_id": "CM-MT-011",
        "category": "config",
        "scenario": "設定管理",
        "session_tags": ["drill_down", "commandmate", "config"],
        "questions": [
            "CommandMate のユーザー設定はどう管理されていますか?",
            "init コマンドが作成する設定ファイルは何ですか?",
            "設定スキーマと TypeScript 型はどう連動していますか?",
            "設定の上書き優先順位 (env / file / CLI flag) は?",
            "デフォルト値はどこに定義されていますか?",
            "設定ファイル更新時の hot reload はありますか?",
        ],
    },
    {
        "session_id": "CM-MT-012",
        "category": "deployment",
        "scenario": "デプロイ・配布",
        "session_tags": ["topic_narrowing", "commandmate", "deployment"],
        "questions": [
            "CommandMate のデプロイ手順を教えてください。",
            "Next.js アプリのビルドと起動のコマンドは?",
            "ローカル開発環境のセットアップ手順は?",
            "Docker でのコンテナ化はサポートされていますか?",
            "self-hosted GitHub Actions runner との連携はどう設定しますか?",
            "リリース版の配布形態は何ですか? (npm / standalone binary)",
        ],
    },
    {
        "session_id": "CM-MT-013",
        "category": "feature_implementation",
        "scenario": "セキュリティ機能",
        "session_tags": ["drill_down", "commandmate", "security"],
        "questions": [
            "CommandMate でのトークン管理はどう設計されていますか?",
            "GitHub App private key はどこに保存されていますか?",
            "API ルートの認証はどうしていますか?",
            "ローカル LLM 呼び出し時の secret leakage 対策は?",
            "ログ出力時の機密情報マスキングはありますか?",
            "セキュリティ関連の lint / scanner は導入されていますか?",
        ],
    },
    {
        "session_id": "CM-MT-014",
        "category": "frontend",
        "scenario": "リアルタイム UI 更新",
        "session_tags": ["drill_down", "commandmate", "frontend"],
        "questions": [
            "worktree status の表示は何秒間隔で更新されますか?",
            "tmux session 出力のストリーミング表示はどう実装されていますか?",
            "WebSocket / SSE / polling のどれを使っていますか?",
            "切断時の再接続ロジックはどうなっていますか?",
            "UI 状態を React で管理する際の最適化 (memo / useCallback) は?",
            "大量ログ表示時のパフォーマンス対策は?",
        ],
    },
    {
        "session_id": "CM-MT-015",
        "category": "feature_implementation",
        "scenario": "agent 自動化フロー",
        "session_tags": ["topic_narrowing", "commandmate", "automation"],
        "questions": [
            "CommandMate でのエージェント自動化フローを教えてください。",
            "issue create → worktree setup → agent run の chain はどう繋がっていますか?",
            "report 自動生成 (daily) のスケジュール実行はどう実装されていますか?",
            "エラー時のリトライロジックはどこにありますか?",
            "agent 出力のキャプチャと永続化はどうしていますか?",
            "完了検知後の next-step トリガーは何で実装されていますか?",
        ],
    },
]


def main() -> None:
    out_path = Path("data/eval_sets/commandmate_multi_turn_eval.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for s in SESSIONS:
            session = {
                "session_id": s["session_id"],
                "category": s["category"],
                "scenario": s["scenario"],
                "session_tags": s["session_tags"],
                "turns": [
                    {
                        "turn_id": i + 1,
                        "question": q,
                        "reference_answer": "",
                        "reference_chunk_ids": [],
                        "grading_notes": "",
                        "tags": [],
                    }
                    for i, q in enumerate(s["questions"])
                ],
            }
            f.write(json.dumps(session, ensure_ascii=False) + "\n")

    total_turns = sum(len(s["questions"]) for s in SESSIONS)
    print(f"Generated {len(SESSIONS)} sessions × 6 turns = {total_turns} turns")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
