"""
generate_eval_sets.py  –  Generate evaluation sets from an ingested repo.

Produces:
  - data/eval_sets/static_eval.jsonl       (120 questions)
  - data/eval_sets/multi_turn_eval.jsonl   (30 sessions × 6 turns)
  - data/eval_sets/stress_eval.jsonl       (8 sessions × 10 turns)

Usage:
    python scripts/generate_eval_sets.py --repo-id fastapi_fastapi
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline_reporag.config import load_config
from baseline_reporag.ingestion.store import ChunkStore

# ================================================================
# Static eval templates (30 per category)
# ================================================================

ONBOARDING = [
    {"q": "この repo の全体構成と主要ディレクトリの役割を教えてください。", "d": "easy"},
    {"q": "FastAPI のエントリポイントはどのファイルですか？", "d": "easy"},
    {"q": "FastAPI の依存性注入の仕組みを説明してください。", "d": "medium"},
    {"q": "Depends() はどのファイルで定義されていますか？", "d": "easy"},
    {"q": "ルーティングの仕組みを説明してください。", "d": "medium"},
    {"q": "APIRouter はどのように動作しますか？", "d": "medium"},
    {"q": "リクエストのバリデーションはどこで行われますか？", "d": "medium"},
    {"q": "Pydantic モデルとの統合はどのように実装されていますか？", "d": "medium"},
    {"q": "OpenAPI スキーマの自動生成はどこで行われますか？", "d": "medium"},
    {"q": "テストの構成を教えてください。", "d": "easy"},
    {"q": "ミドルウェアの仕組みを説明してください。", "d": "medium"},
    {"q": "例外ハンドリングはどのように実装されていますか？", "d": "medium"},
    {"q": "WebSocket サポートはどこにありますか？", "d": "medium"},
    {"q": "BackgroundTasks はどのように動作しますか？", "d": "medium"},
    {"q": "FastAPI アプリケーションのライフサイクルイベントを説明してください。", "d": "medium"},
    {"q": "テンプレートレスポンスの仕組みを教えてください。", "d": "easy"},
    {"q": "静的ファイルの配信はどのように設定しますか？", "d": "easy"},
    {"q": "CORS の設定方法を教えてください。", "d": "easy"},
    {"q": "FastAPI の内部で Starlette をどのように利用していますか？", "d": "hard"},
    {"q": "パスパラメータとクエリパラメータの処理の違いを説明してください。", "d": "medium"},
    {"q": "レスポンスモデルの仕組みを説明してください。", "d": "medium"},
    {"q": "FastAPI のセキュリティユーティリティを一覧してください。", "d": "medium"},
    {"q": "OAuth2 の実装はどのファイルにありますか？", "d": "easy"},
    {"q": "HTTPBearer の実装を説明してください。", "d": "medium"},
    {"q": "FastAPI の型ヒントの活用方法を説明してください。", "d": "medium"},
    {"q": "デコレータベースのルーティングの内部実装を説明してください。", "d": "hard"},
    {"q": "サブアプリケーションのマウントの仕組みを教えてください。", "d": "medium"},
    {"q": "FastAPI の設定管理のベストプラクティスを教えてください。", "d": "easy"},
    {"q": "このリポジトリのドキュメント構成を教えてください。", "d": "easy"},
    {"q": "FastAPI と Flask/Django の設計上の主な違いは何ですか？", "d": "hard"},
]

IMPACT_ANALYSIS = [
    {"q": "`Depends()` の実装を変えると波及先はどこですか？", "d": "hard"},
    {"q": "Request クラスの変更が影響する範囲を教えてください。", "d": "hard"},
    {"q": "ルーティングのパス解決ロジックを変更した場合の影響は？", "d": "hard"},
    {"q": "Pydantic v2 への移行で影響を受けるモジュールはどこですか？", "d": "hard"},
    {"q": "OpenAPI スキーマ生成ロジックを変更した場合の影響は？", "d": "hard"},
    {"q": "ミドルウェアスタックの順序を変更した場合の影響は？", "d": "medium"},
    {"q": "例外ハンドラの登録方法を変えた場合の影響範囲は？", "d": "medium"},
    {"q": "セキュリティスキームの基底クラスを変更した場合の影響は？", "d": "hard"},
    {"q": "JSONResponse のデフォルト挙動を変えた場合の影響は？", "d": "medium"},
    {"q": "Body() のバリデーションロジックを変更した場合の影響は？", "d": "medium"},
    {"q": "Header() パラメータの解析方法を変更した場合の影響は？", "d": "medium"},
    {"q": "Cookie() の処理を変更した場合の影響範囲は？", "d": "medium"},
    {"q": "UploadFile の実装を変更した場合の影響は？", "d": "medium"},
    {"q": "WebSocket の接続ハンドリングを変更した場合の影響は？", "d": "medium"},
    {"q": "テスト用 TestClient の実装を変えた場合の影響は？", "d": "medium"},
    {"q": "FastAPI クラスのコンストラクタ引数を変更した場合の影響は？", "d": "hard"},
    {"q": "APIRouter.include_router の挙動を変更した場合の影響は？", "d": "hard"},
    {"q": "status モジュールの定数を変更した場合の影響は？", "d": "easy"},
    {"q": "encoders.py の jsonable_encoder を変更した場合の影響は？", "d": "medium"},
    {"q": "params.py の Param クラスを変更した場合の影響は？", "d": "hard"},
    {"q": "dependencies/utils.py を変更した場合の影響範囲は？", "d": "hard"},
    {"q": "routing.py の APIRoute を変更した場合の影響は？", "d": "hard"},
    {"q": "applications.py を変更した場合の影響範囲は？", "d": "hard"},
    {"q": "responses.py のレスポンスクラスを変更した場合の影響は？", "d": "medium"},
    {"q": "datastructures.py を変更した場合の影響は？", "d": "medium"},
    {"q": "openapi/utils.py を変更した場合の影響は？", "d": "medium"},
    {"q": "openapi/models.py を変更した場合の影響は？", "d": "medium"},
    {"q": "security/oauth2.py を変更した場合の影響は？", "d": "medium"},
    {"q": "testclient.py を変更した場合の影響は？", "d": "easy"},
    {"q": "concurrency.py を変更した場合の影響範囲は？", "d": "medium"},
]

BUG_LOCALIZATION = [
    {"q": "認証ミドルウェアで 401 が返る原因候補を 3 つ出してください。", "d": "medium"},
    {"q": "POST リクエストで 422 Validation Error が出る場合の原因候補は？", "d": "medium"},
    {"q": "依存性注入で循環参照が発生した場合、どこでエラーになりますか？", "d": "hard"},
    {"q": "CORS エラーが発生する場合の設定ミスの候補を教えてください。", "d": "easy"},
    {"q": "WebSocket 接続が切断される原因候補を教えてください。", "d": "medium"},
    {"q": "レスポンスモデルと実際の返り値が不一致の場合のエラー箇所は？", "d": "medium"},
    {"q": "ファイルアップロードで 413 が返る場合の原因箇所は？", "d": "medium"},
    {"q": "OpenAPI ドキュメントが生成されない場合の原因候補は？", "d": "medium"},
    {"q": "ミドルウェアでリクエストボディが消費される問題の原因は？", "d": "hard"},
    {"q": "BackgroundTasks が実行されない場合の原因候補は？", "d": "medium"},
    {"q": "JSON レスポンスのエンコーディングエラーの原因箇所は？", "d": "medium"},
    {"q": "パスパラメータの型変換エラーの原因箇所はどこですか？", "d": "medium"},
    {"q": "依存性の yield が正しくクリーンアップされない場合の原因は？", "d": "hard"},
    {"q": "サブアプリケーションのルーティングが動かない原因候補は？", "d": "medium"},
    {"q": "HTTPException のカスタムハンドラが機能しない原因は？", "d": "medium"},
    {"q": "OAuth2PasswordBearer のトークン検証が失敗する原因候補は？", "d": "medium"},
    {"q": "レスポンスヘッダが設定されない場合の原因は？", "d": "easy"},
    {"q": "依存性のキャッシュが効かない場合の原因は？", "d": "medium"},
    {"q": "クエリパラメータのデフォルト値が反映されない原因は？", "d": "easy"},
    {"q": "Form データのパースエラーの原因箇所はどこですか？", "d": "medium"},
    {"q": "Enum 型パラメータで不正な値が通る原因は？", "d": "medium"},
    {"q": "Optional パラメータが必須として扱われる原因は？", "d": "medium"},
    {"q": "独自バリデータが呼ばれない場合の原因は？", "d": "medium"},
    {"q": "Response クラスのステータスコードが無視される原因は？", "d": "medium"},
    {"q": "APIRouter のプレフィックスが二重に付く原因は？", "d": "medium"},
    {"q": "テストでリクエストボディが渡されない場合の原因は？", "d": "easy"},
    {"q": "非同期エンドポイントでデッドロックが発生する原因候補は？", "d": "hard"},
    {"q": "Lifespan イベントが発火しない場合の原因は？", "d": "medium"},
    {"q": "カスタムレスポンスクラスが OpenAPI に反映されない原因は？", "d": "medium"},
    {"q": "マルチパートフォームで複数ファイルが受け取れない原因は？", "d": "medium"},
]

CHANGE_PLANNING = [
    {"q": "認可ロジックを middleware に移す案と decorator に残す案を比較してください。", "d": "hard"},
    {"q": "依存性注入を property-based に変更する場合の設計案は？", "d": "hard"},
    {"q": "OpenAPI 3.1 から 3.2 への対応に必要な変更を計画してください。", "d": "hard"},
    {"q": "レスポンスキャッシュを追加する最小の変更案は？", "d": "medium"},
    {"q": "レートリミットをミドルウェアで実装する場合の設計案は？", "d": "medium"},
    {"q": "WebSocket のルーム機能を追加する場合の変更計画は？", "d": "medium"},
    {"q": "グローバルエラーハンドリングを改善する設計案は？", "d": "medium"},
    {"q": "APIのバージョニングを導入する最小の変更は？", "d": "medium"},
    {"q": "非同期データベース接続プールを組み込む設計案は？", "d": "medium"},
    {"q": "テストカバレッジを向上させるための優先順位を計画してください。", "d": "medium"},
    {"q": "GraphQL サポートを追加する場合のアーキテクチャ案は？", "d": "hard"},
    {"q": "ヘルスチェックエンドポイントを標準化する設計案は？", "d": "easy"},
    {"q": "リクエストログの構造化を実装する変更計画は？", "d": "medium"},
    {"q": "依存性の遅延ロードを実装する設計案は？", "d": "hard"},
    {"q": "セキュリティヘッダの自動設定を追加する最小変更は？", "d": "easy"},
    {"q": "レスポンスの圧縮を追加する設計案は？", "d": "easy"},
    {"q": "API キーベースの認証を追加する場合の変更計画は？", "d": "medium"},
    {"q": "テスト用のフィクスチャ管理を改善する設計案は？", "d": "medium"},
    {"q": "OpenTelemetry トレーシングを組み込む変更計画は？", "d": "medium"},
    {"q": "カスタムシリアライザをサポートする設計案は？", "d": "hard"},
    {"q": "エンドポイントの deprecation 機構を追加する最小変更は？", "d": "medium"},
    {"q": "マルチテナント対応を追加する場合のアーキテクチャ案は？", "d": "hard"},
    {"q": "内部イベントバスを導入する設計案は？", "d": "hard"},
    {"q": "サーキットブレーカーパターンを組み込む変更計画は？", "d": "medium"},
    {"q": "リクエストスコープの DI コンテナを実装する設計案は？", "d": "hard"},
    {"q": "API ドキュメントのカスタマイズ方法を拡張する設計案は？", "d": "medium"},
    {"q": "型安全な設定管理を導入する最小変更は？", "d": "medium"},
    {"q": "プラグインシステムを追加する場合のアーキテクチャ案は？", "d": "hard"},
    {"q": "E2E テストフレームワークを統合する変更計画は？", "d": "medium"},
    {"q": "パフォーマンスプロファイリング機能を追加する設計案は？", "d": "medium"},
]


# ================================================================
# Multi-turn session templates
# ================================================================

MULTI_TURN_TEMPLATES = [
    {
        "category": "onboarding", "scenario": "依存性注入の段階的深掘り",
        "tags": ["topic_narrowing"],
        "turns": [
            "FastAPI の全体構成を教えてください。",
            "依存性注入の仕組みをコードで説明してください。",
            "Depends() のソースコードを見せてください。",
            "依存性のキャッシュの仕組みは？",
            "yield を使った依存性のライフサイクルを説明してください。",
            "依存性注入と他のフレームワークとの違いは？",
        ],
    },
    {
        "category": "onboarding", "scenario": "セキュリティ機能の探索",
        "tags": ["topic_narrowing"],
        "turns": [
            "FastAPI のセキュリティ関連モジュールを一覧してください。",
            "OAuth2 の実装はどうなっていますか？",
            "OAuth2PasswordBearer の使い方を教えてください。",
            "JWT トークンの検証はどこで行いますか？",
            "セキュリティスキームの基底クラスを見せてください。",
            "カスタムセキュリティスキームの作り方は？",
        ],
    },
    {
        "category": "impact_analysis", "scenario": "Router 変更の影響調査",
        "tags": ["topic_narrowing"],
        "turns": [
            "APIRouter の主要メソッドを教えてください。",
            "include_router の実装を説明してください。",
            "Router のプレフィックス処理を変えた場合の影響は？",
            "テストのうち Router に依存しているものはどれですか？",
            "Router と Starlette Router の関係を教えてください。",
            "Router のリファクタリングで最もリスクが高い変更は？",
        ],
    },
    {
        "category": "bug_localization", "scenario": "422 エラーの原因調査",
        "tags": ["topic_narrowing"],
        "turns": [
            "422 Validation Error が発生する主な原因は？",
            "リクエストボディのバリデーションはどこで行われますか？",
            "Pydantic のバリデーションエラーのフォーマットは？",
            "カスタムバリデーションエラーの追加方法は？",
            "バリデーションをスキップする方法はありますか？",
            "バリデーションエラーのログを強化する方法は？",
        ],
    },
    {
        "category": "onboarding", "scenario": "話題切り替えテスト",
        "tags": ["topic_shift"],
        "turns": [
            "FastAPI のミドルウェアの仕組みを教えてください。",
            "CORS ミドルウェアの設定方法は？",
            "ところで WebSocket のサポートはどうなっていますか？",
            "WebSocket でメッセージを送受信するコードを見せてください。",
            "話を戻して、ミドルウェアの実行順序はどうなっていますか？",
            "カスタムミドルウェアの作り方を教えてください。",
        ],
    },
    {
        "category": "safe_recgen", "scenario": "exact quote テスト",
        "tags": ["exact_quote"],
        "turns": [
            "FastAPI クラスの __init__ を教えてください。",
            "その __init__ のソースコードを exact quote で出してください。",
            "引数の型を確認してください。",
            "デフォルト値が設定されている引数を一覧してください。",
            "title 引数の使われ方を追跡してください。",
            "FastAPI と Starlette の __init__ の違いは？",
        ],
    },
    {
        "category": "safe_recgen", "scenario": "diff/patch テスト",
        "tags": ["diff_or_patch"],
        "turns": [
            "CORS の設定にカスタムヘッダを追加したい。",
            "変更が必要なファイルを教えてください。",
            "具体的な diff を出してください。",
            "この変更のテストコードを書いてください。",
            "既存のテストへの影響はありますか？",
            "この変更のリスクを評価してください。",
        ],
    },
    {
        "category": "change_planning", "scenario": "認可設計の比較",
        "tags": ["topic_narrowing"],
        "turns": [
            "現在の認可の仕組みを教えてください。",
            "middleware 方式と decorator 方式を比較してください。",
            "middleware 方式のメリットとリスクは？",
            "decorator 方式のメリットとリスクは？",
            "最小変更で実装するならどちらですか？",
            "段階的な移行計画を提案してください。",
        ],
    },
]


# ================================================================
# Stress eval templates
# ================================================================

STRESS_TOPICS = [
    "FastAPI の依存性注入の仕組み",
    "ルーティングの内部実装",
    "OpenAPI スキーマ生成",
    "セキュリティモジュールの構成",
    "ミドルウェアスタック",
    "例外ハンドリングの仕組み",
    "WebSocket サポートの実装",
    "テスト基盤の構成",
]


def generate_stress_turns(topic: str, n: int = 10) -> list[str]:
    base_questions = [
        f"{topic}の概要を教えてください。",
        f"{topic}の主要ファイルはどこですか？",
        f"{topic}のエントリポイントを教えてください。",
        f"そのコードを見せてください。",
        f"依存関係を教えてください。",
        f"変更した場合の影響範囲は？",
        f"テストはどこにありますか？",
        f"改善案を提案してください。",
        f"最もリスクが高い変更は何ですか？",
        f"まとめてください。",
    ]
    return base_questions[:n]


# ================================================================
# Generator
# ================================================================

def generate_static_eval(output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    categories = [
        ("onboarding", ONBOARDING),
        ("impact_analysis", IMPACT_ANALYSIS),
        ("bug_localization", BUG_LOCALIZATION),
        ("change_planning", CHANGE_PLANNING),
    ]
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for cat, questions in categories:
            for i, q in enumerate(questions, 1):
                record = {
                    "id": f"SE-{cat[:3].upper()}-{i:03d}",
                    "category": cat,
                    "difficulty": q["d"],
                    "question": q["q"],
                    "reference_answer": "",
                    "reference_chunk_ids": [],
                    "grading_notes": "",
                    "rubric": {
                        "correctness": {"max": 2, "notes": ""},
                        "grounding": {"max": 2, "notes": ""},
                        "usefulness": {"max": 1, "notes": ""},
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count


def generate_multi_turn_eval(output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i, tmpl in enumerate(MULTI_TURN_TEMPLATES, 1):
            record = {
                "session_id": f"MT-{i:03d}",
                "category": tmpl["category"],
                "scenario": tmpl["scenario"],
                "turns": [
                    {
                        "turn_id": t + 1,
                        "question": q,
                        "reference_answer": "",
                        "reference_chunk_ids": [],
                        "grading_notes": "",
                        "tags": [],
                    }
                    for t, q in enumerate(tmpl["turns"])
                ],
                "session_tags": tmpl.get("tags", []),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

        # Pad to 30 sessions by duplicating with variations
        while count < 30:
            base = MULTI_TURN_TEMPLATES[count % len(MULTI_TURN_TEMPLATES)]
            record = {
                "session_id": f"MT-{count + 1:03d}",
                "category": base["category"],
                "scenario": f"{base['scenario']} (variant {count // len(MULTI_TURN_TEMPLATES) + 1})",
                "turns": [
                    {
                        "turn_id": t + 1,
                        "question": q,
                        "reference_answer": "",
                        "reference_chunk_ids": [],
                        "grading_notes": "",
                        "tags": [],
                    }
                    for t, q in enumerate(base["turns"])
                ],
                "session_tags": base.get("tags", []),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def generate_stress_eval(output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i, topic in enumerate(STRESS_TOPICS, 1):
            turns = generate_stress_turns(topic)
            record = {
                "session_id": f"ST-{i:03d}",
                "concurrency_group": 1,
                "turns": [
                    {"turn_id": t + 1, "question": q, "tags": []}
                    for t, q in enumerate(turns)
                ],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval sets")
    parser.add_argument("--repo-id", default="fastapi_fastapi")
    parser.add_argument("--output-dir", default="data/eval_sets")
    args = parser.parse_args()

    out = Path(args.output_dir)

    n1 = generate_static_eval(out / "static_eval.jsonl")
    print(f"Static eval: {n1} questions -> {out / 'static_eval.jsonl'}")

    n2 = generate_multi_turn_eval(out / "multi_turn_eval.jsonl")
    print(f"Multi-turn eval: {n2} sessions -> {out / 'multi_turn_eval.jsonl'}")

    n3 = generate_stress_eval(out / "stress_eval.jsonl")
    print(f"Stress eval: {n3} sessions -> {out / 'stress_eval.jsonl'}")

    print(f"\nTotal: {n1} static + {n2} multi-turn sessions + {n3} stress sessions")


if __name__ == "__main__":
    main()
