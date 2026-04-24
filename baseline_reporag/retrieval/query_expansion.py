"""Lightweight query expansion without LLM inference.

Expands a query by:
1. Extracting code identifiers (camelCase, snake_case, PascalCase, dotted paths).
2. Mapping common Japanese technical terms to English/code equivalents.

No external model calls — zero latency overhead on top of retrieval.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from baseline_reporag.config import Config

# Mirror of configs/baseline.yaml retrieval.query_expansion.domain_map
# (FastAPI default). Update both when modifying.
_JP_TO_CODE: dict[str, list[str]] = {
    # --- Architecture / infrastructure terms ---
    "ミドルウェア": ["middleware", "Middleware", "BaseHTTPMiddleware"],
    "依存性注入": ["dependency injection", "Depends", "dependencies"],
    "依存性": ["dependency", "Depends", "dependencies"],
    "ルーター": ["router", "APIRouter", "include_router"],
    "エンドポイント": ["endpoint", "route", "path operation"],
    "認証": ["authentication", "auth", "authenticate"],
    "認可": ["authorization", "authorize", "permission"],
    "セキュリティ": ["security", "SecurityScheme", "OAuth2"],
    "バリデーション": ["validation", "validator", "pydantic"],
    "スキーマ": ["schema", "BaseModel", "pydantic"],
    "例外": ["exception", "HTTPException", "error handler"],
    "エラーハンドラ": ["exception handler", "add_exception_handler"],
    "リクエスト": ["request", "Request"],
    "レスポンス": ["response", "Response", "JSONResponse"],
    "セッション": ["session", "SessionMiddleware"],
    "テスト": ["test", "TestClient", "pytest"],
    "移行": ["migration", "migration plan"],
    "リスク": ["risk", "impact"],
    "影響": ["impact", "affected", "dependency"],
    "キャッシュ": ["cache", "use_cache", "dependency_cache"],
    "デコレータ": ["decorator", "@app.get", "@router"],
    "ライフサイクル": ["lifecycle", "lifespan", "startup", "shutdown"],
    "WebSocket": ["websocket", "WebSocket", "ws"],
    "バックグラウンド": ["background", "BackgroundTasks"],
    "静的ファイル": ["static", "StaticFiles", "mount"],
    "設定": ["config", "Settings", "BaseSettings"],
    "ログ": ["log", "logger", "logging"],
    "ストリーミング": ["streaming", "StreamingResponse", "stream"],
    "実行順序": ["execution order", "order", "stack"],
    "移植": ["migration", "porting", "refactor"],
    # --- Bug localization: symptom → implementation file keywords ---
    # BM25 verified: each expansion lifts the relevant impl file into top-5.
    "エンコーディング": ["jsonable_encoder", "ENCODERS_BY_TYPE", "encoders"],
    "デッドロック": ["run_in_threadpool", "asyncio", "anyio", "concurrency"],
    "非同期": ["asyncio", "run_in_threadpool", "anyio", "concurrency"],
    "ステータスコード": ["status_code", "routing"],
    "マルチパート": ["UploadFile", "ImmutableMultiDict", "datastructures", "multipart"],
    "フォームデータ": ["UploadFile", "Form", "datastructures"],
    "ファイルアップロード": ["UploadFile", "File", "datastructures"],
    "パスパラメータ": ["path_params", "solve_dependencies", "routing", "Path"],
    "型変換": ["solve_dependencies", "request_path", "routing", "field"],
    "クエリパラメータ": ["query_params", "Query", "Param", "ParamTypes"],
    "デフォルト値": ["Param", "ParamTypes", "Query", "Body", "Form"],
    "独自バリデータ": ["Param", "ParamTypes", "field_validator", "model_validator"],
    "バリデータ": ["Param", "ParamTypes", "field_validator", "model_validator"],
    "サブアプリ": ["mount", "sub_applications", "routing"],
    "サブアプリケーション": ["mount", "sub_applications", "routing"],
    # Additional bug localization terms - BM25 verified
    "エラーハンドリング": [
        "exception_handler",
        "add_exception_handler",
        "HTTPException",
        "exception_handlers",
    ],
    "ハンドリング": ["exception_handler", "add_exception_handler", "handler"],
    "ルーティング": ["routing", "APIRouter", "include_router", "route"],
    "OpenAPI": ["openapi", "get_openapi", "generate_operation_id", "openapi_utils"],
    "openapi": ["openapi", "get_openapi", "generate_operation_id", "openapi_utils"],
    "スキーマ生成": ["get_openapi", "generate_operation_id", "openapi_utils"],
    "セキュリティスキーム": [
        "SecurityMiddleware",
        "add_middleware",
        "TrustedHostMiddleware",
        "security",
    ],
    "HTTPSリダイレクト": ["HTTPSRedirectMiddleware", "add_middleware", "middleware"],
    "信頼済みホスト": ["TrustedHostMiddleware", "add_middleware", "middleware"],
}

_IDENTIFIER_PATTERN = re.compile(
    r"(?<![a-zA-Z])"
    r"(?:[A-Z][a-z]+(?:[A-Z][a-z]*)+)"  # PascalCase
    r"|(?:[a-z]+(?:_[a-z]+)+)"  # snake_case
    r"|(?:[a-z]+[A-Z][a-zA-Z]*)"  # camelCase
    r"|(?:\w+\.\w+(?:\.\w+)*)"  # dotted.path.notation
)


def _normalize_mapping(
    mapping: "Config | dict[str, list[str]] | None",
) -> dict[str, list[str]]:
    """Normalize mapping into a plain dict.

    - None → built-in `_JP_TO_CODE` fallback (backward compat).
    - Config wrapper → converted via `to_dict()`.
    - dict → returned as-is (identity).
    """
    if mapping is None:
        return _JP_TO_CODE
    from baseline_reporag.config import Config

    if isinstance(mapping, Config):
        return mapping.to_dict()
    return mapping


def expand_query(
    query: str,
    mapping: "Config | dict[str, list[str]] | None" = None,
) -> list[str]:
    """Return a list of expanded search queries derived from *query*.

    Args:
        query: Original user query.
        mapping: Domain-specific Japanese→code-term mapping.
            - None (default): use built-in `_JP_TO_CODE` (backward compat).
            - dict: use as-is.
            - Config wrapper: normalized to dict via `.to_dict()` internally.
            - Empty dict/Config: explicit disable of Japanese expansion
              (identifier extraction still runs).

    Returns:
        List where the first element is always the original query.
        Additional element (if any) is a combined expansion query.
    """
    effective_map = _normalize_mapping(mapping)

    queries: list[str] = [query]
    extra_terms: list[str] = []

    # 1. Japanese → code-term expansion
    for jp, code_terms in effective_map.items():
        if jp in query:
            extra_terms.extend(code_terms)

    # 2. Code identifier extraction (runs on the original query)
    for m in _IDENTIFIER_PATTERN.finditer(query):
        term = m.group(0)
        if term not in query.split() and len(term) > 3:
            extra_terms.append(term)

    # Build one combined expansion query if we found anything useful
    if extra_terms:
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for t in extra_terms:
            if t.lower() not in seen:
                seen.add(t.lower())
                unique.append(t)
        combined = " ".join(unique[:8])  # cap at 8 terms to avoid dilution
        queries.append(combined)

    return queries
