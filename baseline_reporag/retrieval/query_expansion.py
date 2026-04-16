"""Lightweight query expansion without LLM inference.

Expands a query by:
1. Extracting code identifiers (camelCase, snake_case, PascalCase, dotted paths).
2. Mapping common Japanese technical terms to English/code equivalents.

No external model calls — zero latency overhead on top of retrieval.
"""

from __future__ import annotations

import re

# Mapping from Japanese technical terms to English/code equivalents commonly
# found in Python/FastAPI codebases.  Keeps the list minimal and repo-relevant.
_JP_TO_CODE: dict[str, list[str]] = {
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
}

_IDENTIFIER_PATTERN = re.compile(
    r"(?<![a-zA-Z])"
    r"(?:[A-Z][a-z]+(?:[A-Z][a-z]*)+)"  # PascalCase
    r"|(?:[a-z]+(?:_[a-z]+)+)"  # snake_case
    r"|(?:[a-z]+[A-Z][a-zA-Z]*)"  # camelCase
    r"|(?:\w+\.\w+(?:\.\w+)*)"  # dotted.path.notation
)


def expand_query(query: str) -> list[str]:
    """Return a list of expanded search queries derived from *query*.

    The first element is always the original query unchanged.  Additional
    elements are lightweight expansions — no LLM call is made.
    """
    queries: list[str] = [query]
    extra_terms: list[str] = []

    # 1. Japanese → code-term expansion
    for jp, code_terms in _JP_TO_CODE.items():
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
