"""Prompt templates for the 6 institutional-question categories (Issue #110)."""

from __future__ import annotations

CATEGORY_CONFIG: dict[str, dict[str, str]] = {
    "definition": {
        "label": "定義",
        "focus": "用語・主体の定義条項",
        "example": "「賃貸住宅管理業者の定義は？」",
    },
    "article_lookup": {
        "label": "条文検索",
        "focus": "特定条文 (第N条) が定める内容",
        "example": "「第3条に規定されている登録制度の期間は？」",
    },
    "overview": {
        "label": "概要",
        "focus": "法令・制度全体の目的や趣旨",
        "example": "「この法律の目的は？」",
    },
    "scope": {
        "label": "適用範囲",
        "focus": "誰にどの事業に適用されるか",
        "example": "「この法律の適用を受けるのはどのような事業者？」",
    },
    "penalty": {
        "label": "罰則",
        "focus": "違反時の罰則規定",
        "example": "「登録せず管理業務を行った場合の罰則は？」",
    },
    "exception": {
        "label": "例外・但書",
        "focus": "但書・経過措置・適用除外",
        "example": "「国土交通省令で定める例外は？」",
    },
}

SUPPORTED_CATEGORIES: tuple[str, ...] = tuple(CATEGORY_CONFIG.keys())

_SYSTEM_TEMPLATE = (
    "あなたは日本の法令・制度文書のエキスパートです。"
    "次の文書から{label}に関する質問を 1 件生成してください。"
    "焦点: {focus}。例: {example}\n"
    "出力は JSON オブジェクト 1 件で、キーは "
    "question / reference_answer / expected_citation_patterns / grading_notes です。"
    "条文番号 (第N条) が本文に含まれる場合は expected_citation_patterns に "
    'その pattern (例: "第3条", "第3条第1項") を含めてください。'
    "出力は JSON オブジェクト 1 件のみ。前後に markdown フェンスや説明を含めない。"
)


def build_system_prompt(category: str) -> str:
    """Return the system prompt for the given category code."""
    if category not in CATEGORY_CONFIG:
        raise ValueError(f"Unknown category: {category!r}")
    cfg = CATEGORY_CONFIG[category]
    return _SYSTEM_TEMPLATE.format(
        label=cfg["label"], focus=cfg["focus"], example=cfg["example"]
    )


def build_user_prompt(metadata: dict, context: str) -> str:
    """Return the user prompt containing metadata + context."""
    title = metadata.get("title", "")
    header = f"タイトル: {title}\n" if title else ""
    return f"{header}{context}"


def build_prompt(category: str, metadata: dict, context: str) -> str:
    """Convenience wrapper returning ``system + '\\n\\n' + user``."""
    return f"{build_system_prompt(category)}\n\n{build_user_prompt(metadata, context)}"
