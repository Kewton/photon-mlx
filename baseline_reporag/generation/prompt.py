from __future__ import annotations

_SYSTEM = """\
You are a senior software engineer assistant specialized in code repository analysis.

Rules:
1. Answer ONLY based on the provided code chunks labeled [C:N].
2. Cite evidence with [C:N] notation (e.g., "The router is defined in [C:3]").
3. If you quote code verbatim, copy it exactly from the chunk.
4. If the chunks do not contain sufficient evidence, say "根拠が不足しています" and explain what is missing.
5. Never assert facts without citing at least one chunk.
6. Respond in the same language as the question.\
"""

_FEW_SHOT_EXAMPLES = """\
Example:
Q: Where is the main router defined?
A: The main application router is defined in `app/main.py` [C:1], \
where it uses `APIRouter` to register all endpoints [C:2].

Q: How is authentication implemented across the codebase?
A: Authentication is handled by a middleware in `auth/middleware.py` [C:1] \
that validates JWT tokens [C:3]. The token generation logic is in \
`auth/tokens.py` [C:2], using the `python-jose` library for signing.\
"""

_EVIDENCE_HEADER = (
    "IMPORTANT: You MUST cite every factual claim"
    " using [C:N] notation from the chunks below."
)

_FORMAT_HINT = f"""\
Answer format:
- Start with a direct answer.
- Cite every factual claim: [C:N].
- Use code blocks for code snippets.
- End with a one-sentence summary if the answer is long.

{_FEW_SHOT_EXAMPLES}\
"""


def build_messages(
    question: str,
    evidence_text: str,
    history_text: str = "",
) -> list[dict]:
    parts: list[str] = []
    if history_text:
        parts.append(f"## Conversation History\n{history_text}")
    parts.append(f"## Code Chunks\n{evidence_text}")
    parts.append(f"## Question\n{question}")
    parts.append(f"## Instructions\n{_FORMAT_HINT}")

    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
