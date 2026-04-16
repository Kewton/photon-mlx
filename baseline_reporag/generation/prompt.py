from __future__ import annotations

# Marker used by rule 4 in _SYSTEM. Kept as a module-level constant so
# post-processing (see baseline_reporag.pipeline.apply_citation_postprocess)
# and _SYSTEM share the exact same string.
ABSTAIN_MARKER = "根拠が不足しています"

_SYSTEM = f"""\
You are a senior software engineer assistant specialized in code repository analysis.

Rules:
1. Answer ONLY based on the provided code chunks labeled [C:N].
2. Cite evidence with [C:N] notation (e.g., "The router is defined in [C:3]").
3. If you quote code verbatim, copy it exactly from the chunk.
4. If the chunks do not contain sufficient evidence, say "{ABSTAIN_MARKER}" and explain what is missing.
5. Never assert facts without citing at least one chunk.
6. Respond in the same language as the question.
7. For design, comparison, or change-planning questions: you MAY reason from \
patterns visible in the code chunks (e.g., existing middleware, decorator usage, \
class hierarchies). Provide a structured recommendation (pros/cons/recommendation) \
citing the relevant patterns with [C:N]. Do NOT abstain solely because there is no \
explicit design document in the chunks — use the code itself as evidence.\
"""

_FEW_SHOT_EXAMPLES = """\
Example 1 — factual question:
Q: Where is the main router defined?
A: The main application router is defined in `app/main.py` [C:1], \
where it uses `APIRouter` to register all endpoints [C:2].

Example 2 — implementation question:
Q: How is authentication implemented across the codebase?
A: Authentication is handled by a middleware in `auth/middleware.py` [C:1] \
that validates JWT tokens [C:3]. The token generation logic is in \
`auth/tokens.py` [C:2], using the `python-jose` library for signing.

Example 3 — design/comparison question (reason from code patterns):
Q: Should we move authorization logic to middleware or keep it in decorators?
A: Based on the existing patterns in this codebase:

**Option A — Middleware** [C:2]: Centralizes auth checks before any route handler \
runs. The current `SecurityMiddleware` in [C:2] already follows this pattern for \
CORS and trusted hosts. Pros: single enforcement point, no per-route boilerplate. \
Cons: harder to apply per-route scopes (see the granular `Depends(oauth2_scheme)` \
used in [C:5]).

**Option B — Decorators/Depends** [C:5]: Keeps authorization co-located with each \
endpoint, enabling per-route scopes. The existing `Depends(get_current_user)` \
pattern in [C:5] is already used consistently.

**Recommendation**: Retain decorator-based auth for per-route granularity [C:5], \
and add a lightweight middleware only for blanket unauthenticated-request rejection \
(HTTP 401 before hitting route handlers), similar to [C:2].\
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
    session_summary: str = "",
) -> list[dict]:
    parts: list[str] = []
    if session_summary:
        parts.append(f"## Session Summary\n{session_summary}")
    if history_text:
        parts.append(f"## Conversation History\n{history_text}")
    parts.append(f"## Code Chunks\n{evidence_text}")
    parts.append(f"## Question\n{question}")
    parts.append(f"## Instructions\n{_FORMAT_HINT}")

    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
