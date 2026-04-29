from __future__ import annotations

from copy import deepcopy


def is_qwen35_model(model_name: str) -> bool:
    return "qwen3.5" in model_name.lower()


def normalize_qwen_thinking(
    messages: list[dict[str, str]],
    model_name: str,
    enable_thinking: bool | None = None,
) -> tuple[list[dict[str, str]], bool | None]:
    if not is_qwen35_model(model_name):
        return messages, None

    normalized_messages = deepcopy(messages)
    resolved = enable_thinking

    for message in reversed(normalized_messages):
        if message.get("role") != "user":
            continue

        content = message.get("content", "")
        if not isinstance(content, str):
            break

        stripped = content.lstrip()
        if stripped.startswith("/think"):
            resolved = True
            message["content"] = _strip_leading_directive(content, "/think")
            break
        if stripped.startswith("/nothink"):
            resolved = False
            message["content"] = _strip_leading_directive(content, "/nothink")
            break
        if stripped.startswith("/no_think"):
            resolved = False
            message["content"] = _strip_leading_directive(content, "/no_think")
            break
        break

    return normalized_messages, resolved


def _strip_leading_directive(content: str, directive: str) -> str:
    stripped = content.lstrip()
    remainder = stripped[len(directive) :].lstrip()
    return remainder or content
