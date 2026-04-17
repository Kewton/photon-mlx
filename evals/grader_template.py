"""
grader_template.py  –  LLM-as-judge grader for RepoRAG answers.

Rubric (5 points total):
  correctness : 0-2  (factual accuracy against reference)
  grounding   : 0-2  (citations present and pointing to correct chunks)
  usefulness  : 0-1  (actionability / clarity for the user)

Usage:
    python evals/grader_template.py \
        --predictions logs/baseline_fastapi_fastapi_20260412_abc1234.jsonl \
        --eval-set    data/eval_sets/static_eval.jsonl \
        --output      reports/grader_output.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_GRADER_SYSTEM = """\
You are an expert code-review grader evaluating answers from a RepoRAG system.
Score the answer on three dimensions using the rubric below.
Return ONLY a JSON object — no prose, no markdown fences.

Rubric:
  correctness (0-2):
    2 = fully correct, all claims match the reference answer and code
    1 = mostly correct, minor errors or omissions
    0 = incorrect or hallucinated

  grounding (0-2):
    2 = every claim is supported by a cited chunk; citations are valid
    1 = some claims are cited; one or two unsupported assertions
    0 = no citations, or citations are wrong / irrelevant

  usefulness (0-1):
    1 = actionable and clear for the target user
    0 = vague, too long, or not useful

Return format:
{
  "correctness": <int>,
  "grounding": <int>,
  "usefulness": <int>,
  "reason": "<one sentence>"
}
"""

_GRADER_USER = """\
## Question
{question}

## Reference Answer
{reference_answer}

## Grading Notes
{grading_notes}

## System Answer (to grade)
{answer}

## Cited Chunk IDs
{cited_chunk_ids}
"""


def build_grader_messages(
    question: str,
    reference_answer: str,
    grading_notes: str,
    answer: str,
    cited_chunk_ids: list[str],
) -> list[dict]:
    return [
        {"role": "system", "content": _GRADER_SYSTEM},
        {
            "role": "user",
            "content": _GRADER_USER.format(
                question=question,
                reference_answer=reference_answer,
                grading_notes=grading_notes,
                answer=answer,
                cited_chunk_ids=", ".join(cited_chunk_ids) or "(none)",
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Default rubric (correctness 0-2, grounding 0-2, usefulness 0-1)
# ---------------------------------------------------------------------------

_DEFAULT_RUBRIC: dict[str, dict] = {
    "correctness": {"min": 0, "max": 2},
    "grounding": {"min": 0, "max": 2},
    "usefulness": {"min": 0, "max": 1},
}

# ---------------------------------------------------------------------------
# LLM client abstraction
# ---------------------------------------------------------------------------


def _call_llm_judge(messages: list[dict], model_id: str = "qwen3.5:27b") -> str:
    """Call an OpenAI-compatible endpoint to get the judge response text."""
    import httpx

    base_url = os.getenv("LLM_JUDGE_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("LLM_JUDGE_API_KEY", "ollama")

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.0,
    }
    resp = httpx.post(
        f"{base_url}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120.0,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _parse_judge_response(text: str) -> dict:
    """Parse JSON from raw LLM output, stripping markdown fences if present."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`")
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Grader runner
# ---------------------------------------------------------------------------


def grade_one(
    question: str,
    reference_answer: str,
    grading_notes: str,
    answer: str,
    cited_chunk_ids: list[str],
    model_id: str = "qwen3.5:27b",
) -> dict:
    """Call the judge model and return parsed scores."""
    messages = build_grader_messages(
        question, reference_answer, grading_notes, answer, cited_chunk_ids
    )
    response_text = _call_llm_judge(messages, model_id=model_id)
    return _parse_judge_response(response_text)


def compute_aggregate(
    scores: list[dict],
    rubric: dict[str, dict] | None = None,
) -> dict:
    """Aggregate grading scores using rubric-aware dynamic calculation."""
    n = len(scores)
    if n == 0:
        return {}

    rubric = rubric or _DEFAULT_RUBRIC
    dims = list(rubric.keys())
    max_total = sum(rubric[d]["max"] for d in dims)

    agg: dict = {"n": n}
    total_sum = 0.0
    for dim in dims:
        dim_sum = sum(s.get(dim, 0) for s in scores)
        agg[f"{dim}_mean"] = dim_sum / n
        total_sum += dim_sum

    agg["total_mean"] = total_sum / n
    agg["total_100"] = (total_sum / n / max_total * 100) if max_total > 0 else 0.0
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade RepoRAG predictions")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default="qwen3.5:27b")
    args = parser.parse_args()

    eval_items = {
        item["id"]: item
        for line in Path(args.eval_set).read_text(encoding="utf-8").splitlines()
        if line.strip()
        for item in [json.loads(line)]
    }

    results: list[dict] = []
    for line in Path(args.predictions).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        pred = json.loads(line)
        item = eval_items.get(pred.get("eval_id", ""))
        if not item:
            continue
        scores = grade_one(
            question=item["question"],
            reference_answer=item["reference_answer"],
            grading_notes=item.get("grading_notes", ""),
            answer=pred["answer"],
            cited_chunk_ids=pred.get("cited_chunk_ids", []),
            model_id=args.model_id,
        )
        results.append({"eval_id": item["id"], **scores})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    agg = compute_aggregate(results)
    print(json.dumps(agg, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
