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
# Grader runner (stub – wire to LLM client when ready)
# ---------------------------------------------------------------------------


def grade_one(
    question: str,
    reference_answer: str,
    grading_notes: str,
    answer: str,
    cited_chunk_ids: list[str],
    model_id: str = "qwen3.5:27b",
) -> dict:
    """
    Call the judge model and return parsed scores.
    Replace the NotImplementedError with your LLM client call.
    """
    _messages = build_grader_messages(
        question, reference_answer, grading_notes, answer, cited_chunk_ids
    )
    # TODO: call LLM judge here
    # response_text = call_llm(model_id, messages)
    # return json.loads(response_text)
    raise NotImplementedError("Wire grade_one() to an LLM client.")


def compute_aggregate(scores: list[dict]) -> dict:
    n = len(scores)
    if n == 0:
        return {}
    return {
        "n": n,
        "correctness_mean": sum(s["correctness"] for s in scores) / n,
        "grounding_mean": sum(s["grounding"] for s in scores) / n,
        "usefulness_mean": sum(s["usefulness"] for s in scores) / n,
        "total_mean": sum(
            s["correctness"] + s["grounding"] + s["usefulness"] for s in scores
        )
        / n,
        "total_100": sum(
            s["correctness"] + s["grounding"] + s["usefulness"] for s in scores
        )
        / n
        / 5
        * 100,
    }


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
