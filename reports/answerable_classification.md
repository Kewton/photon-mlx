# Answerable Classification Report

## Summary

7 of 120 static eval questions are **unanswerable** -- the concepts they
ask about do not exist in the FastAPI repository.  Marking these allows
us to distinguish "correct abstain" from "true no-citation" failures.

## Unanswerable Questions

| ID | Category | Reason |
|----|----------|--------|
| SE-CHA-003 | change_planning | OpenAPI 3.2 does not exist as a specification |
| SE-CHA-019 | change_planning | OpenTelemetry is not part of the FastAPI codebase |
| SE-CHA-023 | change_planning | FastAPI has no internal event bus |
| SE-CHA-024 | change_planning | Circuit breaker pattern is not in FastAPI |
| SE-CHA-028 | change_planning | FastAPI has no plugin system |
| SE-CHA-030 | change_planning | Performance profiling is not a FastAPI feature |
| SE-IMP-018 | impact_analysis | `status` module belongs to Starlette, not FastAPI |

## NC Rate Comparison

| Metric | Formula | Value |
|--------|---------|-------|
| Raw NC | 21 / 120 | 17.5% |
| True NC (answerable only) | 14 / 113 | 12.4% |
| Correct abstains | 7 / 7 | 100% |

## Methodology

1. Each of the 120 static eval questions was reviewed against the
   FastAPI source repository to determine whether the concept in the
   question actually exists in the codebase.
2. Questions about concepts that do not exist (e.g., OpenAPI 3.2,
   OpenTelemetry, event bus) were marked `"answerable": false` in
   `data/eval_sets/static_eval.jsonl`.
3. `scripts/ci_eval_check.py` was updated to cross-reference the eval
   set and report **True NC** (no-citation rate among answerable
   questions only) alongside the raw NC rate.
4. CI pass/fail thresholds remain based on raw NC to maintain backward
   compatibility.
