# Issue #80 — Aggregation Mode Benchmark (stub)

Status: **methodology-only stub** — the full A/B/C run requires the real
14B Qwen model and was deliberately not executed in the implementation
iteration. Please run before PR merge (design §10 acceptance + work-plan
Phase 5).

## Summary

Issue #80 adds `aggregation: Literal["weighted", "attention", "last"]` to
`WorkingMemoryConfig`. This report compares the three modes on the
multi-turn eval set at roughly equal retrieval / generation budgets.

Default mode is `weighted` (backward compatible with pre-#80 behaviour).
Both `attention` and `last` are opt-in via YAML:

```yaml
session_memory:
  working_memory:
    enabled: true
    aggregation: attention   # or "last" / "weighted"
```

## Variants

The 3 variants live in `configs/eval.yaml` and differ only in
`override.session_memory.working_memory.aggregation`:

| Variant id                     | Aggregation | Base config           |
|--------------------------------|-------------|-----------------------|
| `photon_rag_aggr_weighted`     | `weighted`  | `photon_small.yaml`   |
| `photon_rag_aggr_attention`    | `attention` | `photon_small.yaml`   |
| `photon_rag_aggr_last`         | `last`      | `photon_small.yaml`   |

All three keep `safe_recgen.enabled: false` so the session state
aggregation is the only varying factor.

## Methodology

Use the existing `bench/run_all.py` harness (no new runner — design
judgement #6). The `deep_merge` path in `bench/run_all.py` already
handles nested `override.session_memory.working_memory.aggregation`
transparently.

```bash
python -m bench.run_all \
  --config configs/eval.yaml \
  --variants photon_rag_aggr_weighted,photon_rag_aggr_attention,photon_rag_aggr_last
```

Metrics to collect (design §10 / Gate 2 v4 schema):

- MT no-citation rate (%)
- Retrieval noise rate (%)
- Per-query latency (P50 / P90) in ms
- Peak memory (MB)

## Acceptance criteria (work-plan §Phase 5 Task 5.2)

- Baseline (`weighted`) MT NC rate is the reference.
- `attention` / `last` MT NC rate must stay within `+2 pt` of
  `weighted`. A larger regression → do root-cause analysis in this
  report and decide: design rollback vs. follow-up issue.

## Results

TBD — fill in once the run completes. Template:

| Variant                     | MT NC (%) | Retrieval noise (%) | P50 (ms) | Peak mem (MB) |
|-----------------------------|-----------|---------------------|----------|---------------|
| `photon_rag_aggr_weighted`  | ...       | ...                 | ...      | ...           |
| `photon_rag_aggr_attention` | ...       | ...                 | ...      | ...           |
| `photon_rag_aggr_last`      | ...       | ...                 | ...      | ...           |

## Notes on scope-out

- `bench/run_mt` full harness — follow-up issue.
- `QueryResult` / telemetry `aggregation_mode` field — follow-up issue.
  Source of truth today is `PhotonSessionState.working_memory_cfg.
  aggregation` (read-only).
