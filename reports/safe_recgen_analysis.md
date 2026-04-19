# Safe RecGen Fallback Recall Analysis

## Methodology

This analysis measures how well the Safe RecGen fallback controller detects
turns that would benefit from re-retrieval, using the multi-turn (MT) eval
log data.

### Proxy for Ground Truth

Since there is no hand-labeled "should have re-retrieved" ground truth, we
use **`no_citation=True`** as a proxy signal. The reasoning is:

- A turn that produces **no citations** likely failed to retrieve relevant
  evidence, which is exactly the situation where Safe RecGen should trigger
  a fallback (re-retrieve, re-prefill hierarchy, etc.).
- This is an imperfect heuristic (see Limitations), but provides a
  practical baseline for measuring recall.

### Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **fallback_rate** | `fallback_flag / total` | How often does Safe RecGen fire? |
| **fallback_recall** | `(fallback AND no_citation) / no_citation` | Of turns that *needed* re-retrieval, how many did we catch? |
| **fallback_precision** | `(fallback AND no_citation) / fallback_flag` | Of turns where fallback fired, how many actually needed it? |

### Measurement Script

```bash
python scripts/measure_fallback_recall.py --log logs/mt_eval_*.jsonl
```

## Results

> **TODO**: Fill in after running the measurement script against actual
> MT eval logs.

| Metric | Value |
|--------|-------|
| Total turns | -- |
| fallback_flag=True | -- |
| no_citation=True | -- |
| fallback_rate | -- |
| fallback_recall | -- |
| fallback_precision | -- |

## Limitations

1. **Proxy fidelity**: `no_citation=True` is not a perfect proxy for
   "should have re-retrieved". Some turns may legitimately produce no
   citations (e.g., clarification questions, unanswerable queries), which
   would inflate the denominator and deflate measured recall.

2. **False negatives in proxy**: A turn might have citations yet still
   benefit from re-retrieval (e.g., stale or wrong citations). These cases
   are invisible to the heuristic.

3. **Threshold sensitivity**: The Safe RecGen controller uses fixed
   thresholds (v1) for drift metrics. Results will change as thresholds
   are tuned or a learned calibrator (v2) is introduced.

4. **Log availability**: The current baseline pipeline logs
   `fallback_flag=False` for all turns because Safe RecGen integration
   is not yet wired into the live pipeline. Meaningful results require
   logs from a PHOTON-pipeline run with Safe RecGen enabled.

5. **Single-repo bias**: Eval logs are generated against a single target
   repository (fastapi/fastapi). Generalization to other repos is
   untested.
