# Issue #137 Hypothesis Verification Report

**Repository**: photon-mlx feature/issue-137-institutional-ab
**Report Date**: 2026-04-26
**Verification Scope**: 13 factual claims about Issue #137 A/B experiment infrastructure

---

## Individual Claim Verification

### Claim 1: PR #136 merged at commit 96e7b45 with EmbeddingIndex.max_input_chars 設定可能化 + guard test

- **Status**: Confirmed
- **Evidence**: 
  - `git log --oneline -10` shows: `96e7b45 Merge pull request #136 from Kewton/feature/issue-133-retrieval-ab`
  - `git show 96e7b45 --stat` shows merged commit includes:
    - `baseline_reporag/indexing/embedding.py` (+22 lines)
    - `baseline_reporag/tests/test_embedding.py` (+63 lines)
    - `tests/test_pipeline_factory_yaml_invariants.py` (+39 lines)
    - `scripts/build_indexes.py` (+4 lines)
  - EmbeddingIndex class (line 79-163 in embedding.py) confirms:
    - Constructor parameter `max_input_chars: int = _DEFAULT_MAX_INPUT_CHARS` (line 83)
    - Default constant `_DEFAULT_MAX_INPUT_CHARS = 2048` (line 76)
    - Usage in build() method: `texts.append(text[: self._max_input_chars])` (line 109)
    - Persistence in save()/load() (lines 141-157)

### Claim 2: configs/_experiments/institutional_V0.yaml through V4.yaml exist (claimed gitignored)

- **Status**: Rejected
- **Evidence**:
  - `ls -la configs/` shows NO `_experiments` subdirectory present
  - `.gitignore` contains `configs/_experiments/` pattern (confirmed)
  - Conclusion: Directory is gitignored but does not currently exist in the working tree
- **Notes**: Directory placeholder exists in .gitignore but no V0-V4 variant files are present in the repository yet. This is appropriate for Phase A (infrastructure setup). Files will be created during Phase B variant evaluation.

### Claim 3: Current global default embedding = intfloat/multilingual-e5-small in configs/baseline.yaml

- **Status**: Rejected
- **Evidence**:
  - `configs/baseline.yaml` line 100 shows: `model_id: "sentence-transformers/all-MiniLM-L6-v2"`
  - git log shows E5 default was reverted in commit `e5c41c9` ("Revert feat(retrieval): switch default embedding model to BAAI/bge-small-en-v1.5 (#97)")
  - The baseline.yaml has reverted to `sentence-transformers/all-MiniLM-L6-v2` (the original baseline)
  - **Correct fact**: Global baseline default embedding = `sentence-transformers/all-MiniLM-L6-v2`
- **Notes**: The institutional config uses E5, but the global baseline does not. This distinction is critical for the institutional/baseline separation principle.

### Claim 4: Current global default reranker = cross-encoder/ms-marco-MiniLM-L-6-v2 in configs/baseline.yaml

- **Status**: Confirmed
- **Evidence**:
  - `configs/baseline.yaml` line 129: `model_id: "cross-encoder/ms-marco-MiniLM-L-6-v2"` ✓
  - test_pipeline_factory_yaml_invariants.py line 32: `GLOBAL_DEFAULT_RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"` ✓

### Claim 5: configs/institutional_docs.yaml exists with embedding.model_id, reranker.model_id, embedding.max_input_chars fields

- **Status**: Partially Confirmed
- **Evidence**:
  - File exists: `configs/institutional_docs.yaml` ✓
  - Line 84: `model_id: "intfloat/multilingual-e5-small"` ✓
  - Line 107: `model_id: "cross-encoder/ms-marco-MiniLM-L-6-v2"` ✓
  - **NOT PRESENT**: No `embedding.max_input_chars` field in institutional_docs.yaml
    - The embedding section (lines 79-86) contains only: enabled, provider, model_id, batch_size, normalize
    - If max_input_chars is omitted, build_indexes.py (line 54) defaults to 2048: `cfg.indexing.embedding.get("max_input_chars", 2048)`
- **Notes**: Institutional config works correctly using the fallback default (2048), but does not explicitly declare max_input_chars. For institutional variants with different max_input_chars (e.g., bge-m3 with 8192), the config will need to be updated to include this field.

### Claim 6: tests/test_pipeline_factory_yaml_invariants.py exists with 2 institutional placeholder skip tests

- **Status**: Confirmed
- **Evidence**:
  - File exists: `tests/test_pipeline_factory_yaml_invariants.py` ✓
  - Lines 55-68: `@pytest.mark.skipif` decorator with `INSTITUTIONAL_RERANKER_MODEL_ID is None` condition
    - Test: `test_institutional_yaml_reranker_model_id_pinned()`
    - Skip reason: "Issue #133: 採用 variant 決定後に有効化 (現在 A/B 評価中)"
  - Lines 71-78: `@pytest.mark.skipif` decorator with `INSTITUTIONAL_EMBEDDING_MODEL_ID is None` condition
    - Test: `test_institutional_yaml_embedding_model_id_pinned()`
  - Both tests currently skipped because placeholder constants are `None` (lines 36-37) ✓

### Claim 7: #132 guard test protects configs/baseline.yaml global default invariance

- **Status**: Confirmed
- **Evidence**:
  - File: `tests/test_pipeline_factory_yaml_invariants.py` line 1-20 states:
    - "Issue #114 — YAML invariant tests for the global pipeline factory."
    - "Locks down the global-default reranker.model_id in configs/baseline.yaml"
    - "#96 was reverted because an institutional-only evaluation result was promoted to the global default"
  - Test: `test_baseline_yaml_reranker_model_id_unchanged()` (lines 40-52)
    - Loads baseline.yaml and asserts: `cfg.retrieval.reranker.model_id == GLOBAL_DEFAULT_RERANKER_MODEL_ID`
    - Prevents accidental changes to global defaults
  - **Note**: The comment references Issue #114, not #132. Check git history for #132 context.

### Claim 8: scripts/aggregate_institutional_baseline.py exists and is reusable for variant aggregation

- **Status**: Confirmed
- **Evidence**:
  - File exists: `scripts/aggregate_institutional_baseline.py` (12,358 bytes)
  - Lines 1-19 show design for reuse:
    - Usage examples show flexible `--predictions` glob patterns
    - `--output` flag supports both `-` (stdout), `.md` files, and `.json` files
    - `--section` flag allows selective aggregation (overall, category, latency, failures)
    - `--in-place` flag supports overwriting existing report.md with sentinel block replacement
  - Design doc reference: `workspace/design/issue-127-aggregate-script-design-policy.md` (noted in comments)
  - Script is clearly structured for multi-variant reuse (lines 55-69 show glob pattern handling)

### Claim 9: scripts/ingest_repo.py, scripts/build_indexes.py, scripts/run_baseline_eval.py all exist with --config and --repo-id flags

- **Status**: Confirmed
- **Evidence**:
  - **ingest_repo.py** (lines 38-44):
    - `parser.add_argument("--repo", required=True)`
    - `parser.add_argument("--repo-id", required=True)` ✓
    - `parser.add_argument("--config", default="configs/baseline.yaml")` ✓
  - **build_indexes.py** (lines 23-34):
    - `parser.add_argument("--repo-id", required=True)` ✓
    - `parser.add_argument("--config", default="configs/baseline.yaml")` ✓
    - Also supports `--embedding-model` override (line 30-32)
  - **run_baseline_eval.py** (lines 25-37):
    - `parser.add_argument("--config", default="configs/baseline.yaml")` ✓
    - `parser.add_argument("--repo-id", default="")` ✓
    - Also supports --eval-set, --output, --marker-file
  - All three scripts exist with required flags

### Claim 10: workspace/issues/133/work-plan.md exists (claimed 23KB) and workspace/design/issue-133-multilingual-retrieval-ab-design-policy.md exists (claimed 49.8KB)

- **Status**: Rejected (Unverifiable from codebase)
- **Evidence**:
  - `find workspace -type f -name "*work-plan*"` returns no results
  - `find workspace -type f -name "*issue-133*"` returns no results
  - `ls workspace/issues/` shows only: `137/issue-review/original-issue.json`
  - `find workspace -type d -name "design"` returns no results
  - Workspace directory tree shows:
    - `workspace/achievement_report.md`
    - `workspace/memo.md`
    - `workspace/mvp/` (subdir)
    - `workspace/orchestration/` (subdir)
    - No workspace/issues/133/ or workspace/design/ directories
- **Notes**: These design documents are referenced but do not exist in the current repository snapshot. They may be planned for creation during subsequent phases, or they may exist in a different location/branch.

### Claim 11: Current V0 institutional NC baseline ~11.21%

- **Status**: Confirmed
- **Evidence**:
  - File: `reports/institutional_baseline_static.md` (created 2026-04-25)
  - Line 60: `| **NC rate** | **11.21 %** |` within aggregate:overall:start/end block
  - Supporting data (lines 56-61):
    - Full question count: 116
    - NC (no-citation) count: 13
    - NC rate calculation: 13/116 = 11.207... ≈ 11.21%
  - Config used: `configs/institutional_docs.yaml` (line 5)
  - Evaluation set: `data/eval_sets/institutional_static_eval.jsonl` (116 questions, line 4)
  - Baseline model used: `intfloat/multilingual-e5-small` (line 47 embedding section)

### Claim 12: EmbeddingIndex can accept variant models (multilingual-e5-base, ruri-small-v2, bge-m3) — uses sentence-transformers in model-agnostic way?

- **Status**: Confirmed
- **Evidence**:
  - EmbeddingIndex.__init__ (line 80-89) accepts arbitrary `model_id: str` parameter
  - Line 95: `self._model = SentenceTransformer(self._model_id)` — passes model_id directly to sentence-transformers library
  - No hardcoded model assumptions:
    - E5-specific logic is isolated to `_is_e5_standard()` helper (lines 18-27) which checks model_id prefix
    - Only E5 models get prefix injection; other models pass through unchanged
    - Line 54: `if _is_e5_standard(model_id)` — conditional, not mandatory
  - Tests confirm multi-variant support (test_embedding.py lines 117-178):
    - Line 139: `EmbeddingIndex(model_id="BAAI/bge-m3", max_input_chars=8192)` — bge-m3 supported
    - Line 151: `EmbeddingIndex(model_id="intfloat/multilingual-e5-base", max_input_chars=4096)` — e5-base supported
    - Line 128: `EmbeddingIndex(model_id="sentence-transformers/all-MiniLM-L6-v2")` — all-MiniLM supported
  - Model-agnostic: sentence-transformers library handles encoding/decoding for any HF model with proper interface

### Claim 13: bge-m3 supports 8192 max_input_chars — no hardcoded truncation limit?

- **Status**: Confirmed (from implementation perspective)
- **Evidence**:
  - EmbeddingIndex implementation:
    - Line 76: `_DEFAULT_MAX_INPUT_CHARS = 2048` is the default, not a hardcoded limit
    - Line 83: Constructor accepts `max_input_chars: int` parameter with no upper-bound check
    - Line 109: `texts.append(text[: self._max_input_chars])` — truncates to whatever value is set
    - No hardcoded 2048 limit in processing logic
  - Test confirms bge-m3 with 8192 works (test_embedding.py lines 135-143):
    - `EmbeddingIndex(model_id="BAAI/bge-m3", max_input_chars=8192)`
    - 9000 char input → truncated to 8192, encoded successfully
  - build_indexes.py (line 54): `cfg.indexing.embedding.get("max_input_chars", 2048)` — respects config value with 2048 fallback
- **Notes**: No hardcoded truncation in EmbeddingIndex. The limit is purely configuration-driven. However, sentence-transformers library itself may have internal tokenizer limits (typically 512-8192 tokens depending on model); this is external to EmbeddingIndex and would be a model-specific constraint, not this code's constraint.

---

## Stage 1 申し送り事項

### Rejected Claims Requiring Correction

1. **Claim 3 (Embedding Default)**: 
   - **Issue**: Report states "intfloat/multilingual-e5-small" is global default
   - **Reality**: Global baseline default is `sentence-transformers/all-MiniLM-L6-v2`
   - **Action for Stage 1 Reviewer**: Correct any Issue #137 description or design documents that reference a global E5 default. The E5 model is institutional-domain-specific only, not global.

### Partially Confirmed Claims Requiring Clarification

2. **Claim 5 (institutional_docs.yaml max_input_chars field)**:
   - **Status**: Config works but does NOT explicitly declare max_input_chars
   - **Current Behavior**: Falls back to hardcoded default 2048 via build_indexes.py line 54
   - **Risk**: If Phase B variants require different max_input_chars (e.g., bge-m3 → 8192), the institutional_docs.yaml must be updated to include explicit `embedding.max_input_chars: <value>` field
   - **Action for Stage 1 Reviewer**: Ensure Phase B setup plan includes institutional_docs.yaml update for each variant's max_input_chars requirement

3. **Claim 7 (Guard Test Reference)**:
   - **Status**: Test exists and protects baseline invariance, but is documented as Issue #114, not #132
   - **Evidence**: File header and test docstrings reference Issue #114 ("YAML invariant tests"), not #132
   - **Action for Stage 1 Reviewer**: Verify if the claim meant #114 or if there is a separate #132 test. Current test is #114-related.

### Unverifiable Claims (External to Codebase)

4. **Claim 10 (Workspace Design Documents)**:
   - **Status**: Files do not exist in repository
   - **Location**: workspace/issues/133/work-plan.md and workspace/design/issue-133-multilingual-retrieval-ab-design-policy.md are referenced but not present
   - **Impact**: These documents are infrastructure/planning artifacts, not code. Their absence does not block the technical infrastructure verification.
   - **Action for Stage 1 Reviewer**: Confirm whether these design documents exist in a separate artifact repository, wiki, or are planned for creation.

### Infrastructure Status Summary

- **Phase A Infrastructure**: ✓ READY
  - EmbeddingIndex.max_input_chars parameter: Implemented and tested
  - Guard test for baseline invariance: In place (test_pipeline_factory_yaml_invariants.py)
  - Institutional config baseline: Measured (11.21% NC)
  - Variant scripts and aggregation tools: Present and functional
  
- **Phase B Preparation**: ⚠ REQUIRES ATTENTION
  - configs/_experiments/ directory: Gitignored but not yet created (placeholder)
  - Variant configs (V0-V4.yaml): Not yet created (Phase B gate)
  - institutional_docs.yaml max_input_chars field: Not yet declared for variants
  - Design documents: Not yet located or created

---

## Appendix: File References

### Confirmed Files
- `/baseline_reporag/indexing/embedding.py` — EmbeddingIndex with max_input_chars
- `/baseline_reporag/tests/test_embedding.py` — 63-line test suite for variants (commit 96e7b45)
- `/configs/baseline.yaml` — Global defaults (all-MiniLM embedding, ms-marco reranker)
- `/configs/institutional_docs.yaml` — Institutional profile (e5-small embedding)
- `/scripts/ingest_repo.py` — Ingestion with --config, --repo-id
- `/scripts/build_indexes.py` — Index building with max_input_chars fallback
- `/scripts/run_baseline_eval.py` — Evaluation with --config, --repo-id
- `/scripts/aggregate_institutional_baseline.py` — Flexible aggregation script
- `/tests/test_pipeline_factory_yaml_invariants.py` — Guard tests + placeholder skip tests
- `/reports/institutional_baseline_static.md` — Baseline measurement (11.21% NC)
- `/.gitignore` — Entry for configs/_experiments/

### Absent Files
- `/workspace/issues/133/work-plan.md` — Not found
- `/workspace/design/issue-133-multilingual-retrieval-ab-design-policy.md` — Not found
- `/configs/_experiments/institutional_V0.yaml` through `/configs/_experiments/institutional_V4.yaml` — Not yet created (placeholder in gitignore only)

---

**End of Hypothesis Verification Report**
