# PHOTON-RepoRAG — Self-use Makefile (A-1 / 自分用試行錯誤環境)
#
# 使い方:
#   make help              # 全 target を一覧表示
#   make setup             # venv + pip install
#   make ingest REPO=/path/to/target REPO_ID=fastapi_fastapi
#   make indexes REPO_ID=fastapi_fastapi
#   make ask Q="認証処理の入口は？"
#   make eval CONFIG=configs/photon_small.yaml
#   make check             # ruff + pytest
#
# 別 config を使うときは CONFIG=... を渡す:
#   make ask CONFIG=configs/photon_small.yaml Q="..."
#   make ingest CONFIG=configs/institutional_docs.yaml REPO=...

CONFIG ?= configs/baseline.yaml
REPO ?=
REPO_ID ?= fastapi_fastapi
COMMIT ?= HEAD
SCENARIO ?= demo-01
Q ?=

PYTHON ?= python
RUFF ?= ruff
PYTEST ?= $(PYTHON) -m pytest

# default ターゲットは help
.DEFAULT_GOAL := help

.PHONY: help setup ingest indexes graph prepare serve ask ask-photon cli compare \
        demo demo-list eval eval-baseline eval-photon \
        check lint fmt fmt-check test test-fast clean

help: ## 全 target を一覧表示
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "  Variables (上書き可):"
	@echo "    CONFIG=$(CONFIG)"
	@echo "    REPO_ID=$(REPO_ID)  COMMIT=$(COMMIT)"
	@echo "    SCENARIO=$(SCENARIO)"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

setup: ## venv 作成 + 依存インストール (.venv が無い場合)
	@if [ ! -d .venv ]; then \
		$(PYTHON) -m venv .venv; \
		echo "✓ .venv 作成完了。'source .venv/bin/activate' で activate してください"; \
	fi
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Ingestion / indexing
# ---------------------------------------------------------------------------

ingest: ## repo を ingest (REPO=/path/to/target REPO_ID=id 必須)
	@if [ -z "$(REPO)" ]; then echo "ERROR: REPO=/path/to/target が必要です"; exit 1; fi
	$(PYTHON) scripts/ingest_repo.py --repo "$(REPO)" --repo-id $(REPO_ID) --commit $(COMMIT) --config $(CONFIG)

indexes: ## BM25 + embedding index を構築 (REPO_ID 指定可)
	$(PYTHON) scripts/build_indexes.py --repo-id $(REPO_ID) --config $(CONFIG)

graph: ## symbol graph を構築 (Python 中心 repo のみ)
	$(PYTHON) scripts/build_symbol_graph.py --repo-id $(REPO_ID) --config $(CONFIG)

prepare: ingest indexes graph ## ingest + indexes + graph を一括実行

# ---------------------------------------------------------------------------
# Serve / Ask
# ---------------------------------------------------------------------------

serve: ## FastAPI server を起動 (configs/baseline.yaml の serving セクション参照)
	$(PYTHON) -m baseline_reporag.server --config $(CONFIG)

ask: ## CLI から 1 問 (Q="..." で質問を渡す)
	@if [ -z "$(Q)" ]; then echo 'ERROR: Q="質問" を渡してください'; exit 1; fi
	$(PYTHON) -m baseline_reporag.cli --config $(CONFIG) --repo-id $(REPO_ID) --question "$(Q)"

cli: ask ## ask の alias

ask-photon: ## CLI from PHOTON pipeline (--use-photon shortcut)
	@if [ -z "$(Q)" ]; then echo 'ERROR: Q="質問" を渡してください'; exit 1; fi
	$(PYTHON) -m baseline_reporag.cli --use-photon --repo-id $(REPO_ID) --question "$(Q)"

compare: ## baseline と PHOTON を 1 質問で並列比較 (Q="..." 必須)
	@if [ -z "$(Q)" ]; then echo 'ERROR: Q="質問" を渡してください'; exit 1; fi
	$(PYTHON) scripts/compare_baseline_photon.py --repo-id $(REPO_ID) --question "$(Q)"

# ---------------------------------------------------------------------------
# Demo scenarios
# ---------------------------------------------------------------------------

demo: ## demo シナリオを実行 (SCENARIO=demo-01 がデフォルト)
	$(PYTHON) demo/run_demo.py --scenario $(SCENARIO) --config $(CONFIG)

demo-list: ## demo シナリオの一覧を表示
	$(PYTHON) demo/run_demo.py --list

# ---------------------------------------------------------------------------
# Eval / Benchmark
# ---------------------------------------------------------------------------

eval: ## bench を実行 (CONFIG=configs/eval.yaml がデフォルト)
	$(PYTHON) bench/run_all.py --config configs/eval.yaml

eval-baseline: ## baseline 単独で eval (qwen-25 vs qwen-35 など)
	$(PYTHON) bench/run_all.py --config configs/eval.yaml --variants baseline

eval-photon: ## PHOTON 単独で eval
	$(PYTHON) bench/run_all.py --config configs/eval.yaml --variants photon

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

check: lint fmt-check test ## ruff check + format check + pytest

lint: ## ruff check
	$(RUFF) check .

fmt: ## ruff format で auto-fix
	$(RUFF) format .

fmt-check: ## ruff format --check のみ (差分なしを確認)
	$(RUFF) format --check .

test: ## 全テスト実行 (重め)
	$(PYTEST) torch_ref/tests/ photon_mlx/tests/ baseline_reporag/tests/ tests/ -v

test-fast: ## 軽量テストのみ (CI 不要、コミット前確認用)
	$(PYTEST) tests/ baseline_reporag/tests/ -q -x --ignore=tests/integration

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

clean: ## __pycache__ / .pytest_cache を削除
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
