# PHOTON-RepoRAG アプリ 使い方ガイド

## 起動

```bash
cd /path/to/photon-mlx
streamlit run app/photon_app.py --server.port 3012 --server.baseUrlPath /proxy/photon
```

ブラウザで http://localhost:3012/proxy/photon を開く。

---

## 画面構成

左サイドバーの 4 つのメニューで操作。

| メニュー | 用途 |
|---------|------|
| 💬 チャット | リポジトリに質問する（drift / turn_history パネル付き） |
| 📦 ベクトルDB作成 | リポジトリを検索可能にする |
| 🧠 PHOTON学習 | follow-up 高速化モデルを学習（+ 評価ジョブ実行） |
| 📋 プロジェクト登録 | DB + モデルを紐づけて管理（+ PHOTON wizard） |

---

## 初回セットアップ (3 ステップ)

### Step 1: ベクトルDB作成

**目的**: リポジトリのコードを検索可能にする

1. 左メニュー **📦 ベクトルDB作成** を選択
2. 入力:
   - **対象リポジトリのディレクトリ**: 対象リポジトリの絶対パス
     - 例: `/Users/maenokota/share/work/github_kewton/mulmoclaude`
   - **リポジトリ ID**: 英数字の識別子 (スペース・ハイフン不可)
     - 例: `mulmoclaude`
   - **Embedding モデル**: ベクトル検索に使うモデルを選択
     - `multilingual-e5-small` — 日本語を含む場合に推奨 (デフォルト)
     - `all-MiniLM-L6-v2` — 英語のみの場合に軽量
     - `multilingual-e5-base` — 多言語で高精度が必要な場合
     - `all-MiniLM-L12-v2` — 英語で高精度が必要な場合
3. **作成開始** をクリック
4. バックグラウンドで 3 フェーズが実行される:
   - Phase 1: Ingest (コードをチャンク分割)
   - Phase 2: BM25 + Embedding (検索インデックス構築)
   - Phase 3: Symbol Graph (import/call 関係の解析)
5. ステータスが `completed` になるまで待つ (通常 2-5 分)

### Step 2: プロジェクト登録

**目的**: DB とモデルを紐づけて「プロジェクト」として使えるようにする

1. 左メニュー **📋 プロジェクト登録** を選択
2. 入力:
   - **プロジェクト名**: 任意の名前 (例: `mulmoclaude`)
     - ⚠ 制約: 英数字・ハイフン・アンダースコアのみ（path traversal 防止）
   - **ベクトルデータベース**: Step 1 で作った repo_id を選択
   - **PHOTON モデル**: 
     - 初回は `(なし — baseline のみ)` を選択
     - PHOTON 学習済みの場合は checkpoint を選択
   - **Config ファイル**:
     - baseline のみ: `configs/baseline.yaml`
     - PHOTON あり: `configs/photon_small.yaml`
     - あるいは **PHOTON settings (Wave 2-4 toggles)** で wizard 使用（§PHOTON wizard 参照）
3. **登録** をクリック

### Step 3: チャット

**目的**: リポジトリに質問する

1. 左メニュー **💬 チャット** を選択
2. 上部のドロップダウンでプロジェクトを選択
3. 下部の入力欄に質問を入力
4. 回答が表示される
   - `[C:N]` は引用 (どのコードチャンクを根拠にしたか)
   - **メトリクス** を展開すると Latency / Citations 数が見える
   - **Drift metrics** を展開すると PHOTON の内部指標が見える（PHOTON プロジェクトのみ、§Drift metrics 参照）
   - **Turn history** を展開するとターンごとの引用数などが見える（§Turn history 参照）
5. 続けて質問すると前の回答を踏まえたマルチターン会話になる
6. **会話をクリア** で新しいセッションを開始

---

## PHOTON wizard（プロジェクト登録画面、Wave 2-4 機能 toggle）

baseline から一歩進めて PHOTON 拡張機能を試す場合、手動 YAML 編集不要で使える wizard。

### 使い方

1. **📋 プロジェクト登録** ページで **PHOTON モデル** を選択
2. **PHOTON settings (Wave 2-4 toggles)** expander を展開
3. **この form で PHOTON YAML を生成して保存** をオンにする
4. 下記の toggle を設定
5. **登録** クリックで `projects/<name>/photon.yaml` が自動生成・保存される

### Toggle 一覧

| Toggle | YAML path | 効果 |
|--------|-----------|------|
| **Config template** | — | `photon_small` / `photon_tiny` / `photon_long_context` から base を選択 |
| **RecGen enabled** | `inference.photon_generation_enabled` | PHOTON で回答生成（⚠ 実測で +6.1pp 悪化、推奨せず）|
| **Fallback policy** | `inference.generation_fallback_policy` | RecGen 失敗時の fallback（`qwen` / `abort`）|
| **2-pass search enabled** | `retrieval.two_pass_search.enabled` | PHOTON 再ランク（⚠ 実測で +4.2pp 悪化、推奨せず）|
| **pass1_top_k / pass2_top_k** | `retrieval.two_pass_search.*_top_k` | 2-pass 検索の 1/2 段目数 |
| **Working memory enabled** | `session_memory.working_memory.enabled` | PHOTON working memory（推奨: ON）|
| **max_turns** | `session_memory.working_memory.max_turns` | 保持するターン数（default 8）|
| **aggregation** | `session_memory.working_memory.aggregation` | `weighted`（推奨）/ `attention` / `last` |
| **storage_mode** | `session_memory.working_memory.storage_mode` | `full`（推奨 2048 コンテキスト）/ `top_level_only`（長コンテキスト）|
| **past_turn_pinning_enabled** | `session_memory.working_memory.past_turn_pinning_enabled` | 過去ターン引用 pin（⚠ 実測で +1.4pp 悪化、推奨せず）|
| **Apply best-practice** | — | 5 キーの推奨値をマージ（下記）|

### Apply best-practice の中身

5 キーを「Gate 2 v4 実測で最良」の組合せに上書き:

- `inference.safe_recgen_enabled: true`
- `inference.evidence_pruning_enabled: true`
- `session_memory.working_memory.enabled: true`
- `inference.photon_generation_enabled: false`（RecGen 非推奨）
- `retrieval.two_pass_search.enabled: false`（2-pass 非推奨）

手動 toggle で敢えて true にしたものと衝突する場合は UI で警告表示。

---

## Drift metrics パネル（チャット画面、PHOTON プロジェクトのみ）

各回答直後、**Drift metrics** expander で PHOTON の内部状態を可視化。

### 表示される指標

| 指標 | 意味 | 色分け |
|------|------|--------|
| **token_level** | token 埋め込みのドリフト | `ok` / `warn` / `alert` |
| **mid_level** | 中間層 coarse state のドリフト | 同上 |
| **top_level** | 最上位 coarse state のドリフト | 同上 |
| **topic_shift** | 話題転換スコア | 同上 |

閾値は config の `drift_thresholds` から読み込み。色分けは Safe RecGen の発火条件と同一基準のため、**ドリフト追跡とフォールバック判定が一致**。

### 使いどころ

- 話題が大きく変わった直後に `topic_shift` が alert → PHOTON 判断で retrieval やり直しが発生している（正常）
- `top_level` が安定して低い → マルチターン文脈継続が効いている

---

## Turn history パネル（チャット画面）

ターンごとの情報を 1 行ずつ表示:

| 列 | 意味 |
|---|------|
| turn_id | ターン番号（1 始まり）|
| question (head) | 質問冒頭 |
| cited | 引用した chunk ID 数 |
| timestamp | 実行時刻 |

PHOTON `turn_history` と baseline `SessionManager.turns` を `turn_id` で join して表示。fail-closed 経路で片方だけ更新された場合は cited が空で表示される（UI が壊れない保証）。

---

## 評価ジョブ実行（training ページ、Wave 4）

PHOTON 学習結果の質を数値で確認するための async eval runner。

### 使い方

1. 左メニュー **🧠 PHOTON学習** を選択
2. 既存の training ジョブの expander 内（または専用セクション）で「評価ジョブ (Issue #82 Wave 4)」を確認
3. **評価対象プロジェクト** を選択
4. **Run Static Eval** または **Run Multi-Turn Eval** をクリック
5. 進捗バー（`done_q / total_q · p50 latency · NC rate`）で状態確認

### eval_type

| タイプ | 対象 eval set | 所要時間 | 用途 |
|--------|-------------|---------|------|
| **Static Eval** | `data/eval_sets/static_eval.jsonl`（120Q）| ~30 分 | 単発質問の NC rate |
| **Multi-Turn Eval** | `data/eval_sets/multi_turn_eval.jsonl`（30 session × 6 turn = 180Q）| ~90 分 | 会話の NC rate |

### 制約

- 同時実行は **1 eval のみ**（LLM ロード競合防止、`MAX_CONCURRENT_EVAL=1`）
- 実行中は両ボタン無効化
- 結果 JSON は `reports/eval_runs/<job_id>.json`、ログは `logs/eval/<job_id>.log`（allowlist 外に書き込まない security guardrail）

### 最近の eval（同一プロジェクト）

最大 5 件まで status・開始時刻・p50・NC rate が表示される。succeeded なら result 場所、failed なら最終エラー 1 行を表示。

---

## PHOTON 学習 (オプション)

**目的**: follow-up ターンを高速化する PHOTON モデルを学習する

> baseline のみでも使えます。PHOTON は「2 問目以降を速くしたい」場合のオプション。

1. 左メニュー **🧠 PHOTON学習** を選択
2. 入力:
   - **対象リポジトリのディレクトリ**: ベクトルDB と同じパス
   - **最大ステップ数**: 学習の長さ
     - 小さいリポジトリ (~100 ファイル): 500
     - 中程度 (~500 ファイル): 1000
     - 大きい (~2000 ファイル): 2000
   - **バッチサイズ**: 2 (メモリ不足なら 1)
   - **学習率**: 0.00015 (デフォルト推奨)
   - **評価間隔**: 100 (デフォルト推奨)
3. **学習開始** をクリック
4. バックグラウンドで学習が進行
   - プログレスバーで進捗を確認
   - val_loss が表示される (低いほど良い)
   - 所要時間: 500 steps で約 1 時間、1000 steps で約 2 時間
5. 完了後、**📋 プロジェクト登録** で PHOTON モデルを選択して再登録
6. 登録後、**評価ジョブ** セクションで Static / Multi-Turn NC を計測し質を確認

### Resume / 追加学習

既存 checkpoint から学習を継続することも可能。CLI から:

```bash
python scripts/train_photon.py --resume checkpoints/<run>/step_000600
```

step 数と optimizer state（AdamW momentum）が完全に復元される。

---

## 質問の例

### Repo オンボーディング

```
このリポジトリの主要モジュールは何ですか？
エントリーポイントはどこですか？
テストはどう構成されていますか？
```

### コードの理解

```
認証処理の流れを教えてください
データベース接続はどこで管理されていますか？
設定ファイルの読み込み方法は？
```

### 影響範囲分析

```
この関数を変更したら何が壊れますか？
このモジュールに依存しているファイルは？
APIの引数を変えた場合の影響は？
```

### バグ調査

```
ファイルアップロードで 413 が返る原因は？
リクエストボディが消費される問題の原因は？
テストでエラーになる箇所はどこですか？
```

### 変更計画

```
キャッシュ機能を追加するならどこに入れるべき？
非同期処理に移行する場合の設計案は？
ログ出力を改善する方法は？
```

---

## マルチターン会話のコツ

PHOTON の効果はマルチターン (深掘り質問) で発揮されます。

```
良い使い方 (ドリルダウン型):
  Q1: 「認証処理の全体像は？」        ← 広い質問
  Q2: 「その中の JWT 検証の詳細は？」 ← 絞り込み
  Q3: 「そこを変えたら何が壊れる？」  ← さらに深掘り
  Q4: 「修正案を出して」              ← 具体的

あまり効果がない使い方 (単発質問の繰り返し):
  Q1: 「認証処理は？」
  Q2: 「データベース接続は？」  ← 話題が変わる
  Q3: 「テストの書き方は？」    ← また変わる
```

PHOTON ならではの効果が出るシーン:
- Turn 5-6 で NC rate 0%（baseline は 6-7%）
- follow-up latency 20s → 13-14s（-34%、KV cache #54）

---

## トラブルシューティング

### ベクトルDB作成が失敗する

**ステータスが `failed` になった場合:**

1. ログを確認:
   ```bash
   cat logs/idx_YYYYMMDD_HHMMSS.log
   ```
2. よくある原因:
   - ディレクトリパスが間違っている
   - Git リポジトリでない (`.git` がない)
   - ディスク容量不足

### チャットでエラーが出る

**「エラーが発生しました」と表示される場合:**

1. ベクトルDB が `completed` になっているか確認
2. プロジェクト登録で正しい repo_id を選択しているか確認
3. Qwen 14B モデルの初回ダウンロードに時間がかかる場合がある (初回のみ ~8GB)

### Drift metrics が "N/A" 表示

- PHOTON プロジェクト（`use_photon=True`）でないと drift は取得されない
- baseline プロジェクトでは正常に "N/A" 表示（エラーではない）

### 評価ジョブが失敗する

- 他の eval 実行中は新規実行不可（1 並列制限）
- 失敗時は UI に最終エラー行が表示される、詳細は `logs/eval/<job_id>.log`
- eval set が `data/eval_sets/` に存在するか確認

### PHOTON 学習が止まる

1. メモリ不足: バッチサイズを 1 に下げる
2. コーパスが小さすぎる: 100 ファイル未満の場合は baseline のみ推奨
3. ログ確認:
   ```bash
   cat logs/train_YYYYMMDD_HHMMSS.log
   ```

### 回答に引用 [C:N] が付かない

- 質問が曖昧すぎる → より具体的に聞く
- リポジトリに該当するコードがない → ABSTAIN は正常動作
- 日本語の質問で検索が弱い → Embedding モデルに `multilingual-e5-small` を使用

### wizard で保存した YAML が反映されない

- `PHOTON モデル` が選択されているか確認（baseline のみでは wizard は無効）
- `projects/<name>/photon.yaml` が生成されているか確認
- 「この form で PHOTON YAML を生成して保存」のチェックが必要

---

## Security Notes（Wave 2）

Wave 2 で以下の guardrail を実装:

- **project_name**: 英数字 + `-_` のみ、path traversal 防止（`_safe_id`）
- **eval paths**: `reports/eval_runs/` と `logs/eval/` のみ書き込み許可
- **state loading**: JSON schema 検証、不正値で起動失敗
- **YAML**: `yaml.safe_load` のみ許可、`!!python/object` 等のカスタムタグ拒否

不審な入力は UI でエラー表示され、ディスク上に不正なファイルは残らない設計。

---

## 技術仕様

### 使用モデル一覧

| モデル | 用途 | サイズ | DL タイミング |
|--------|------|--------|-------------|
| Embedding (選択式) | ベクトル検索 | 100-400 MB | DB 作成時 |
| ms-marco-MiniLM-L-6-v2 | 検索結果の再ランキング | 100 MB | 初回チャット時 |
| PHOTON Small | evidence pruning + session state | 1.5 GB | 学習時に生成 |
| Qwen2.5-Coder-14B-4bit | 回答生成 | 8 GB | 初回チャット時 |

### 現在の実測メトリクス（weighted aggregation、2-run average）

| metric | baseline | PHOTON（default）| PHOTON の効果 |
|--------|---------|-------------------|--------------|
| Static NC | 16.7% | 17.5% | ±ノイズ（Static には寄与せず）|
| MT NC | 9.4% | 7-8% | **-2〜4pp 改善** |
| Turn 5-6 NC | 6-7% | **0%** | **working memory で長期文脈維持** |
| Follow-up latency | 20.1s | 13-14s | **-34%（KV cache #54）**|

### システム要件

| 項目 | 最小 | 推奨 |
|------|------|------|
| OS | macOS 14+ (Apple Silicon) | macOS 15 |
| RAM | 32 GB | 64 GB |
| Storage | 15 GB | 30 GB |
| Python | 3.12+ | 3.12+ |
