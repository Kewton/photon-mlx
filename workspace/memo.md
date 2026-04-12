結論から固定します。

primary target repo は fastapi/fastapi にします。FastAPI は README で Python ベースの API フレームワークだと明示されており、リポジトリ直下に docs、docs_src、tests が揃っています。さらに公式ドキュメントやチュートリアルには依存性注入、セキュリティ、認証など RepoRAG の評価問題を作りやすい題材があります。初回の ingest、評価問題作成、benchmark freeze を安定させるには、まずこの種の「中規模・単一言語・ドキュメント豊富」な repo のほうが向いています。ml-explore/mlx は Apple silicon 向けで魅力的ですが、Python/C++/C/Swift API を持つため、最初の freeze 対象としては複雑すぎます。なので FastAPI を primary、MLX を secondary holdout にするのが安全です。repo_commit は HEAD ではなく固定 SHA にしてください。 

baseline model は 2 本立てで決めるのがよいです。
プロダクト baseline は mlx-community/Qwen2.5-Coder-14B-Instruct-4bit を採用します。元の Qwen2.5-Coder-14B-Instruct は code-specific な instruct model で、14.7B パラメータ、48 層、128K context を持ちます。MLX 版 4-bit 量子化モデルが公開されていて、mlx_lm.load(...) でそのまま読み込め、モデルサイズは約 8.31 GB です。さらに MLX-LM は Apple silicon 向けの生成・微調整パッケージで、prompt caching をサポートしており、同じ長文脈を何度も使う multi-turn / multi-query ワークロードに向いています。RepoRAG の実用 baseline としては、これが最も始めやすいです。 

ただし、PHOTON の研究比較用 control baseline は別に置いてください。 こちらは Qwen 系ではなく、paper に合わせた LLaMA-style decoder-only baseline を matched parameter で持つべきです。PHOTON 論文では vanilla / Block Transformer / PHOTON の比較を LLaMA ベースで行い、Llama tokenizer を使っています。したがって、RepoRAG の「使える baseline」と、PHOTON の「公平なアーキ比較用 baseline」は分けたほうが設計がきれいです。 

Safe RecGen の fallback threshold は v1 では fixed にしてください。 learned は v2 です。 その理由は、PHOTON 論文自身が「RecGen では recursive consistency が重要」であり、「RecGen 誘導過程の厳密 likelihood 評価は standard teacher forcing にそのまま乗せにくい」と述べているからです。ここで threshold まで学習に入れると、PHOTON 本体の改善と controller の改善が混ざって、何が効いたのか分かりにくくなります。最初は ルールベース + 固定閾値 で運用し、ログと human labels が十分に溜まってから小さな calibrator を学習するほうが、比較実験としても安全です。 

v1 の固定閾値は、私は次で始めます。
必須 fallback は exact_quote、diff/patch、security/auth/billing/delete 系。
数値閾値 は latent_cosine_drift > 0.18、logit_kl > 0.75、topic_shift_score > 0.65、confidence < 0.40。
この数値自体は論文の既定値ではなく、最初の安全側ヒューリスティクスです。benchmark freeze の前に dev split だけで軽く調整し、本番比較中は固定してください。

PHOTON の具体設定については、はい、d_model / layers / heads は “論文準拠の downscale” で決めるのが正しいです。ただし、paper の 600M/900M/1.2B をそのまま tiny/small にコピーするのではなく、構造を保ったまま縮小します。論文のアーキテクチャ上の固定点はかなり明確です。PHOTON は 2-level hierarchy を使い、level 1 chunker は concatenate、上位 chunker は linear、converter は 1D convolution で短い conditioning prefix を作り、encoder/decoder ブロックは LLaMA 系です。さらに論文の parameter tables では、600M が base embed d=416 -> hidden 1664 -> FFN 4096 -> 32 heads -> 各 encoder/decoder 4層、900M が 448 -> 1792 -> 4608 -> 32 heads -> 5層、1.2B が 480 -> 1920 -> 5120 -> 32 heads -> 6層 になっています。ここから、base_embed_dim = hidden_size / 4、2-level、chunk size はまず [4,4]、q heads と kv heads は最初は同数という scaling rule を取るのが自然です。 

ここで重要なのは、config に base_embed_dim と hidden_size の 2 つが必要だという点です。論文の 600M Table では encoder 側の最初の embedder が d=416 なのに対して、decoder 側と主要ブロックは d=1664 です。つまり hidden_size だけ持つ単純な LLaMA config では PHOTON を忠実に表現しきれません。 

私なら初期値はこう切ります。

photon_tiny.yaml
base_embed_dim=160
hidden_size=640
intermediate_size=1664
num_attention_heads=10
num_key_value_heads=10
encoder_layers_per_level=[2,2]
decoder_layers_per_level=[2,2]
chunk_sizes=[4,4]
converter_prefix_lengths=[2,2]
vocab_size=32000
context_length=2048
recursive_loss_weight=0.0

これは まず 80M 前後の忠実実装を作るための設定です。紙の構造を保ちつつ、Mac 上で overfit、teacher-forced eval、drift logging を早く回せます。

photon_small.yaml
base_embed_dim=256
hidden_size=1024
intermediate_size=2816
num_attention_heads=16
num_key_value_heads=16
encoder_layers_per_level=[3,3]
decoder_layers_per_level=[3,3]
chunk_sizes=[4,4]
converter_prefix_lengths=[2,2]
vocab_size=32000
context_length=2048
recursive_loss_weight=0.0

これは 200M 前後の small を狙う設定で、tiny よりも multi-turn の挙動が見やすくなります。

そして 直接の論文再現用に photon_600m_paper.yaml を別で持つのがよいです。
そこは論文表に合わせて、
base_embed_dim=416
hidden_size=1664
intermediate_size=4096
num_attention_heads=32
num_key_value_heads=32
encoder_layers_per_level=[4,4]
decoder_layers_per_level=[4,4]
chunk_sizes=[4,4]
をそのまま使います。 

chunk size は、デフォルトを [4,4] にしてください。 論文の chunk-length ablation では、より細かい [2,2] 相当の設定は PPL と zero-shot accuracy を改善しますが、throughput と TPM は大きく落ちます。なので最初の control は [4,4]、品質寄り ablation として [2,2] を後から切るのが順番として正しいです。 

recursive loss の重みは、最初は 0.0 でよいです。論文の ablation でも、moderate な nonzero weight が downstream を少し改善する可能性はありますが、効果は単調ではありません。なので v1 は 0.0 で forward / training / HierGen / RecGen を安定化し、その後に 0.1、0.3 を足すのが実装順として安全です。 

要するに、今回の未解決点はこう閉じるのがいいです。

primary target repo は fastapi/fastapi。
secondary holdout は ml-explore/mlx。
product baseline model は mlx-community/Qwen2.5-Coder-14B-Instruct-4bit。
research control baseline は matched-size の LLaMA-style decoder-only。
Safe RecGen は v1 fixed thresholds、v2 learned calibrator。
PHOTON configs は paper-conformal downscaling で作成し、tiny、small、paper600m の 3 本を分ける。

この方針で tasks.md の Open Questions は実質クローズできます。

必要なら次に、この決定を反映した configs/photon_tiny.yaml / configs/photon_small.yaml / configs/photon_600m_paper.yaml をそのままコピペできる形で出します。

以下で一旦閉じます。
前提はこれで固定です。primary repo は fastapi/fastapi、product baseline は mlx-community/Qwen2.5-Coder-14B-Instruct-4bit、Safe RecGen は v1 では fixed thresholds にします。FastAPI は Python API フレームワークの公式リポジトリで、baseline 候補の MLX 量子化 Qwen2.5-Coder-14B-Instruct は Hugging Face 上で MLX 4-bit 版が公開され、mlx_lm から直接ロードできます。PHOTON 側は、論文本文のアーキテクチャ説明と 600M/900M/1.2B の表を保ったまま縮小した tiny / small と、表10準拠の paper600m を分ける方針です。PHOTON の chunker は concat ベース、converter は 1D convolution ベース、decoder/encoder は LLaMA 系で、論文の訓練設定は Llama tokenizer、vocab 32000、context 2048 です。

Safe RecGen を fixed で始めるのは、論文自体が RecGen の厳密 likelihood / perplexity 評価を standard teacher forcing に素直に載せにくいと述べているためです。最初から learned controller にすると、PHOTON 本体の効果と controller の効果が混ざります。なので v1 はルールベース + 固定閾値、v2 で learned calibrator に進む設計で十分です。


---

configs/photon_tiny.yaml

version: 1

project:
  name: "photon-reporag"
  mode: "photon_tiny"

run:
  name_prefix: "photon_tiny"
  seed: 42
  deterministic: true

paths:
  data_root: "./data"
  raw_root: "./data/raw"
  processed_root: "./data/processed"
  index_root: "./data/indexes"
  eval_root: "./data/eval_sets"
  log_root: "./logs"
  report_root: "./reports"
  checkpoint_root: "./checkpoints"
  cache_root: "./.cache"

runtime:
  device: "auto"
  dtype: "float16"
  num_workers: 8
  profile_latency: true
  profile_memory: true
  compile_decode_step: false

repo:
  repo_id: "fastapi_fastapi"
  repo_path: "/ABSOLUTE/PATH/TO/fastapi"
  repo_commit: "SET_ME_TO_FIXED_SHA"

tokenizer:
  tokenizer_id: "meta-llama/Llama-2-7b-hf"
  vocab_size: 32000

model:
  architecture: "photon_decoder"

  # PHOTON-specific:
  # - base_embed_dim = encoder bottom embed dim
  # - hidden_size    = main model width used by enc/dec blocks and LM head
  base_embed_dim: 160
  hidden_size: 640
  intermediate_size: 1664

  num_attention_heads: 10
  num_key_value_heads: 10
  head_dim: 64

  max_position_embeddings: 2048
  rope_theta: 1000000.0
  norm_eps: 0.00001
  tie_word_embeddings: false
  dropout: 0.0
  bias: false

hierarchy:
  levels: 2

  # Paper-conformal default
  chunk_sizes: [4, 4]
  converter_prefix_lengths: [2, 2]

  chunker:
    level1_type: "concat"
    upper_type: "linear"

  encoder_layers_per_level: [2, 2]
  decoder_layers_per_level: [2, 2]

  context_encoder_arch: "llama_decoder_style"
  context_decoder_arch: "llama_decoder_style"
  context_converter_type: "conv1d"

  recursive_loss_weight: 0.0
  bottleneck_consistency_target: "top_level"

training:
  enabled: true
  stage: "tiny"

  train_corpus: "./data/processed/train_tiny.jsonl"
  val_corpus: "./data/processed/val_tiny.jsonl"

  context_length: 2048
  micro_batch_size: 4
  gradient_accumulation_steps: 8

  learning_rate: 0.0002
  min_learning_rate: 0.00002
  warmup_ratio: 0.03
  weight_decay: 0.1
  max_grad_norm: 1.0

  max_steps: 5000
  eval_every_steps: 200
  save_every_steps: 500
  log_every_steps: 20

retrieval:
  mode: "hybrid"
  lexical_top_k: 20
  embedding_top_k: 20
  fused_top_k: 16
  rerank_top_k: 12

  weights:
    lexical: 0.45
    embedding: 0.45
    graph: 0.10

evidence_pack:
  max_chunks: 16
  max_tokens: 16000

  local_refresh:
    enabled: true
    top_k: 4
    refresh_before_answer: true
    refresh_on_exact_quote: true
    refresh_on_diff_or_patch: true

session_memory:
  mode: "photon"
  max_turns: 8
  summary_max_tokens: 800
  pin_recent_chunks_max: 8
  pin_cited_chunks_max: 12
  track_topic_state: true

inference:
  hierarchical_prefill: true
  recgen_enabled: false
  safe_recgen_enabled: false

  answer_max_new_tokens: 768
  temperature: 0.2
  top_p: 0.9
  do_sample: false

drift_monitoring:
  enabled: true
  track_latent_cosine_drift: true
  track_token_agreement: true
  track_logit_kl: true

logging:
  level: "INFO"
  save_hidden_stats: true
  save_drift_metrics: true
  save_latency_breakdown: true
  save_memory_metrics: true


---

configs/photon_small.yaml

version: 1

project:
  name: "photon-reporag"
  mode: "photon_small"

run:
  name_prefix: "photon_small"
  seed: 42
  deterministic: true

paths:
  data_root: "./data"
  raw_root: "./data/raw"
  processed_root: "./data/processed"
  index_root: "./data/indexes"
  eval_root: "./data/eval_sets"
  log_root: "./logs"
  report_root: "./reports"
  checkpoint_root: "./checkpoints"
  cache_root: "./.cache"

runtime:
  device: "auto"
  dtype: "float16"
  num_workers: 8
  profile_latency: true
  profile_memory: true
  compile_decode_step: true

repo:
  repo_id: "fastapi_fastapi"
  repo_path: "/ABSOLUTE/PATH/TO/fastapi"
  repo_commit: "SET_ME_TO_FIXED_SHA"

tokenizer:
  tokenizer_id: "meta-llama/Llama-2-7b-hf"
  vocab_size: 32000

model:
  architecture: "photon_decoder"

  base_embed_dim: 256
  hidden_size: 1024
  intermediate_size: 2816

  num_attention_heads: 16
  num_key_value_heads: 16
  head_dim: 64

  max_position_embeddings: 2048
  rope_theta: 1000000.0
  norm_eps: 0.00001
  tie_word_embeddings: false
  dropout: 0.0
  bias: false

hierarchy:
  levels: 2
  chunk_sizes: [4, 4]
  converter_prefix_lengths: [2, 2]

  chunker:
    level1_type: "concat"
    upper_type: "linear"

  encoder_layers_per_level: [3, 3]
  decoder_layers_per_level: [3, 3]

  context_encoder_arch: "llama_decoder_style"
  context_decoder_arch: "llama_decoder_style"
  context_converter_type: "conv1d"

  recursive_loss_weight: 0.0
  bottleneck_consistency_target: "top_level"

training:
  enabled: true
  stage: "small"

  train_corpus: "./data/processed/train_small.jsonl"
  val_corpus: "./data/processed/val_small.jsonl"

  context_length: 2048
  micro_batch_size: 2
  gradient_accumulation_steps: 16

  learning_rate: 0.00015
  min_learning_rate: 0.000015
  warmup_ratio: 0.03
  weight_decay: 0.1
  max_grad_norm: 1.0

  max_steps: 12000
  eval_every_steps: 250
  save_every_steps: 1000
  log_every_steps: 20

retrieval:
  mode: "hybrid"
  lexical_top_k: 20
  embedding_top_k: 20
  fused_top_k: 16
  rerank_top_k: 12

  weights:
    lexical: 0.45
    embedding: 0.45
    graph: 0.10

evidence_pack:
  max_chunks: 16
  max_tokens: 16000

  local_refresh:
    enabled: true
    top_k: 4
    refresh_before_answer: true
    refresh_on_exact_quote: true
    refresh_on_diff_or_patch: true

session_memory:
  mode: "photon"
  max_turns: 8
  summary_max_tokens: 800
  pin_recent_chunks_max: 8
  pin_cited_chunks_max: 12
  track_topic_state: true

inference:
  hierarchical_prefill: true
  recgen_enabled: true
  safe_recgen_enabled: true

  answer_max_new_tokens: 768
  temperature: 0.2
  top_p: 0.9
  do_sample: false

safe_recgen:
  enabled: true

  # v1: fixed thresholds
  triggers:
    exact_quote: true
    diff_or_patch: true
    high_risk_query: true
    topic_shift: true
    latent_drift: true
    low_confidence: true

  thresholds:
    latent_cosine_drift: 0.18
    topic_shift_score: 0.65
    confidence_floor: 0.40
    logit_kl: 0.75

  fallback_actions:
    re_retrieve: true
    strengthen_local_refresh: true
    reprefill_hierarchy: true
    fallback_to_baseline_path: true

drift_monitoring:
  enabled: true
  track_latent_cosine_drift: true
  track_token_agreement: true
  track_logit_kl: true
  save_turn_level_metrics: true

logging:
  level: "INFO"
  save_hidden_stats: true
  save_drift_metrics: true
  save_latency_breakdown: true
  save_memory_metrics: true
  save_fallback_reasons: true


---

configs/photon_600m_paper.yaml

version: 1

project:
  name: "photon-reporag"
  mode: "photon_600m_paper"

run:
  name_prefix: "photon_600m_paper"
  seed: 42
  deterministic: true

paths:
  data_root: "./data"
  raw_root: "./data/raw"
  processed_root: "./data/processed"
  index_root: "./data/indexes"
  eval_root: "./data/eval_sets"
  log_root: "./logs"
  report_root: "./reports"
  checkpoint_root: "./checkpoints"
  cache_root: "./.cache"

runtime:
  device: "auto"
  dtype: "float16"
  num_workers: 8
  profile_latency: true
  profile_memory: true
  compile_decode_step: true

repo:
  repo_id: "fastapi_fastapi"
  repo_path: "/ABSOLUTE/PATH/TO/fastapi"
  repo_commit: "SET_ME_TO_FIXED_SHA"

tokenizer:
  tokenizer_id: "meta-llama/Llama-2-7b-hf"
  vocab_size: 32000

model:
  architecture: "photon_decoder"

  # Table 10 aligned
  base_embed_dim: 416
  hidden_size: 1664
  intermediate_size: 4096

  num_attention_heads: 32
  num_key_value_heads: 32
  head_dim: 52

  max_position_embeddings: 2048
  rope_theta: 1000000.0
  norm_eps: 0.00001
  tie_word_embeddings: false
  dropout: 0.0
  bias: false

hierarchy:
  levels: 2
  chunk_sizes: [4, 4]
  converter_prefix_lengths: [2, 2]

  chunker:
    level1_type: "concat"
    upper_type: "linear"

  encoder_layers_per_level: [4, 4]
  decoder_layers_per_level: [4, 4]

  context_encoder_arch: "llama_decoder_style"
  context_decoder_arch: "llama_decoder_style"
  context_converter_type: "conv1d"

  recursive_loss_weight: 0.0
  bottleneck_consistency_target: "top_level"

training:
  # Architecture-matched bench config.
  # This is not intended to reproduce the paper's full H200 training run.
  enabled: false
  stage: "paper600m_arch_only"

  train_corpus: "./data/processed/train_small.jsonl"
  val_corpus: "./data/processed/val_small.jsonl"

  context_length: 2048
  micro_batch_size: 1
  gradient_accumulation_steps: 32

  learning_rate: 0.00012
  min_learning_rate: 0.000012
  warmup_ratio: 0.03
  weight_decay: 0.1
  max_grad_norm: 1.0

  max_steps: 2000
  eval_every_steps: 200
  save_every_steps: 500
  log_every_steps: 20

retrieval:
  mode: "hybrid"
  lexical_top_k: 20
  embedding_top_k: 20
  fused_top_k: 16
  rerank_top_k: 12

  weights:
    lexical: 0.45
    embedding: 0.45
    graph: 0.10

evidence_pack:
  max_chunks: 16
  max_tokens: 16000

  local_refresh:
    enabled: true
    top_k: 4
    refresh_before_answer: true
    refresh_on_exact_quote: true
    refresh_on_diff_or_patch: true

session_memory:
  mode: "photon"
  max_turns: 8
  summary_max_tokens: 800
  pin_recent_chunks_max: 8
  pin_cited_chunks_max: 12
  track_topic_state: true

inference:
  hierarchical_prefill: true
  recgen_enabled: true
  safe_recgen_enabled: false

  answer_max_new_tokens: 768
  temperature: 0.2
  top_p: 0.9
  do_sample: false

safe_recgen:
  enabled: false

drift_monitoring:
  enabled: true
  track_latent_cosine_drift: true
  track_token_agreement: true
  track_logit_kl: true
  save_turn_level_metrics: true

logging:
  level: "INFO"
  save_hidden_stats: true
  save_drift_metrics: true
  save_latency_breakdown: true
  save_memory_metrics: true


---

configs/local.baseline.yaml の最小差し替え

これは configs/baseline.yaml をコピーして、最初に最低限ここだけ変える想定です。mlx-community/Qwen2.5-Coder-14B-Instruct-4bit は MLX 4-bit 版が公開されていて、mlx_lm.load(...) でそのまま使えます。元モデルの Qwen2.5-Coder-14B-Instruct は 14.7B、48層、128K context の code 特化 instruct モデルです。

repo:
  repo_id: "fastapi_fastapi"
  repo_path: "/ABSOLUTE/PATH/TO/fastapi"
  repo_commit: "SET_ME_TO_FIXED_SHA"

model:
  provider: "mlx_lm"
  model_id: "mlx-community/Qwen2.5-Coder-14B-Instruct-4bit"
  tokenizer_id: ""
  trust_remote_code: false
  use_quantized: true
  quantization: "4bit"


---

tasks.md の Open Questions を閉じる追記

これを tasks.md 末尾の Open Questions の代わりに置けば、未決項目は一旦閉じられます。FastAPI を primary、MLX を secondary holdout にするのは、最初の ingest / eval / benchmark freeze を単一言語寄りで安定させるためです。MLX 自体は Apple silicon 向けの公式フレームワークで、別系統の holdout repo としてちょうど良いです。

# Resolved Decisions

- [x] primary target repo は `fastapi/fastapi`
- [x] secondary holdout repo は `ml-explore/mlx`
- [x] product baseline model は `mlx-community/Qwen2.5-Coder-14B-Instruct-4bit`
- [x] research control baseline は matched-size の LLaMA-style decoder-only
- [x] Safe RecGen の fallback threshold は v1 では fixed、v2 で learned calibrator を検討
- [x] PHOTON config は `tiny` / `small` / `paper600m` の 3 本で管理
- [x] PHOTON default chunk size は `[4, 4]`
- [x] 初期 `recursive_loss_weight` は `0.0`


---

これで固定されること

この状態で、tasks.md 末尾の未解決 3 点は実質閉じられます。
また、PHOTON の初期 config は論文準拠の縮尺で決めた、と答えて問題ありません。論文上の 600M/900M/1.2B は、600M が base embed 416 / hidden 1664 / FFN 4096 / 32 heads / enc-dec 各4層、900M が 448 / 1792 / 4608 / 32 heads / 各5層、1.2B が 480 / 1920 / 5120 / 32 heads / 各6層 です。今回の tiny と small は、その構造を崩さず Mac 実装向けに縮めたものです。

これで spec.md、README.md、tasks.md、baseline.yaml、eval.yaml、photon_tiny.yaml、photon_small.yaml、photon_600m_paper.yaml が一本につながります。