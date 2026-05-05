# PHOTON Technical Overview

このドキュメントは、PHOTON-RepoRAG の内部で PHOTON をどのように使っているかを説明します。README はプロダクト概要と始め方に絞り、PHOTON の技術詳細はこのドキュメントに分離しています。

## PHOTON は何をしているのか

このドキュメントでいう PHOTON は、`doc/paper/list.md` に記載している論文 [PHOTON: Hierarchical Autoregressive Modeling for Lightspeed and Memory-Efficient Language Generation](https://arxiv.org/pdf/2512.20687) の技術を指します。

論文上の PHOTON は、Transformer のように長い履歴をフラットな token sequence として毎ステップ参照するのではなく、文脈を複数解像度の latent stream に圧縮し、粗い文脈から細かい token 表現へ戻す hierarchical autoregressive model です。狙いは、長文脈・multi-query で支配的になる KV cache の読み書きを減らし、memory-efficient な生成を行うことです。

この OSS では、その PHOTON の「階層的に文脈を圧縮する」「粗い状態と局所状態を分ける」「文脈候補を latent state として比較できる」という性質を Multi-turn RAG に転用しています。つまり、PHOTON を検索エンジンの代替として使うのではなく、retrieval 後の evidence 候補や過去質問を階層表現に変換し、現在質問に対して残すべき文脈を選ぶための scoring / working memory layer として使います。

## 論文 PHOTON の技術要素

論文で提案されている PHOTON の中核は次の通りです。

| 技術要素 | 論文上の役割 |
|---|---|
| Multi-resolution latent streams | token 列を複数段の latent stream に分け、粗い解像度で長距離文脈を持つ |
| Hierarchical Encoder | token-level state を chunk 単位にまとめ、低レートの contextual state へ圧縮する |
| Context Chunker | 下位レベルの連続表現を chunk にまとめ、上位レベルの入力表現を作る |
| Autoregressive Context Encoder | chunk-level state 間の依存関係を causal に文脈化する |
| Hierarchical Decoder | 上位 latent から下位 stream を top-down に再構成する |
| Context Converter | 上位 latent を、下位 local decoder の短い conditioning prefix に変換する |
| Local Causal Decoder | 各 chunk 内だけを bounded attention で autoregressive に復元する |
| HierGen | 階層 prefill 後、粗い global state と bounded local attention で生成する |
| RecGen | 生成中に bottom-up re-encoding を避け、coarsest latent stream だけを更新する |
| Recursive Loss | bottom-up で得た状態と top-down 再構成を近づけ、階層状態の一貫性を高める補助損失 |

標準 Transformer は、decode 時に伸び続ける token-level KV cache を読むため、長い文脈では memory bandwidth がボトルネックになります。PHOTON は、グローバル文脈を低レート latent stream に逃がし、token-level 復元は固定長の local window に閉じ込めます。これにより、global KV cache は短くなり、local decoder は chunk ごとに並列化しやすくなります。

論文では、2-level hierarchy、chunk size `[4, 4]`、LLaMA-style decoder block、Llama tokenizer などを用いた 600M / 900M / 1.2B 規模の比較が示されています。評価では、KV cache memory と throughput per memory の改善、HierGen に対する RecGen の効率改善が報告されています。

## この OSS での活用方法

この OSS では、論文 PHOTON の技術をそのまま大規模言語モデル置換として使うだけでなく、RAG の文脈選択に転用しています。

| 論文技術 | OSS での対応 |
|---|---|
| Hierarchical Encoder | evidence chunk や質問文を階層的に prefill し、token / mid / top level の表現を得る |
| Multi-resolution latent streams | chunk の粗い意味、局所的な意味、session-level state を分けて扱う |
| Hierarchical Decoder / local reconstruction | PHOTON standalone では生成に使える。RAG 統合では主に scoring / state 更新に使う |
| Recursive Loss | PHOTON checkpoint の学習側で利用可能。RAG runtime では直接の判定ロジックではない |
| HierGen / RecGen | PHOTON standalone の生成技術。RAG 統合では最終回答生成を Qwen fallback / Qwen generation に任せる構成も取る |
| KV-cache 削減の発想 | Multi-turn で全履歴をプロンプトへ詰め込まず、必要な過去質問/evidence だけを選ぶ設計に転用 |

実装上は、`photon_mlx/` が論文 PHOTON の runtime / training layer です。`PhotonModel` は bottom-up encoder と top-down decoder を持ち、`PhotonInference` は session state、hierarchical prefill、chunk scoring を扱います。

`baseline_reporag/photon_pipeline.py` は RAG 統合層です。ここでは retrieval / reranker で得た候補を PHOTON に渡し、現在質問と session 文脈に対する score を取得します。その score を使って、関連過去質問の選択、evidence pruning、support check、citation eligibility を行います。

## モジュール境界

このリポジトリでは、PHOTON 本体と RAG 統合層を分けています。

| 層 | 場所 | 責務 |
|---|---|---|
| PHOTON standalone | `photon_mlx/` | PHOTON model、inference、session state、checkpoint、training |
| RAG baseline | `baseline_reporag/pipeline.py` | ingest/index 済み corpus に対する retrieval、evidence pack、generation、citation |
| PHOTON-RAG integration | `baseline_reporag/photon_pipeline.py` | RAG の evidence selection / session carryover / citation 制御に PHOTON score を組み込む |
| UI / orchestration | `app/`, `scripts/` | Streamlit、CLI、評価、比較レポート |

依存方向は `baseline_reporag` から `photon_mlx` です。`photon_mlx` は `baseline_reporag` に依存しません。OSS MVP では Multi-turn RAG が主プロダクトで、PHOTON 単体利用は experimental standalone API として提供します。

## Multi-turn RAG での使い方

具体的には、次の流れで使います。

1. 通常の retrieval / reranker で候補チャンクを取得する
2. retrieval / reranker 上位 N 件は、検索側の強い根拠として保護する
3. 過去質問と現在質問の関連度を PHOTON で評価する
4. 関連する過去質問から追加 evidence を取得する
5. 候補 evidence を PHOTON score と retrieval/reranker score の両方で評価する
6. 重要な evidence だけを LLM に渡す
7. 回答後、引用が現在質問と回答内容を支えているかを citation eligibility と citation budget で再確認する

PHOTON は検索エンジンの代替ではありません。検索結果を土台にしながら、multi-turn の会話文脈に照らして evidence を選び直すための判断レイヤーです。

## 現在の multi-turn 制御

最新の実装では、PHOTON score だけで evidence を選ぶのではなく、retrieval/reranker と PHOTON の役割を分けて組み合わせます。

| 制御 | 目的 | PHOTON の使い方 |
|---|---|---|
| Segment Memory Retrieval | 短い follow-up 質問で、直近の話題セグメントを復元する | 現在質問と最近の質問群の関連を見て、検索クエリに必要な過去質問を足す |
| Dual Score Pruning | 古い話題の evidence が残り続けることを抑える | retrieval/reranker score、現在質問に対する PHOTON score、session 内 PHOTON score を組み合わせて選別する |
| Support Check for Claims | 根拠が弱いのに断定回答することを抑える | 選ばれた evidence が現在質問をどれだけ支えるかを score 化し、弱い場合は慎重回答を促す |
| Citation Eligibility Scoring | 回答中の citation が本当に現在質問・回答内容を支えているかを確認する | PHOTON current/session score と retrieval score を補助情報として使い、引用ごとの適格性を計算する |
| Citation Budget Re-ranking | citation が多すぎる、または古い話題の citation が混ざることを抑える | 適格性の高い citation を優先し、低い citation を削除または高い citation へ置換する |

これらは制度文書専用のパターンマッチングではなく、現在質問、過去質問、回答文、retrieval score、PHOTON score、引用周辺文脈から汎用的に判断します。

## なぜ PHOTON で実現できるのか

通常の retrieval / reranker は、基本的には現在質問と文書チャンクの関係を評価します。multi-turn では、現在質問だけでは意味が足りないことがあります。

例えば、次の会話では 2 ターン目の質問だけを見ると対象が曖昧です。

```text
1ターン目:
セーフティネット保証1号の認定基準を教えて

2ターン目:
2号との違いは？
```

この場合、2 ターン目では「セーフティネット保証1号について話していた」という過去文脈を使う必要があります。ただし、会話履歴をすべて渡すと、関係の薄い話題まで混ざります。

PHOTON は、現在質問、過去質問、evidence 候補を working memory に読み込ませたうえで、会話文脈上の関連度を score として取り出します。その score を使うことで、次の判断ができます。

- 現在の質問と、過去のどの質問が関係しているか
- 現在の質問に対して、どの evidence 候補が重要か
- retrieval / reranker の上位結果を守りつつ、追加でどのチャンクを残すべきか
- 関係の薄い過去会話や不要な evidence を落としてよいか
- 回答後に、どの citation を残すべきか

つまり PHOTON は、回答文を作る前に、LLM に渡す材料を整理する役割を持ちます。

## PHOTON を使わない方法との比較

PHOTON を使わなくても、multi-turn RAG はある程度実現できます。ただし、それぞれに限界があります。

| 方法 | 概要 | 課題 |
|---|---|---|
| 会話履歴を全部プロンプトに入れる | 過去のやり取りをそのまま LLM に渡す | 不要な履歴も混ざり、コスト・レイテンシが増える。長くなるほど重要情報が埋もれる |
| 直前の質問だけ結合する | 「前の質問 + 現在質問」で検索する | 2 ターン程度なら効くが、長い会話や話題転換に弱い |
| LLM で質問を書き換える | 「2号との違いは？」を完全な検索クエリに変換する | 書き換え品質に依存し、誤った前提を補う可能性がある |
| embedding 類似度で過去質問を選ぶ | 現在質問に近い過去質問をベクトル検索する | 表層的な類似には強いが、会話上の関係や意図の変化を捉えにくい |
| reranker で候補を並べ替える | 検索結果を query-passage 類似度で再順位付けする | 現在質問単体に対する評価になりやすく、過去文脈込みの判断は別途必要 |
| ルールで上位チャンクを固定する | retrieval 上位 K 件を常に使う | 安定するが、multi-turn で不足する文脈を補いにくい |

PHOTON を使うメリットは、現在質問・過去質問・evidence 候補をまとめて「会話文脈上の重要度」として扱える点です。

- 会話履歴を全部渡さず、関連する過去質問だけ選べる
- 現在質問だけでは拾えない evidence を、関連過去質問から補える
- retrieval / reranker 上位を保護しつつ、PHOTON score で追加 evidence を選べる
- topic drift がある場合に、関係の薄い過去文脈を落としやすい
- Debug UI で、PHOTON がどの evidence を評価したか確認できる

PHOTON を活用する価値は、RAG の文脈を単純に増やすことではありません。multi-turn の会話では、必要なのは「全部覚えること」ではなく、「今の質問に必要な過去文脈と evidence を選び直すこと」です。

## PHOTON で目指す改善

PHOTON-RepoRAG では、現在質問と過去質問の関連性を評価し、関連する過去質問から追加 evidence を取得します。そのうえで、retrieval / reranker の上位チャンクを保護しながら、PHOTON score によって補助的な evidence を選別します。

これにより、multi-turn RAG において次の改善を目指します。

- 省略された質問でも、過去の質問文脈を補って解釈する
- 関連する過去質問だけを選び、不要な会話履歴の混入を抑える
- 現在質問だけでは不足する evidence を、関連過去質問から補完する
- 重要な retrieval 上位チャンクを守りつつ、PHOTON によって追加文脈を選別する
- 分割チャンクや近傍チャンクを補完し、根拠の欠落を減らす
- retrieval score と PHOTON score を併用し、現在質問に弱い古い session evidence を落とす
- citation eligibility と citation budget により、回答中の不要引用や stale citation を減らす
- baseline RAG と PHOTON RAG の回答・引用・メトリクスを比較する

## 実装上の位置づけ

PHOTON は、RAG パイプライン全体を置き換えるものではありません。既存の retrieval、reranker、evidence pack、answer generator を活かしながら、生成前の文脈選択を補助します。

```text
現在質問
    -> retrieval / reranker
    -> Segment Memory Retrieval
    -> 関連過去質問の選択
    -> 関連過去質問から evidence 取得
    -> Dual Score Pruning
    -> Support Check
    -> answer generation
    -> Citation Eligibility / Citation Budget
```

この構成にすることで、検索結果を Source of Truth として維持しつつ、multi-turn で不足しやすい文脈引き継ぎと evidence 補完を追加できます。
