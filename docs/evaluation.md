# Evaluation Guide

このドキュメントは、Multi-turn RAG の品質を baseline と PHOTON で比較するための評価手順をまとめます。

## 評価対象

MVP では、単発質問だけでなく、会話の流れを含むシナリオで評価します。

- 省略質問で前提を引き継げるか
- 話題が切り替わったときに古い evidence を引きずらないか
- 比較質問で両方の根拠を拾えているか
- 回答本文の citation が実際に回答内容を支えているか
- 不明確な根拠で断定していないか

代表シナリオは `workspace/テストシナリオ2.md` にあります。

## 実行単位

評価は次の 2 種類で実施します。

1. **シナリオ単独実行**
   - 各シナリオを新しい session として実行する
   - そのシナリオ単体で前提引き継ぎと citation が妥当かを見る

2. **連続シナリオ実行**
   - 任意の複数シナリオを同じ session で続けて実行する
   - 話題切り替え後に古い evidence や citation が混ざらないかを見る

## スコア化

`scripts/score_scenario2_comparison.py` は、baseline と PHOTON の JSONL 実行ログを突き合わせ、ターンごとに 10 点満点で評価します。

```bash
python scripts/score_scenario2_comparison.py \
  --baseline-log logs/scenario2_eval/<baseline-run>.jsonl \
  --photon-log logs/scenario2_eval/<photon-run>.jsonl \
  --output-json reports/scenario2_baseline_vs_photon.json \
  --output-csv reports/scenario2_baseline_vs_photon.csv \
  --output-md reports/scenario2_baseline_vs_photon.md
```

出力される主な項目:

| 項目 | 意味 |
|---|---|
| `baseline_total` / `photon_total` | 10 点満点のターン別スコア |
| `delta` | PHOTON - baseline |
| `winner` | スコアまたは同点時の latency を加味した勝者 |
| `answer` | 回答内容が期待要素を含むか |
| `evidence_recall` | 必要な根拠文書を引用できているか |
| `evidence_precision` | 不要な文書を引用していないか |
| `citation` | wrong citation や citation 過多がないか |
| `safety` | 根拠不足の断定を避けているか |

この scorer は deterministic な regression 評価用です。人手レビューを置き換えるものではなく、変更前後の比較を再現可能にするための補助です。

## 最新の評価例

最新の scenario-2 評価では、51 ターンを照合し、PHOTON が平均スコアと平均 latency の両方で baseline を上回りました。

- Report: `reports/scenario2_baseline_vs_photon_citation_eligibility_20260502.md`
- JSON: `reports/scenario2_baseline_vs_photon_citation_eligibility_20260502.json`
- CSV: `reports/scenario2_baseline_vs_photon_citation_eligibility_20260502.csv`

概要:

| 指標 | baseline | PHOTON |
|---|---:|---:|
| 平均スコア | 8.98 / 10 | 9.941 / 10 |
| 平均 latency | 11804.1 ms | 9749.8 ms |

主な改善:

- 「必要書類」などの曖昧な follow-up で、直近の話題に沿った evidence に絞りやすくなった
- 話題転換後の stale citation が減った
- 根拠不足の inclusion 質問で慎重回答しやすくなった

残課題:

- 一部の比較質問では、PHOTON 側で必要 evidence が 1 件不足するケースが残る
- 「事業計画書」と「起業計画書」のように、語が近いが業務上の扱いが異なる項目は人手レビューが必要
- scorer はテストシナリオに対する期待値ベースなので、未知ドメインの品質保証には別シナリオ追加が必要

## Debug で見る項目

Streamlit の Retrieval debug では、次を確認します。

| 列 | 見方 |
|---|---|
| `source` | retrieval, neighbor, related_past, related_past_neighbor, photon_pruned などの由来 |
| `PHOTON score` | PHOTON が evidence candidate に付けた総合スコア |
| `PHOTON current` | 現在質問に対する PHOTON score |
| `PHOTON session` | session 文脈に対する PHOTON score |
| `Used` | 最終 evidence pack に入ったか |
| `Citation` | 回答本文で `[C:N]` として使われたか |

`Used` は「回答生成に渡された evidence」、`Citation` は「回答本文で引用マーカーとして使われた evidence」です。生成に渡されても、LLM が回答内で引用しなければ `Citation` は空になります。
