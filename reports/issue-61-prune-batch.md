# Issue #61 — `prune_evidence` バッチ化ベンチマーク結果

## 実行条件

| 項目 | 値 |
|------|-----|
| N chunks | 64 |
| max_len (max_position_embeddings) | 2048 |
| max_chunks (top-K) | 8 |
| warmup runs | 3 |
| measure runs | 10 |
| seed | 42 |

## 実機環境

| 項目 | 値 |
|------|-----|
| Python | 3.12.3 |
| Platform | macOS-26.4.1-arm64-arm-64bit |
| Machine | arm64 |
| Processor | arm |
| MLX version | 0.31.1 |

## レイテンシ（per call）

| 経路 | min | p50 | p95 | max | mean | n |
|-----|-----|-----|-----|-----|-----|---|
| 逐次（legacy） | 55.43 ms | 56.35 ms | 61.18 ms | 64.37 ms | 57.90 ms | 10 |
| バッチ（new） | 8.32 ms | 8.52 ms | 8.94 ms | 9.50 ms | 8.64 ms | 10 |

## 高速化倍率

- p50 speedup: **6.611x**
- mean speedup: **6.698x**

## 受入判定

**PASS (>= 1.5x speedup)**

## 選択結果の同等性

- 逐次選択: `[11, 18, 22, 35, 47, 50, 51, 63]`
- バッチ選択: `[11, 18, 22, 35, 47, 50, 51, 63]`
- top-K 一致: **True**

## OOM チェック

実行が完了し、上記レイテンシが記録できていることが OOM していないことの実機証拠である。
M2 Pro / M3 Max など実機での再現は本ファイルを直接実行して結果欄を更新してください。

## 補足

- 本レポートは `bench/issue61_prune_batch.py` の `--report-path` オプションで自動生成・上書きされる。
- 数値は `_tiny_cfg` 相当の小型 PHOTON config を使用しており、production の絶対値とは異なる。倍率（speedup）は同一インスタンス・同一入力での比較なので、実装上の高速化効果を直接示す。
- E2E follow-up latency への影響は本スクリプト単体では計測しない（既存 profiler の `total_ms` / `generation_ms` で間接確認）。
