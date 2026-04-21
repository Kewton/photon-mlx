# Issue #55 長コンテキスト実測レポート

**実施日**: 2026-04-21
**対象**: `configs/photon_long_context.yaml` (`max_position_embeddings: 65536`, `rope_scaling: ntk`, `rope_scale_factor: 32.0`)
**ハード**: Apple Silicon（統合メモリ 256GB）
**PHOTON**: 学習済みチェックポイントなし（ランダム重み）での実測
**関連コミット**: branch `feature/issue-55-context-length-65k`

## TL;DR

1. **長コンテキスト構築自体は動く**: `max_position_embeddings: 65536` + `rope_scaling: ntk` で `PhotonModel.__init__` が例外なく完了、16,384 トークン prompt での `generate()` も成功。
2. **メモリ見積りは大幅に過小**だった。設計方針書の見積り（16,384 で ~0.75GB）に対し、実測は **cached=20.7GB / nocache=13.1GB**。約 20 倍。設計方針書のメモリテーブルは更新が必要。
3. **KV cache は長コンテキストで負の speedup**: 16,384 prompt / 8 gen で speedup **0.89x**（cache の方が遅い）。原因は encoder_replay / top_level_increment / local_tail_decode の合計が nocache prefill を上回る。実運用では use_kv_cache=False の方が良いケースあり → troubleshooting.md に記載。
4. **`bench/issue61_prune_batch.py` は 32768 実測不可**: PR #69 で `PhotonInference.__init__(..., tokenizer)` が必須化されたが、bench script が未更新で TypeError。**これは Issue #55 とは無関係の pre-existing bug**。別 Issue で修正が必要。

## 詳細

### 1. kv_cache_speedup.py @ prompt-len=1024（smoke test）

```
python bench/kv_cache_speedup.py --config configs/photon_long_context.yaml --prompt-len 1024 --gen 4 --runs 1
```

| 項目 | cached | nocache |
|---|---|---|
| avg_sec | 0.2163 | 0.2016 |
| per_token_ms | 54.0 | 50.4 |
| tokens_per_sec | 18.5 | 19.8 |
| peak_mb | **2,940.9** | **3,162.4** |

Speedup: **0.93x** (cache わずかに遅い)

### 2. kv_cache_speedup.py @ prompt-len=16384（本命）

```
python bench/kv_cache_speedup.py --config configs/photon_long_context.yaml --prompt-len 16384 --gen 8 --runs 1
```

| 項目 | cached | nocache |
|---|---|---|
| avg_sec | 6.9233 | 6.1589 |
| per_token_ms | 865.4 | 769.9 |
| tokens_per_sec | 1.16 | 1.30 |
| peak_mb | **20,773.7** | **13,138.1** |

Speedup: **0.89x** (cache が nocache より 11% 遅い)

#### cached 経路の phase breakdown

| phase | count | mean_ms | total_ms |
|---|---|---|---|
| prefill | 1 | 862.5 | 862.5 |
| encoder_replay | 7 | 77.6 | 543.0 |
| top_level_increment | 7 | 431.6 | 3,020.9 |
| local_tail_decode | 7 | 356.1 | 2,492.9 |

top-level increment が dominant。16,384 prompt 時の T/16 = 1,024 において top-level attention が二次的コストを取る。

### 3. issue61_prune_batch.py @ max-len=32768（未実施）

**エラー**: pre-existing bug により実行不可。

```
python bench/issue61_prune_batch.py --max-len 32768 --n-chunks 16 --warmup 1 --measure 3
```

```
TypeError: PhotonInference.__init__() missing 1 required positional argument: 'tokenizer'
```

原因は `bench/issue61_prune_batch.py:242` の `PhotonInference(model, cfg)` 呼び出し。PR #69（commit `720953d`）で `PhotonInference.__init__` に tokenizer 引数が必須化されたが、bench script は追随更新されていない。**Issue #55 とは無関係**。別 Issue（例: "fix(bench): update issue61_prune_batch.py for tokenizer-required PhotonInference"）として切り出す。

## メモリ見積りの再計算（設計方針書への follow-up）

| コンテキスト長 | 設計見積り (KV cache 込) | 実測 (nocache) | 実測 (cached) | 見積りとの乖離 |
|---|---|---|---|---|
| 1,024 | - | 3.2 GB | 2.9 GB | - |
| 16,384 | ~0.8 GB | 13.1 GB | 20.8 GB | **約 25 倍** |
| 32,768 | ~0.95 GB | - | - | 未実測 |
| 65,536 | ~1.30 GB | - | - | 未実測 |

**原因の推定**:
- 設計見積りでは `top-level attention score = layers × heads × (T/16)^2 × 2B (fp16)` のみ計上していたが、hidden_size=1024 の `cached_top_out` および local RoPE / encoder 側のメモリが含まれていない
- cached 経路の `cached_top_out (B, T/16, hidden_size) × 2B` = `1 × 1024 × 1024 × 2B` = 2MB（これ自体は小さい）
- 実測の 20GB は、おそらく (a) local decoder の中間 tensor (b) attention score の precompute (c) 結合した大きな buffer の一時確保 による

**Issue #55 の実用性判断**:
- 16,384 prompt で nocache 13GB、cached 21GB — **256GB 統合メモリでは十分実用可能**（設計は『2桁誤っても余裕』と明記済み、これは 1.5 桁程度の乖離に収まっている）
- ただし、M3/M4 laptop の 16–48GB RAM 環境では cached 経路は厳しく、16,384 超の prompt で制限される可能性
- **結論**: v1（推論時のみ拡張）は 256GB 環境で動作確認済み。小メモリ環境向けには `use_kv_cache=False` を推奨

## 推奨事項

1. **docs/troubleshooting.md** に以下を追記:
   - 長コンテキスト入力でメモリ不足になる場合は `use_kv_cache=False` を指定
   - 実測では 16,384 prompt で nocache が 7GB 節約 + 11% 高速（このワークロード条件下）
2. **docs/tutorial.md** の「長コンテキスト推論」節に実測値（16,384 で約 20GB）を記載
3. **別 Issue** として `bench/issue61_prune_batch.py` の tokenizer 追加を切り出し
4. **Future Work**: 32,768 / 65,536 での実測は、学習済み checkpoint が利用可能になってから再実施

## Raw JSON 出力

- `bench/reports/kv_cache_speedup_20260421_041655.json` (prompt-len=1024)
- `bench/reports/kv_cache_speedup_20260421_041723.json` (prompt-len=16384)
