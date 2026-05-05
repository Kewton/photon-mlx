# Minimal Demo FAQ

## PHOTON checkpoint は必要ですか？

Baseline mode の ingest / index / ask には PHOTON checkpoint は不要です。PHOTON checkpoint は、multi-turn context と evidence candidates を PHOTON score で ranking / pruning する場合に使います。

## 外部モデルは同梱されていますか？

MVP v0.1.0 の release artifact には PHOTON checkpoint、LLM weight、embedding model weight、reranker model weight は含まれません。利用者は、自分の環境で利用条件を確認したうえで外部モデルを取得します。

## Streamlit と CLI はどう使い分けますか？

Streamlit app は、対象 repository の ingest / index、training、project registration、chat、comparison mode を画面上で確認するための管理 UI です。CLI は、Streamlit で作成した repo_id、config、checkpoint を再利用して batch 実行や smoke check を行うために使います。

## 比較モードでは何を比べますか？

比較モードでは baseline pipeline と PHOTON pipeline に同じ質問を送り、回答、引用、metrics、retrieval debug を turn ごとに比較します。PHOTON は generator や retriever を置き換えるものではなく、会話文脈と evidence candidates を接続する判断レイヤーとして使われます。
