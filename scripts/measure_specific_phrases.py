"""Issue #175: directly count specific phrase occurrences in corpus.

Given a list of candidate phrases (from Issue #175 examples + seed clusters),
count how many documents each phrase appears in (DF).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# Candidate dictionary structure: {key: [synonym1, synonym2, ...]}
# All entries here are CANDIDATES — final selection happens after frequency check
CANDIDATE_DICT: dict[str, list[str]] = {
    # === 認定 / 適用 系 ===
    "認定基準": ["認定の条件", "認定要件", "認定の基準", "対象事業者"],
    "対象": ["対象者", "対象事業者", "対象事業", "適用", "該当"],
    "適用": ["対象", "該当", "適用範囲"],
    # === 申請 / 手続 系 ===
    "申請方法": ["申請手続", "申込方法", "提出方法", "申請の方法", "申し込み方法"],
    "申請": ["申込", "申し込み", "応募", "提出"],
    "手続": ["手続き", "プロセス", "手順"],
    "申込": ["申込み", "申し込み", "応募", "申請"],
    # === 必要書類 / 提出資料 系 ===
    "必要書類": ["提出書類", "添付書類", "必要な書類", "提出資料"],
    "書類": ["資料", "様式", "書面", "ドキュメント"],
    "添付": ["添付書類", "添付資料", "添付ファイル"],
    "様式": ["書類", "フォーマット", "テンプレート"],
    # === 補助 / 助成 系 ===
    "補助率": ["補助割合", "補助の割合", "助成率", "助成割合"],
    "補助金額": ["助成額", "補助上限額", "補助の上限", "上限額", "補助金"],
    "補助金": ["助成金", "支援金", "交付金", "給付金"],
    "助成": ["補助", "支援", "支給", "給付"],
    "支援": ["援助", "サポート", "助成", "補助"],
    # === 期限 / 受付 系 ===
    "申請期限": ["申請受付期限", "提出期限", "締切", "受付終了", "応募期限"],
    "期限": ["締切", "期日", "終期"],
    "募集期間": ["公募期間", "受付期間", "応募期間"],
    "公募": ["募集", "応募受付"],
    # === 報告 / 連絡 系 ===
    "報告": ["届出", "通知", "申告", "連絡"],
    "通知": ["連絡", "報告", "案内", "周知"],
    "届出": ["申告", "提出", "報告"],
    # === 選定 / 評価 系 ===
    "選定": ["選考", "決定", "採択", "選択"],
    "審査": ["評価", "審理", "検討", "判定"],
    "評価": ["審査", "判定", "査定"],
    "採択": ["選定", "選考", "決定"],
    # === 賃貸 / 契約 系 (corpus theme) ===
    "賃貸": ["貸借", "賃借", "リース"],
    "契約": ["契約書", "合意", "取り決め"],
    "賃料": ["家賃", "賃借料", "貸料"],
    # === 原状回復 / 退去 系 ===
    "原状回復": ["原状復旧", "現状回復", "復旧"],
    "退去": ["明渡", "明け渡し", "解約"],
    "敷金": ["保証金", "預り金"],
    # === 改修 / 修繕 系 ===
    "改修": ["修繕", "リフォーム", "補修", "修理"],
    "修繕": ["改修", "補修", "リフォーム"],
    "整備": ["改修", "改善", "充実"],
    # === 住宅 / 建物 系 ===
    "住宅": ["建物", "住居", "家屋", "物件"],
    "物件": ["建物", "住宅", "不動産"],
    # === 事業 / 業務 系 ===
    "事業": ["事業活動", "業務", "プロジェクト", "事業実施"],
    "業務": ["事業", "活動", "業"],
    # === 管理 / 運営 系 ===
    "管理": ["運営", "運用", "管理運営"],
    "運営": ["管理", "運用"],
    # === 確認 / 検査 系 ===
    "確認": ["検査", "点検", "チェック", "確認事項"],
    "検査": ["点検", "確認", "監査"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    md_files = sorted(args.source.rglob("*.md"))
    print(f"Scanning {len(md_files)} documents...")

    # Collect all unique phrases (keys + values)
    all_phrases: set[str] = set()
    for k, vals in CANDIDATE_DICT.items():
        all_phrases.add(k)
        all_phrases.update(vals)

    df_counter: Counter[str] = Counter()
    files_processed = 0

    for f in md_files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        seen = {p for p in all_phrases if p in text}
        for p in seen:
            df_counter[p] += 1
        files_processed += 1

    # Build report grouped by candidate dict structure
    result = {"files_processed": files_processed, "candidate_dict_with_df": {}}
    for key, syns in CANDIDATE_DICT.items():
        key_df = df_counter.get(key, 0)
        # Filter syns that have meaningful DF (>= 50 docs)
        syn_with_df = [{"phrase": s, "df": df_counter.get(s, 0)} for s in syns]
        # Sort syns by DF
        syn_with_df.sort(key=lambda x: -x["df"])
        result["candidate_dict_with_df"][key] = {
            "key_df": key_df,
            "synonyms": syn_with_df,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Candidate dictionary with DF ===\n")
    for key, info in result["candidate_dict_with_df"].items():
        print(f"key: {key:<10}  df={info['key_df']:>5}")
        for s in info["synonyms"]:
            mark = " ✓" if s["df"] >= 50 else "  "
            print(f"   {mark} {s['phrase']:<14}  df={s['df']:>5}")
        print()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
