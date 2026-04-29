"""Issue #175: extract synonym candidate clusters from N-gram frequency JSON.

For each seed concept (申請, 認定, 補助, 期限, 書類, 対象 ...), grep the N-gram
output for related compound nouns and rank by DF.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

# Seed concepts from Issue #175 + corpus theme (housing subsidy / rental policy)
SEED_KEYWORDS: dict[str, list[str]] = {
    "申請系": ["申請", "申込", "提出", "応募", "公募"],
    "認定系": ["認定", "承認", "確認", "認可", "登録"],
    "対象系": ["対象", "適用", "該当", "受け", "受給"],
    "補助系": ["補助", "助成", "支援", "支給", "給付", "交付"],
    "条件系": ["条件", "要件", "基準", "規定", "規則"],
    "書類系": ["書類", "書面", "様式", "資料", "証明", "添付"],
    "期限系": ["期限", "期間", "締切", "受付", "終期", "始期", "実施"],
    "事業系": ["事業", "業務", "活動", "計画", "プロジェクト"],
    "支払系": ["支払", "支払い", "支出", "費用", "経費", "金額"],
    "報告系": ["報告", "提出", "通知", "連絡", "届出", "申告"],
    "選定系": ["選定", "決定", "判定", "評価", "審査"],
    "改修系": ["改修", "修繕", "改善", "工事", "整備"],
    "賃貸系": ["賃貸", "賃借", "貸借", "契約", "賃料", "家賃"],
    "原状回復系": ["原状", "回復", "復旧", "退去", "敷金", "保証金"],
    "住宅系": ["住宅", "建物", "建築", "物件", "居室"],
    "管理系": ["管理", "運営", "運用", "保守"],
    "募集系": ["募集", "公募", "応募", "二次募集"],
    "確認系": ["確認", "検査", "点検", "監査"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    phrases = data["top_phrases"]

    # Index phrases by all keyword occurrences
    clusters: dict[str, list[dict]] = defaultdict(list)
    for entry in phrases:
        p = entry["phrase"]
        # Filter: 2-6 char phrases (synonym candidates)
        if not (2 <= len(p) <= 6):
            continue
        for cluster_name, keywords in SEED_KEYWORDS.items():
            for kw in keywords:
                if kw in p:
                    clusters[cluster_name].append(
                        {
                            "phrase": p,
                            "tf": entry["tf"],
                            "df": entry["df"],
                            "matched_keyword": kw,
                        }
                    )
                    break  # avoid duplicate within same cluster

    # Sort each cluster by DF desc
    sorted_clusters = {
        name: sorted(items, key=lambda x: -x["df"]) for name, items in clusters.items()
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(sorted_clusters, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== Cluster summary (top 12 phrases per cluster, by DF) ===\n")
    for name, items in sorted_clusters.items():
        if not items:
            continue
        print(f"--- {name} ---")
        for entry in items[:12]:
            print(
                f"  {entry['phrase']:<10} tf={entry['tf']:>6}  df={entry['df']:>5}  (matched: {entry['matched_keyword']})"
            )
        print()


if __name__ == "__main__":
    main()
