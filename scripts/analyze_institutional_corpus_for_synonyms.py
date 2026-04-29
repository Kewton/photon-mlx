"""Issue #175 corpus 分析スクリプト.

institutional_documents corpus (4,228 .md files) から日本語 N-gram を抽出し、
synonym 辞書候補を選定する手がかりを生成する.

依存ゼロ (regex のみ、MeCab/Sudachi 不要、N-gram sliding window).

使用例:
    python scripts/analyze_institutional_corpus_for_synonyms.py \\
        --source /Users/maenokota/share/work/github_kewton/myWebData/markdowndb/institutional_documents \\
        --output workspace/issues/175/corpus-analysis/ngram-frequency.json \\
        --top-n 500
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# 日本語の連続 (漢字 + ひらがな + カタカナ + 長音). 句読点・記号で区切る.
JP_TOKEN_PATTERN = re.compile(r"[一-鿿぀-ゟ゠-ヿー]+")

# 単独で頻出するが synonym として無意味な語 (ストップワード)
NOISE_TOKENS: set[str] = {
    "について",
    "における",
    "により",
    "として",
    "もの",
    "こと",
    "その",
    "この",
    "ため",
    "など",
    "まで",
    "から",
    "また",
    "なお",
    "及び",
    "または",
    "もしくは",
    "場合",
    "ある",
    "する",
    "なる",
    "いる",
    "れる",
    "られる",
    "せる",
    "させる",
    "ます",
    "です",
    "した",
    "して",
    "され",
    "により",
    "者",
    "等",
    "上",
    "下",
    "中",
    "内",
    "外",
    "前",
    "後",
    "年",
    "月",
    "日",
    "時",
    "分",
    "秒",
    "回",
    "件",
    "名",
    "人",
    "号",
    "条",
    "項",
}


def extract_noun_phrases(text: str, min_len: int = 2, max_len: int = 8) -> list[str]:
    """Extract Japanese noun phrase candidates by sliding-window N-grams.

    For each Japanese token block (e.g. "認定基準について"), enumerate
    all sub-strings of length [min_len, max_len].
    """
    phrases: list[str] = []
    for token in JP_TOKEN_PATTERN.findall(text):
        if len(token) < min_len:
            continue
        # Generate N-grams of varying lengths
        for length in range(min_len, min(max_len + 1, len(token) + 1)):
            for start in range(0, len(token) - length + 1):
                phrase = token[start : start + length]
                if phrase in NOISE_TOKENS:
                    continue
                # Skip if entirely hiragana (likely particles/conjunctions)
                if all("぀" <= ch <= "ゟ" for ch in phrase):
                    continue
                phrases.append(phrase)
    return phrases


def analyze_corpus(source_dir: Path, top_n: int = 500) -> dict:
    md_files = sorted(source_dir.rglob("*.md"))
    print(f"Found {len(md_files)} .md files")

    counter: Counter[str] = Counter()
    file_count_per_phrase: Counter[str] = Counter()
    files_processed = 0

    for f in md_files:
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        phrases = extract_noun_phrases(text)
        counter.update(phrases)
        # Document frequency (DF)
        unique_phrases = set(phrases)
        for p in unique_phrases:
            file_count_per_phrase[p] += 1
        files_processed += 1

    # Filter: phrases appearing in >= 5 files (meaningful) and overall frequency >= 10
    filtered = {
        p: {"tf": cnt, "df": file_count_per_phrase[p]}
        for p, cnt in counter.most_common()
        if file_count_per_phrase[p] >= 5 and cnt >= 10
    }

    # Sort by DF (more documents = more general term)
    sorted_phrases = sorted(filtered.items(), key=lambda x: (-x[1]["df"], -x[1]["tf"]))[
        :top_n
    ]

    return {
        "files_processed": files_processed,
        "total_unique_phrases": len(counter),
        "filtered_phrase_count": len(filtered),
        "top_phrases": [
            {"phrase": p, "tf": d["tf"], "df": d["df"]} for p, d in sorted_phrases
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--top-n", type=int, default=500)
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = analyze_corpus(args.source, args.top_n)

    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== Top 50 phrases (sorted by DF) ===")
    for entry in result["top_phrases"][:50]:
        print(f"  {entry['phrase']:<15} tf={entry['tf']:>6}  df={entry['df']:>5}")
    print(f"\nWrote {len(result['top_phrases'])} phrases to {args.output}")


if __name__ == "__main__":
    main()
