#!/bin/bash
# Issue番号のバリデーション関数

# Issue番号を検証し、ISSUE_NUMBER変数にセットする
# 使用方法: validate_issue_number "$1"
validate_issue_number() {
    local input="$1"

    # 空チェック
    if [ -z "$input" ]; then
        echo "エラー: Issue番号を指定してください。"
        echo "使用方法: /command [Issue番号]"
        return 1
    fi

    # '#' プレフィックスを除去
    local number="${input#\#}"

    # 数値チェック
    if ! [[ "$number" =~ ^[0-9]+$ ]]; then
        echo "エラー: Issue番号は数値で指定してください。入力値: '$input'"
        return 1
    fi

    # 0チェック
    if [ "$number" -eq 0 ]; then
        echo "エラー: Issue番号は1以上の数値で指定してください。"
        return 1
    fi

    # グローバル変数にセット
    export ISSUE_NUMBER="$number"
    echo "Issue #${ISSUE_NUMBER} を対象にします。"
    return 0
}

# ブランチ名を生成する
# 使用方法: generate_branch_name "$ISSUE_NUMBER" "$DESCRIPTION"
generate_branch_name() {
    local issue_number="$1"
    local description="$2"

    # 説明文をケバブケースに変換
    local kebab
    kebab=$(echo "$description" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | sed 's/^-//' | sed 's/-$//' | cut -c1-40)

    if [ -z "$kebab" ]; then
        echo "feature/issue-${issue_number}"
    else
        echo "feature/issue-${issue_number}-${kebab}"
    fi
}

# worktreeディレクトリ名を生成する
# 使用方法: get_worktree_dir "$ISSUE_NUMBER"
get_worktree_dir() {
    local issue_number="$1"
    echo "../photon-mlx-issue-${issue_number}"
}
