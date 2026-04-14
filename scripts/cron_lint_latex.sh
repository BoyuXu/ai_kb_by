#!/bin/bash
# LaTeX 公式定期巡检 — 每天 10:00 JST 自动运行
# 发现问题时通过 TG 通知 Boyu
#
# cron 配置:
#   0 10 * * * /Users/boyu/Documents/ai-kb/scripts/cron_lint_latex.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
KB_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/lint_$(date +%Y%m%d).log"
TG_SEND="$HOME/.agents/skills/tg-group-send/scripts/send"

cd "$KB_ROOT"

# 运行巡检
python3 scripts/lint_latex.py --json > "$LOG_FILE" 2>&1
ISSUES=$(python3 -c "
import json, sys
try:
    data = json.load(open('$LOG_FILE'))
    errors = sum(1 for f in data.values() for i in f if i['code'].startswith('E'))
    warns = sum(1 for f in data.values() for i in f if i['code'].startswith('W'))
    fixable = sum(1 for f in data.values() for i in f if i.get('fixable'))
    files = len(data)
    print(f'{errors}|{warns}|{fixable}|{files}')
except:
    print('0|0|0|0')
")

IFS='|' read -r ERRORS WARNS FIXABLE FILES <<< "$ISSUES"

# 只在有错误时通知
if [ "$ERRORS" -gt 0 ]; then
    # 自动修复可修复的
    python3 scripts/lint_latex.py --fix >> "$LOG_FILE" 2>&1
    FIXED=$?

    "$TG_SEND" main "🔍 ai-kb LaTeX 巡检 $(date +%m/%d)
错误: $ERRORS | 警告: $WARNS | 涉及文件: $FILES
已自动修复: $FIXABLE 处
详情: scripts/logs/lint_$(date +%Y%m%d).log"

    # 如果有修复，自动 commit + push
    if [ "$FIXABLE" -gt 0 ]; then
        cd "$KB_ROOT"
        git add -A
        git commit -m "fix: auto-fix $FIXABLE LaTeX issues (lint_latex.py)" 2>/dev/null || true
        git push 2>/dev/null || true
    fi
fi
