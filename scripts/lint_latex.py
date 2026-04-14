#!/usr/bin/env python3
"""
LaTeX 公式巡检 + 自动修复工具

用法：
    python3 scripts/lint_latex.py                    # 扫描报告
    python3 scripts/lint_latex.py --fix              # 自动修复可安全修复的问题
    python3 scripts/lint_latex.py --path fundamentals # 只扫描指定目录
    python3 scripts/lint_latex.py --fix --dry-run    # 预览修复，不写文件

错误类型：
    E1: 奇数个 $ (未闭合行内公式)
    E2: 未闭合 $$ 块 (行内 $$ 不配对)
    E3: 常见 LaTeX 命令拼写错误 (\frac 写成 \fac 等)
    E4: 下划线未转义 (Obsidian 可能误解为斜体)
    E5: 空 $ $ 对 ($ $ 中间无内容)
    E6: 嵌套 $ ($$...$...$$)
    W1: \\ 后跟小写字母 (可能是 \\text 写成 \\ text)
    W2: \frac{} 缺少参数
    W3: 行内公式过长 (> 80 字符，建议改 $$ 块)
"""

import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
SKIP_DIRS = {"_archive", "repos", "node_modules", ".git", "scripts", "_data", "_config", "~"}

# 常见 LaTeX 命令拼写错误映射
TYPO_MAP = {
    r"\fac{": r"\frac{",
    r"\frc{": r"\frac{",
    r"\delt ": r"\delta ",
    r"\alph ": r"\alpha ",
    r"\sigm ": r"\sigma ",
    r"\thet ": r"\theta ",
    r"\lambd ": r"\lambda ",
    r"\epsiln": r"\epsilon",
    r"\matbb": r"\mathbb",
    r"\mathb{": r"\mathbb{",
    r"\text {": r"\text{",
    r"\operatorname {": r"\operatorname{",
    r"\log_": r"\log_",  # OK, skip
    r"\sum_": r"\sum_",  # OK, skip
}

# 需要在公式外转义的下划线模式（常见误用）
# 不处理：代码块内、URL 内的下划线


class Issue:
    def __init__(self, file: str, line: int, code: str, msg: str, fixable: bool = False, fix_fn=None):
        self.file = file
        self.line = line
        self.code = code
        self.msg = msg
        self.fixable = fixable
        self.fix_fn = fix_fn

    def __str__(self):
        tag = "🔧" if self.fixable else "⚠️"
        return f"  {tag} {self.code} L{self.line}: {self.msg}"


def scan_file(fpath: Path) -> list[Issue]:
    """扫描单个 md 文件的 LaTeX 问题"""
    issues = []
    rel = str(fpath.relative_to(PROJECT_ROOT))

    try:
        content = fpath.read_text(encoding="utf-8")
    except Exception:
        return issues

    lines = content.split("\n")

    # ── E1: 奇数个 $ (逐行检查) ──
    in_code_block = False
    in_math_block = False
    math_block_start = -1

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # 跳过代码块
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        # 跟踪 $$ 块
        dd_count = len(re.findall(r"(?<!\$)\$\$(?!\$)", line))
        if dd_count % 2 == 1:
            if in_math_block:
                in_math_block = False
            else:
                in_math_block = True
                math_block_start = i
            continue

        if in_math_block:
            continue

        # 行内 $ 检查：去掉 $$ 后数剩余的 $
        cleaned = re.sub(r"\$\$", "", line)
        # 去掉转义的 \$
        cleaned = re.sub(r"\\\$", "", cleaned)
        # 去掉代码 `...` 中的 $
        cleaned = re.sub(r"`[^`]*`", "", cleaned)
        single_count = cleaned.count("$")
        if single_count % 2 == 1:
            issues.append(Issue(rel, i, "E1", f"奇数个 $ (={single_count})，可能未闭合: {line.strip()[:80]}"))

    # ── E2: 未闭合 $$ 块 ──
    if in_math_block:
        issues.append(Issue(rel, math_block_start, "E2", f"$$ 块未闭合（从第 {math_block_start} 行开始）"))

    # ── E3: LaTeX 命令拼写错误 ──
    for i, line in enumerate(lines, 1):
        for typo, correct in TYPO_MAP.items():
            if typo == correct:
                continue
            if typo in line:
                def make_fix(t=typo, c=correct):
                    return lambda l: l.replace(t, c)
                issues.append(Issue(rel, i, "E3", f"拼写错误 '{typo}' → '{correct}'",
                                    fixable=True, fix_fn=make_fix()))

    # ── E4: 美元金额误触发公式 ($5.57M, $100B 等) ──
    # 严格模式：只匹配 $数字+小数点+单位 且后面不是 LaTeX 命令
    for i, line in enumerate(lines, 1):
        # 匹配 $数字.数字+单位 (如 $5.57M, $100B, $2.5k) — 真正的货币金额
        # 跳过已转义的 \$ (lookbehind 排除反斜杠)
        money_matches = list(re.finditer(r"(?<!\\)\$(\d+[\d,.]*\s*[MBKTmk](?:illion)?)\b", line))
        for m in money_matches:
            before = line[:m.start()]
            after = line[m.end():]
            cleaned_before = re.sub(r"\\\$", "", before)
            cleaned_before = re.sub(r"\$\$", "", cleaned_before)
            # 检查是否在 LaTeX 公式内部（后面有匹配的 $）
            # 如果后文含 LaTeX 命令（\times, ^, \frac 等）+ 关闭 $，说明是公式不是金额
            cleaned_after = re.sub(r"\\\$", "", after)
            has_closing_dollar = "$" in cleaned_after
            has_latex_commands = bool(re.search(r"\\[a-zA-Z]|[\^_]", after.split("$")[0] if "$" in after else ""))
            if has_closing_dollar and has_latex_commands:
                continue  # 这是合法 LaTeX 公式（如 $70B^2 \times 4$），跳过
            if cleaned_before.count("$") % 2 == 0:
                def make_money_fix(pos=m.start()):
                    return lambda l: l[:pos] + "\\$" + l[pos+1:]
                issues.append(Issue(rel, i, "E4",
                    f"美元金额 '{m.group()}' 会被 Obsidian 解析为公式，需转义为 \\$",
                    fixable=True, fix_fn=make_money_fix()))

    # ── E5: 空 $$ 对 ──
    for i, line in enumerate(lines, 1):
        if "$$" in line and re.search(r"\$\s*\$", line):
            # 区分 $$ (display math) 和 $ $ (empty inline)
            if re.search(r"(?<!\$)\$\s+\$(?!\$)", line):
                issues.append(Issue(rel, i, "E5", f"空公式 '$ $': {line.strip()[:60]}"))

    # ── W1: \\ 后跟小写字母 ──
    for i, line in enumerate(lines, 1):
        if re.search(r"(?<!\$)\$.*\\ [a-z].*\$", line):
            issues.append(Issue(rel, i, "W1", f"反斜杠后有空格（可能拼写错误）: {line.strip()[:60]}"))

    # ── W2: \frac{} 缺少第二个参数（只检查行内公式，跳过 $$ 块内的跨行 frac）──
    in_code = False
    in_display = False
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
        if in_code:
            continue
        if re.match(r"^\$\$\s*$", s):
            in_display = not in_display
            continue
        if in_display:
            continue
        # 只检查行内 $...$ 中的 \frac
        for m in re.finditer(r"(?<!\$)\$([^\$]+)\$(?!\$)", line):
            formula = m.group(1)
            # 检查 \frac{...} 后面不是 { 的情况
            frac_matches = list(re.finditer(r"\\frac\{[^}]*\}([^{\\])", formula))
            for fm in frac_matches:
                issues.append(Issue(rel, i, "W2", f"行内 \\frac 可能缺少第二个参数: {line.strip()[:60]}"))

    # ── W3: 行内公式过长 ──
    for i, line in enumerate(lines, 1):
        inline_formulas = re.findall(r"(?<!\$)\$([^\$]+)\$(?!\$)", line)
        for f in inline_formulas:
            if len(f) > 80:
                issues.append(Issue(rel, i, "W3", f"行内公式过长({len(f)}字符)，建议改为 $$ 块"))

    return issues


def scan_all(root: Path, subpath: str = None) -> dict[str, list[Issue]]:
    """扫描所有 md 文件"""
    results = {}
    scan_root = root / subpath if subpath else root

    for fpath in sorted(scan_root.rglob("*.md")):
        # 跳过排除目录
        rel_parts = fpath.relative_to(root).parts
        if any(p in SKIP_DIRS for p in rel_parts):
            continue

        issues = scan_file(fpath)
        if issues:
            results[str(fpath.relative_to(root))] = issues

    return results


def apply_fixes(root: Path, results: dict, dry_run: bool = False) -> int:
    """应用自动修复"""
    fixed_count = 0
    for rel_path, issues in results.items():
        fixable = [iss for iss in issues if iss.fixable and iss.fix_fn]
        if not fixable:
            continue

        fpath = root / rel_path
        lines = fpath.read_text(encoding="utf-8").split("\n")
        changed = False

        for iss in fixable:
            idx = iss.line - 1
            if 0 <= idx < len(lines):
                new_line = iss.fix_fn(lines[idx])
                if new_line != lines[idx]:
                    if dry_run:
                        print(f"  [DRY] {rel_path}:L{iss.line}: {iss.msg}")
                    lines[idx] = new_line
                    changed = True
                    fixed_count += 1

        if changed and not dry_run:
            fpath.write_text("\n".join(lines), encoding="utf-8")
            print(f"  🔧 修复 {rel_path}: {len(fixable)} 处")

    return fixed_count


def print_report(results: dict):
    """打印巡检报告"""
    total_errors = 0
    total_warnings = 0
    total_fixable = 0
    by_code = defaultdict(int)

    for rel_path, issues in sorted(results.items()):
        print(f"\n📄 {rel_path}")
        for iss in issues:
            print(str(iss))
            by_code[iss.code] += 1
            if iss.code.startswith("E"):
                total_errors += 1
            else:
                total_warnings += 1
            if iss.fixable:
                total_fixable += 1

    print(f"\n{'='*50}")
    print(f"📊 巡检结果汇总")
    print(f"  文件数: {len(results)}")
    print(f"  错误(E): {total_errors}")
    print(f"  警告(W): {total_warnings}")
    print(f"  可自动修复: {total_fixable}")
    print(f"\n  按类型:")
    for code in sorted(by_code):
        print(f"    {code}: {by_code[code]}")


def main():
    parser = argparse.ArgumentParser(description="LaTeX 公式巡检工具")
    parser.add_argument("--fix", action="store_true", help="自动修复可修复的问题")
    parser.add_argument("--dry-run", action="store_true", help="预览修复，不写文件")
    parser.add_argument("--path", type=str, default=None, help="只扫描指定子目录")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    args = parser.parse_args()

    print(f"🔍 LaTeX 公式巡检 — {PROJECT_ROOT}")
    print(f"   范围: {args.path or '全部'}\n")

    results = scan_all(PROJECT_ROOT, args.path)

    if not results:
        print("✅ 未发现公式问题")
        return

    if args.json:
        import json
        out = {}
        for f, issues in results.items():
            out[f] = [{"line": i.line, "code": i.code, "msg": i.msg, "fixable": i.fixable} for i in issues]
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print_report(results)

    if args.fix:
        print(f"\n{'='*50}")
        print("🔧 开始自动修复...")
        count = apply_fixes(PROJECT_ROOT, results, dry_run=args.dry_run)
        print(f"\n{'✅' if not args.dry_run else '📋'} {'修复' if not args.dry_run else '预览'}: {count} 处")


if __name__ == "__main__":
    main()
