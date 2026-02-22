#!/usr/bin/env python3
"""
Reads Criterion JSON results from target/criterion/ and formats them as a
Markdown report.  Raw cargo-bench output is read from stdin and appended as
a collapsible raw section.

Requires Python 3.9+.

Usage (called by scripts/bench.sh):
    cargo bench --features wgpu 2>&1 | python3 scripts/export_bench.py
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# ── Formatting helpers ────────────────────────────────────────────────────────

def _fmt_time(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.2f} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.2f} µs"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    return f"{ns / 1_000_000_000:.2f} s"


def _fmt_throughput(elem_per_s: float) -> str:
    if elem_per_s < 1_000:
        return f"{elem_per_s:.2f} elem/s"
    if elem_per_s < 1_000_000:
        return f"{elem_per_s / 1_000:.2f} Kelem/s"
    if elem_per_s < 1_000_000_000:
        return f"{elem_per_s / 1_000_000:.2f} Melem/s"
    return f"{elem_per_s / 1_000_000_000:.2f} Gelem/s"


# ── JSON reader ───────────────────────────────────────────────────────────────

def collect_results(criterion_dir: Path) -> list:
    """Walk target/criterion/ and return one dict per benchmark variant."""
    rows = []
    for group_dir in sorted(criterion_dir.iterdir()):
        if not group_dir.is_dir() or group_dir.name == "report":
            continue
        group = group_dir.name

        size_dirs = sorted(
            (d for d in group_dir.iterdir() if d.is_dir()),
            key=lambda d: int(d.name) if d.name.isdigit() else 0,
        )
        for size_dir in size_dirs:
            est_file = size_dir / "new" / "estimates.json"
            bm_file  = size_dir / "new" / "benchmark.json"
            if not (est_file.exists() and bm_file.exists()):
                continue

            est = json.loads(est_file.read_text())
            bm  = json.loads(bm_file.read_text())

            mean_ns = est["mean"]["point_estimate"]
            lo_ns   = est["mean"]["confidence_interval"]["lower_bound"]
            hi_ns   = est["mean"]["confidence_interval"]["upper_bound"]
            std_ns  = est["std_dev"]["point_estimate"]

            n = int(bm.get("value_str", 0))
            throughput = ""
            tp = bm.get("throughput")
            if tp and "Elements" in tp:
                throughput = _fmt_throughput(tp["Elements"] / (mean_ns / 1e9))

            rows.append(dict(
                group=group, n=n,
                mean=mean_ns, lo=lo_ns, hi=hi_ns, std=std_ns,
                throughput=throughput,
            ))
    return rows


# ── Git helpers ───────────────────────────────────────────────────────────────

def _git(args: list) -> str:
    try:
        return subprocess.check_output(
            ["git"] + args, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


# ── Markdown renderer ─────────────────────────────────────────────────────────

def render(rows: list, raw: str) -> str:
    now    = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    commit = _git(["rev-parse", "--short", "HEAD"])
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])

    lines = [
        "# Benchmark Results",
        "",
        "| | |",
        "|---|---|",
        f"| **Date** | {now} |",
        f"| **Commit** | `{commit}` ({branch}) |",
        "",
        "## Summary",
        "",
        "| Benchmark | N | Mean | 95% CI | Std dev | Throughput |",
        "|-----------|--:|-----:|--------|--------:|------------|",
    ]

    prev_group = None
    for r in rows:
        # Blank separator row between groups for readability
        if prev_group is not None and r["group"] != prev_group:
            lines.append("| | | | | | |")
        prev_group = r["group"]

        mean = _fmt_time(r["mean"])
        ci   = f"[{_fmt_time(r['lo'])} … {_fmt_time(r['hi'])}]"
        std  = _fmt_time(r["std"])
        lines.append(
            f"| {r['group']} | {r['n']:>5} | {mean:>10} | {ci} | {std:>10} | {r['throughput']} |"
        )

    # ── Raw output (strip compilation noise, keep bench lines) ────────────────
    result_lines = []
    capturing = False
    for line in raw.splitlines():
        # Criterion result lines start after the "Running benches/…" header
        if line.startswith("     Running") and "bench" in line:
            capturing = True
        if capturing:
            result_lines.append(line)

    if result_lines:
        lines += [
            "",
            "## Raw Output",
            "",
            "<details>",
            "<summary>expand</summary>",
            "",
            "```",
            *result_lines,
            "```",
            "",
            "</details>",
        ]

    return "\n".join(lines) + "\n"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    crit = Path("target/criterion")
    if not crit.exists():
        sys.exit(
            "No Criterion results found.\n"
            "Run `cargo bench --features wgpu` first, then re-run this script."
        )

    rows = collect_results(crit)
    raw  = "" if sys.stdin.isatty() else sys.stdin.read()
    print(render(rows, raw), end="")
