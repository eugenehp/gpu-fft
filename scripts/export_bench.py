#!/usr/bin/env python3
"""
Reads Criterion JSON results from target/criterion/, generates SVG performance
charts, and formats everything as a Markdown report.  Raw cargo-bench output is
read from stdin and appended as a collapsible raw section.

Charts are written to bench-results/charts/ next to the report.

Requires Python 3.9+, matplotlib.

Usage (called by scripts/bench.sh):
    cargo bench --features wgpu 2>&1 | python3 scripts/export_bench.py
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")          # non-interactive, no display required
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT       = SCRIPT_DIR.parent
CHARTS_DIR = ROOT / "bench-results" / "charts"

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

            throughput_str  = ""
            throughput_mels = 0.0
            tp = bm.get("throughput")
            if tp and "Elements" in tp:
                elem_per_s      = tp["Elements"] / (mean_ns / 1e9)
                throughput_str  = _fmt_throughput(elem_per_s)
                throughput_mels = elem_per_s / 1e6

            rows.append(dict(
                group=group, n=n,
                mean=mean_ns, lo=lo_ns, hi=hi_ns, std=std_ns,
                throughput=throughput_str,
                throughput_mels=throughput_mels,
            ))
    return rows


# ── Chart generator ───────────────────────────────────────────────────────────

# Palette and markers — consistent across both charts
_PALETTE  = {"fft": "#2563eb", "ifft": "#059669", "roundtrip": "#dc2626"}
_MARKERS  = {"fft": "o",       "ifft": "s",        "roundtrip": "^"}
_FALLBACK = "#6b7280"


def _apply_style() -> None:
    """Apply a clean, GitHub-friendly rcParams style."""
    plt.rcParams.update({
        "figure.facecolor":    "white",
        "axes.facecolor":      "white",
        "axes.edgecolor":      "#d1d5db",
        "axes.linewidth":      0.8,
        "grid.color":          "#e5e7eb",
        "grid.linewidth":      0.7,
        "grid.linestyle":      "--",
        "font.family":         "sans-serif",
        "font.size":           11,
        "axes.titlesize":      13,
        "axes.titleweight":    "bold",
        "axes.labelsize":      11,
        "axes.labelcolor":     "#374151",
        "xtick.color":         "#6b7280",
        "ytick.color":         "#6b7280",
        "legend.fontsize":     10,
        "legend.framealpha":   0.95,
        "legend.edgecolor":    "#d1d5db",
        "lines.linewidth":     2.0,
        "lines.markersize":    6,
    })


def _x_formatter(x, _):
    v = int(x)
    return f"{v:,}"


def generate_charts(rows: list) -> tuple[Path, Path]:
    """
    Generate latency.svg and throughput.svg inside CHARTS_DIR.
    Returns (latency_path, throughput_path).
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    _apply_style()

    # Group rows by benchmark name
    groups: dict[str, dict] = {}
    for r in rows:
        g = r["group"]
        if g not in groups:
            groups[g] = {"n": [], "mean_us": [], "lo_us": [], "hi_us": [],
                         "throughput_mels": []}
        groups[g]["n"].append(r["n"])
        groups[g]["mean_us"].append(r["mean"] / 1_000)
        groups[g]["lo_us"].append(r["lo"]   / 1_000)
        groups[g]["hi_us"].append(r["hi"]   / 1_000)
        groups[g]["throughput_mels"].append(r["throughput_mels"])

    # ── Latency chart ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for gname in sorted(groups):
        d      = groups[gname]
        color  = _PALETTE.get(gname, _FALLBACK)
        marker = _MARKERS.get(gname, "o")
        ax.plot(d["n"], d["mean_us"],
                marker=marker, color=color, label=gname, zorder=3)
        ax.fill_between(d["n"], d["lo_us"], d["hi_us"],
                        alpha=0.12, color=color, zorder=2)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Signal length  N")
    ax.set_ylabel("Latency  (µs)")
    ax.set_title("GPU FFT/IFFT — Latency vs Signal Length")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_x_formatter))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}"
    ))
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    latency_path = CHARTS_DIR / "latency.svg"
    fig.savefig(latency_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    # ── Throughput chart ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for gname in sorted(groups):
        d      = groups[gname]
        color  = _PALETTE.get(gname, _FALLBACK)
        marker = _MARKERS.get(gname, "o")
        ax.plot(d["n"], d["throughput_mels"],
                marker=marker, color=color, label=gname, zorder=3)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Signal length  N")
    ax.set_ylabel("Throughput  (Melem/s)")
    ax.set_title("GPU FFT/IFFT — Throughput vs Signal Length")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_x_formatter))
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    throughput_path = CHARTS_DIR / "throughput.svg"
    fig.savefig(throughput_path, format="svg", bbox_inches="tight")
    plt.close(fig)

    return latency_path, throughput_path


# ── Git helpers ───────────────────────────────────────────────────────────────

def _git(args: list) -> str:
    try:
        return subprocess.check_output(
            ["git"] + args, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


# ── Markdown renderer ─────────────────────────────────────────────────────────

def render(rows: list, raw: str, chart_paths: tuple | None) -> str:
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
    ]

    # ── Embedded charts ───────────────────────────────────────────────────────
    if chart_paths:
        lat_path, thr_path = chart_paths
        # Paths relative to bench-results/ (where latest.md lives)
        lat_rel = lat_path.relative_to(ROOT / "bench-results")
        thr_rel = thr_path.relative_to(ROOT / "bench-results")
        lines += [
            "## Charts",
            "",
            f"![Latency]({lat_rel})",
            "",
            f"![Throughput]({thr_rel})",
            "",
        ]

    # ── Summary table ─────────────────────────────────────────────────────────
    lines += [
        "## Summary",
        "",
        "| Benchmark | N | Mean | 95% CI | Std dev | Throughput |",
        "|-----------|--:|-----:|--------|--------:|------------|",
    ]

    prev_group = None
    for r in rows:
        if prev_group is not None and r["group"] != prev_group:
            lines.append("| | | | | | |")
        prev_group = r["group"]

        mean = _fmt_time(r["mean"])
        ci   = f"[{_fmt_time(r['lo'])} … {_fmt_time(r['hi'])}]"
        std  = _fmt_time(r["std"])
        lines.append(
            f"| {r['group']} | {r['n']:>5} | {mean:>10} | {ci} | {std:>10}"
            f" | {r['throughput']} |"
        )

    # ── Raw output ────────────────────────────────────────────────────────────
    result_lines = []
    capturing = False
    for line in raw.splitlines():
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
    crit = ROOT / "target" / "criterion"
    if not crit.exists():
        sys.exit(
            "No Criterion results found.\n"
            "Run `cargo bench --features wgpu` first, then re-run this script."
        )

    rows = collect_results(crit)
    raw  = "" if sys.stdin.isatty() else sys.stdin.read()

    chart_paths = None
    if HAS_MATPLOTLIB:
        chart_paths = generate_charts(rows)
        lat, thr = chart_paths
        print(f"✓  Charts     → {lat}", file=sys.stderr)
        print(f"              → {thr}", file=sys.stderr)
    else:
        print(
            "Warning: matplotlib not found — skipping chart generation.",
            file=sys.stderr,
        )

    print(render(rows, raw, chart_paths), end="")
