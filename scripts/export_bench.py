#!/usr/bin/env python3
"""
Reads Criterion JSON results from target/criterion/, generates SVG performance
charts, and formats everything as a Markdown report.  Raw cargo-bench output is
read from stdin and appended as a collapsible raw section.

Charts are written to bench-results/charts/ next to the report.

Requires Python 3.9+, matplotlib.

Usage (called by scripts/bench.sh):
    cargo bench --features wgpu 2>&1 | python3 scripts/export_bench.py

Directory layout produced by Criterion
───────────────────────────────────────
Criterion sanitises '/' in group names to '_', so:

  Rust benchmark group          Criterion directory
  ────────────────────────────  ──────────────────────────────
  "fft"                         fft/<param>/new/
  "fft_batch/batch_size"   →    fft_batch_batch_size/<param>/new/
  "fft_batch/signal_len"   →    fft_batch_signal_len/<param>/new/
  "fft_batch_vs_sequential"     fft_batch_vs_sequential/<sub>/<param>/new/
  (BenchmarkId::new("batch",n)) └─ sub = "batch" | "sequential"

collect_results() handles all three layouts by inspecting path depth.
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

# ── Color / marker palette ────────────────────────────────────────────────────
#
# Group names here match the *Criterion directory names* (underscored), not the
# Rust benchmark group strings (which use '/').
#
# Design intent
# ─────────────
# Scalar baselines        blue / emerald / red          saturated, high contrast
# Batch FFT family        violet / purple               same cool hue as blue, darker
# Batch IFFT family       cyan / teal                   same cool family as emerald
# Batch round-trip        amber / orange                warm, clearly distinct from red
#
# vs_sequential groups receive TWO colours each via _VS_PALETTE:
#   batch      = saturated batch colour  (matches the batch_size chart)
#   sequential = lighter tint of the matching scalar baseline
# so the two sub-series are immediately distinguishable without shape alone.

_PALETTE: dict[str, str] = {
    # ── scalar baselines ──────────────────────────────────────────────────────
    "fft":                        "#2563eb",  # blue-600
    "ifft":                       "#059669",  # emerald-600
    "roundtrip":                  "#dc2626",  # red-600

    # ── batch FFT — violet / purple family ───────────────────────────────────
    "fft_batch_batch_size":       "#7c3aed",  # violet-700
    "fft_batch_signal_len":       "#a855f7",  # purple-500

    # ── batch IFFT — cyan / teal family ──────────────────────────────────────
    "ifft_batch_batch_size":      "#0891b2",  # cyan-600
    "ifft_batch_signal_len":      "#14b8a6",  # teal-500

    # ── batch round-trip — amber / orange family ──────────────────────────────
    "roundtrip_batch":            "#d97706",  # amber-600
    "roundtrip_batch_signal_len": "#ea580c",  # orange-600
}

# Sub-series colours for the head-to-head comparison groups.
# Keys are Criterion directory names; values map sub-series label → hex colour.
_VS_PALETTE: dict[str, dict[str, str]] = {
    "fft_batch_vs_sequential": {
        "batch":      "#7c3aed",  # violet-700  — matches fft_batch_batch_size
        "sequential": "#93c5fd",  # blue-300    — lighter echo of scalar fft
    },
    "ifft_batch_vs_sequential": {
        "batch":      "#0891b2",  # cyan-600    — matches ifft_batch_batch_size
        "sequential": "#6ee7b7",  # emerald-300 — lighter echo of scalar ifft
    },
}

_MARKERS: dict[str, str] = {
    # scalar
    "fft":                        "o",   # circle
    "ifft":                       "s",   # square
    "roundtrip":                  "^",   # triangle-up
    # batch FFT
    "fft_batch_batch_size":       "D",   # diamond
    "fft_batch_signal_len":       "D",
    # batch IFFT
    "ifft_batch_batch_size":      "P",   # filled-plus
    "ifft_batch_signal_len":      "P",
    # batch round-trip
    "roundtrip_batch":            "v",   # triangle-down
    "roundtrip_batch_signal_len": "v",
}

# Sub-series markers / line-styles for vs_sequential groups.
_VS_MARKERS: dict[str, str] = {"batch": "D", "sequential": "o"}
_VS_LS:      dict[str, str] = {"batch": "-", "sequential": "--"}

_FALLBACK_COLOR  = "#6b7280"  # gray-500 — shown for any unmapped group
_FALLBACK_MARKER = "o"

# Human-readable legend labels.
_LABELS: dict[str, str] = {
    "fft":                        "fft",
    "ifft":                       "ifft",
    "roundtrip":                  "roundtrip",
    "fft_batch_batch_size":       "fft_batch  (sweep batch)",
    "fft_batch_signal_len":       "fft_batch  (batch=16)",
    "ifft_batch_batch_size":      "ifft_batch  (sweep batch)",
    "ifft_batch_signal_len":      "ifft_batch  (batch=16)",
    "roundtrip_batch":            "roundtrip_batch  (sweep batch)",
    "roundtrip_batch_signal_len": "roundtrip_batch  (batch=16)",
}

# Groups whose primary x-axis is signal length N.
_SIGNAL_LEN_GROUPS = frozenset({
    "fft", "ifft", "roundtrip",
    "fft_batch_signal_len",
    "ifft_batch_signal_len",
    "roundtrip_batch_signal_len",
})

# Groups whose primary x-axis is batch size.
_BATCH_SIZE_GROUPS = frozenset({
    "fft_batch_batch_size",
    "ifft_batch_batch_size",
    "roundtrip_batch",
})

# Groups that contain two named sub-series (batch vs sequential).
_VS_GROUPS = frozenset({
    "fft_batch_vs_sequential",
    "ifft_batch_vs_sequential",
})


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

def collect_results(criterion_dir: Path) -> list[dict]:
    """
    Recursively walk target/criterion/ and return one dict per benchmark leaf.

    Three path shapes are handled (see module docstring):

    depth 2  →  group / param / new / estimates.json
               e.g.  fft / 1024 / new / …
               group = parts[0], sub_series = None, n = int(parts[1])

    depth 3  →  group / sub_series / param / new / estimates.json
               e.g.  fft_batch_vs_sequential / batch / 16 / new / …
               group = parts[0], sub_series = parts[1], n = int(parts[2])

    depth > 3: treated as depth 2 (group = all-but-last, param = last).
    """
    rows: list[dict] = []
    _walk(criterion_dir, criterion_dir, rows)
    return rows


def _walk(root: Path, node: Path, rows: list[dict]) -> None:
    if (node / "new" / "estimates.json").exists():
        _parse_leaf(root, node, rows)
        return
    for child in sorted(node.iterdir()):
        if child.is_dir() and child.name not in ("report", "new"):
            _walk(root, child, rows)


def _parse_leaf(root: Path, node: Path, rows: list[dict]) -> None:
    est_file = node / "new" / "estimates.json"
    bm_file  = node / "new" / "benchmark.json"
    if not (est_file.exists() and bm_file.exists()):
        return

    rel   = node.relative_to(root)
    parts = rel.parts   # e.g. ("fft","1024") or ("fft_batch_vs_sequential","batch","16")
    depth = len(parts)

    if depth < 2:
        return

    if depth == 2:
        # Standard: group / param
        group, raw_param = parts[0], parts[1]
        sub_series: str | None = None
    elif depth == 3:
        # vs-style: group / sub_series / param
        group, sub_series, raw_param = parts[0], parts[1], parts[2]
    else:
        # Deep nesting: group = all but last, no sub_series
        group, raw_param, sub_series = "/".join(parts[:-1]), parts[-1], None

    try:
        n = int(raw_param)
    except ValueError:
        n = 0

    est = json.loads(est_file.read_text())
    bm  = json.loads(bm_file.read_text())

    mean_ns = est["mean"]["point_estimate"]
    lo_ns   = est["mean"]["confidence_interval"]["lower_bound"]
    hi_ns   = est["mean"]["confidence_interval"]["upper_bound"]
    std_ns  = est["std_dev"]["point_estimate"]

    throughput_str  = ""
    throughput_mels = 0.0
    tp = bm.get("throughput")
    if tp and "Elements" in tp:
        elem_per_s      = tp["Elements"] / (mean_ns / 1e9)
        throughput_str  = _fmt_throughput(elem_per_s)
        throughput_mels = elem_per_s / 1e6

    rows.append(dict(
        group=group, raw_param=raw_param, sub_series=sub_series, n=n,
        mean=mean_ns, lo=lo_ns, hi=hi_ns, std=std_ns,
        throughput=throughput_str,
        throughput_mels=throughput_mels,
    ))


# ── Chart helpers ─────────────────────────────────────────────────────────────

def _apply_style() -> None:
    """Apply a clean, GitHub-friendly rcParams style."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#d1d5db",
        "axes.linewidth":    0.8,
        "grid.color":        "#e5e7eb",
        "grid.linewidth":    0.7,
        "grid.linestyle":    "--",
        "font.family":       "sans-serif",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "axes.labelcolor":   "#374151",
        "xtick.color":       "#6b7280",
        "ytick.color":       "#6b7280",
        "legend.fontsize":   9,
        "legend.framealpha": 0.95,
        "legend.edgecolor":  "#d1d5db",
        "lines.linewidth":   2.0,
        "lines.markersize":  6,
    })


def _x_formatter(x, _):
    return f"{int(x):,}"


def _build_series(rows: list[dict], groups: list[str]) -> dict[str, dict]:
    """
    Aggregate rows for the given group names into per-group series dicts.
    Each dict has: n, mean_us, lo_us, hi_us, throughput_mels — sorted by n.
    Only rows with sub_series=None are included (vs groups handled separately).
    """
    series: dict[str, dict] = {}
    for r in rows:
        g = r["group"]
        if g not in groups or r["sub_series"] is not None:
            continue
        if g not in series:
            series[g] = {"n": [], "mean_us": [], "lo_us": [], "hi_us": [],
                         "throughput_mels": []}
        series[g]["n"].append(r["n"])
        series[g]["mean_us"].append(r["mean"] / 1_000)
        series[g]["lo_us"].append(r["lo"]   / 1_000)
        series[g]["hi_us"].append(r["hi"]   / 1_000)
        series[g]["throughput_mels"].append(r["throughput_mels"])
    for d in series.values():
        order = sorted(range(len(d["n"])), key=lambda i: d["n"][i])
        for k in d:
            d[k] = [d[k][i] for i in order]
    return series


def _line_chart(
    series: dict[str, dict],
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    metric: str,            # "mean_us" or "throughput_mels"
    xscale: str = "log2",
    scalar_groups: frozenset[str] = frozenset(),
) -> Path:
    """
    Render a single line chart and save it as SVG.

    If scalar_groups is non-empty, those groups get solid lines and the rest
    get dashed lines, so scalar baselines stand out in comparison charts.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for gname in sorted(series):
        d      = series[gname]
        color  = _PALETTE.get(gname, _FALLBACK_COLOR)
        marker = _MARKERS.get(gname, _FALLBACK_MARKER)
        label  = _LABELS.get(gname, gname)
        ls     = "-" if (not scalar_groups or gname in scalar_groups) else "--"

        ax.plot(d["n"], d[metric], marker=marker, color=color,
                label=label, linestyle=ls, zorder=3)

        if metric == "mean_us":
            ax.fill_between(d["n"], d["lo_us"], d["hi_us"],
                            alpha=0.12, color=color, zorder=2)

    if xscale == "log2":
        ax.set_xscale("log", base=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_x_formatter))
    if metric == "mean_us":
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x:.0f}"
        ))
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()

    path = CHARTS_DIR / filename
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return path


def _vs_chart(rows: list[dict], group_names: list[str]) -> Path:
    """
    Render the batch-vs-sequential comparison chart.

    Each group becomes one subplot with two lines:
    • batch      — solid diamond,  saturated colour
    • sequential — dashed circle,  lighter tint of the scalar baseline
    """
    n_plots = len(group_names)
    fig, axes = plt.subplots(1, n_plots, figsize=(6.5 * n_plots, 4.5),
                             sharey=False)
    if n_plots == 1:
        axes = [axes]

    for ax, gname in zip(axes, sorted(group_names)):
        sub: dict[str, list[dict]] = {}
        for r in rows:
            if r["group"] != gname or r["sub_series"] is None:
                continue
            sub.setdefault(r["sub_series"], []).append(r)

        vs_colors = _VS_PALETTE.get(gname, {})

        for series_name in sorted(sub):
            sr = sorted(sub[series_name], key=lambda r: r["n"])
            xs = [r["n"] for r in sr]
            ys = [r["throughput_mels"] for r in sr]

            color  = vs_colors.get(series_name, _FALLBACK_COLOR)
            marker = _VS_MARKERS.get(series_name, _FALLBACK_MARKER)
            ls     = _VS_LS.get(series_name, "-")

            ax.plot(xs, ys, marker=marker, color=color,
                    label=series_name, linestyle=ls, linewidth=2, zorder=3)

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Throughput  (Melem/s)")
        transform = gname.replace("_batch_vs_sequential", "").upper()
        ax.set_title(f"{transform}: batch vs sequential")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(_x_formatter))
        ax.set_ylim(bottom=0)
        ax.grid(True, which="both")
        ax.legend()

    fig.suptitle("Batch vs Sequential Throughput", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()

    path = CHARTS_DIR / "vs_sequential.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return path


# ── Chart generator ───────────────────────────────────────────────────────────

def generate_charts(rows: list[dict]) -> dict[str, Path]:
    """
    Generate all SVG charts in CHARTS_DIR.

    Returns a dict:
      "latency"       latency.svg          latency   vs N, scalar baselines
      "throughput"    throughput.svg        throughput vs N, scalar baselines
      "batch_signal"  batch_signal.svg      throughput vs N, scalar + batch×16
      "batch_size"    batch_size.svg        throughput vs batch size
      "vs_sequential" vs_sequential.svg     batch vs sequential comparison
    """
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    _apply_style()

    present_groups = {r["group"] for r in rows}
    paths: dict[str, Path] = {}

    # ── 1 & 2. Latency + throughput — scalar baselines only ───────────────────
    scalar_names  = sorted(_SIGNAL_LEN_GROUPS & {"fft", "ifft", "roundtrip"}
                           & present_groups)
    scalar_series = _build_series(rows, scalar_names)

    if scalar_series:
        paths["latency"] = _line_chart(
            scalar_series,
            "latency.svg",
            "GPU FFT/IFFT — Latency vs Signal Length",
            "Signal length  N", "Latency  (µs)",
            metric="mean_us",
        )
        paths["throughput"] = _line_chart(
            scalar_series,
            "throughput.svg",
            "GPU FFT/IFFT — Throughput vs Signal Length",
            "Signal length  N", "Throughput  (Melem/s)",
            metric="throughput_mels",
        )

    # ── 3. Throughput vs N — scalar + batch/signal_len side-by-side ──────────
    sig_names  = sorted(_SIGNAL_LEN_GROUPS & present_groups)
    sig_series = _build_series(rows, sig_names)
    batch_sig_present = any(g not in {"fft", "ifft", "roundtrip"}
                            for g in sig_series)
    if batch_sig_present:
        paths["batch_signal"] = _line_chart(
            sig_series,
            "batch_signal.svg",
            "Throughput vs Signal Length — scalar vs batch×16",
            "Signal length  N", "Throughput  (Melem/s)",
            metric="throughput_mels",
            scalar_groups=frozenset({"fft", "ifft", "roundtrip"}),
        )

    # ── 4. Throughput vs batch size ───────────────────────────────────────────
    bs_names  = sorted(_BATCH_SIZE_GROUPS & present_groups)
    bs_series = _build_series(rows, bs_names)
    if bs_series:
        paths["batch_size"] = _line_chart(
            bs_series,
            "batch_size.svg",
            "Batch FFT/IFFT — Throughput vs Batch Size",
            "Batch size", "Throughput  (Melem/s)",
            metric="throughput_mels",
        )

    # ── 5. Batch vs sequential comparison ─────────────────────────────────────
    vs_names = sorted(_VS_GROUPS & present_groups)
    if vs_names:
        paths["vs_sequential"] = _vs_chart(rows, vs_names)

    return paths


# ── Git helpers ───────────────────────────────────────────────────────────────

def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git"] + args, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


# ── Markdown renderer ─────────────────────────────────────────────────────────

# Chart display metadata: key → (section heading, image alt-text)
_CHART_META: dict[str, tuple[str, str]] = {
    "latency":       ("Scalar baselines",                "Latency vs N"),
    "throughput":    ("Scalar baselines",                "Throughput vs N"),
    "batch_signal":  ("Batch vs scalar (batch=16)",      "Throughput vs N"),
    "batch_size":    ("Batch size sweep (N=4 096 fixed)","Throughput vs batch size"),
    "vs_sequential": ("Batch vs sequential",             "Batch vs sequential throughput"),
}

# Ordered sections for the summary table (group names = Criterion dir names).
_TABLE_SECTIONS: list[tuple[str, frozenset[str]]] = [
    ("Scalar",           frozenset({"fft", "ifft", "roundtrip"})),
    ("Batch FFT",        frozenset({"fft_batch_batch_size", "fft_batch_signal_len",
                                    "fft_batch_vs_sequential"})),
    ("Batch IFFT",       frozenset({"ifft_batch_batch_size", "ifft_batch_signal_len",
                                    "ifft_batch_vs_sequential"})),
    ("Batch round-trip", frozenset({"roundtrip_batch", "roundtrip_batch_signal_len"})),
]


def render(rows: list[dict], raw: str, chart_paths: dict[str, Path] | None) -> str:
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
        bench_results = ROOT / "bench-results"
        emitted: set[str] = set()

        def _img(key: str) -> str:
            p   = chart_paths[key]
            rel = p.relative_to(bench_results)
            _, alt = _CHART_META.get(key, ("", key))
            return f"![{alt}]({rel})"

        lines.append("## Charts")
        lines.append("")

        # Scalar pair — side-by-side via Markdown table.
        if "latency" in chart_paths and "throughput" in chart_paths:
            lines += [
                "### Scalar baselines",
                "",
                "| Latency | Throughput |",
                "|---------|------------|",
                f"| {_img('latency')} | {_img('throughput')} |",
                "",
            ]
            emitted |= {"latency", "throughput"}

        # Remaining charts full-width, each with its own sub-heading.
        for key in ("batch_signal", "batch_size", "vs_sequential"):
            if key not in chart_paths:
                continue
            heading, _ = _CHART_META.get(key, (key, key))
            lines += [f"### {heading}", "", _img(key), ""]
            emitted.add(key)

        # Catch-all for any future charts not listed above.
        for key in chart_paths:
            if key in emitted:
                continue
            heading, _ = _CHART_META.get(key, (key, key))
            lines += [f"### {heading}", "", _img(key), ""]

    # ── Summary table ─────────────────────────────────────────────────────────
    lines += [
        "## Summary",
        "",
        "| Benchmark | Param | Mean | 95% CI | Std dev | Throughput |",
        "|-----------|------:|-----:|--------|--------:|------------|",
    ]

    by_group: dict[str, list[dict]] = {}
    for r in rows:
        by_group.setdefault(r["group"], []).append(r)
    for g in by_group:
        # Sort: sub_series first, then n.
        by_group[g].sort(key=lambda r: (r["sub_series"] or "", r["n"]))

    emitted_groups: set[str] = set()
    first_section = True

    def _emit_group(g: str) -> None:
        if g not in by_group:
            return
        for r in by_group[g]:
            mean  = _fmt_time(r["mean"])
            ci    = f"[{_fmt_time(r['lo'])} … {_fmt_time(r['hi'])}]"
            std   = _fmt_time(r["std"])
            param = r["raw_param"]
            if r["sub_series"]:
                param = f"{r['sub_series']} {param}"
            lines.append(
                f"| {r['group']} | {param:>12} | {mean:>10} | {ci}"
                f" | {std:>10} | {r['throughput']} |"
            )
        emitted_groups.add(g)

    for section_name, section_groups in _TABLE_SECTIONS:
        present = [g for g in sorted(section_groups) if g in by_group]
        if not present:
            continue
        if not first_section:
            lines.append("| | | | | | |")
        first_section = False
        for g in present:
            _emit_group(g)

    for g in sorted(by_group):
        if g not in emitted_groups:
            lines.append("| | | | | | |")
            _emit_group(g)

    # ── Raw output ────────────────────────────────────────────────────────────
    result_lines: list[str] = []
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

    chart_paths: dict[str, Path] | None = None
    if HAS_MATPLOTLIB:
        chart_paths = generate_charts(rows)
        for key, path in chart_paths.items():
            print(f"✓  Chart ({key:<14}) → {path}", file=sys.stderr)
    else:
        print(
            "Warning: matplotlib not found — skipping chart generation.",
            file=sys.stderr,
        )

    print(render(rows, raw, chart_paths), end="")
