#!/usr/bin/env bash
#
# Run benchmarks and save results to bench-results/.
#
# Usage:
#   ./scripts/bench.sh                           # run all benchmarks
#   ./scripts/bench.sh -- fft/1024               # run a single benchmark
#   ./scripts/bench.sh -- --save-baseline before # save a Criterion baseline
#   ./scripts/bench.sh -- --baseline before      # compare against a baseline
#
# Extra arguments after -- are forwarded exclusively to the fft_bench binary,
# so Criterion flags never reach the lib unit-test runner.
#
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$ROOT/bench-results"
ARCHIVE_DIR="$RESULTS_DIR/archive"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LATEST="$RESULTS_DIR/latest.md"
TMP="$(mktemp)"

mkdir -p "$ARCHIVE_DIR"
trap 'rm -f "$TMP"' EXIT

cd "$ROOT"

echo "▶  Running benchmarks…"
echo ""

# Show output on the terminal in real-time *and* capture it for the report.
cargo bench --features wgpu --bench fft_bench "$@" 2>&1 | tee "$TMP"

echo ""
echo "▶  Generating report…"

python3 scripts/export_bench.py < "$TMP" | tee "$LATEST" > "$ARCHIVE_DIR/$TIMESTAMP.md"

echo ""
echo "✓  Latest  → $LATEST"
echo "✓  Archive → $ARCHIVE_DIR/$TIMESTAMP.md"
echo "✓  Charts  → $(dirname "$LATEST")/charts/"
