#!/usr/bin/env bash
# D8 CPU baseline runner.
#
# Re-builds Negar's Go binary with our csv_emit.go + memsnap.go shims,
# then drives the menu via stdin to run all five HE suites in turn.
# Output lands in benchmarks/matmul_encodings/results/cpu_baseline_paper.csv,
# in the same schema as our Python harness CSV (see runners/_common.py).
#
# Prerequisites:
#   * Go >= 1.22 on PATH (`go version`).
#   * matmul-encoding-material/ checked out at $ORION_ROOT/matmul-encoding-material/
#     (the gitignored sibling clone).
#
# Idempotent: re-running overwrites the output CSV. Pass MATMULT_TRIALS to
# bump the n_trials per shape (default 1; bigger = better stdev, slower).

set -euo pipefail

ORION_ROOT="${ORION_ROOT:-$HOME/orion}"
MAT_REPO="$ORION_ROOT/matmul-encoding-material/MatMult/matmult"
PATCH_DIR="$ORION_ROOT/benchmarks/matmul_encodings/cpu_baseline"
OUT_DIR="$ORION_ROOT/benchmarks/matmul_encodings/results"
TRIALS="${MATMULT_TRIALS:-1}"

[[ -d "$MAT_REPO" ]] || {
  echo "FATAL: $MAT_REPO not found. Clone matmul-encoding-material/ first." >&2
  exit 1
}

# Drop the csv_emit + memsnap shims into the gitignored clone if missing.
# (The runners/*_runner.go diffs are in matmult_csv_emit.patch -- apply
# manually if you need them.)
for f in csv_emit.go memsnap.go; do
  if ! diff -q "$PATCH_DIR/$f" "$MAT_REPO/$f" > /dev/null 2>&1; then
    echo "  installing $f -> $MAT_REPO/"
    cp "$PATCH_DIR/$f" "$MAT_REPO/$f"
  fi
done

# Apply the runner patch if there are no csvEmit calls already.
if ! grep -q "csvEmit" "$MAT_REPO/rowenc_runner.go"; then
  echo "  applying $PATCH_DIR/matmult_csv_emit.patch"
  ( cd "$MAT_REPO/.." && git apply --reject "$PATCH_DIR/matmult_csv_emit.patch" )
fi

mkdir -p "$OUT_DIR"
OUT_CSV="$OUT_DIR/cpu_baseline_paper.csv"
rm -f "$OUT_CSV"

echo "  building binary..."
( cd "$MAT_REPO" && go build -o /tmp/matmult_runner . )

echo "  running suites: rowenc, bmm3, thor, moai, bmm1 (-trials $TRIALS)"
echo "  CSV -> $OUT_CSV"
# Menu options: 10=Row HE, 8=BMM-III HE, 2=THOR HE, 4=MOAI HE, 6=BMM-I HE.
echo -e "10\n8\n2\n4\n6\n0" \
  | BENCH_CSV_OUT="$OUT_CSV" /tmp/matmult_runner -trials "$TRIALS" -verify

echo
echo "  rows in CSV: $(wc -l < "$OUT_CSV")"
