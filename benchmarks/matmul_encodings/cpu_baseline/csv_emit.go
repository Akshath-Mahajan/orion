// csv_emit.go
//
// Append-mode CSV writer used by the Run*HE functions to emit one row per
// shape, alongside their existing pretty-print output. Schema is matched
// to the orion harness CSV (benchmarks/matmul_encodings/runners/_common.py)
// so direct CPU-vs-GPU comparison is a single load_csv call away.
//
// Activation: set the BENCH_CSV_OUT env var to a writable file path.
// Empty / unset means csv emit is a no-op (the menu prints unchanged).
//
// Added by Akshath while wiring D8 (CPU baseline rerun) -- not part of
// Negar's intended source. The header is written once on first call.
//
// Schema: backend, device, kernel, shape, n_he, n_trials,
//         mean_seconds, std_seconds, rotations, ct_ct_muls, ct_pt_muls,
//         peak_hbm_mb, max_abs_err

package main

import (
	"fmt"
	"os"
	"sync"
	"time"
)

var (
	csvFile   *os.File
	csvOnce   sync.Once
	csvHeader = "backend,device,kernel,shape,n_he,n_trials,mean_seconds,std_seconds,rotations,ct_ct_muls,ct_pt_muls,peak_hbm_mb,max_abs_err"
)

func csvOpen() {
	path := os.Getenv("BENCH_CSV_OUT")
	if path == "" {
		return
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		fmt.Fprintf(os.Stderr, "  [csv] failed to open %s: %v\n", path, err)
		return
	}
	csvFile = f
	fmt.Fprintln(csvFile, csvHeader)
}

// csvEmit writes one row to BENCH_CSV_OUT (no-op if unset).
//
//   kernel      one of {bmm1, bmm3, thor, moai_alg3, moai_alg4, rowenc}
//   shape       human-readable shape label, e.g. "(516,540,528)/blk(43,45,44)"
//   nHE         CKKS slot count (typically ctx.NHE)
//   nTrials     number of timed iterations
//   mean, std   wall-clock per call
//   rot, ctCt   counted (or theoretical) op counts
//   ctPt
//   maxErr      negative => skip column (not verified)
func csvEmit(
	kernel, shape string,
	nHE, nTrials int,
	mean, std time.Duration,
	rot, ctCt, ctPt int,
	maxErr float64,
) {
	csvOnce.Do(csvOpen)
	if csvFile == nil {
		return
	}
	errStr := ""
	if maxErr >= 0 {
		errStr = fmt.Sprintf("%.6e", maxErr)
	}
	fmt.Fprintf(csvFile,
		"lattigo,cpu,%s,%q,%d,%d,%.6f,%.6f,%d,%d,%d,,%s\n",
		kernel, shape, nHE, nTrials,
		mean.Seconds(), std.Seconds(),
		rot, ctCt, ctPt,
		errStr,
	)
}
