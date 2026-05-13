// memsnap.go
//
// Local stub for the diagnostic memory-snapshot helpers referenced
// throughout the *_cipher.go and *_runner.go files. The upstream repo
// references TakeMemSnap / PrintMemDelta but never ships their
// definitions, so a fresh `go build .` in matmult/ fails. This file
// adds minimal implementations so the binary builds and runs.
//
// Added by Akshath while wiring D8 (CPU baseline rerun) -- not part of
// Negar's intended source. Print output goes to stderr so it doesn't
// pollute the timing lines on stdout that the harness wrapper parses.

package main

import (
	"fmt"
	"os"
	"runtime"
)

// MemSnap is a lightweight wrapper around runtime.MemStats.HeapAlloc
// (the metric that's most useful for "how much did this kernel grow
// the live Go heap").
type MemSnap struct {
	HeapAlloc uint64
}

// TakeMemSnap forces a GC and returns a snapshot of HeapAlloc.
func TakeMemSnap() MemSnap {
	runtime.GC()
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return MemSnap{HeapAlloc: m.HeapAlloc}
}

// PrintMemDelta prints the (after - before) HeapAlloc delta in MB to
// stderr, prefixed with `tag` for grep-ability.
func PrintMemDelta(tag string, before, after MemSnap) {
	delta := int64(after.HeapAlloc) - int64(before.HeapAlloc)
	fmt.Fprintf(os.Stderr, "  [mem] %-40s %+8.2f MB\n",
		tag, float64(delta)/(1<<20))
}
