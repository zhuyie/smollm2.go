# Inference Performance Notes

This document summarizes the inference-side optimizations that currently matter
in `smollm2.go`, why they help, and where they live in the codebase.

The project still aims to stay small and readable, so the optimizations here
favor simple hot-path wins over large framework-style abstractions.

## Measurement Setup

The benchmark numbers below were collected with:

```sh
env GOCACHE=/tmp/smollm2-go-cache \
  go test ./internal/model -bench 'Benchmark(Decode|Prefill)' -benchtime=1x -run '^$'
```

Machine used for the measurements in this document:

- Apple M2 Max
- `darwin/arm64`

These are single-sample benchmark numbers, so treat them as directional rather
than perfectly stable.

## Current Effective Optimizations

### 1. Bounds-check-elimination-friendly loops

Code references:

- [`internal/model/matmul_scalar.go`](../internal/model/matmul_scalar.go)
- [`internal/model/model.go`](../internal/model/model.go)

Several hot loops are written in a style that helps the Go compiler eliminate
repeated bounds checks:

- slices are trimmed to their exact working length up front
- row slices are made explicit before inner loops
- tight loops use simple index progression

Example from `matmulScalar`:

```go
out = out[:d]
x = x[:n]
w = w[:d*n]
for i := range out {
    row := w[:n]
    w = w[n:]
    out[i] = dotF32Scalar(row, x)
}
```

This is not as flashy as SIMD or threading, but it matters because these loops
run constantly during both prefill and decode.

### 2. ARM64 SIMD dot product for matrix-vector work

Code references:

- [`internal/model/matmul_arm64.go`](../internal/model/matmul_arm64.go)
- [`internal/model/dot_arm64.s`](../internal/model/dot_arm64.s)
- [`internal/model/dot_arm64.go`](../internal/model/dot_arm64.go)

The dominant math shape in this runtime is matrix-vector multiply:

```text
out[d] = W[d x n] * x[n]
```

On `arm64`, rows with a suitable width use a handwritten NEON/FMLA dot kernel
instead of a scalar Go loop:

- `dotF32ARM64` handles the actual SIMD dot product
- `matmulKernel` applies it row-by-row for projection matrices
- `dotF32` also reuses it for attention score computation

This gives a meaningful speedup without changing the high-level model code.

### 3. Parallel matmul across output rows

Code reference:

- [`internal/model/matmul.go`](../internal/model/matmul.go)

Large matrix-vector multiplies are split across worker goroutines:

- row chunks are distributed through a job channel
- the worker count is capped
- small problems stay single-threaded to avoid overhead

Relevant tuning constants:

```go
const (
    matmulMinParallelOps = 1 << 20
    matmulRowsPerWorker  = 256
    matmulMaxWorkers     = 16
)
```

This is especially important for larger projections such as:

- `WQ`, `WK`, `WV`, `WO`
- `W1`, `W2`, `W3`
- final logits projection

### 4. Precomputed RoPE lookup tables

Code references:

- [`internal/model/model.go`](../internal/model/model.go)

RoPE originally computed `pow/cos/sin` inside `Forward()` for every token and
every layer. That work is now moved to load time:

- `buildRopeTables()` precomputes `cos` and `sin`
- `Transformer.Tables` stores the lookup tables
- `Forward()` just indexes the table for the current `pos`

This removes repeated transcendental math from a hot decode path.

### 5. Faster attention score computation

Code references:

- [`internal/model/model.go`](../internal/model/model.go)
- [`internal/model/dot_arm64.go`](../internal/model/dot_arm64.go)
- [`internal/model/dot_generic.go`](../internal/model/dot_generic.go)

The attention score loop used to do a scalar `q·k` dot product directly inside
the nested head/time loop. It now:

- reuses `dotF32`
- uses the ARM64 SIMD kernel where available
- precomputes `attScale = 1/sqrt(headSize)` once per forward call
- hoists `kvHead` / `headOff` out of the innermost loop

This is a good example of a focused hot-loop optimization that stays readable
while cutting repeated work.

## Observed Benchmark Direction

Using the benchmark command above on the M2 Max, the model runtime moved from
the original benchmark baseline at commit `18a36ef` (before this batch of
inference optimizations) to the current optimized runtime on `main`:

| Benchmark | Original Baseline | Current | Improvement |
| --- | ---: | ---: | ---: |
| `Prefill/128` | `7.818 tok/s` | `32.33 tok/s` | `+313.5%` |
| `Prefill/512` | `7.233 tok/s` | `27.56 tok/s` | `+281.0%` |
| `Decode/128` | `7.503 tok/s` | `30.58 tok/s` | `+307.6%` |
| `Decode/512` | `6.685 tok/s` | `22.92 tok/s` | `+242.9%` |

The exact percentages will vary run to run, but the gains are real enough to
show up consistently in both prefill and decode, especially at longer context
lengths.

## What Has Not Proven Worthwhile Yet

Not every plausible optimization wins in this codebase.

One example: directly calling Apple Accelerate `sgemv` for the current single
token decode workload turned out slower than the existing ARM64 SIMD path. The
likely reason is that this runtime is dominated by many matrix-vector operations
rather than large matrix-matrix workloads where BLAS usually shines more.
