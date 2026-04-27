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

The model benchmarks include both `f32` and `int8` subcases. The int8 benchmark
case converts the loaded float32 test checkpoint in memory so the kernel paths
can be compared in one command. In normal CLI usage, int8 inference is selected
by loading a version 2 checkpoint with `weight_type == 1`; there is no runtime
quantization flag.

Machine used for the measurements in this document:

- Apple M2 Max
- `darwin/arm64`

These are single-sample benchmark numbers, so treat them as directional rather
than perfectly stable.

## Current Effective Optimizations

### 1. Bounds-check-elimination-friendly loops

Code references:

- [`internal/model/kernel_scalar.go`](../internal/model/kernel_scalar.go)
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

- [`internal/model/kernel_arm64.s`](../internal/model/kernel_arm64.s)
- [`internal/model/kernel_arm64.go`](../internal/model/kernel_arm64.go)

The dominant math shape in this runtime is matrix-vector multiply:

```text
out[d] = W[d x n] * x[n]
```

On `arm64`, rows with a suitable width use a handwritten NEON/FMLA dot kernel
instead of a scalar Go loop:

- `dotF32ARM64` handles the actual SIMD dot product
- `matmulF32` applies it row-by-row for projection matrices
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

### 5. Batched prompt prefill

Code references:

- [`internal/model/model.go`](../internal/model/model.go)
- [`internal/model/matmul.go`](../internal/model/matmul.go)
- [`internal/model/kernel_arm64.s`](../internal/model/kernel_arm64.s)
- [`cmd/smollm2/main.go`](../cmd/smollm2/main.go)

Prompt ingestion now uses `Transformer.Prefill` instead of repeatedly calling
`Forward` for every prompt token. The prefill path keeps all prompt-token
hidden states in batch-shaped scratch buffers and applies the projection
matrices with `matmulBatch`.

The important shape is:

```text
out[batch x d] = x[batch x n] * W[d x n]^T
```

For ARM64, `matmulBatchRows` uses `dotF32Batch4ARM64` to compute four prompt
rows against the same weight row while loading that weight row once. The kernel
uses two accumulator groups to reduce dependent FMLA chains.

Prefill also only computes final logits for the last prompt token. Intermediate
prompt tokens still fill the KV cache and feed later layers, but they do not
pay for the final vocabulary projection.

Measured on Apple M2 Max during the batched-prefill optimization pass:

| Benchmark | Forward loop | Batched prefill | Improvement |
| --- | ---: | ---: | ---: |
| `Prefill/128` | `32.74 tok/s` | `124.3 tok/s` | `+279.7%` |
| `Prefill/512` | `28.59 tok/s` | `84.38 tok/s` | `+195.1%` |

This optimization improves time to first generated token for longer prompts. It
does not improve steady-state decode throughput, because decode still processes
one token at a time and therefore uses the regular single-token `Forward` path.

### 6. Faster attention score computation

Code references:

- [`internal/model/model.go`](../internal/model/model.go)
- [`internal/model/kernel_arm64.go`](../internal/model/kernel_arm64.go)
- [`internal/model/kernel_generic.go`](../internal/model/kernel_generic.go)

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
inference optimizations) to the optimized float32 runtime. Prefill numbers
include the batched prefill path; decode numbers are from the earlier
single-token optimization pass and are shown for context.

| Benchmark | Original Baseline | Current | Improvement |
| --- | ---: | ---: | ---: |
| `Prefill/128` | `7.818 tok/s` | `124.3 tok/s` | `+1489.9%` |
| `Prefill/512` | `7.233 tok/s` | `84.38 tok/s` | `+1066.7%` |
| `Decode/128` | `7.503 tok/s` | `30.58 tok/s` | `+307.6%` |
| `Decode/512` | `6.685 tok/s` | `22.92 tok/s` | `+242.9%` |

The exact percentages will vary run to run, but the gains are real enough to
show up consistently in both prefill and decode, especially at longer context
lengths.

## Weight-only int8 checkpoints and kernels

Code references:

- [`internal/model/quant.go`](../internal/model/quant.go)
- [`internal/model/model.go`](../internal/model/model.go)
- [`internal/model/kernel_arm64.go`](../internal/model/kernel_arm64.go)
- [`internal/model/kernel_arm64.s`](../internal/model/kernel_arm64.s)
- [`tools/quantize.py`](../tools/quantize.py)

The quantized path is intentionally narrow: projection matrices are stored as
per-row symmetric int8, while token embeddings, activations, KV cache, and
normalization weights remain float32. This keeps the model code close to the
float32 path while reducing memory bandwidth for the large matrix-vector
projection work.

SML2 version 2 checkpoints carry an explicit `weight_type` field:

- `0`: float32 weights
- `1`: int8 projection weights

The loader reads that field and fills either the float32 matrix fields or their
quantized counterparts. Runtime matmul dispatch then uses `matmulWeight` /
`matmulBatchWeight`, so higher-level `Forward` and `Prefill` code does not need
separate branches for every projection.

The decode path multiplies float32 activations by int8 weights and applies the
per-row scale after the dot product:

```text
out[row] = dot_f32_int8(x, qweight[row]) * scale[row]
```

On ARM64, `dotF32Int8ARM64` loads the float32 activation vector and signed int8
weights, widens the int8 lanes, converts them to float32, and accumulates with
NEON FMLA. The generic path keeps the same behavior in scalar Go.

The prefill path reuses the same quantized storage but dequantizes one weight
row into reusable scratch space before applying the existing batch-4 float32 dot
kernel. This measured better than keeping a dynamic int8 activation path in this
codebase, and avoids repeated steady-state allocation by pooling the scratch
buffer.

Measured on Apple M2 Max with the benchmark command above:

| Benchmark | Float32 | Int8 | Improvement |
| --- | ---: | ---: | ---: |
| `Decode/128` | `31.2 tok/s` | `56.7 tok/s` | `+81.7%` |
| `Decode/512` | `24.6 tok/s` | `36.4 tok/s` | `+48.0%` |
| `Prefill/128` | `114.5 tok/s` | `164.8 tok/s` | `+43.9%` |
| `Prefill/512` | `85.5 tok/s` | `109.9 tok/s` | `+28.5%` |

The main reason this helps is memory traffic: the large projection weights are
read constantly, and int8 storage cuts those reads by roughly 4x before scale
loads and other overheads are considered.

## What Has Not Proven Worthwhile Yet

Not every plausible optimization wins in this codebase.

One example: directly calling Apple Accelerate `sgemv` for the current single
token decode workload turned out slower than the existing ARM64 SIMD path. The
likely reason is that this runtime is dominated by many matrix-vector operations
rather than large matrix-matrix workloads where BLAS usually shines more.

Another example: dynamically quantizing activations to int8 for an
`int8 x int8 -> int32` dot path did not earn its complexity here. The extra
quantization work and scratch management outweighed the gains, so the retained
path keeps activations float32 and stores only the weights as int8.
