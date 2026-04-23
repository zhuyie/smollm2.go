//go:build arm64

package model

const matmulSIMDMinN = 64

func dotF32ARM64(a []float32, b []float32) float32

func matmul(out []float32, x []float32, w []float32, n int, d int) {
	if n >= matmulSIMDMinN && n&3 == 0 {
		out = out[:d]
		x = x[:n]
		w = w[:d*n]
		for i := range out {
			row := w[:n]
			w = w[n:]
			out[i] = dotF32ARM64(x, row)
		}
		return
	}
	out = out[:d]
	x = x[:n]
	w = w[:d*n]
	for i := range out {
		// Four independent accumulators shorten the dependency chain.
		var v0, v1, v2, v3 float32
		// Keep row slicing explicit so the compiler's BCE pass can prove bounds.
		row := w[:n]
		w = w[n:]
		j := 0
		for ; j+3 < n; j += 4 {
			v0 += row[j] * x[j]
			v1 += row[j+1] * x[j+1]
			v2 += row[j+2] * x[j+2]
			v3 += row[j+3] * x[j+3]
		}
		val := v0 + v1 + v2 + v3
		for ; j < n; j++ {
			val += row[j] * x[j]
		}
		out[i] = val
	}
}
