//go:build arm64

package model

const matmulSIMDMinN = 64

func dotF32ARM64(a []float32, b []float32) float32

func matmulKernel(out []float32, x []float32, w []float32, n int, d int) {
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
	matmulScalar(out, x, w, n, d)
}
