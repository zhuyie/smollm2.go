//go:build arm64

package model

const matmulSIMDMinN = 64

func dotF32ARM64(a []float32, b []float32) float32
func addScaledF32ARM64(dst []float32, src []float32, scale float32)

func dotF32(a []float32, b []float32) float32 {
	n := min(len(a), len(b))
	if n >= matmulSIMDMinN && n&3 == 0 {
		return dotF32ARM64(a[:n], b[:n])
	}
	return dotF32Scalar(a[:n], b[:n])
}

func matmulF32(out []float32, x []float32, w []float32, n int, d int) {
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

func addScaledF32(dst []float32, src []float32, scale float32) {
	n := min(len(dst), len(src))
	vecN := n &^ 3
	if vecN > 0 {
		addScaledF32ARM64(dst[:vecN], src[:vecN], scale)
	}
	if vecN < n {
		addScaledF32Scalar(dst[vecN:n], src[vecN:n], scale)
	}
}
