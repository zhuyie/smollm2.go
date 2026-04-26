//go:build !arm64

package model

func dotF32(a []float32, b []float32) float32 {
	n := min(len(a), len(b))
	return dotF32Scalar(a[:n], b[:n])
}

func matmulF32(out []float32, x []float32, w []float32, n int, d int) {
	matmulScalar(out, x, w, n, d)
}

func addScaledF32(dst []float32, src []float32, scale float32) {
	addScaledF32Scalar(dst, src, scale)
}
