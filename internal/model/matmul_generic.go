//go:build !arm64

package model

func matmulKernel(out []float32, x []float32, w []float32, n int, d int) {
	matmulScalar(out, x, w, n, d)
}
