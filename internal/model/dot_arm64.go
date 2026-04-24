//go:build arm64

package model

func dotF32(a []float32, b []float32) float32 {
	n := min(len(a), len(b))
	if n >= matmulSIMDMinN && n&3 == 0 {
		return dotF32ARM64(a[:n], b[:n])
	}
	return dotF32Scalar(a[:n], b[:n])
}
