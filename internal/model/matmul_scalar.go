package model

func dotF32Scalar(a []float32, b []float32) float32 {
	var v0, v1, v2, v3 float32
	j := 0
	n := len(a)
	for ; j+3 < n; j += 4 {
		v0 += a[j] * b[j]
		v1 += a[j+1] * b[j+1]
		v2 += a[j+2] * b[j+2]
		v3 += a[j+3] * b[j+3]
	}
	val := v0 + v1 + v2 + v3
	for ; j < n; j++ {
		val += a[j] * b[j]
	}
	return val
}

func matmulScalar(out []float32, x []float32, w []float32, n int, d int) {
	out = out[:d]
	x = x[:n]
	w = w[:d*n]
	for i := range out {
		// Keep row slicing explicit so the compiler's BCE pass can prove bounds.
		row := w[:n]
		w = w[n:]
		out[i] = dotF32Scalar(row, x)
	}
}
