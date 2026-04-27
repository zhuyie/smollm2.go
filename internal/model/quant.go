package model

import "math"

type QuantizedMatrix struct {
	Data   []int8
	Scale  []float32
	Inputs int
	Rows   int
}

// QuantizeInt8 converts the dense projection weights to per-row symmetric int8.
// Token embeddings and normalization weights remain float32 because they are
// read directly rather than consumed through matrix multiplication.
func (t *Transformer) QuantizeInt8() {
	cfg := t.Config
	kvDim := cfg.Dim * cfg.NKVHeads / cfg.NHeads
	for i := range t.Weights.Layers {
		lw := &t.Weights.Layers[i]
		lw.QWQ = quantizeMatrixInt8(lw.WQ, cfg.Dim, cfg.Dim)
		lw.WQ = nil
		lw.QWK = quantizeMatrixInt8(lw.WK, cfg.Dim, kvDim)
		lw.WK = nil
		lw.QWV = quantizeMatrixInt8(lw.WV, cfg.Dim, kvDim)
		lw.WV = nil
		lw.QWO = quantizeMatrixInt8(lw.WO, cfg.Dim, cfg.Dim)
		lw.WO = nil
		lw.QW1 = quantizeMatrixInt8(lw.W1, cfg.Dim, cfg.HiddenDim)
		lw.W1 = nil
		lw.QW2 = quantizeMatrixInt8(lw.W2, cfg.HiddenDim, cfg.Dim)
		lw.W2 = nil
		lw.QW3 = quantizeMatrixInt8(lw.W3, cfg.Dim, cfg.HiddenDim)
		lw.W3 = nil
	}
	t.Weights.QWCls = quantizeMatrixInt8(t.Weights.WCls, cfg.Dim, cfg.VocabSize)
	if !t.Weights.SharedWeights {
		t.Weights.WCls = nil
	}
}

func quantizeMatrixInt8(w []float32, n int, d int) *QuantizedMatrix {
	q := &QuantizedMatrix{
		Data:   make([]int8, n*d),
		Scale:  make([]float32, d),
		Inputs: n,
		Rows:   d,
	}
	for row := 0; row < d; row++ {
		src := w[row*n : (row+1)*n]
		var maxAbs float32
		for _, v := range src {
			abs := float32(math.Abs(float64(v)))
			if abs > maxAbs {
				maxAbs = abs
			}
		}
		scale := float32(1)
		if maxAbs > 0 {
			scale = maxAbs / 127
		}
		q.Scale[row] = scale
		dst := q.Data[row*n : (row+1)*n]
		invScale := float32(1) / scale
		for i, v := range src {
			quantized := int(math.Round(float64(v * invScale)))
			if quantized > 127 {
				quantized = 127
			} else if quantized < -127 {
				quantized = -127
			}
			dst[i] = int8(quantized)
		}
	}
	return q
}

func matmulWeight(out []float32, x []float32, w []float32, q *QuantizedMatrix, n int, d int) {
	if q != nil {
		matmulInt8(out, x, q, n, d)
		return
	}
	matmul(out, x, w, n, d)
}

func matmulBatchWeight(out []float32, x []float32, w []float32, q *QuantizedMatrix, batch int, n int, d int) {
	if q != nil {
		matmulBatchInt8(out, x, q, batch, n, d)
		return
	}
	matmulBatch(out, x, w, batch, n, d)
}

func matmulInt8(out []float32, x []float32, q *QuantizedMatrix, n int, d int) {
	out = out[:d]
	x = x[:n]
	data := q.Data[:d*n]
	for row := range out {
		weights := data[row*n : (row+1)*n]
		var v0, v1, v2, v3 float32
		i := 0
		for ; i+3 < n; i += 4 {
			v0 += x[i] * float32(weights[i])
			v1 += x[i+1] * float32(weights[i+1])
			v2 += x[i+2] * float32(weights[i+2])
			v3 += x[i+3] * float32(weights[i+3])
		}
		val := v0 + v1 + v2 + v3
		for ; i < n; i++ {
			val += x[i] * float32(weights[i])
		}
		out[row] = val * q.Scale[row]
	}
}

func matmulBatchInt8(out []float32, x []float32, q *QuantizedMatrix, batch int, n int, d int) {
	if batch == 1 {
		matmulInt8(out[:d], x[:n], q, n, d)
		return
	}
	out = out[:batch*d]
	x = x[:batch*n]
	for row := 0; row < d; row++ {
		weights := q.Data[row*n : (row+1)*n]
		scale := q.Scale[row]
		for b := 0; b < batch; b++ {
			out[b*d+row] = dotInt8(x[b*n:(b+1)*n], weights) * scale
		}
	}
}

func dotInt8(x []float32, weights []int8) float32 {
	var v0, v1, v2, v3 float32
	i := 0
	n := len(x)
	for ; i+3 < n; i += 4 {
		v0 += x[i] * float32(weights[i])
		v1 += x[i+1] * float32(weights[i+1])
		v2 += x[i+2] * float32(weights[i+2])
		v3 += x[i+3] * float32(weights[i+3])
	}
	val := v0 + v1 + v2 + v3
	for ; i < n; i++ {
		val += x[i] * float32(weights[i])
	}
	return val
}
