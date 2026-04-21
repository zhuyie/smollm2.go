package model

import (
	"math"
	"testing"
)

func TestValidateConfig(t *testing.T) {
	valid := Config{
		Dim:       4,
		HiddenDim: 8,
		NLayers:   1,
		NHeads:    2,
		NKVHeads:  1,
		VocabSize: 3,
		SeqLen:    16,
		RopeTheta: 10000,
	}
	if err := validateConfig(valid); err != nil {
		t.Fatalf("validateConfig(valid) returned error: %v", err)
	}
	invalid := valid
	invalid.NHeads = 3
	if err := validateConfig(invalid); err == nil {
		t.Fatal("validateConfig(invalid) returned nil")
	}
}

func TestRMSNorm(t *testing.T) {
	out := make([]float32, 2)
	x := []float32{3, 4}
	weight := []float32{1, 2}
	rmsnorm(out, x, weight)

	scale := float32(1.0 / math.Sqrt(float64((3*3+4*4)/float32(2)+1e-5)))
	want := []float32{3 * scale, 8 * scale}
	for i := range want {
		if math.Abs(float64(out[i]-want[i])) > 1e-6 {
			t.Fatalf("out[%d] = %f, want %f", i, out[i], want[i])
		}
	}
}

func TestSoftmax(t *testing.T) {
	x := []float32{1, 2, 3}
	softmax(x)
	var sum float32
	for _, v := range x {
		sum += v
	}
	if math.Abs(float64(sum-1)) > 1e-6 {
		t.Fatalf("sum = %f, want 1", sum)
	}
	if !(x[0] < x[1] && x[1] < x[2]) {
		t.Fatalf("softmax probabilities not ordered: %v", x)
	}
}

func TestMatmul(t *testing.T) {
	x := []float32{2, 3, 5, 7, 11}
	w := []float32{
		1, 0, 1, 0, 1,
		0, 1, 0, 1, 0,
	}
	out := make([]float32, 2)
	matmul(out, x, w, 5, 2)
	want := []float32{18, 10}
	for i := range want {
		if out[i] != want[i] {
			t.Fatalf("out[%d] = %f, want %f", i, out[i], want[i])
		}
	}
}
