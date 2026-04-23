package model

import (
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"
	"time"
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

var benchmarkLogits []float32

func loadBenchmarkTransformer(b *testing.B) *Transformer {
	b.Helper()
	path := filepath.Join("..", "..", "models", "smollm2-360m-instruct-f32.bin")
	if _, err := os.Stat(path); err != nil {
		b.Skipf("model checkpoint not found: %s", path)
	}
	t, err := Load(path)
	if err != nil {
		b.Fatal(err)
	}
	return t
}

func benchmarkTokens(vocabSize int, count int) []int {
	tokens := make([]int, count)
	for i := range tokens {
		tokens[i] = (i*131 + 17) % vocabSize
	}
	return tokens
}

// BenchmarkForwardPositionSweep measures one-token Forward calls while cycling
// through all cache positions. It is useful as a broad Forward regression test,
// but it is not shaped like a real prefill or decode workload.
func BenchmarkForwardPositionSweep(b *testing.B) {
	t := loadBenchmarkTransformer(b)
	token := 0
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchmarkLogits = t.Forward(token, i%t.Config.SeqLen)
	}
}

// BenchmarkPrefill measures prompt ingestion from position 0. This matches the
// model layer's current prefill behavior, where prompt tokens are processed one
// at a time while filling the KV cache.
func BenchmarkPrefill(b *testing.B) {
	for _, promptLen := range []int{128, 512} {
		b.Run(strconv.Itoa(promptLen), func(b *testing.B) {
			t := loadBenchmarkTransformer(b)
			if promptLen > t.Config.SeqLen {
				b.Skipf("prompt length %d exceeds sequence length %d", promptLen, t.Config.SeqLen)
			}
			tokens := benchmarkTokens(t.Config.VocabSize, promptLen)

			b.ReportAllocs()
			b.ResetTimer()
			start := time.Now()
			for i := 0; i < b.N; i++ {
				for pos, token := range tokens {
					benchmarkLogits = t.Forward(token, pos)
				}
			}
			elapsed := time.Since(start)
			b.StopTimer()
			b.ReportMetric(float64(b.N*promptLen)/elapsed.Seconds(), "tok/s")
		})
	}
}

// BenchmarkDecode measures the cost of generating one token after an existing
// context has already populated the KV cache. Setup prefill is intentionally
// outside the timed region.
func BenchmarkDecode(b *testing.B) {
	for _, contextLen := range []int{128, 512} {
		b.Run(strconv.Itoa(contextLen), func(b *testing.B) {
			t := loadBenchmarkTransformer(b)
			if contextLen >= t.Config.SeqLen {
				b.Skipf("context length %d leaves no decode position in sequence length %d", contextLen, t.Config.SeqLen)
			}
			tokens := benchmarkTokens(t.Config.VocabSize, contextLen+1)
			for pos := 0; pos < contextLen; pos++ {
				benchmarkLogits = t.Forward(tokens[pos], pos)
			}
			decodeToken := tokens[contextLen]

			b.ReportAllocs()
			b.ResetTimer()
			start := time.Now()
			for i := 0; i < b.N; i++ {
				benchmarkLogits = t.Forward(decodeToken, contextLen)
			}
			elapsed := time.Since(start)
			b.StopTimer()
			b.ReportMetric(float64(b.N)/elapsed.Seconds(), "tok/s")
		})
	}
}
