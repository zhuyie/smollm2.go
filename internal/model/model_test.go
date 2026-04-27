package model

import (
	"math"
	"os"
	"path/filepath"
	"strconv"
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

func TestAddScaledF32(t *testing.T) {
	for _, n := range []int{1, 4, 7, 8, 9, 16, 17} {
		dst := make([]float32, n)
		src := make([]float32, n)
		want := make([]float32, n)
		for i := range dst {
			dst[i] = float32(i + 1)
			src[i] = float32(i*2 + 3)
			want[i] = dst[i] + 0.5*src[i]
		}
		addScaledF32(dst, src, 0.5)
		for i := range want {
			if dst[i] != want[i] {
				t.Fatalf("n=%d dst[%d] = %f, want %f", n, i, dst[i], want[i])
			}
		}
	}
}

func TestDotF32Batch4(t *testing.T) {
	for _, n := range []int{4, 8, 64, 65} {
		x0 := make([]float32, n)
		x1 := make([]float32, n)
		x2 := make([]float32, n)
		x3 := make([]float32, n)
		w := make([]float32, n)
		for i := range w {
			x0[i] = float32((i%7)-3) / 7
			x1[i] = float32((i%11)-5) / 11
			x2[i] = float32((i%13)-6) / 13
			x3[i] = float32((i%17)-8) / 17
			w[i] = float32((i%19)-9) / 19
		}

		got0, got1, got2, got3 := dotF32Batch4(x0, x1, x2, x3, w)
		want := []float32{
			dotF32Scalar(x0, w),
			dotF32Scalar(x1, w),
			dotF32Scalar(x2, w),
			dotF32Scalar(x3, w),
		}
		got := []float32{got0, got1, got2, got3}
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 1e-4 {
				t.Fatalf("n=%d got[%d] = %f, want %f", n, i, got[i], want[i])
			}
		}
	}
}

func TestDotF32Int8(t *testing.T) {
	for _, n := range []int{7, 16, 64, 80} {
		x := make([]float32, n)
		w := make([]int8, n)
		for i := range w {
			x[i] = float32((i%13)-6) / 13
			w[i] = int8((i % 17) - 8)
		}
		got := dotF32Int8(x, w)
		want := dotF32Int8Scalar(x, w)
		if math.Abs(float64(got-want)) > 1e-4 {
			t.Fatalf("n=%d got %f, want %f", n, got, want)
		}
	}
}

func TestBuildRopeTables(t *testing.T) {
	seqLen := 4
	headSize := 8
	ropeTheta := float32(10000)
	cosTable, sinTable := buildRopeTables(seqLen, headSize, ropeTheta)
	headPairs := headSize / 2
	for pos := 0; pos < seqLen; pos++ {
		for pair := 0; pair < headPairs; pair++ {
			headDim := pair * 2
			freq := float32(1.0 / math.Pow(float64(ropeTheta), float64(headDim)/float64(headSize)))
			val := float32(pos) * freq
			idx := pos*headPairs + pair
			if got, want := cosTable[idx], float32(math.Cos(float64(val))); math.Abs(float64(got-want)) > 1e-6 {
				t.Fatalf("cosTable[%d] = %f, want %f", idx, got, want)
			}
			if got, want := sinTable[idx], float32(math.Sin(float64(val))); math.Abs(float64(got-want)) > 1e-6 {
				t.Fatalf("sinTable[%d] = %f, want %f", idx, got, want)
			}
		}
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

func TestQuantizedMatmulApproximatesFloat(t *testing.T) {
	n := 9
	d := 4
	x := fillTestWeights(n)
	w := fillTestWeights(n * d)
	q := quantizeMatrixInt8(w, n, d)
	got := make([]float32, d)
	want := make([]float32, d)
	matmulInt8(got, x, q, n, d)
	matmul(want, x, w, n, d)
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > 0.02 {
			t.Fatalf("out[%d] = %f, want near %f", i, got[i], want[i])
		}
	}
}

func TestQuantizeInt8ForwardApproximatesFloat(t *testing.T) {
	cfg := Config{
		Dim:       8,
		HiddenDim: 16,
		NLayers:   2,
		NHeads:    2,
		NKVHeads:  1,
		VocabSize: 17,
		SeqLen:    16,
		RopeTheta: 10000,
	}
	floatModel := newTestTransformer(cfg)
	quantModel := newTestTransformer(cfg)
	quantModel.QuantizeInt8()

	floatLogits := floatModel.Forward(3, 0)
	quantLogits := quantModel.Forward(3, 0)
	for i := range floatLogits {
		if math.Abs(float64(quantLogits[i]-floatLogits[i])) > 0.05 {
			t.Fatalf("logit[%d] = %f, want near %f", i, quantLogits[i], floatLogits[i])
		}
	}
}

func TestQuantizeInt8PrefillApproximatesFloat(t *testing.T) {
	cfg := Config{
		Dim:       8,
		HiddenDim: 16,
		NLayers:   2,
		NHeads:    2,
		NKVHeads:  1,
		VocabSize: 17,
		SeqLen:    16,
		RopeTheta: 10000,
	}
	floatModel := newTestTransformer(cfg)
	quantModel := newTestTransformer(cfg)
	quantModel.QuantizeInt8()

	tokens := []int{1, 5, 9, 3, 7}
	floatLogits := floatModel.Prefill(tokens, 0)
	quantLogits := quantModel.Prefill(tokens, 0)
	for i := range floatLogits {
		if math.Abs(float64(quantLogits[i]-floatLogits[i])) > 0.05 {
			t.Fatalf("logit[%d] = %f, want near %f", i, quantLogits[i], floatLogits[i])
		}
	}
}

func TestPrefillMatchesForwardLoop(t *testing.T) {
	cfg := Config{
		Dim:       8,
		HiddenDim: 16,
		NLayers:   2,
		NHeads:    2,
		NKVHeads:  1,
		VocabSize: 17,
		SeqLen:    16,
		RopeTheta: 10000,
	}
	t1 := newTestTransformer(cfg)
	t2 := newTestTransformer(cfg)
	tokens := []int{1, 5, 9, 3, 7}

	var loopLogits []float32
	for pos, token := range tokens {
		loopLogits = t1.Forward(token, pos)
	}
	prefillLogits := t2.Prefill(tokens, 0)

	for i := range loopLogits {
		if math.Abs(float64(loopLogits[i]-prefillLogits[i])) > 1e-4 {
			t.Fatalf("logit[%d] = %f, want %f", i, prefillLogits[i], loopLogits[i])
		}
	}
}

func newTestTransformer(cfg Config) *Transformer {
	kvDim := cfg.Dim * cfg.NKVHeads / cfg.NHeads
	weights := Weights{
		TokenEmbeddingTable: fillTestWeights(cfg.VocabSize * cfg.Dim),
		Layers:              make([]LayerWeights, cfg.NLayers),
		RMSFinalWeight:      fillTestWeights(cfg.Dim),
	}
	for i := range weights.Layers {
		lw := &weights.Layers[i]
		lw.RMSAttWeight = fillTestWeights(cfg.Dim)
		lw.RMSFFNWeight = fillTestWeights(cfg.Dim)
		lw.WQ = fillTestWeights(cfg.Dim * cfg.Dim)
		lw.WK = fillTestWeights(cfg.Dim * kvDim)
		lw.WV = fillTestWeights(cfg.Dim * kvDim)
		lw.WO = fillTestWeights(cfg.Dim * cfg.Dim)
		lw.W1 = fillTestWeights(cfg.Dim * cfg.HiddenDim)
		lw.W2 = fillTestWeights(cfg.HiddenDim * cfg.Dim)
		lw.W3 = fillTestWeights(cfg.Dim * cfg.HiddenDim)
	}
	weights.WCls = weights.TokenEmbeddingTable

	headSize := cfg.Dim / cfg.NHeads
	ropeCos, ropeSin := buildRopeTables(cfg.SeqLen, headSize, cfg.RopeTheta)
	return &Transformer{
		Config:  cfg,
		Weights: weights,
		State: State{
			X:          make([]float32, cfg.Dim),
			XB:         make([]float32, cfg.Dim),
			XB2:        make([]float32, cfg.Dim),
			HB:         make([]float32, cfg.HiddenDim),
			HB2:        make([]float32, cfg.HiddenDim),
			Q:          make([]float32, cfg.Dim),
			K:          make([]float32, kvDim),
			V:          make([]float32, kvDim),
			Att:        make([]float32, cfg.NHeads*cfg.SeqLen),
			Logits:     make([]float32, cfg.VocabSize),
			KeyCache:   make([]float32, cfg.NLayers*cfg.SeqLen*kvDim),
			ValueCache: make([]float32, cfg.NLayers*cfg.SeqLen*kvDim),
		},
		Tables: Tables{RopeCos: ropeCos, RopeSin: ropeSin},
	}
}

func fillTestWeights(n int) []float32 {
	values := make([]float32, n)
	for i := range values {
		values[i] = float32((i%17)-8) / 100
	}
	return values
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

func loadBenchmarkTransformerMode(b *testing.B, quantize bool) *Transformer {
	b.Helper()
	t := loadBenchmarkTransformer(b)
	if quantize {
		t.QuantizeInt8()
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

// BenchmarkPrefill measures batched prompt ingestion from position 0.
func BenchmarkPrefill(b *testing.B) {
	for _, promptLen := range []int{128, 512} {
		for _, mode := range []struct {
			name     string
			quantize bool
		}{
			{name: "f32"},
			{name: "int8", quantize: true},
		} {
			b.Run(strconv.Itoa(promptLen)+"/"+mode.name, func(b *testing.B) {
				t := loadBenchmarkTransformerMode(b, mode.quantize)
				if promptLen > t.Config.SeqLen {
					b.Skipf("prompt length %d exceeds sequence length %d", promptLen, t.Config.SeqLen)
				}
				tokens := benchmarkTokens(t.Config.VocabSize, promptLen)
				benchmarkLogits = t.Prefill(tokens, 0)

				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					benchmarkLogits = t.Prefill(tokens, 0)
				}
				b.ReportMetric(float64(b.N*promptLen)*1e9/float64(b.Elapsed().Nanoseconds()), "tok/s")
			})
		}
	}
}

// BenchmarkDecode measures the cost of generating one token after an existing
// context has already populated the KV cache. Setup prefill is intentionally
// outside the timed region.
func BenchmarkDecode(b *testing.B) {
	for _, contextLen := range []int{128, 512} {
		for _, mode := range []struct {
			name     string
			quantize bool
		}{
			{name: "f32"},
			{name: "int8", quantize: true},
		} {
			b.Run(strconv.Itoa(contextLen)+"/"+mode.name, func(b *testing.B) {
				t := loadBenchmarkTransformerMode(b, mode.quantize)
				if contextLen >= t.Config.SeqLen {
					b.Skipf("context length %d leaves no decode position in sequence length %d", contextLen, t.Config.SeqLen)
				}
				tokens := benchmarkTokens(t.Config.VocabSize, contextLen+1)
				benchmarkLogits = t.Prefill(tokens[:contextLen], 0)
				decodeToken := tokens[contextLen]

				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					benchmarkLogits = t.Forward(decodeToken, contextLen)
				}
				b.ReportMetric(float64(b.N)*1e9/float64(b.Elapsed().Nanoseconds()), "tok/s")
			})
		}
	}
}
