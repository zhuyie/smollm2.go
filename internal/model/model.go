package model

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
)

const (
	checkpointMagic      uint32 = 0x324c4d53 // SML2
	checkpointVersion           = int32(1)
	checkpointHeaderSize        = int64(256)
)

type Config struct {
	// Dim is the residual stream width. Every token position carries one vector
	// of this size through all transformer blocks.
	Dim int
	// HiddenDim is the intermediate width of the feed-forward network.
	HiddenDim int
	// NLayers is the number of transformer blocks.
	NLayers int
	// NHeads is the number of query heads.
	NHeads int
	// NKVHeads is the number of key/value heads. This value may be smaller
	// than NHeads when the model uses grouped-query attention.
	NKVHeads int
	// VocabSize is the number of tokenizer ids and output logits.
	VocabSize int
	// SeqLen is the maximum KV cache length supported by this checkpoint.
	SeqLen int
	// RopeTheta is the base frequency used by rotary positional embeddings.
	RopeTheta float32
}

type LayerWeights struct {
	// RMSNorm weights for the pre-attention and pre-FFN normalizations.
	RMSAttWeight []float32
	RMSFFNWeight []float32
	// Attention projections. WQ outputs Dim values, while WK/WV output kvDim
	// values because K/V only have NKVHeads heads.
	WQ []float32
	WK []float32
	WV []float32
	// WO projects concatenated attention head outputs back to Dim.
	WO []float32
	// Feed-forward projections for SwiGLU: W2(silu(W1(x)) * W3(x)).
	W1 []float32
	W2 []float32
	W3 []float32
}

// Weights owns all model parameters.
type Weights struct {
	TokenEmbeddingTable []float32
	Layers              []LayerWeights
	RMSFinalWeight      []float32
	WCls                []float32
	SharedWeights       bool
}

// State contains the reusable scratch buffers and KV cache.
type State struct {
	X          []float32
	XB         []float32
	XB2        []float32
	HB         []float32
	HB2        []float32
	Q          []float32
	K          []float32
	V          []float32
	Att        []float32
	Logits     []float32
	KeyCache   []float32
	ValueCache []float32
}

type Tables struct {
	RopeCos []float32
	RopeSin []float32
}

type Transformer struct {
	Config  Config
	Weights Weights
	State   State
	Tables  Tables
}

func Load(path string) (*Transformer, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var magic uint32
	var version int32
	if err := binary.Read(file, binary.LittleEndian, &magic); err != nil {
		return nil, err
	}
	if err := binary.Read(file, binary.LittleEndian, &version); err != nil {
		return nil, err
	}
	if magic != checkpointMagic || version != checkpointVersion {
		return nil, fmt.Errorf("bad checkpoint header: magic=%#x version=%d", magic, version)
	}

	var fields [8]int32
	for i := range fields {
		if err := binary.Read(file, binary.LittleEndian, &fields[i]); err != nil {
			return nil, err
		}
	}
	var ropeTheta float32
	if err := binary.Read(file, binary.LittleEndian, &ropeTheta); err != nil {
		return nil, err
	}

	cfg := Config{
		Dim:       int(fields[0]),
		HiddenDim: int(fields[1]),
		NLayers:   int(fields[2]),
		NHeads:    int(fields[3]),
		NKVHeads:  int(fields[4]),
		VocabSize: int(fields[5]),
		SeqLen:    int(fields[6]),
		RopeTheta: ropeTheta,
	}
	sharedWeights := fields[7] != 0
	if err := validateConfig(cfg); err != nil {
		return nil, err
	}
	if _, err := file.Seek(checkpointHeaderSize, io.SeekStart); err != nil {
		return nil, err
	}

	kvDim := cfg.Dim * cfg.NKVHeads / cfg.NHeads

	// Weight order must match tools/export.py exactly. Matrices are stored row-major:
	// each output channel owns one contiguous row consumed by matmul.
	weights := Weights{SharedWeights: sharedWeights}
	weights.TokenEmbeddingTable = readFloat32s(file, cfg.VocabSize*cfg.Dim)
	weights.Layers = make([]LayerWeights, cfg.NLayers)
	for i := range weights.Layers {
		lw := &weights.Layers[i]
		lw.RMSAttWeight = readFloat32s(file, cfg.Dim)
		lw.WQ = readFloat32s(file, cfg.Dim*cfg.Dim)
		lw.WK = readFloat32s(file, cfg.Dim*kvDim)
		lw.WV = readFloat32s(file, cfg.Dim*kvDim)
		lw.WO = readFloat32s(file, cfg.Dim*cfg.Dim)
		lw.RMSFFNWeight = readFloat32s(file, cfg.Dim)
		lw.W1 = readFloat32s(file, cfg.Dim*cfg.HiddenDim)
		lw.W2 = readFloat32s(file, cfg.HiddenDim*cfg.Dim)
		lw.W3 = readFloat32s(file, cfg.Dim*cfg.HiddenDim)
	}
	weights.RMSFinalWeight = readFloat32s(file, cfg.Dim)
	if sharedWeights {
		weights.WCls = weights.TokenEmbeddingTable
	} else {
		weights.WCls = readFloat32s(file, cfg.VocabSize*cfg.Dim)
	}

	state := State{
		X:      make([]float32, cfg.Dim),
		XB:     make([]float32, cfg.Dim),
		XB2:    make([]float32, cfg.Dim),
		HB:     make([]float32, cfg.HiddenDim),
		HB2:    make([]float32, cfg.HiddenDim),
		Q:      make([]float32, cfg.Dim),
		K:      make([]float32, kvDim),
		V:      make([]float32, kvDim),
		Att:    make([]float32, cfg.NHeads*cfg.SeqLen),
		Logits: make([]float32, cfg.VocabSize),
		// KV cache layout: [layer][position][kvDim]. It is append-only along the
		// position dimension during autoregressive decoding.
		KeyCache:   make([]float32, cfg.NLayers*cfg.SeqLen*kvDim),
		ValueCache: make([]float32, cfg.NLayers*cfg.SeqLen*kvDim),
	}

	headSize := cfg.Dim / cfg.NHeads
	ropeCos, ropeSin := buildRopeTables(cfg.SeqLen, headSize, cfg.RopeTheta)

	return &Transformer{
		Config:  cfg,
		Weights: weights,
		State:   state,
		Tables: Tables{
			RopeCos: ropeCos,
			RopeSin: ropeSin,
		},
	}, nil
}

func validateConfig(cfg Config) error {
	if cfg.Dim <= 0 || cfg.HiddenDim <= 0 || cfg.NLayers <= 0 || cfg.NHeads <= 0 ||
		cfg.NKVHeads <= 0 || cfg.VocabSize <= 0 || cfg.SeqLen <= 0 || cfg.RopeTheta <= 0 {
		return fmt.Errorf("invalid config: %+v", cfg)
	}
	if cfg.Dim%cfg.NHeads != 0 || cfg.NHeads%cfg.NKVHeads != 0 {
		return fmt.Errorf("invalid attention dimensions: %+v", cfg)
	}
	return nil
}

func readFloat32s(r io.Reader, count int) []float32 {
	data := make([]float32, count)
	if err := binary.Read(r, binary.LittleEndian, data); err != nil {
		panic(err)
	}
	return data
}

func buildRopeTables(seqLen int, headSize int, ropeTheta float32) ([]float32, []float32) {
	headPairs := headSize / 2
	cosTable := make([]float32, seqLen*headPairs)
	sinTable := make([]float32, seqLen*headPairs)
	for pair := 0; pair < headPairs; pair++ {
		headDim := pair * 2
		freq := float32(1.0 / math.Pow(float64(ropeTheta), float64(headDim)/float64(headSize)))
		for pos := 0; pos < seqLen; pos++ {
			val := float32(pos) * freq
			idx := pos*headPairs + pair
			cosTable[idx] = float32(math.Cos(float64(val)))
			sinTable[idx] = float32(math.Sin(float64(val)))
		}
	}
	return cosTable, sinTable
}

// Forward evaluates one autoregressive decoding step.
//
// The caller passes the token id at absolute sequence position pos. During
// prompt prefill, this function is called once per prompt token with increasing
// pos. During generation, it is called once for each sampled token. In both
// cases the function appends the current token's K/V vectors into the cache and
// returns logits for the next token.
//
// The high-level flow is:
//  1. Copy the token embedding into the residual stream s.X.
//  2. For each layer:
//     - RMS-normalize the residual stream for attention.
//     - Project Q, K, and V; write K/V directly into the current cache slot.
//     - Apply RoPE to Q and the newly written K.
//     - Run causal self-attention over cached positions [0, pos].
//     - Apply the attention output projection and residual add.
//     - Run the RMSNorm + SwiGLU feed-forward block and residual add.
//  3. Apply final RMSNorm and project to vocabulary logits.
//
// State buffers are reused in place; callers should treat the returned logits
// slice as owned by the Transformer until the next Forward call.
func (t *Transformer) Forward(token int, pos int) []float32 {
	cfg := t.Config
	w := t.Weights
	s := &t.State
	dim := cfg.Dim
	kvDim := cfg.Dim * cfg.NKVHeads / cfg.NHeads
	kvMul := cfg.NHeads / cfg.NKVHeads
	hiddenDim := cfg.HiddenDim
	headSize := dim / cfg.NHeads
	headPairs := headSize / 2
	ropeCos := t.Tables.RopeCos[pos*headPairs : (pos+1)*headPairs]
	ropeSin := t.Tables.RopeSin[pos*headPairs : (pos+1)*headPairs]

	// Start from the token embedding. The residual stream then flows through all
	// transformer blocks in s.X.
	copy(s.X, w.TokenEmbeddingTable[token*dim:(token+1)*dim])

	for layer := 0; layer < cfg.NLayers; layer++ {
		lw := w.Layers[layer]
		// Attention sublayer starts with pre-norm, following the Llama-style
		// transformer block layout.
		rmsnorm(s.XB, s.X, lw.RMSAttWeight)

		// Select the cache row for this layer and position. K/V matmul outputs
		// below are written straight into these slices.
		loff := layer * cfg.SeqLen * kvDim
		kcache := s.KeyCache[loff+pos*kvDim : loff+(pos+1)*kvDim]
		vcache := s.ValueCache[loff+pos*kvDim : loff+(pos+1)*kvDim]

		// Query is transient for the current token; K/V persist in the cache so
		// future positions can attend back to them.
		matmul(s.Q, s.XB, lw.WQ, dim, dim)
		matmul(kcache, s.XB, lw.WK, dim, kvDim)
		matmul(vcache, s.XB, lw.WV, dim, kvDim)

		// Apply RoPE to Q and K. Q has one vector per query head, while K only
		// has NKVHeads vectors for grouped-query attention.
		for i := 0; i < dim; i += 2 {
			pair := (i % headSize) / 2
			fcr, fci := ropeCos[pair], ropeSin[pair]
			rotn := 1
			if i < kvDim {
				rotn = 2
			}
			for v := 0; v < rotn; v++ {
				vec := s.Q
				if v == 1 {
					vec = kcache
				}
				v0, v1 := vec[i], vec[i+1]
				vec[i] = v0*fcr - v1*fci
				vec[i+1] = v0*fci + v1*fcr
			}
		}

		// Causal self-attention. Query heads share KV heads when NHeads>NKVHeads.
		for h := 0; h < cfg.NHeads; h++ {
			q := s.Q[h*headSize : (h+1)*headSize]
			att := s.Att[h*cfg.SeqLen : (h+1)*cfg.SeqLen]
			// Score this query head against every cached key up to pos. This is
			// the only O(sequence length) part of a single-token decoding step.
			for ts := 0; ts <= pos; ts++ {
				k := s.KeyCache[loff+ts*kvDim+(h/kvMul)*headSize : loff+ts*kvDim+(h/kvMul+1)*headSize]
				var score float32
				for i := 0; i < headSize; i++ {
					score += q[i] * k[i]
				}
				att[ts] = score / float32(math.Sqrt(float64(headSize)))
			}
			softmax(att[:pos+1])

			// Weighted sum of cached values produces this head's slice of s.XB.
			xb := s.XB[h*headSize : (h+1)*headSize]
			clear(xb)
			for ts := 0; ts <= pos; ts++ {
				v := s.ValueCache[loff+ts*kvDim+(h/kvMul)*headSize : loff+ts*kvDim+(h/kvMul+1)*headSize]
				a := att[ts]
				for i := 0; i < headSize; i++ {
					xb[i] += a * v[i]
				}
			}
		}

		// Attention output projection plus residual connection.
		matmul(s.XB2, s.XB, lw.WO, dim, dim)
		for i := 0; i < dim; i++ {
			s.X[i] += s.XB2[i]
		}

		// SwiGLU feed-forward block: W2(silu(W1(x)) * W3(x)).
		rmsnorm(s.XB, s.X, lw.RMSFFNWeight)
		matmul(s.HB, s.XB, lw.W1, dim, hiddenDim)
		matmul(s.HB2, s.XB, lw.W3, dim, hiddenDim)
		for i := 0; i < hiddenDim; i++ {
			val := s.HB[i]
			val *= 1.0 / (1.0 + float32(math.Exp(float64(-val))))
			s.HB[i] = val * s.HB2[i]
		}
		matmul(s.XB, s.HB, lw.W2, hiddenDim, dim)
		for i := 0; i < dim; i++ {
			s.X[i] += s.XB[i]
		}
	}

	rmsnorm(s.X, s.X, w.RMSFinalWeight)
	matmul(s.Logits, s.X, w.WCls, dim, cfg.VocabSize)
	return s.Logits
}

func rmsnorm(out []float32, x []float32, weight []float32) {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	ss = ss/float32(len(x)) + 1e-5
	scale := float32(1.0 / math.Sqrt(float64(ss)))
	for i := range x {
		out[i] = weight[i] * scale * x[i]
	}
}

func softmax(x []float32) {
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum float32
	for i, v := range x {
		x[i] = float32(math.Exp(float64(v - maxVal)))
		sum += x[i]
	}
	for i := range x {
		x[i] /= sum
	}
}
