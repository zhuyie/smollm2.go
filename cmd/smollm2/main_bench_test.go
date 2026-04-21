package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"smollm2go/internal/model"
	"smollm2go/internal/tokenizer"
)

const benchPrompt = "Hello, my name is"

func benchPaths(b *testing.B) (string, string) {
	b.Helper()
	modelPath := filepath.Join("..", "..", "models", "smollm2-360m-instruct-f32.bin")
	tokenizerPath := filepath.Join("..", "..", "models", "smollm2-tokenizer.bin")
	if _, err := os.Stat(modelPath); err != nil {
		b.Skipf("model checkpoint not found: %s", modelPath)
	}
	if _, err := os.Stat(tokenizerPath); err != nil {
		b.Skipf("tokenizer not found: %s", tokenizerPath)
	}
	return modelPath, tokenizerPath
}

func BenchmarkInitialize(b *testing.B) {
	modelPath, tokenizerPath := benchPaths(b)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		t, err := model.Load(modelPath)
		if err != nil {
			b.Fatal(err)
		}
		if _, err := tokenizer.Load(tokenizerPath, t.Config.VocabSize); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEncodePrompt(b *testing.B) {
	benchmarkEncode(b, benchPrompt)
}

func BenchmarkEncodeLongPrompt(b *testing.B) {
	benchmarkEncode(b, strings.Repeat(benchPrompt+". ", 512))
}

func benchmarkEncode(b *testing.B, prompt string) {
	modelPath, tokenizerPath := benchPaths(b)
	t, err := model.Load(modelPath)
	if err != nil {
		b.Fatal(err)
	}
	tok, err := tokenizer.Load(tokenizerPath, t.Config.VocabSize)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tok.Encode(prompt, false, false)
	}
}
