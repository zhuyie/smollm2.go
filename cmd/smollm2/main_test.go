package main

import (
	"os"
	"path/filepath"
	"smollm2go/internal/model"
	"smollm2go/internal/tokenizer"
	"strings"
	"testing"
)

func TestRenderChatPromptIncludesHistory(t *testing.T) {
	messages := []chatMessage{
		{role: "user", content: "hello"},
		{role: "assistant", content: "hi"},
		{role: "user", content: "again"},
	}
	got := renderChatPrompt(messages, "system")
	want := "<|im_start|>system\nsystem<|im_end|>\n" +
		"<|im_start|>user\nhello<|im_end|>\n" +
		"<|im_start|>assistant\nhi<|im_end|>\n" +
		"<|im_start|>user\nagain<|im_end|>\n" +
		"<|im_start|>assistant\n"
	if got != want {
		t.Fatalf("renderChatPrompt() = %q, want %q", got, want)
	}
}

func TestRenderChatPromptUsesDefaultSystemPrompt(t *testing.T) {
	got := renderChatPrompt(nil, "")
	if !strings.Contains(got, "You are a helpful AI assistant named SmolLM") {
		t.Fatalf("renderChatPrompt() = %q, want default system prompt", got)
	}
}

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
