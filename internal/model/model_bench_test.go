package model

import (
	"os"
	"path/filepath"
	"testing"
)

func BenchmarkForward(b *testing.B) {
	path := filepath.Join("..", "..", "models", "smollm2-360m-instruct-f32.bin")
	if _, err := os.Stat(path); err != nil {
		b.Skipf("model checkpoint not found: %s", path)
	}
	t, err := Load(path)
	if err != nil {
		b.Fatal(err)
	}
	token := 0
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		t.Forward(token, i%t.Config.SeqLen)
	}
}
