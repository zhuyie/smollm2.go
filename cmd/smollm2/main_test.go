package main

import (
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
