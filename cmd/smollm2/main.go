package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"sort"
	"strings"
	"time"

	"smollm2go/internal/model"
	"smollm2go/internal/sampler"
	"smollm2go/internal/tokenizer"
)

type chatMessage struct {
	role    string
	content string
}

func main() {
	modelPath := flag.String("model", "", "SML2 model path")
	tokenizerPath := flag.String("tokenizer", "", "TOK2 tokenizer path")
	mode := flag.String("mode", "generate", "generate|chat|encode|logits")
	prompt := flag.String("prompt", "", "input prompt")
	systemPrompt := flag.String("system", "", "optional system prompt for chat")
	maxNew := flag.Int("n", 256, "maximum new tokens")
	temperature := flag.Float64("temp", 1.0, "sampling temperature, 0 for greedy")
	topP := flag.Float64("top-p", 0.9, "top-p nucleus sampling")
	seed := flag.Int64("seed", time.Now().UnixNano(), "random seed")
	flag.Parse()

	if *modelPath == "" || *tokenizerPath == "" {
		flag.Usage()
		os.Exit(2)
	}

	transformer, err := model.Load(*modelPath)
	if err != nil {
		log.Fatal(err)
	}
	tok, err := tokenizer.Load(*tokenizerPath, transformer.Config.VocabSize)
	if err != nil {
		log.Fatal(err)
	}
	samp := sampler.New(float32(*temperature), float32(*topP), *seed)

	switch *mode {
	case "generate":
		generate(transformer, tok, samp, *prompt, *maxNew)
	case "chat":
		chat(transformer, tok, samp, *prompt, *systemPrompt, *maxNew)
	case "encode":
		ids := tok.Encode(*prompt, false, false)
		for i, id := range ids {
			if i > 0 {
				fmt.Print(" ")
			}
			fmt.Print(id)
		}
		fmt.Println()
	case "logits":
		printTopLogits(transformer, tok, *prompt, 10)
	default:
		log.Fatalf("unknown mode %q", *mode)
	}
}

func generate(t *model.Transformer, tok *tokenizer.Tokenizer, samp *sampler.Sampler, prompt string, maxNew int) {
	ids := tok.Encode(prompt, false, false)
	if len(ids) == 0 {
		ids = append(ids, tok.EOS())
	}
	var logits []float32
	pos := 0
	// Prefill consumes the whole prompt and leaves logits for the next token.
	for ; pos < len(ids) && pos < t.Config.SeqLen; pos++ {
		logits = t.Forward(ids[pos], pos)
	}
	generated := 0
	token := -1
	start := time.Time{}
	for generated < maxNew && pos < t.Config.SeqLen {
		next := samp.Sample(logits)
		if next == tok.EOS() {
			break
		}
		fmt.Print(tok.Decode(next))
		// Decode one token at a time, appending its KV entries to the cache.
		if token = next; token >= 0 {
			logits = t.Forward(token, pos)
			pos++
		}
		generated++
		if start.IsZero() {
			start = time.Now()
		}
	}
	fmt.Println()
	if generated > 1 && !start.IsZero() {
		tokPerSec := float64(generated) / time.Since(start).Seconds()
		fmt.Fprintf(os.Stderr, "achieved tok/s: %.6f\n", tokPerSec)
	}
}

func chat(t *model.Transformer, tok *tokenizer.Tokenizer, samp *sampler.Sampler, userPrompt string, systemPrompt string, maxNew int) {
	if userPrompt != "" {
		messages := []chatMessage{{role: "user", content: userPrompt}}
		fmt.Print("Assistant: ")
		chatReply(t, tok, samp, messages, systemPrompt, maxNew, os.Stdout)
		fmt.Println()
		return
	}

	var messages []chatMessage
	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("User: ")
		if !scanner.Scan() {
			break
		}
		userPrompt := strings.TrimSpace(scanner.Text())
		if userPrompt == "" {
			continue
		}
		if userPrompt == "/exit" || userPrompt == "/quit" {
			break
		}
		messages = append(messages, chatMessage{role: "user", content: userPrompt})
		fmt.Print("Assistant: ")
		reply := chatReply(t, tok, samp, messages, systemPrompt, maxNew, os.Stdout)
		fmt.Println()
		messages = append(messages, chatMessage{role: "assistant", content: reply})
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
}

func chatReply(t *model.Transformer, tok *tokenizer.Tokenizer, samp *sampler.Sampler, messages []chatMessage, systemPrompt string, maxNew int, w io.Writer) string {
	rendered := renderChatPrompt(messages, systemPrompt)
	ids := tok.Encode(rendered, false, false)
	pos := 0
	var logits []float32
	// Chat mode differs from generate mode only in prompt rendering.
	for ; pos < len(ids) && pos < t.Config.SeqLen; pos++ {
		logits = t.Forward(ids[pos], pos)
	}
	var out strings.Builder
	generated := 0
	token := -1
	for generated < maxNew && pos < t.Config.SeqLen {
		next := samp.Sample(logits)
		if next == tok.EOS() {
			break
		}
		piece := tok.Decode(next)
		fmt.Fprint(w, piece)
		out.WriteString(piece)
		token = next
		logits = t.Forward(token, pos)
		pos++
		generated++
	}
	return out.String()
}

func renderChatPrompt(messages []chatMessage, systemPrompt string) string {
	if systemPrompt == "" {
		systemPrompt = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
	}
	var b strings.Builder
	b.WriteString("<|im_start|>system\n")
	b.WriteString(systemPrompt)
	b.WriteString("<|im_end|>\n")
	for _, msg := range messages {
		b.WriteString("<|im_start|>")
		b.WriteString(msg.role)
		b.WriteByte('\n')
		b.WriteString(msg.content)
		b.WriteString("<|im_end|>\n")
	}
	b.WriteString("<|im_start|>assistant\n")
	return b.String()
}

func printTopLogits(t *model.Transformer, tok *tokenizer.Tokenizer, prompt string, k int) {
	ids := tok.Encode(prompt, false, false)
	if len(ids) == 0 {
		log.Fatal("empty prompt")
	}
	var logits []float32
	for pos, id := range ids {
		logits = t.Forward(id, pos)
	}
	type item struct {
		id  int
		val float32
	}
	items := make([]item, len(logits))
	for i, v := range logits {
		items[i] = item{id: i, val: v}
	}
	sort.Slice(items, func(i, j int) bool { return items[i].val > items[j].val })
	for i := 0; i < k && i < len(items); i++ {
		fmt.Printf("%d %.6f %s\n", items[i].id, items[i].val, tok.Decode(items[i].id))
	}
}
