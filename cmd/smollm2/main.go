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

const (
	ansiReset     = "\x1b[0m"
	ansiPrompt    = "\x1b[33m"
	ansiUserInput = "\x1b[1m\x1b[32m"
)

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
		printAssistantPrefix(os.Stdout)
		chatReply(t, tok, samp, renderChatPrompt(messages, systemPrompt), maxNew, os.Stdout)
		fmt.Println(ansiReset)
		return
	}

	pos := 0
	var logits []float32
	logits, pos = forwardTokens(t, tok.Encode(renderSystemPrompt(systemPrompt), false, false), pos)
	totalGenerated := 0
	totalDuration := time.Duration(0)
	scanner := bufio.NewScanner(os.Stdin)
	for {
		printUserPrefix(os.Stdout)
		if !scanner.Scan() {
			fmt.Print(ansiReset)
			break
		}
		fmt.Print(ansiReset)
		userPrompt := strings.TrimSpace(scanner.Text())
		if userPrompt == "" {
			continue
		}
		if userPrompt == "/exit" || userPrompt == "/quit" {
			fmt.Println()
			break
		}
		logits, pos = forwardTokens(t, tok.Encode(renderUserTurn(userPrompt), false, false), pos)
		printAssistantPrefix(os.Stdout)
		var generated int
		var duration time.Duration
		_, pos, generated, duration = generateAssistant(t, tok, samp, logits, pos, maxNew, os.Stdout)
		totalGenerated += generated
		totalDuration += duration
		fmt.Println(ansiReset)
	}
	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
	if totalGenerated > 1 && totalDuration > 0 {
		tokPerSec := float64(totalGenerated) / totalDuration.Seconds()
		fmt.Fprintf(os.Stderr, "achieved tok/s: %.6f\n", tokPerSec)
	}
}

func printUserPrefix(w io.Writer) {
	fmt.Fprint(w, ansiPrompt, "User: ", ansiUserInput)
}

func printAssistantPrefix(w io.Writer) {
	fmt.Fprint(w, ansiPrompt, "Assistant: ", ansiReset)
}

func chatReply(t *model.Transformer, tok *tokenizer.Tokenizer, samp *sampler.Sampler, rendered string, maxNew int, w io.Writer) string {
	ids := tok.Encode(rendered, false, false)
	pos := 0
	// Chat mode differs from generate mode only in prompt rendering.
	logits, pos := forwardTokens(t, ids, pos)
	out, _, _, _ := generateAssistant(t, tok, samp, logits, pos, maxNew, w)
	return out
}

func generateAssistant(t *model.Transformer, tok *tokenizer.Tokenizer, samp *sampler.Sampler, logits []float32, pos int, maxNew int, w io.Writer) (string, int, int, time.Duration) {
	var out strings.Builder
	generated := 0
	start := time.Now()
	for generated < maxNew && pos < t.Config.SeqLen {
		next := samp.Sample(logits)
		if next == tok.EOS() {
			pos = closeAssistantTurn(t, tok, pos)
			break
		}
		piece := tok.Decode(next)
		fmt.Fprint(w, piece)
		out.WriteString(piece)
		logits = t.Forward(next, pos)
		pos++
		generated++
	}
	if generated == maxNew && pos < t.Config.SeqLen {
		pos = closeAssistantTurn(t, tok, pos)
	}
	duration := time.Duration(0)
	if generated > 0 {
		duration = time.Since(start)
	}
	return out.String(), pos, generated, duration
}

func closeAssistantTurn(t *model.Transformer, tok *tokenizer.Tokenizer, pos int) int {
	if pos < t.Config.SeqLen {
		t.Forward(tok.EOS(), pos)
		pos++
	}
	ids := tok.Encode("\n", false, false)
	for i := 0; i < len(ids) && pos < t.Config.SeqLen; i++ {
		t.Forward(ids[i], pos)
		pos++
	}
	return pos
}

func forwardTokens(t *model.Transformer, ids []int, pos int) ([]float32, int) {
	var logits []float32
	for i := 0; i < len(ids) && pos < t.Config.SeqLen; i++ {
		logits = t.Forward(ids[i], pos)
		pos++
	}
	return logits, pos
}

func renderChatPrompt(messages []chatMessage, systemPrompt string) string {
	var b strings.Builder
	b.WriteString(renderSystemPrompt(systemPrompt))
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

func renderSystemPrompt(systemPrompt string) string {
	if systemPrompt == "" {
		systemPrompt = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
	}
	return "<|im_start|>system\n" + systemPrompt + "<|im_end|>\n"
}

func renderUserTurn(userPrompt string) string {
	return "<|im_start|>user\n" + userPrompt + "<|im_end|>\n<|im_start|>assistant\n"
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
