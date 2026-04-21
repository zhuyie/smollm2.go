# smollm2.go

Minimal Go implementation for inference `HuggingFaceTB/SmolLM2-360M-Instruct`.
Inspired by Andrej Karpathy's `llama2.c`.

The project is self-contained:

- Python scripts convert Hugging Face files into compact local binary formats.
- Go code loads those binary files and runs inference.

## Layout

```text
cmd/smollm2/          CLI entry point
internal/model/       SML2 loader, weights, KV cache, forward pass
internal/tokenizer/   TOK2 loader and byte-level BPE tokenizer
internal/sampler/     greedy, multinomial, and top-p sampling
export.py             Hugging Face model -> SML2 checkpoint
export_tokenizer.py   Hugging Face tokenizer -> TOK2 tokenizer
docs/CHECKPOINT.md    SML2/TOK2 binary format notes
```

## Prepare Python Environment

The export scripts need `torch`, `transformers`, and `sentencepiece`.

One option is to create a local venv:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip torch transformers sentencepiece safetensors accelerate numpy
```

If you already have a Python environment with those packages, use that instead.

## Export Model And Tokenizer

```sh
mkdir -p models

.venv/bin/python export.py models/smollm2-360m-instruct-f32.bin \
  --hf HuggingFaceTB/SmolLM2-360M-Instruct

.venv/bin/python export_tokenizer.py models/smollm2-tokenizer.bin \
  --hf HuggingFaceTB/SmolLM2-360M-Instruct
```

## Build

```sh
mkdir -p bin
go build -o bin/smollm2 ./cmd/smollm2
```

## Run

Encode a prompt:

```sh
bin/smollm2 \
  -model models/smollm2-360m-instruct-f32.bin \
  -tokenizer models/smollm2-tokenizer.bin \
  -mode encode \
  -prompt "Hello, my name is"
```

Chat:

```sh
bin/smollm2 \
  -model models/smollm2-360m-instruct-f32.bin \
  -tokenizer models/smollm2-tokenizer.bin \
  -mode chat \
  -prompt "What is 2+2?" \
  -n 32 \
  -temp 0
```

Expected output:

```text
Assistant: 2 + 2 = 4.
```

Generate:

```sh
bin/smollm2 \
  -model models/smollm2-360m-instruct-f32.bin \
  -tokenizer models/smollm2-tokenizer.bin \
  -mode generate \
  -prompt "Hello, my name is" \
  -n 64 \
  -temp 0
```
