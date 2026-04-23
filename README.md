# smollm2.go

Minimal Go implementation for inference with Hugging Face SmolLM2 Instruct models:
[360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) and [1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct).

Inspired by Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c).

The project is self-contained:

- Python scripts convert Hugging Face weights into compact local binary formats.
- Go code loads those binary files and runs inference.

## Layout

```text
cmd/smollm2/          CLI entry point
internal/model/       SML2 loader, weights, KV cache, forward pass
internal/tokenizer/   TOK2 loader and byte-level BPE tokenizer
internal/sampler/     greedy, multinomial, and top-p sampling
tools/                Hugging Face model/tokenizer export scripts
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

Create the output directory first:

```sh
mkdir -p models
```

Export the small 360M model:

```sh
.venv/bin/python tools/export.py models/smollm2-360m-instruct-f32.bin \
  --hf HuggingFaceTB/SmolLM2-360M-Instruct

.venv/bin/python tools/export_tokenizer.py models/smollm2-tokenizer.bin \
  --hf HuggingFaceTB/SmolLM2-360M-Instruct
```

For a smarter model, export the 1.7B checkpoint:

```sh
.venv/bin/python tools/export.py models/smollm2-1.7b-instruct-f32.bin \
  --hf HuggingFaceTB/SmolLM2-1.7B-Instruct
```

The tokenizer is shared by the SmolLM2 Instruct family, so it only needs to be exported once.

## Build

```sh
mkdir -p bin
go build -o bin/smollm2 ./cmd/smollm2
```

## Run

### Chat

```sh
bin/smollm2 \
  -model models/smollm2-360m-instruct-f32.bin \
  -tokenizer models/smollm2-tokenizer.bin \
  -mode chat \
  -prompt "What is 2+2?" \
  -temp 0
```

Expected output:

```text
Assistant: 2 + 2 = 4.
```

### Raw Completion

```sh
bin/smollm2 \
  -model models/smollm2-360m-instruct-f32.bin \
  -tokenizer models/smollm2-tokenizer.bin \
  -mode generate \
  -prompt "A tiny robot found a lost key and" \
  -n 64 \
  -temp 0
```
