# Binary Formats

This project uses a simple float32 checkpoint format.

## Header

The first 256 bytes are reserved for the header.

| Field | Type | Notes |
| --- | --- | --- |
| `magic` | `uint32` | `0x324C4D53`, bytes spell `SML2` |
| `version` | `int32` | currently `1` |
| `dim` | `int32` | model hidden size |
| `hidden_dim` | `int32` | MLP hidden size |
| `n_layers` | `int32` | Transformer block count |
| `n_heads` | `int32` | query attention heads |
| `n_kv_heads` | `int32` | key/value heads |
| `vocab_size` | `int32` | tokenizer vocabulary size |
| `seq_len` | `int32` | maximum sequence length |
| `shared_classifier` | `int32` | nonzero means `wcls` aliases token embeddings |
| `rope_theta` | `float32` | RoPE base frequency |

All remaining bytes up to byte 256 are zero padding.

## Tensor Order

All tensors are written as contiguous float32 values in row-major order.

1. `token_embedding_table`: `(vocab_size, dim)`
2. For each layer:
   - `rms_att_weight`: `(dim,)`
   - `wq`: `(dim, dim)`
   - `wk`: `(n_kv_heads * head_size, dim)`
   - `wv`: `(n_kv_heads * head_size, dim)`
   - `wo`: `(dim, dim)`
   - `rms_ffn_weight`: `(dim,)`
   - `w1`: `(hidden_dim, dim)`
   - `w2`: `(dim, hidden_dim)`
   - `w3`: `(hidden_dim, dim)`
3. `rms_final_weight`: `(dim,)`
4. If `shared_classifier == 0`, `wcls`: `(vocab_size, dim)`

`head_size = dim / n_heads`.

## Tokenizer Format

The tokenizer file is a compact binary form of SmolLM2's GPT-2 byte-level BPE
tokenizer.

The first 256 bytes are reserved for the header.

| Field | Type | Notes |
| --- | --- | --- |
| `magic` | `uint32` | `0x324B4F54`, bytes spell `TOK2` |
| `version` | `int32` | currently `1` |
| `vocab_size` | `int32` | number of token strings |
| `merge_count` | `int32` | number of BPE merge rules |
| `max_token_length` | `int32` | maximum UTF-8 byte length of a token string |
| `bos_id` | `int32` | usually `<|im_start|>` |
| `eos_id` | `int32` | usually `<|im_end|>` |
| `pad_id` | `int32` | usually `<|im_end|>` |
| `unk_id` | `int32` | usually `<|endoftext|>` |

The body stores:

1. `vocab_size` token strings, each as `uint32 byte_length` followed by UTF-8 bytes.
2. `merge_count` merge rules, each as three `int32` values:
   - left token id
   - right token id
   - merged output token id
