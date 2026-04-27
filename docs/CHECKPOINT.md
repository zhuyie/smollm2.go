# Binary Formats

This project uses a simple SML2 checkpoint format. Version 1 checkpoints are
float32-only. Version 2 checkpoints add an explicit `weight_type` field so the
runtime can select the inference path from the file itself.

All numeric fields are little-endian.

## Header

The first 256 bytes are reserved for the header.

| Field | Type | Notes |
| --- | --- | --- |
| `magic` | `uint32` | `0x324C4D53`, bytes spell `SML2` |
| `version` | `int32` | `1` or `2` |
| `dim` | `int32` | model hidden size |
| `hidden_dim` | `int32` | MLP hidden size |
| `n_layers` | `int32` | Transformer block count |
| `n_heads` | `int32` | query attention heads |
| `n_kv_heads` | `int32` | key/value heads |
| `vocab_size` | `int32` | tokenizer vocabulary size |
| `seq_len` | `int32` | maximum sequence length |
| `shared_classifier` | `int32` | nonzero means `wcls` aliases token embeddings |
| `rope_theta` | `float32` | RoPE base frequency |
| `weight_type` | `int32` | version 2 only: `0` = float32, `1` = int8 |

All remaining bytes up to byte 256 are zero padding.

Version 1 has no `weight_type` field; it is equivalent to version 2 with
`weight_type == 0`.

## Float32 Weights

For `weight_type == 0`, all tensors are written as contiguous float32 values in
row-major order.

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

## Int8 Weights

For `weight_type == 1`, embeddings and RMSNorm weights stay float32. Projection
matrices are stored as per-row symmetric int8.

Each int8 matrix is written as:

1. `int8 data`: `(rows, inputs)` in row-major order
2. `float32 scale`: `(rows,)`

The dequantized value is `float32(data[row, col]) * scale[row]`.

The tensor order is:

1. `token_embedding_table`: `(vocab_size, dim)` as float32
2. For each layer:
   - `rms_att_weight`: `(dim,)` as float32
   - `wq`: int8 matrix `(dim, dim)`
   - `wk`: int8 matrix `(n_kv_heads * head_size, dim)`
   - `wv`: int8 matrix `(n_kv_heads * head_size, dim)`
   - `wo`: int8 matrix `(dim, dim)`
   - `rms_ffn_weight`: `(dim,)` as float32
   - `w1`: int8 matrix `(hidden_dim, dim)`
   - `w2`: int8 matrix `(dim, hidden_dim)`
   - `w3`: int8 matrix `(hidden_dim, dim)`
3. `rms_final_weight`: `(dim,)` as float32
4. `wcls`: int8 matrix `(vocab_size, dim)`

Int8 checkpoints always include a quantized `wcls`, even when the float32 source
checkpoint used shared classifier weights.

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
