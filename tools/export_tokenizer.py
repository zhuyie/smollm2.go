import argparse
import struct

from transformers import AutoTokenizer


TOKENIZER_MAGIC = 0x324B4F54  # "TOK2" in little-endian bytes
TOKENIZER_VERSION = 1
TOKENIZER_HEADER_SIZE = 256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="output tokenizer .bin path")
    parser.add_argument("--hf", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.hf)
    vocab_by_token = tokenizer.get_vocab()
    vocab_size = len(vocab_by_token)
    vocab = [None] * vocab_size
    for token, idx in vocab_by_token.items():
        vocab[idx] = token
    if any(token is None for token in vocab):
        raise ValueError("vocabulary ids are not contiguous")

    merges = []
    for left, right in tokenizer._merges:
        out = left + right
        if left in vocab_by_token and right in vocab_by_token and out in vocab_by_token:
            merges.append((vocab_by_token[left], vocab_by_token[right], vocab_by_token[out]))

    max_token_length = max(len(token.encode("utf-8")) for token in vocab)
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else -1
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else -1
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else -1

    with open(args.output, "wb") as f:
        header = struct.pack(
            "<Iiiiiiiii",
            TOKENIZER_MAGIC,
            TOKENIZER_VERSION,
            vocab_size,
            len(merges),
            max_token_length,
            bos_id,
            eos_id,
            pad_id,
            unk_id,
        )
        f.write(header)
        f.write(b"\0" * (TOKENIZER_HEADER_SIZE - len(header)))

        for token in vocab:
            data = token.encode("utf-8")
            f.write(struct.pack("<I", len(data)))
            f.write(data)

        for left, right, out in merges:
            f.write(struct.pack("<iii", left, right, out))

    print(f"wrote {args.output}")
    print(f"vocab_size={vocab_size} merges={len(merges)} eos_id={eos_id}")


if __name__ == "__main__":
    main()
