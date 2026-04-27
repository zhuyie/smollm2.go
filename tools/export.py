"""Export a Hugging Face SmolLM2/LlamaForCausalLM model to SML2 fp32 format."""

import argparse
import struct

import torch
from transformers import AutoModelForCausalLM


CHECKPOINT_MAGIC = 0x324C4D53  # "SML2" in little-endian bytes
CHECKPOINT_VERSION_V1 = 1
CHECKPOINT_HEADER_SIZE = 256


def serialize_fp32(file, tensor):
    data = tensor.detach().cpu().reshape(-1).to(torch.float32).numpy()
    file.write(data.tobytes())


def reverse_rope_permute(weight, n_heads, out_dim, in_dim):
    return (
        weight.view(n_heads, 2, out_dim // n_heads // 2, in_dim)
        .transpose(1, 2)
        .reshape(out_dim, in_dim)
    )


def config_value(config, name, default):
    value = getattr(config, name, default)
    return default if value is None else value


def export_hf(model_name, output_path):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    state = model.state_dict()
    config = model.config

    dim = config.hidden_size
    hidden_dim = config.intermediate_size
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = config_value(config, "num_key_value_heads", n_heads)
    vocab_size = config.vocab_size
    seq_len = config.max_position_embeddings
    rope_theta = float(config_value(config, "rope_theta", 10000.0))
    shared_classifier = bool(config_value(config, "tie_word_embeddings", True))
    head_dim = dim // n_heads
    kv_dim = n_kv_heads * head_dim

    with open(output_path, "wb") as out:
        header = struct.pack(
            "<Iiiiiiiiiif",
            CHECKPOINT_MAGIC,
            CHECKPOINT_VERSION_V1,
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
            int(shared_classifier),
            rope_theta,
        )
        out.write(header)
        out.write(b"\0" * (CHECKPOINT_HEADER_SIZE - len(header)))

        serialize_fp32(out, state["model.embed_tokens.weight"])
        for layer in range(n_layers):
            prefix = f"model.layers.{layer}"
            serialize_fp32(out, state[f"{prefix}.input_layernorm.weight"])
            serialize_fp32(
                out,
                reverse_rope_permute(
                    state[f"{prefix}.self_attn.q_proj.weight"],
                    n_heads,
                    dim,
                    dim,
                ),
            )
            serialize_fp32(
                out,
                reverse_rope_permute(
                    state[f"{prefix}.self_attn.k_proj.weight"],
                    n_kv_heads,
                    kv_dim,
                    dim,
                ),
            )
            serialize_fp32(out, state[f"{prefix}.self_attn.v_proj.weight"])
            serialize_fp32(out, state[f"{prefix}.self_attn.o_proj.weight"])
            serialize_fp32(out, state[f"{prefix}.post_attention_layernorm.weight"])
            serialize_fp32(out, state[f"{prefix}.mlp.gate_proj.weight"])
            serialize_fp32(out, state[f"{prefix}.mlp.down_proj.weight"])
            serialize_fp32(out, state[f"{prefix}.mlp.up_proj.weight"])

        serialize_fp32(out, state["model.norm.weight"])
        if not shared_classifier:
            serialize_fp32(out, state["lm_head.weight"])

    print(f"wrote {output_path}")
    print(
        "config: "
        f"dim={dim} hidden_dim={hidden_dim} layers={n_layers} "
        f"heads={n_heads} kv_heads={n_kv_heads} vocab={vocab_size} "
        f"seq_len={seq_len} rope_theta={rope_theta} shared_classifier={int(shared_classifier)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", help="output SML2 checkpoint path")
    parser.add_argument("--hf", default="HuggingFaceTB/SmolLM2-360M-Instruct")
    args = parser.parse_args()
    export_hf(args.hf, args.output)


if __name__ == "__main__":
    main()
