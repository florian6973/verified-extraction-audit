"""Build a tiny, randomly-initialized Llama model for smoke testing.

The real pipeline fine-tunes a downloaded base model (e.g. Llama-3.2-1B). For a
smoke test we only need *something* with the Llama architecture that trains and
generates in seconds. This builds a ~1M-parameter ``LlamaForCausalLM`` with a
real tokenizer and saves it as a self-contained HuggingFace model directory.

The directory name should contain ``Llama`` so ``finetune.py`` picks the right
FSDP decoder layer (``decoding_layers``).

Example
-------
    python -m src.jobs.smoke.build_tiny_model --out models/base/Llama_tiny --tokenizer gpt2
"""

import argparse


def build(out_dir, tokenizer_name, hidden_size, layers, heads, max_pos):
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 2,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        max_position_embeddings=max_pos,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = LlamaForCausalLM(config)
    n_params = sum(p.numel() for p in model.parameters())

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"Wrote tiny Llama ({n_params/1e6:.2f}M params, vocab {len(tokenizer)}) to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Output model directory (name should contain 'Llama')")
    parser.add_argument("--tokenizer", default="gpt2",
                        help="Tokenizer to reuse (any HF tokenizer id or local path)")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-pos", type=int, default=2048)
    args = parser.parse_args()
    build(args.out, args.tokenizer, args.hidden_size, args.layers, args.heads, args.max_pos)


if __name__ == "__main__":
    main()
