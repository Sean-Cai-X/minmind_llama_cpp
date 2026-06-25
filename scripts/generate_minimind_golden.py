#!/usr/bin/env python3
"""Generate tiny MiniMind golden fixtures for the C++ test suite.

This script is intentionally small and deterministic. It uses the Python
MiniMind reference implementation in `other/minimind-master` to produce JSON
fixtures that C++ tests can consume without importing Python at test runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MiniMind golden fixtures.")
    parser.add_argument(
        "--minimind-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "minimind-master",
        help="Path to the Python MiniMind reference repository.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "testdata" / "golden" / "phase0_forward.json",
        help="Output JSON fixture path.",
    )
    parser.add_argument(
        "--rmsnorm-output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "testdata" / "golden" / "phase0_rmsnorm.json",
        help="Output RMSNorm JSON fixture path.",
    )
    parser.add_argument(
        "--rope-output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "testdata" / "golden" / "phase0_rope.json",
        help="Output RoPE JSON fixture path.",
    )
    parser.add_argument(
        "--ffn-output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "testdata" / "golden" / "phase0_ffn.json",
        help="Output gated FFN JSON fixture path.",
    )
    parser.add_argument(
        "--attention-output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "testdata" / "golden" / "phase0_attention.json",
        help="Output minimal causal attention JSON fixture path.",
    )
    parser.add_argument(
        "--block-output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "testdata" / "golden" / "phase0_block.json",
        help="Output minimal transformer block JSON fixture path.",
    )
    parser.add_argument(
        "--causallm-output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "testdata" / "golden" / "phase0_causallm.json",
        help="Output minimal CausalLM JSON fixture path.",
    )
    parser.add_argument(
        "--loss-output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "testdata" / "golden" / "phase0_loss.json",
        help="Output minimal cross-entropy loss JSON fixture path.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Deterministic torch seed.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.minimind_root))

    import torch
    from model.model_minimind import (
        MiniMindConfig,
        MiniMindForCausalLM,
        FeedForward,
        RMSNorm,
        apply_rotary_pos_emb,
        precompute_freqs_cis,
    )

    torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    config = MiniMindConfig(
        hidden_size=32,
        num_hidden_layers=1,
        vocab_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        flash_attn=False,
        dropout=0.0,
    )
    model = MiniMindForCausalLM(config).eval()

    input_ids = torch.tensor([[1, 10, 11, 12]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, -100, 11, 12]], dtype=torch.long)

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            do_sample=False,
            eos_token_id=config.eos_token_id,
        )

    fixture = {
        "name": "phase0_forward",
        "source": "scripts/generate_minimind_golden.py",
        "seed": args.seed,
        "dtype": "float32",
        "config": {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "max_seq_len": config.max_position_embeddings,
            "bos_token_id": config.bos_token_id,
            "eos_token_id": config.eos_token_id,
            "pad_token_id": config.pad_token_id,
        },
        "prompt": "1 10 11 12",
        "input_ids": input_ids[0].tolist(),
        "labels": labels[0].tolist(),
        "attention_mask": attention_mask[0].tolist(),
        "loss": float(output.loss.item()),
        "logits_last": output.logits[0, -1, :16].tolist(),
        "greedy_ids": generated[0].tolist(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")

    rmsnorm = RMSNorm(4, eps=1e-6)
    rmsnorm.weight.data = torch.tensor([1.0, 1.5, 0.5, 2.0], dtype=torch.float32)
    rmsnorm_input = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        rmsnorm_output = rmsnorm(rmsnorm_input)

    rmsnorm_fixture = {
        "name": "phase0_rmsnorm",
        "source": "scripts/generate_minimind_golden.py",
        "seed": args.seed,
        "dtype": "float32",
        "rows": 2,
        "cols": 4,
        "eps": 1e-6,
        "rmsnorm_input": rmsnorm_input.reshape(-1).tolist(),
        "rmsnorm_weight": rmsnorm.weight.detach().reshape(-1).tolist(),
        "rmsnorm_output": rmsnorm_output.reshape(-1).tolist(),
    }

    args.rmsnorm_output.parent.mkdir(parents=True, exist_ok=True)
    args.rmsnorm_output.write_text(json.dumps(rmsnorm_fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.rmsnorm_output}")

    rope_theta = 1e6
    rope_input = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]],
        dtype=torch.float32,
    )
    rope_cos, rope_sin = precompute_freqs_cis(dim=4, end=2, rope_base=rope_theta)
    q = rope_input.reshape(1, 2, 1, 4)
    rope_output, _ = apply_rotary_pos_emb(q, q, rope_cos, rope_sin)
    rope_output = rope_output.reshape(2, 4)
    rope_fixture = {
        "name": "phase0_rope",
        "source": "scripts/generate_minimind_golden.py",
        "seed": args.seed,
        "dtype": "float32",
        "rows": 2,
        "cols": 4,
        "rope_theta": rope_theta,
        "rope_input": rope_input.reshape(-1).tolist(),
        "rope_cos": rope_cos.reshape(-1).tolist(),
        "rope_sin": rope_sin.reshape(-1).tolist(),
        "rope_output": rope_output.reshape(-1).tolist(),
    }

    args.rope_output.parent.mkdir(parents=True, exist_ok=True)
    args.rope_output.write_text(json.dumps(rope_fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.rope_output}")

    ffn_config = MiniMindConfig(
        hidden_size=3,
        intermediate_size=4,
        hidden_act="silu",
        vocab_size=64,
        num_attention_heads=1,
        num_key_value_heads=1,
    )
    ffn = FeedForward(ffn_config).eval()
    ffn.gate_proj.weight.data = torch.tensor(
        [
            [0.1, 0.2, -0.3],
            [-0.4, 0.5, 0.6],
            [0.7, -0.8, 0.9],
            [-1.0, 0.3, 0.2],
        ],
        dtype=torch.float32,
    )
    ffn.up_proj.weight.data = torch.tensor(
        [
            [0.2, -0.1, 0.4],
            [0.5, 0.6, -0.2],
            [-0.3, 0.8, 0.1],
            [0.9, -0.7, 0.3],
        ],
        dtype=torch.float32,
    )
    ffn.down_proj.weight.data = torch.tensor(
        [
            [0.4, -0.5, 0.6, -0.7],
            [-0.2, 0.1, 0.3, 0.5],
            [0.8, -0.4, 0.2, -0.1],
        ],
        dtype=torch.float32,
    )
    ffn_input = torch.tensor(
        [[1.0, -2.0, 0.5], [0.0, 1.0, 2.0]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        gate_output = ffn.gate_proj(ffn_input)
        up_output = ffn.up_proj(ffn_input)
        ffn_output = ffn(ffn_input)

    ffn_fixture = {
        "name": "phase0_ffn",
        "source": "scripts/generate_minimind_golden.py",
        "seed": args.seed,
        "dtype": "float32",
        "rows": 2,
        "hidden_size": 3,
        "intermediate_size": 4,
        "ffn_input": ffn_input.reshape(-1).tolist(),
        "gate_weight": ffn.gate_proj.weight.detach().reshape(-1).tolist(),
        "up_weight": ffn.up_proj.weight.detach().reshape(-1).tolist(),
        "down_weight": ffn.down_proj.weight.detach().reshape(-1).tolist(),
        "gate_output": gate_output.reshape(-1).tolist(),
        "up_output": up_output.reshape(-1).tolist(),
        "ffn_output": ffn_output.reshape(-1).tolist(),
    }

    args.ffn_output.parent.mkdir(parents=True, exist_ok=True)
    args.ffn_output.write_text(json.dumps(ffn_fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.ffn_output}")

    attention_input = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]],
        dtype=torch.float32,
    )
    identity = torch.eye(4, dtype=torch.float32)
    q_proj = attention_input @ identity.t()
    k_proj = attention_input @ identity.t()
    v_proj = attention_input @ identity.t()
    attn_cos, attn_sin = precompute_freqs_cis(dim=4, end=2, rope_base=rope_theta)
    q_rot, k_rot = apply_rotary_pos_emb(
        q_proj.reshape(1, 2, 1, 4),
        k_proj.reshape(1, 2, 1, 4),
        attn_cos,
        attn_sin,
    )
    q_rot = q_rot.reshape(2, 4)
    k_rot = k_rot.reshape(2, 4)
    scores = (q_rot @ k_rot.t()) / (4 ** 0.5)
    scores = scores.masked_fill(torch.triu(torch.ones(2, 2), diagonal=1).bool(), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    attention_output = probs @ v_proj

    attention_fixture = {
        "name": "phase0_attention",
        "source": "scripts/generate_minimind_golden.py",
        "seed": args.seed,
        "dtype": "float32",
        "seq_len": 2,
        "hidden_size": 4,
        "num_heads": 1,
        "rope_theta": rope_theta,
        "attention_input": attention_input.reshape(-1).tolist(),
        "q_weight": identity.reshape(-1).tolist(),
        "k_weight": identity.reshape(-1).tolist(),
        "v_weight": identity.reshape(-1).tolist(),
        "o_weight": identity.reshape(-1).tolist(),
        "attention_output": attention_output.reshape(-1).tolist(),
    }

    args.attention_output.parent.mkdir(parents=True, exist_ok=True)
    args.attention_output.write_text(json.dumps(attention_fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.attention_output}")

    def rms_norm_tensor(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))

    block_input = attention_input
    norm_weight = torch.ones(4, dtype=torch.float32)
    block_attn_input = rms_norm_tensor(block_input, norm_weight)
    q_proj = block_attn_input @ identity.t()
    k_proj = block_attn_input @ identity.t()
    v_proj = block_attn_input @ identity.t()
    q_rot, k_rot = apply_rotary_pos_emb(
        q_proj.reshape(1, 2, 1, 4),
        k_proj.reshape(1, 2, 1, 4),
        attn_cos,
        attn_sin,
    )
    q_rot = q_rot.reshape(2, 4)
    k_rot = k_rot.reshape(2, 4)
    block_scores = (q_rot @ k_rot.t()) / (4 ** 0.5)
    block_scores = block_scores.masked_fill(torch.triu(torch.ones(2, 2), diagonal=1).bool(), float("-inf"))
    block_probs = torch.softmax(block_scores, dim=-1)
    block_attention_output = block_probs @ v_proj
    block_residual = block_input + block_attention_output
    block_ffn_input = rms_norm_tensor(block_residual, norm_weight)
    block_ffn_output = (block_ffn_input / (1.0 + torch.exp(-block_ffn_input))) * block_ffn_input
    block_output_tensor = block_residual + block_ffn_output

    block_fixture = {
        "name": "phase0_block",
        "source": "scripts/generate_minimind_golden.py",
        "seed": args.seed,
        "dtype": "float32",
        "seq_len": 2,
        "hidden_size": 4,
        "num_heads": 1,
        "intermediate_size": 4,
        "eps": 1e-6,
        "rope_theta": rope_theta,
        "block_input": block_input.reshape(-1).tolist(),
        "attn_norm_weight": norm_weight.reshape(-1).tolist(),
        "ffn_norm_weight": norm_weight.reshape(-1).tolist(),
        "q_weight": identity.reshape(-1).tolist(),
        "k_weight": identity.reshape(-1).tolist(),
        "v_weight": identity.reshape(-1).tolist(),
        "o_weight": identity.reshape(-1).tolist(),
        "gate_weight": identity.reshape(-1).tolist(),
        "up_weight": identity.reshape(-1).tolist(),
        "down_weight": identity.reshape(-1).tolist(),
        "block_attention_output": block_attention_output.reshape(-1).tolist(),
        "block_output": block_output_tensor.reshape(-1).tolist(),
    }

    args.block_output.parent.mkdir(parents=True, exist_ok=True)
    args.block_output.write_text(json.dumps(block_fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.block_output}")

    causallm_input_ids = torch.tensor([0, 1], dtype=torch.long)
    token_embedding = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [-1.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    causallm_hidden = token_embedding[causallm_input_ids]
    causallm_attn_input = rms_norm_tensor(causallm_hidden, norm_weight)
    q_proj = causallm_attn_input @ identity.t()
    k_proj = causallm_attn_input @ identity.t()
    v_proj = causallm_attn_input @ identity.t()
    q_rot, k_rot = apply_rotary_pos_emb(
        q_proj.reshape(1, 2, 1, 4),
        k_proj.reshape(1, 2, 1, 4),
        attn_cos,
        attn_sin,
    )
    q_rot = q_rot.reshape(2, 4)
    k_rot = k_rot.reshape(2, 4)
    causallm_scores = (q_rot @ k_rot.t()) / (4 ** 0.5)
    causallm_scores = causallm_scores.masked_fill(torch.triu(torch.ones(2, 2), diagonal=1).bool(), float("-inf"))
    causallm_probs = torch.softmax(causallm_scores, dim=-1)
    causallm_attention_output = causallm_probs @ v_proj
    causallm_residual = causallm_hidden + causallm_attention_output
    causallm_ffn_input = rms_norm_tensor(causallm_residual, norm_weight)
    causallm_ffn_output = (causallm_ffn_input / (1.0 + torch.exp(-causallm_ffn_input))) * causallm_ffn_input
    causallm_hidden = causallm_residual + causallm_ffn_output
    causallm_hidden = rms_norm_tensor(causallm_hidden, norm_weight)
    causallm_logits = causallm_hidden @ identity.t()
    causallm_generated = causallm_input_ids.tolist() + [int(torch.argmax(causallm_logits[-1]).item())]
    loss_labels = torch.tensor([-100, 3], dtype=torch.long)
    causallm_loss = torch.nn.functional.cross_entropy(
        causallm_logits,
        loss_labels,
        ignore_index=-100,
        reduction="mean",
    )

    causallm_fixture = {
        "name": "phase0_causallm",
        "source": "scripts/generate_minimind_golden.py",
        "seed": args.seed,
        "dtype": "float32",
        "vocab_size": 4,
        "seq_len": 2,
        "hidden_size": 4,
        "num_heads": 1,
        "intermediate_size": 4,
        "eps": 1e-6,
        "rope_theta": rope_theta,
        "input_ids": causallm_input_ids.tolist(),
        "labels": loss_labels.tolist(),
        "loss": float(causallm_loss.item()),
        "greedy_ids": causallm_generated,
        "token_embedding": token_embedding.reshape(-1).tolist(),
        "attn_norm_weight": norm_weight.reshape(-1).tolist(),
        "ffn_norm_weight": norm_weight.reshape(-1).tolist(),
        "final_norm_weight": norm_weight.reshape(-1).tolist(),
        "q_weight": identity.reshape(-1).tolist(),
        "k_weight": identity.reshape(-1).tolist(),
        "v_weight": identity.reshape(-1).tolist(),
        "o_weight": identity.reshape(-1).tolist(),
        "gate_weight": identity.reshape(-1).tolist(),
        "up_weight": identity.reshape(-1).tolist(),
        "down_weight": identity.reshape(-1).tolist(),
        "lm_head_weight": identity.reshape(-1).tolist(),
        "logits": causallm_logits.reshape(-1).tolist(),
    }

    args.causallm_output.parent.mkdir(parents=True, exist_ok=True)
    args.causallm_output.write_text(json.dumps(causallm_fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.causallm_output}")

    loss_fixture = {
        "name": "phase0_loss",
        "source": "scripts/generate_minimind_golden.py",
        "seed": args.seed,
        "dtype": "float32",
        "rows": 2,
        "vocab_size": 4,
        "labels": loss_labels.tolist(),
        "loss": float(causallm_loss.item()),
        "logits": causallm_logits.reshape(-1).tolist(),
    }

    args.loss_output.parent.mkdir(parents=True, exist_ok=True)
    args.loss_output.write_text(json.dumps(loss_fixture, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.loss_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
