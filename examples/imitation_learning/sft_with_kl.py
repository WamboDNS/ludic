"""
Imitation Learning with KL Regularization (SFT+KL).

This example demonstrates how to warm up models using imitation learning with
KL regularization:

    L_SFT(π) = E[ (1/|a|) * Σ_i ( -log π(a_i|s, a_{<i}) + β * KL_token ) ]

where:
    - The first term is standard SFT (imitate demonstrations)
    - KL_token penalizes deviation from the reference/base policy
    - β controls regularization strength

This is useful for:
    1. Warming up small models before RL training
    2. Bootstrapping agent/attacker policies from capable model demonstrations
    3. Transfer learning while preserving base model capabilities

Workflow:
    1. Collect demonstrations using a capable model (e.g., GPT-4, Claude)
    2. Annotate with reference logprobs from the base model
    3. Train with SFT+KL using this script

Usage:
    # Step 1: Annotate data with reference logprobs
    python sft_with_kl.py annotate \
        --model Qwen/Qwen2.5-3B-Instruct \
        --input demos_raw.jsonl \
        --output demos_with_logprobs.jsonl

    # Step 2: Train with SFT+KL
    torchrun --nproc_per_node=2 sft_with_kl.py train \
        --model Qwen/Qwen2.5-3B-Instruct \
        --data demos_with_logprobs.jsonl \
        --kl-coeff 0.1
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed import fsdp
from transformers import AutoModelForCausalLM, AutoTokenizer

from ludic.training import (
    OfflineBatchSource,
    Trainer,
    TrainerConfig,
    CheckpointConfig,
    make_sft_with_kl,
    make_chat_template_step_to_item,
    annotate_jsonl_with_ref_logprobs,
)


def init_dist(local_rank: int) -> int:
    """Initialize distributed training."""
    if dist.is_initialized():
        return dist.get_rank()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    else:
        dist.init_process_group(backend="gloo", init_method="env://")
    return dist.get_rank()


def cmd_annotate(args: argparse.Namespace) -> None:
    """Annotate demonstration data with reference policy logprobs."""
    print(f"Loading model: {args.model}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    print(f"Annotating {input_path} -> {output_path}")

    num_steps = annotate_jsonl_with_ref_logprobs(
        input_path,
        output_path,
        model,
        tokenizer,
        device=device,
    )

    print(f"Done! Annotated {num_steps} steps with reference logprobs.")
    print(f"Output saved to: {output_path}")


def cmd_train(args: argparse.Namespace) -> None:
    """Train with SFT+KL."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = init_dist(local_rank)
    world_size = dist.get_world_size()

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    data_path = Path(args.data)
    if not data_path.exists():
        if rank == 0:
            print(f"Error: Data file not found: {data_path}")
            print("Run 'python sft_with_kl.py annotate' first to prepare data.")
        dist.destroy_process_group()
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # FSDP2 mixed precision policy
    mp_policy = fsdp.MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    # Load model for FSDP2
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Apply FSDP2 sharding
    blocks = None
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        blocks = model.model.layers
    elif hasattr(model, "layers"):
        blocks = model.layers
    if blocks is not None:
        for layer in blocks:
            fsdp.fully_shard(layer, mp_policy=mp_policy)
    fsdp.fully_shard(model, mp_policy=mp_policy)

    # Create imitation learning algorithm with KL regularization
    algo = make_sft_with_kl(
        kl_coeff=args.kl_coeff,
        length_normalize=True,
    )

    # Create step_to_item with reference logprob extraction
    step_to_item = make_chat_template_step_to_item(
        tokenizer,
        extract_ref_logprobs=True,  # Extract logprobs from trace.completion_logprobs
    )

    batch_source = OfflineBatchSource(
        jsonl_paths=[data_path],
        step_to_item=step_to_item,
        credit_assigner=algo.credit_assigner,
        batch_size=args.batch_size,
        shuffle=True,
    )

    # Calculate training steps
    batches_per_epoch = batch_source.num_batches_per_epoch
    total_steps = args.epochs * batches_per_epoch

    if rank == 0:
        print(f"=" * 60)
        print(f"Imitation Learning with KL Regularization")
        print(f"=" * 60)
        print(f"Model: {args.model}")
        print(f"Data: {data_path}")
        print(f"KL coefficient (β_SFT): {args.kl_coeff}")
        print(f"World size: {world_size}")
        print(f"Samples loaded: {len(batch_source)}")
        print(f"Batches per epoch: {batches_per_epoch}")
        print(f"Total steps: {total_steps} ({args.epochs} epochs)")
        print(f"=" * 60)

    cfg = TrainerConfig(
        model_device=str(device),
        max_seq_len=args.max_seq_len,
        micro_token_budget=args.micro_token_budget,
        max_grad_norm=args.max_grad_norm,
        pad_token_id=tokenizer,
        lr=args.lr,
        reduce_stats_across_ranks=True,
        eval_at_start=False,
        eval_every_n_steps=None,
        sync_every_steps=0,
    )

    checkpoint_cfg = None
    if args.checkpoint_every > 0:
        checkpoint_cfg = CheckpointConfig(
            output_dir=args.checkpoint_dir,
            every_n_steps=args.checkpoint_every,
            max_to_keep=args.max_to_keep,
            save_optimizer=True,
        )
        if rank == 0:
            Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        dist.barrier()

    trainer = Trainer(
        model=model,
        algo=algo,
        batch_source=batch_source,
        publisher=None,
        cfg=cfg,
        checkpoint_config=checkpoint_cfg,
        enable_gradient_checkpointing=bool(args.gradient_checkpointing),
        train_logger=None,
        evaluator=None,
    )

    async def train_loop() -> None:
        for step in range(total_steps):
            stats = await trainer.train_step()
            if rank == 0 and (step % args.log_every == 0):
                loss = stats.get("train/loss", float("nan"))
                sft_loss = stats.get("train/sft/loss", float("nan"))
                kl_loss = stats.get("train/kl/loss", float("nan"))
                kl_mean = stats.get("train/kl/kl_mean", float("nan"))
                print(
                    f"[step {step + 1}/{total_steps}] "
                    f"loss={loss:.4f} sft={sft_loss:.4f} kl={kl_loss:.4f} "
                    f"kl_mean={kl_mean:.4f}",
                    flush=True,
                )

    asyncio.run(train_loop())

    if rank == 0:
        print("Training complete!")
        if args.final_save:
            try:
                ckpt_path = trainer.save_checkpoint()
                print(f"Final checkpoint saved to: {ckpt_path}")
            except RuntimeError:
                pass

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Imitation Learning with KL Regularization"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Annotate command
    annotate_parser = subparsers.add_parser(
        "annotate",
        help="Annotate demonstration data with reference logprobs"
    )
    annotate_parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Base model to compute reference logprobs"
    )
    annotate_parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file with demonstrations"
    )
    annotate_parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL file with reference logprobs"
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train with SFT+KL"
    )
    train_parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    train_parser.add_argument("--data", required=True, help="JSONL file with annotated data")
    train_parser.add_argument("--kl-coeff", type=float, default=0.1,
                              help="KL regularization coefficient (β_SFT)")
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=2e-5)
    train_parser.add_argument("--max-seq-len", type=int, default=2048)
    train_parser.add_argument("--micro-token-budget", type=int, default=8192)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument("--checkpoint-dir", default="checkpoints_sft_kl")
    train_parser.add_argument("--checkpoint-every", type=int, default=100)
    train_parser.add_argument("--max-to-keep", type=int, default=2)
    train_parser.add_argument("--final-save", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--log-every", type=int, default=1)
    train_parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    if args.command == "annotate":
        cmd_annotate(args)
    elif args.command == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()
