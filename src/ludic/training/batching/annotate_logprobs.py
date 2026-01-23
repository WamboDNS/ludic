"""
Utilities for annotating demonstration data with reference policy logprobs.

For imitation learning with KL regularization (make_sft_with_kl), you need to
compute logprobs under the reference/base policy for each demonstrated action.
This module provides utilities to:

1. Compute per-token logprobs for a batch of sequences
2. Annotate SAWItems with reference logprobs
3. Process JSONL data files to add reference logprobs

Typical workflow:
    1. Collect demonstrations from a capable model
    2. Use compute_reference_logprobs() to compute base model logprobs
    3. Store in SAWItem.attachments.actor_logps
    4. Train with make_sft_with_kl()
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, TYPE_CHECKING

import torch
from torch import Tensor, nn

from ludic.training.types import SAWItem, SampleAttachments, ActorTokenLogps
from ludic.training.loss import selective_log_softmax

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass
class LogprobBatch:
    """Batch of sequences for logprob computation."""
    input_ids: Tensor  # [B, T]
    attention_mask: Tensor  # [B, T]
    action_mask: Tensor  # [B, T]


def compute_reference_logprobs(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    action_mask: Tensor,
    *,
    device: Optional[torch.device] = None,
) -> List[List[float]]:
    """
    Compute per-token log-probabilities under a reference policy.

    For each sequence, returns logprobs only for tokens where action_mask == 1.
    These can be stored in SAWItem.attachments.actor_logps for KL regularization.

    Args:
        model: The reference model (typically the base model before fine-tuning).
        input_ids: [B, T] token IDs.
        attention_mask: [B, T] attention mask (1 for real tokens).
        action_mask: [B, T] action mask (1 for action/completion tokens).
        device: Device to run on. If None, uses model's device.

    Returns:
        List of B lists, where each inner list contains logprobs for action tokens.
        Length of each inner list equals the number of 1s in the corresponding action_mask.

    Example:
        ```python
        # Prepare batch
        input_ids = tokenizer(texts, return_tensors="pt", padding=True)["input_ids"]
        attention_mask = ...  # from tokenizer
        action_mask = ...  # computed from prompt/completion split

        # Compute reference logprobs
        ref_logprobs = compute_reference_logprobs(
            model, input_ids, attention_mask, action_mask
        )

        # Store in SAWItem
        item.attachments = SampleAttachments(
            actor_logps=ActorTokenLogps(token_logps=ref_logprobs[0])
        )
        ```
    """
    if device is None:
        # Infer device from model parameters
        device = next(model.parameters()).device

    # Move tensors to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    action_mask = action_mask.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, V]

        # Shift for next-token prediction: logits[t] predicts input_ids[t+1]
        logits_shifted = logits[:, :-1, :]  # [B, T-1, V]
        targets = input_ids[:, 1:]  # [B, T-1]
        action_mask_shifted = action_mask[:, 1:]  # [B, T-1]

        # Get log-probs for the actual tokens
        token_logps = selective_log_softmax(logits_shifted, targets)  # [B, T-1]

    # Extract logprobs for action tokens only
    result: List[List[float]] = []
    for b in range(input_ids.size(0)):
        mask = action_mask_shifted[b] > 0
        logps = token_logps[b][mask].cpu().tolist()
        result.append(logps)

    return result


def annotate_saw_item_with_ref_logprobs(
    item: SAWItem,
    model: nn.Module,
    *,
    device: Optional[torch.device] = None,
) -> SAWItem:
    """
    Add reference policy logprobs to a SAWItem.

    Computes logprobs under the given model and stores them in
    item.attachments.actor_logps. This is the format expected by
    make_sft_with_kl() for KL regularization.

    Args:
        item: SAWItem to annotate.
        model: Reference model to compute logprobs.
        device: Device to run on.

    Returns:
        New SAWItem with actor_logps populated.
    """
    input_ids = torch.tensor([item.input_ids], dtype=torch.long)
    attention_mask = torch.tensor([item.attention_mask], dtype=torch.long)
    action_mask = torch.tensor([item.action_mask], dtype=torch.float32)

    ref_logprobs = compute_reference_logprobs(
        model, input_ids, attention_mask, action_mask, device=device
    )

    attachments = SampleAttachments(
        actor_logps=ActorTokenLogps(token_logps=ref_logprobs[0])
    )

    return SAWItem(
        input_ids=item.input_ids,
        attention_mask=item.attention_mask,
        action_mask=item.action_mask,
        weight=item.weight,
        meta=item.meta,
        attachments=attachments,
    )


def annotate_saw_items_batch(
    items: List[SAWItem],
    model: nn.Module,
    *,
    device: Optional[torch.device] = None,
    pad_token_id: int = 0,
) -> List[SAWItem]:
    """
    Batch-annotate SAWItems with reference policy logprobs.

    More efficient than calling annotate_saw_item_with_ref_logprobs() one by one.

    Args:
        items: List of SAWItems to annotate.
        model: Reference model to compute logprobs.
        device: Device to run on.
        pad_token_id: Token ID for padding.

    Returns:
        List of new SAWItems with actor_logps populated.
    """
    if not items:
        return []

    # Collate into padded tensors
    max_len = max(len(it.input_ids) for it in items)
    batch_size = len(items)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    action_mask = torch.zeros((batch_size, max_len), dtype=torch.float32)

    for b, it in enumerate(items):
        L = len(it.input_ids)
        input_ids[b, :L] = torch.tensor(it.input_ids, dtype=torch.long)
        attention_mask[b, :L] = torch.tensor(it.attention_mask, dtype=torch.long)
        action_mask[b, :L] = torch.tensor(it.action_mask, dtype=torch.float32)

    # Compute logprobs
    ref_logprobs = compute_reference_logprobs(
        model, input_ids, attention_mask, action_mask, device=device
    )

    # Create annotated items
    result: List[SAWItem] = []
    for it, logps in zip(items, ref_logprobs):
        attachments = SampleAttachments(
            actor_logps=ActorTokenLogps(token_logps=logps)
        )
        result.append(SAWItem(
            input_ids=it.input_ids,
            attention_mask=it.attention_mask,
            action_mask=it.action_mask,
            weight=it.weight,
            meta=it.meta,
            attachments=attachments,
        ))

    return result


def iter_jsonl_rollouts(path: Path) -> Iterator[dict]:
    """Iterate over rollouts in a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def annotate_jsonl_with_ref_logprobs(
    input_path: Path,
    output_path: Path,
    model: nn.Module,
    tokenizer: "PreTrainedTokenizerBase",
    *,
    device: Optional[torch.device] = None,
    batch_size: int = 8,
) -> int:
    """
    Annotate a JSONL file with reference policy logprobs.

    Reads rollouts from input_path, computes reference logprobs for each step,
    and writes annotated rollouts to output_path.

    The logprobs are stored in step["trace"]["completion_logprobs"], which
    gets loaded into SAWItem.attachments.actor_logps by OfflineBatchSource.

    Args:
        input_path: Path to input JSONL with demonstrations.
        output_path: Path to write annotated JSONL.
        model: Reference model to compute logprobs.
        tokenizer: Tokenizer for the model.
        device: Device to run on.
        batch_size: Batch size for processing.

    Returns:
        Number of steps annotated.

    Example:
        ```python
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

        annotate_jsonl_with_ref_logprobs(
            Path("demos_raw.jsonl"),
            Path("demos_with_logprobs.jsonl"),
            model,
            tokenizer,
            device=torch.device("cuda"),
        )
        ```
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_steps = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for rollout in iter_jsonl_rollouts(input_path):
            # Process each step
            for step in rollout.get("steps", []):
                info = step.get("info", {})
                chat_messages = info.get("chat_prompt_messages", [])
                chat_completion = info.get("chat_completion", {})

                if not chat_messages:
                    continue

                # Build completion message
                if chat_completion and isinstance(chat_completion, dict):
                    completion_msg = chat_completion
                else:
                    completion_msg = {"role": "assistant", "content": step.get("action", "")}

                # Tokenize prompt and full sequence
                prompt_ids: List[int] = tokenizer.apply_chat_template(
                    chat_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                )
                full_ids: List[int] = tokenizer.apply_chat_template(
                    chat_messages + [completion_msg],
                    add_generation_prompt=False,
                    tokenize=True,
                )

                prompt_ids = list(prompt_ids)
                full_ids = list(full_ids)

                if full_ids[:len(prompt_ids)] != prompt_ids:
                    # Alignment issue, skip this step
                    continue

                action_ids = full_ids[len(prompt_ids):]
                if not action_ids:
                    continue

                # Build tensors
                input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
                attention_mask = torch.ones_like(input_ids)
                action_mask = torch.zeros_like(input_ids, dtype=torch.float32)
                action_mask[0, len(prompt_ids):] = 1.0

                # Compute logprobs
                ref_logprobs = compute_reference_logprobs(
                    model, input_ids, attention_mask, action_mask, device=device
                )

                # Store in trace
                trace = step.get("trace", {})
                trace["completion_logprobs"] = ref_logprobs[0]
                step["trace"] = trace

                total_steps += 1

            # Write annotated rollout
            out_f.write(json.dumps(rollout) + "\n")

    return total_steps
