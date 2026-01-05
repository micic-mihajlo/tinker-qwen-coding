"""
Phase 1: Supervised Learning on rStar-Coder dataset.

This script fine-tunes Qwen3-30B-A3B on competitive programming examples
using the Tinker API.

Usage:
    # First, prepare the data
    python data/prepare_rstar.py

    # Then run training
    python train_coding_sl.py
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import tinker
from tinker import types
from tinker.types.tensor_data import TensorData
from transformers import AutoTokenizer
import numpy as np

from config import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_conversations(data_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load conversations from JSONL file."""
    conversations = []
    with open(data_path, "r") as f:
        for line in f:
            conversations.append(json.loads(line))
            if max_samples and len(conversations) >= max_samples:
                break
    return conversations


def conversation_to_datum(
    conversation: Dict,
    tokenizer,
    max_length: int = 2048
) -> Optional[types.Datum]:
    """
    Convert a conversation to a Tinker Datum for training.

    Args:
        conversation: Dict with 'messages' key
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length

    Returns:
        A Tinker Datum or None if conversion fails
    """
    messages = conversation.get("messages", [])
    if not messages:
        return None

    # Build the full text using chat template
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    except Exception as e:
        # Fallback for models without chat template
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
        text = "\n".join(parts)

    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=True)

    # Truncate if needed
    if len(tokens) > max_length:
        tokens = tokens[:max_length]

    if len(tokens) < 2:
        return None

    # For SL, we train on assistant responses
    # Find assistant turn boundaries and create weights
    weights = [0.0] * (len(tokens) - 1)  # weights for target tokens

    # Simple heuristic: weight everything after "assistant" tokens
    # For proper implementation, use the renderer from tinker_cookbook
    text_so_far = ""
    in_assistant = False
    for i, token in enumerate(tokens[:-1]):
        decoded = tokenizer.decode([token])
        text_so_far += decoded

        # Check if we're entering or leaving assistant turn
        if "assistant" in text_so_far[-20:].lower() and "<|im_start|>" in text_so_far[-30:]:
            in_assistant = True
        elif "<|im_end|>" in decoded:
            in_assistant = False
        elif "<|im_start|>" in decoded and "assistant" not in text_so_far[-30:].lower():
            in_assistant = False

        if in_assistant:
            weights[i] = 1.0

    # If no assistant weights found, train on everything after first message
    if sum(weights) == 0:
        # Find first assistant response
        for i in range(len(weights) // 4, len(weights)):
            weights[i] = 1.0

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_numpy(np.array(target_tokens, dtype=np.int64)),
            "weights": TensorData.from_numpy(np.array(weights, dtype=np.float32)),
        }
    )


def compute_mean_nll(logprobs: List, weights: List) -> float:
    """Compute mean negative log-likelihood."""
    total_loss = 0.0
    total_weight = 0.0
    for lp, w in zip(logprobs, weights):
        lp_arr = np.array(lp) if not isinstance(lp, np.ndarray) else lp
        w_arr = np.array(w) if not isinstance(w, np.ndarray) else w
        total_loss += float(np.sum(-lp_arr * w_arr))
        total_weight += float(np.sum(w_arr))
    if total_weight == 0:
        return 0.0
    return total_loss / total_weight


async def train(config):
    """Main training loop."""
    phase1 = config.phase1

    # Check if data exists
    data_path = Path(phase1.data_path)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        logger.error("Run 'python data/prepare_rstar.py' first to prepare the data.")
        sys.exit(1)

    # Create log directory
    log_path = Path(phase1.log_path)
    log_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Logging to: {log_path}")

    # Load tokenizer
    logger.info(f"Loading tokenizer for {config.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=True)

    # Load conversations
    logger.info(f"Loading conversations from {data_path}")
    conversations = load_conversations(
        str(data_path),
        max_samples=phase1.max_samples
    )
    logger.info(f"Loaded {len(conversations)} conversations")

    # Convert to datums
    logger.info("Converting to training datums...")
    datums = []
    for conv in conversations:
        datum = conversation_to_datum(conv, tokenizer, phase1.max_length)
        if datum:
            datums.append(datum)
    logger.info(f"Created {len(datums)} training datums")

    if not datums:
        logger.error("No valid training datums created!")
        sys.exit(1)

    # Calculate number of batches
    n_batches = len(datums) // phase1.batch_size
    total_steps = n_batches * phase1.num_epochs
    logger.info(f"Training for {phase1.num_epochs} epochs = {total_steps} steps")

    # Create Tinker client
    logger.info("Creating Tinker training client...")
    service_client = tinker.ServiceClient()

    # Check if resuming from checkpoint
    checkpoint_file = log_path / "checkpoints.jsonl"
    resume_path = None
    start_step = 0

    if checkpoint_file.exists():
        with open(checkpoint_file, "r") as f:
            lines = f.readlines()
            if lines:
                last_checkpoint = json.loads(lines[-1])
                resume_path = last_checkpoint.get("state_path")
                start_step = last_checkpoint.get("step", 0) + 1
                logger.info(f"Resuming from step {start_step}, checkpoint: {resume_path}")

    if resume_path:
        training_client = await service_client.create_training_client_from_state_with_optimizer_async(
            path=resume_path
        )
    else:
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model.name,
            rank=config.model.lora_rank
        )

    # Training metrics file
    metrics_file = log_path / "metrics.jsonl"

    # Training loop
    logger.info("Starting training...")
    step = start_step

    for epoch in range(phase1.num_epochs):
        # Shuffle datums each epoch
        np.random.shuffle(datums)

        for batch_idx in range(n_batches):
            if step < start_step:
                step += 1
                continue

            start_time = time.time()

            # Get batch
            batch_start = batch_idx * phase1.batch_size
            batch_end = batch_start + phase1.batch_size
            batch = datums[batch_start:batch_end]

            # Linear learning rate schedule
            lr_mult = max(0.0, 1.0 - step / total_steps)
            current_lr = phase1.learning_rate * lr_mult

            adam_params = types.AdamParams(
                learning_rate=current_lr,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8
            )

            # Forward-backward pass
            fwd_bwd_future = await training_client.forward_backward_async(
                batch,
                loss_fn="cross_entropy"
            )

            # Optimizer step
            optim_future = await training_client.optim_step_async(adam_params)

            # Get results
            fwd_bwd_result = await fwd_bwd_future
            await optim_future

            # Compute metrics
            train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
            train_weights = [d.loss_fn_inputs["weights"] for d in batch]
            train_nll = compute_mean_nll(train_logprobs, train_weights)

            step_time = time.time() - start_time

            # Log metrics
            metrics = {
                "step": step,
                "epoch": epoch,
                "train_nll": train_nll,
                "learning_rate": current_lr,
                "batch_size": len(batch),
                "step_time": step_time,
            }

            with open(metrics_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")

            if step % 10 == 0:
                logger.info(
                    f"Step {step}/{total_steps} | "
                    f"Epoch {epoch+1}/{phase1.num_epochs} | "
                    f"NLL: {train_nll:.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Time: {step_time:.2f}s"
                )

            # Save checkpoint
            if phase1.save_every > 0 and step % phase1.save_every == 0 and step > 0:
                logger.info(f"Saving checkpoint at step {step}...")
                save_result = await training_client.save_state_async(name=f"{step:06d}")
                save_path = (await save_result).path

                with open(checkpoint_file, "a") as f:
                    f.write(json.dumps({
                        "step": step,
                        "state_path": save_path,
                        "train_nll": train_nll,
                    }) + "\n")

                logger.info(f"Saved checkpoint: {save_path}")

            step += 1

    # Save final checkpoint
    logger.info("Saving final checkpoint...")
    final_save = await training_client.save_state_async(name="final")
    final_path = (await final_save).path

    with open(checkpoint_file, "a") as f:
        f.write(json.dumps({
            "step": step,
            "state_path": final_path,
            "final": True,
        }) + "\n")

    # Also save weights for sampling
    sampler_save = await training_client.save_weights_for_sampler_async(name="final-sampler")
    sampler_path = (await sampler_save).path

    logger.info(f"Final checkpoint: {final_path}")
    logger.info(f"Sampler weights: {sampler_path}")
    logger.info("Phase 1 training complete!")

    # Save paths for Phase 2
    paths_file = log_path / "paths.json"
    with open(paths_file, "w") as f:
        json.dump({
            "final_state_path": final_path,
            "sampler_path": sampler_path,
        }, f, indent=2)

    logger.info(f"Paths saved to: {paths_file}")


def main():
    """Entry point."""
    config = get_config()
    asyncio.run(train(config))


if __name__ == "__main__":
    main()
