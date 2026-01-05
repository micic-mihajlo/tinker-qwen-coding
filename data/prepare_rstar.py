"""
Prepare rStar-Coder dataset for training.

Converts the rStar-Coder dataset to conversation format suitable for Tinker SL training.

Dataset: microsoft/rStar-Coder
- seed_sft: 592K competitive programming examples with verified solutions
- synthetic_sft: 398K additional synthetic examples

Output: JSONL file with conversation format:
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
"""

import json
import os
import sys
from pathlib import Path
from typing import Iterator

from datasets import load_dataset
from tqdm import tqdm

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config, Phase1Config


def rstar_to_conversation(row: dict) -> dict:
    """
    Convert a rStar-Coder row to conversation format.

    Args:
        row: A row from the rStar-Coder dataset containing:
            - question: The problem statement
            - response: The model's reasoning/explanation
            - code: The solution code
            - verified: Whether the solution is verified
            - is_passed: Whether the solution passes tests

    Returns:
        A dictionary with 'messages' key containing the conversation.
    """
    question = row.get("question", "")
    response = row.get("response", "")
    code = row.get("code", "")

    # Build assistant response with reasoning and code
    if response and code:
        assistant_content = f"{response}\n\n```python\n{code}\n```"
    elif code:
        assistant_content = f"```python\n{code}\n```"
    elif response:
        assistant_content = response
    else:
        return None  # Skip if no response or code

    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_content}
        ]
    }


def process_dataset(
    config: Phase1Config,
    filter_verified: bool = True,
    filter_passed: bool = True
) -> Iterator[dict]:
    """
    Load and process the rStar-Coder dataset.

    Args:
        config: Phase1 configuration
        filter_verified: Only include verified examples
        filter_passed: Only include examples that passed tests

    Yields:
        Conversation dictionaries
    """
    print(f"Loading dataset: {config.dataset_name} ({config.dataset_config})")
    dataset = load_dataset(config.dataset_name, config.dataset_config, split="train")

    print(f"Total examples in dataset: {len(dataset)}")

    # Apply filters if the columns exist
    if filter_verified and "verified" in dataset.column_names:
        dataset = dataset.filter(lambda x: x.get("verified", True))
        print(f"After verified filter: {len(dataset)}")

    if filter_passed and "is_passed" in dataset.column_names:
        dataset = dataset.filter(lambda x: x.get("is_passed", True))
        print(f"After is_passed filter: {len(dataset)}")

    # Limit samples if configured
    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
        print(f"After max_samples limit: {len(dataset)}")

    # Convert to conversation format
    for row in tqdm(dataset, desc="Converting to conversations"):
        conv = rstar_to_conversation(row)
        if conv:
            yield conv


def main():
    """Main function to prepare the dataset."""
    config = get_config()
    phase1 = config.phase1

    # Create output directory
    output_path = Path(phase1.data_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Output path: {output_path}")
    print(f"Model: {config.model.name}")
    print()

    # Process and save
    count = 0
    with open(output_path, "w") as f:
        for conv in process_dataset(phase1):
            f.write(json.dumps(conv) + "\n")
            count += 1

    print(f"\nSaved {count} conversations to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
