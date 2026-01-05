"""
Prepare SWE-Bench tool-use dataset for training.

Converts the SWE-Bench-Verified-O1 dataset to conversation format suitable for Tinker SL training.

Dataset: AlexCuadron/SWE-Bench-Verified-O1-native-tool-calling-reasoning-high-results
- 500 SWE-bench issues with O1 reasoning traces
- Multi-turn conversations (5-66 turns each) with CodeAct tool calling
- Filter for success=True to get 229 high-quality examples

Output: JSONL file with conversation format including tool calls.
"""

import json
import os
import sys
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional

from datasets import load_dataset
from tqdm import tqdm

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config, Phase2Config


def parse_codeact_message(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse a CodeAct message into standard chat format.

    CodeAct format uses structured content with type annotations.
    We convert this to standard role/content format with tool_calls.

    Args:
        msg: A message from the CodeAct conversation

    Returns:
        A standardized message dict or None if invalid
    """
    # Handle different message formats
    if isinstance(msg, str):
        # Plain string message
        return {"role": "user", "content": msg}

    if not isinstance(msg, dict):
        return None

    # Extract role - CodeAct may use different field names
    role = msg.get("role", msg.get("type", "assistant"))

    # Map roles
    role_map = {
        "user": "user",
        "assistant": "assistant",
        "tool": "tool",
        "system": "system",
        "observation": "tool",  # CodeAct uses "observation" for tool results
        "action": "assistant",  # CodeAct uses "action" for assistant tool calls
    }
    role = role_map.get(role, role)

    # Extract content
    content = msg.get("content", msg.get("text", ""))

    # Handle content that's a list of parts
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "tool_use":
                    # Tool call - add to content or extract separately
                    tool_name = part.get("name", "unknown_tool")
                    tool_input = part.get("input", {})
                    text_parts.append(f"[Tool: {tool_name}]\n{json.dumps(tool_input, indent=2)}")
                elif part.get("type") == "tool_result":
                    text_parts.append(f"[Tool Result]\n{part.get('content', '')}")
            elif isinstance(part, str):
                text_parts.append(part)
        content = "\n".join(text_parts)

    if not content:
        return None

    return {"role": role, "content": content}


def parse_conversation(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse a full SWE-bench conversation from the dataset.

    The dataset format is:
    - full_conversation_jsonl: JSON array of turns, each with:
      - messages: list of message dicts with 'content' (list of parts)
      - response: the assistant response
      - args, kwargs: tool arguments
      - timestamp, cost: metadata

    Args:
        row: A row from the dataset

    Returns:
        A dictionary with 'messages' key containing the conversation,
        or None if parsing fails.
    """
    conv_str = row.get("full_conversation_jsonl", "")
    if not conv_str:
        return None

    try:
        # Parse the conversation as JSON array
        turns = json.loads(conv_str)
        if not isinstance(turns, list):
            return None

        messages = []

        for turn in turns:
            if not isinstance(turn, dict):
                continue

            # Extract messages from the turn
            turn_messages = turn.get("messages", [])
            for msg in turn_messages:
                if not isinstance(msg, dict):
                    continue

                # Get content - it's a list of content parts
                content_parts = msg.get("content", [])
                if isinstance(content_parts, str):
                    content = content_parts
                elif isinstance(content_parts, list):
                    # Extract text from content parts
                    text_parts = []
                    for part in content_parts:
                        if isinstance(part, dict):
                            if part.get("type") == "text":
                                text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    content = "\n".join(text_parts)
                else:
                    continue

                if not content.strip():
                    continue

                # Determine role - OpenHands uses specific formatting
                role = msg.get("role", "assistant")
                if role not in ["user", "assistant", "system"]:
                    role = "assistant"

                messages.append({"role": role, "content": content.strip()})

            # Also extract the response if present
            response = turn.get("response", "")
            if response and isinstance(response, str) and response.strip():
                messages.append({"role": "assistant", "content": response.strip()})

        if not messages:
            return None

        # Clean up consecutive same-role messages by merging
        cleaned_messages = []
        for msg in messages:
            if cleaned_messages and cleaned_messages[-1]["role"] == msg["role"]:
                # Merge with previous message
                cleaned_messages[-1]["content"] += "\n\n" + msg["content"]
            else:
                cleaned_messages.append(msg)

        if not cleaned_messages:
            return None

        # Add system message with context about the issue
        issue_name = row.get("issue_name", "Unknown Issue")
        project = row.get("project", "Unknown Project")

        system_msg = {
            "role": "system",
            "content": (
                f"You are a helpful coding assistant working on issue {issue_name} "
                f"in project {project}. You have access to tools for reading and writing files, "
                f"executing bash commands, and searching code. Use these tools to fix the issue."
            )
        }

        return {
            "messages": [system_msg] + cleaned_messages,
            "metadata": {
                "issue_name": issue_name,
                "project": project,
                "resolved": row.get("resolved", False),
                "num_turns": row.get("num_turns", len(cleaned_messages)),
            }
        }

    except Exception as e:
        print(f"Error parsing conversation for {row.get('issue_name', 'unknown')}: {e}")
        return None


def process_dataset(config: Phase2Config) -> Iterator[Dict[str, Any]]:
    """
    Load and process the SWE-bench tool-use dataset.

    Args:
        config: Phase2 configuration

    Yields:
        Conversation dictionaries
    """
    print(f"Loading dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name, split="test")

    print(f"Total examples in dataset: {len(dataset)}")

    # Filter for resolved examples if configured
    # Note: 'success' is always False in this dataset, use 'resolved' instead
    if config.filter_success:
        dataset = dataset.filter(lambda x: x.get("resolved", False) == True)
        print(f"After resolved filter: {len(dataset)}")

    # Convert to conversation format
    for row in tqdm(dataset, desc="Converting to conversations"):
        conv = parse_conversation(row)
        if conv:
            # Remove metadata for the training file (keep messages only)
            yield {"messages": conv["messages"]}


def main():
    """Main function to prepare the dataset."""
    config = get_config()
    phase2 = config.phase2

    # Create output directory
    output_path = Path(phase2.data_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Output path: {output_path}")
    print(f"Model: {config.model.name}")
    print()

    # Process and save
    count = 0
    total_turns = 0
    with open(output_path, "w") as f:
        for conv in process_dataset(phase2):
            f.write(json.dumps(conv) + "\n")
            count += 1
            total_turns += len(conv["messages"])

    print(f"\nSaved {count} conversations to {output_path}")
    if count > 0:
        print(f"Average turns per conversation: {total_turns / count:.1f}")
        print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print("Warning: No conversations were extracted. Check the data format.")


if __name__ == "__main__":
    main()
