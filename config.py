"""
Shared configuration for Qwen3-30B-A3B coding model training.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "Qwen/Qwen3-30B-A3B"
    lora_rank: int = 32


@dataclass
class Phase1Config:
    """Phase 1: Coding SL training on rStar-Coder."""
    log_path: str = "/tmp/tinker-training/phase1-coding"
    data_path: str = "/tmp/tinker-training/data/rstar_conversations.jsonl"

    # Training hyperparameters
    learning_rate: float = 5e-5
    batch_size: int = 128
    max_length: int = 2048
    num_epochs: int = 1
    eval_every: int = 50
    save_every: int = 100

    # Dataset config
    dataset_name: str = "microsoft/rStar-Coder"
    dataset_config: str = "seed_sft"
    max_samples: Optional[int] = None  # None = use all


@dataclass
class Phase2Config:
    """Phase 2: Tool-use SL training on SWE-bench data."""
    log_path: str = "/tmp/tinker-training/phase2-tooluse"
    data_path: str = "/tmp/tinker-training/data/swebench_tool_conversations.jsonl"

    # Training hyperparameters
    learning_rate: float = 5e-5
    batch_size: int = 32  # Smaller due to smaller dataset
    max_length: int = 8192  # Longer for multi-turn tool-use
    num_epochs: int = 3  # More epochs for small dataset
    eval_every: int = 10
    save_every: int = 20

    # Dataset config
    dataset_name: str = "AlexCuadron/SWE-Bench-Verified-O1-native-tool-calling-reasoning-high-results"
    filter_success: bool = True  # Only use successful examples


@dataclass
class Config:
    """Main configuration combining all phases."""
    model: ModelConfig = field(default_factory=ModelConfig)
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)

    # API configuration
    tinker_api_key: Optional[str] = field(
        default_factory=lambda: os.environ.get("TINKER_API_KEY")
    )

    def __post_init__(self):
        if not self.tinker_api_key:
            raise ValueError(
                "TINKER_API_KEY environment variable must be set. "
                "Run: export TINKER_API_KEY=<your-key>"
            )


def get_config() -> Config:
    """Get the default configuration."""
    return Config()


if __name__ == "__main__":
    # Print config for verification
    config = get_config()
    print(f"Model: {config.model.name}")
    print(f"LoRA Rank: {config.model.lora_rank}")
    print(f"\nPhase 1 (Coding):")
    print(f"  Dataset: {config.phase1.dataset_name}")
    print(f"  Batch Size: {config.phase1.batch_size}")
    print(f"  Learning Rate: {config.phase1.learning_rate}")
    print(f"\nPhase 2 (Tool Use):")
    print(f"  Dataset: {config.phase2.dataset_name}")
    print(f"  Batch Size: {config.phase2.batch_size}")
    print(f"  Learning Rate: {config.phase2.learning_rate}")
