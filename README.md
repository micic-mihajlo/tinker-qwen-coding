# Qwen3-30B-A3B Coding Model Training

Train Qwen3-30B-A3B as a coding assistant with tool-use capabilities using the Tinker API.

## Overview

This project implements a two-phase supervised learning approach:

1. **Phase 1 (Coding)**: Fine-tune on rStar-Coder dataset (592K competitive programming examples)
2. **Phase 2 (Tool Use)**: Continue fine-tuning on SWE-bench tool-use conversations (229 examples)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Tinker API key
export TINKER_API_KEY=<your-key>
```

## Usage

### Step 1: Prepare the Data

```bash
# Prepare rStar-Coder dataset (Phase 1)
python data/prepare_rstar.py

# Prepare SWE-bench tool-use dataset (Phase 2)
python data/prepare_swebench_tools.py
```

### Step 2: Run Phase 1 Training

```bash
python train_coding_sl.py
```

This will:
- Train on 592K coding examples from rStar-Coder
- Save checkpoints to `/tmp/tinker-training/phase1-coding/`
- Log metrics to `metrics.jsonl`

### Step 3: Run Phase 2 Training

```bash
python train_tool_use_sl.py
```

This will:
- Load the Phase 1 checkpoint
- Continue training on 229 tool-use examples
- Save final model to `/tmp/tinker-training/phase2-tooluse/`

## Configuration

Edit `config.py` to customize:

- Model name and LoRA rank
- Learning rate, batch size, epochs
- Data paths and max sequence length
- Checkpoint frequency

## File Structure

```
tinker-test/
├── config.py                    # Shared configuration
├── train_coding_sl.py           # Phase 1: Coding SL training
├── train_tool_use_sl.py         # Phase 2: Tool-use SL training
├── data/
│   ├── prepare_rstar.py         # rStar-Coder data processing
│   └── prepare_swebench_tools.py # SWE-bench data processing
├── requirements.txt
└── README.md
```

## Hyperparameters

| Parameter | Phase 1 | Phase 2 |
|-----------|---------|---------|
| Learning Rate | 5e-5 | 5e-5 |
| Batch Size | 128 | 32 |
| LoRA Rank | 32 | 32 |
| Max Tokens | 2048 | 8192 |
| Epochs | 1 | 3 |

## After Training

The final model weights are saved as a Tinker path. You can use it with the Tinker sampling API:

```python
import tinker

service_client = tinker.ServiceClient()
sampling_client = service_client.create_sampling_client(
    model_path="tinker://<your-model-path>"
)

# Sample from the model
result = sampling_client.sample(
    prompt=prompt,
    num_samples=1,
    sampling_params=tinker.types.SamplingParams(max_tokens=1024)
).result()
```

## Datasets

- **rStar-Coder**: [microsoft/rStar-Coder](https://huggingface.co/datasets/microsoft/rStar-Coder)
- **SWE-bench Tool-use**: [AlexCuadron/SWE-Bench-Verified-O1-native-tool-calling-reasoning-high-results](https://huggingface.co/datasets/AlexCuadron/SWE-Bench-Verified-O1-native-tool-calling-reasoning-high-results)
