# VizSage - Vision Language Model Fine-tuning

A comprehensive framework for fine-tuning vision-language models using LoRA (Low-Rank Adaptation) with support for streaming datasets, advanced evaluation metrics, and robust training pipelines.

## 🚀 Features

- **Multi-modal Training**: Fine-tune vision-language models on image-text pairs
- **Streaming Support**: Handle large datasets efficiently with streaming data loading
- **Advanced Evaluation**: Configurable text normalization and exact match evaluation
- **LoRA Fine-tuning**: Memory-efficient training with Low-Rank Adaptation
- **External Knowledge**: Support for external knowledge integration
- **Robust Training**: Early stopping, memory management, and comprehensive logging
- **Wandb Integration**: Full experiment tracking and monitoring

## 📁 Project Structure

```
VizSage/
├── src/                           # Source code
│   ├── train.py                   # Main training script (unified)
│   ├── test.py                    # Testing and evaluation script
│   ├── config_utils.py            # Configuration utilities
│   ├── data_utils.py              # Data processing utilities
│   ├── model.py                   # Model utilities
│   ├── evaluation_utils.py        # Evaluation metrics and functions
│   ├── training_utils.py          # Training callbacks and utilities
│   └── formatting_utils.py        # Text formatting utilities
├── config/                        # Configuration files
│   ├── example_config.yaml        # Example configuration
│   └── debug_config.yaml          # Debug configuration
├── data/                          # Dataset directory
│   ├── AQUA/                      # Dataset files
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   ├── Images/                    # Image files
│   └── semart.csv                 # External knowledge (optional)
├── outputs/                       # Training outputs
└── README.md                      # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ GPU memory for 11B models

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd VizSage
```

2. **Install dependencies:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
pip install transformers datasets accelerate peft trl
pip install wandb pyyaml tqdm pandas pillow
```

3. **Prepare your dataset:**
```bash
# Place your dataset in the data/ directory
mkdir -p data/AQUA data/Images
# Copy your JSON files and images
```

## 🎯 Quick Start

### Basic Training

1. **Configure your training:**
```bash
# Copy and edit the example configuration
cp config/example_config.yaml config/my_config.yaml
# Edit my_config.yaml with your settings
```

2. **Start training:**
```bash
cd src
python train.py ../config/my_config.yaml
```

### Testing Your Model

```bash
cd src
python test.py ../config/my_config.yaml
```

## ⚙️ Configuration

### Key Configuration Parameters

#### Model Settings
```yaml
model_name: "unsloth/Llama-3.2-11B-Vision-Instruct"
load_in_4bit: true
use_gradient_checkpointing: "unsloth"
```

#### LoRA Parameters
```yaml
lora_r: 32                    # LoRA rank
lora_alpha: 32                # LoRA alpha
lora_dropout: 0.05            # LoRA dropout
bias: "none"                  # Bias configuration
```

#### Training Parameters
```yaml
batch_size: 2                 # Per-device batch size
grad_accum: 4                 # Gradient accumulation steps
epochs: 3                     # Number of training epochs
lr: 2e-4                      # Learning rate
max_seq_length: 2048          # Maximum sequence length
```

#### Evaluation Settings (New!)
```yaml
use_text_normalization: true  # SQuAD-style text normalization
debug_exact_match: false      # Show detailed evaluation debug
```

#### Early Stopping
```yaml
use_early_stopping: true      # Enable early stopping
early_stopping_patience_ratio: 0.1  # Patience as ratio of total steps
early_stopping_threshold: 0.001     # Minimum improvement threshold
```

#### Dataset Configuration
```yaml
dataset: "AQUA"               # Dataset name
base_path: "data"             # Base data directory
use_streaming: false          # Use streaming for large datasets
external_knowledge: false     # Use external knowledge integration
```

### Full Configuration Example

```yaml
# Model parameters
model_name: "unsloth/Llama-3.2-11B-Vision-Instruct"
load_in_4bit: true
name_trained_model: "VizSage_final_model"

# LoRA parameters
lora_r: 32
lora_alpha: 32
lora_dropout: 0.05
bias: "none"

# Training parameters
batch_size: 2
grad_accum: 4
epochs: 3
lr: 2e-4
use_bf16: true

# Evaluation parameters
use_text_normalization: true
debug_exact_match: false

# Early stopping
use_early_stopping: true
early_stopping_patience_ratio: 0.1

# Dataset
dataset: "AQUA"
base_path: "data"
use_streaming: false

# Logging
use_wandb: true
wandb_project: "vizsage-training"

# Output
output_dir: "outputs"
```

## 📊 Evaluation Metrics

The framework supports two evaluation modes:

### 1. Standard Research Evaluation (Recommended)
```yaml
use_text_normalization: true
```
- Applies SQuAD-style text normalization
- Removes articles (a, an, the)
- Removes punctuation
- Normalizes whitespace and case
- **Compatible with research standards**

### 2. Rigorous Evaluation
```yaml
use_text_normalization: false
```
- Exact character-by-character matching
- No text transformations
- **More strict evaluation**

### Debug Mode
```yaml
debug_exact_match: true
```
Shows detailed comparison for the first 5 evaluation samples:
```
--- DEBUG SAMPLE 1 (WITH normalization) ---
Original Prediction: 'The sky is blue.'
Normalized Pred:    'sky is blue'
Original Label:     'The sky is blue!'
Normalized Label:   'sky is blue'
Match: True
```

## 🔄 Training Modes

### Regular Training
For smaller datasets that fit in memory:
```yaml
use_streaming: false
```

### Streaming Training
For large datasets:
```yaml
use_streaming: true
stream_buffer_size: 1000
```

## 📈 Monitoring and Logging

### Wandb Integration
```yaml
use_wandb: true
wandb_project: "your-project"
wandb_run_name: "experiment-1"  # Optional
wandb_tags: ["vision", "llama", "fine-tuning"]
```

### Local Logging
- Training logs saved to `outputs/`
- Model checkpoints saved automatically
- Detailed evaluation results in JSON format

## 🧪 Testing and Evaluation

### Run Full Evaluation
```bash
cd src
python test.py ../config/my_config.yaml
```

### Evaluation Features
- **Comprehensive metrics**: Exact match, breakdown by external knowledge
- **Sample preview**: Shows individual predictions vs ground truth
- **Error analysis**: Detailed error reporting
- **Results saving**: JSON format for further analysis

### Example Test Output
```
📊 EVALUATION RESULTS (WITH normalization)
============================================================
Total Samples: 1000
Correct Predictions: 847
Incorrect Predictions: 153
Exact Match Score: 0.8470
Exact Match Percentage: 84.70%

🧠 With External Knowledge:
  • Samples: 450
  • Exact Match: 82.22%

📝 Without External Knowledge:
  • Samples: 550
  • Exact Match: 86.73%
```

## 🔧 Advanced Features

### Memory Management
- Automatic GPU memory clearing
- Memory usage monitoring
- Gradient checkpointing support

### Early Stopping
```yaml
use_early_stopping: true
early_stopping_patience_ratio: 0.1  # Stop if no improvement in 10% of steps
early_stopping_threshold: 0.001     # Minimum improvement required
```

### External Knowledge Integration
```yaml
external_knowledge: true
external_knowledge_path: "data/semart.csv"
```

## 🐛 Troubleshooting

### Common Issues

**Unsloth Logits Error (2024.11+)**
The framework automatically sets `UNSLOTH_RETURN_LOGITS=1` for evaluation compatibility. If you still encounter logits errors, manually set:
```bash
export UNSLOTH_RETURN_LOGITS=1
```

**Out of Memory (OOM)**
- Reduce `batch_size`
- Increase `grad_accum` to maintain effective batch size
- Use `load_in_4bit: true`
- Enable gradient checkpointing

**Slow Training**
- Enable `use_bf16: true` if supported
- Use streaming for large datasets
- Optimize `num_proc` for your system

**Validation Issues**
- Check dataset paths in configuration
- Ensure image files exist in `data/Images/`
- Validate JSON file structure

### Debug Mode
Enable debug mode for troubleshooting:
```yaml
debug_exact_match: true
logging_steps: 1
batch_size: 1  # For debugging
epochs: 1      # Quick test
```

## 📋 Dataset Format

### JSON Structure
```json
[
  {
    "image": "image001.jpg",
    "question": "What is shown in this image?",
    "answer": "A beautiful landscape",
    "need_external_knowledge": false
  }
]
```

### Directory Structure
```
data/
├── AQUA/
│   ├── train.json
│   ├── val.json
│   └── test.json
└── Images/
    ├── image001.jpg
    ├── image002.jpg
    └── ...
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for efficient model training
- [Hugging Face](https://huggingface.co/) for transformers and datasets
- [Weights & Biases](https://wandb.ai/) for experiment tracking

---