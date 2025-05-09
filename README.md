# VizSage: Vision-Language Model Fine-tuning Framework

VizSage is a streamlined framework for fine-tuning vision-language models (VLMs) with a focus on simplicity, flexibility, and reproducibility. The project provides an easy way to customize model parameters and training configurations through a simple YAML file.

## 🌟 Features

- **Easy Configuration**: Fine-tune models by modifying a single YAML file without changing code
- **Vision Model Support**: Built specifically for vision-language models like Llama-3-Vision, Qwen2-VL, and others
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning through LoRA
- **Memory Optimization**: Process large datasets in manageable chunks
- **Experiment Tracking**: Optional Weights & Biases integration for tracking experiments
- **Reproducibility**: Save and reuse configurations for consistent results
- **Validation Support**: Automatically use validation sets to select the best model
- **Advanced Metrics**: Comprehensive evaluation with BLEU, ROUGE, and exact match metrics
- **Detailed Reports**: Organized results with visual charts and summaries

## 🚀 Getting Started

### Prerequisites

- Python 3.10
- CUDA-compatible GPU with 16GB+ VRAM (recommended)
- Hugging Face API key (required for accessing model weights)
- Weights & Biases API key (optional, for experiment tracking)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bianccc01/vizsage.git
   cd vizsage
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create your `.env`:
   ```bash
   touch .env
   ```

4. Edit the `.env` file to add your API keys:
   ```
   HF_TOKEN=your_hf_token_here  # Required for downloading models
   WANDB_API_KEY=your_wandb_api_key_here  # Optional for experiment tracking
   ```

5. Download the AQUA dataset:
   - Visit the [AQUA GitHub repository](https://github.com/noagarcia/ArtVQA/tree/master/AQUA)
   - Download the dataset files (train.json, val.json, test.json)
   - Create a directory structure: `data/AQUA/` and place the files there

6. Download the SemArt dataset:
   - Download the image dataset from the [SemArt website](https://noagarcia.github.io/SemArt/)
   - Download the SemArt JSON files (train.json, test.json, val.json)
   - Place the images in your desired directory (you'll configure this path in the next step)
   - Create a single `semart.csv` file that combines all train, test, and val data from SemArt (this file is required for providing descriptions for questions that need external knowledge)
   - Place the `semart.csv` file in the `data/` directory

## 📋 Usage

### Prepare Your Dataset

VizSage is currently configured to work with the [AQUA dataset](https://github.com/noagarcia/ArtVQA/tree/master/AQUA), which contains visual question answering pairs for artwork.

The dataset uses the following structure:
```
/path/to/your/data/
├── AQUA/
│   ├── train.json
│   ├── val.json
│   └── test.json
├── semart.csv  # Combined SemArt dataset for external knowledge descriptions
└── Images/
    ├── 1234_painting.jpg
    ├── 5678_painting.jpg
    └── ...
```

Each JSON file contains entries with the following format:
```json
{
  "image": "1234_painting.jpg",
  "question": "What technique was used in this painting?",
  "answer": "The painting uses oil on canvas...",
  "need_external_knowledge": true
}
```

Where:
- `image`: Image filename as in SemArt dataset
- `question`: Question about the artwork
- `answer`: Answer to the question
- `need_external_knowledge`: Whether the question requires external knowledge (`True` for QAs generated from comments and `False` for QAs generated from paintings). When `True`, the system will retrieve additional descriptions from `semart.csv`

### Configure Training

Edit `config.yaml` to customize your training:

```yaml
# Model parameters
model_name: "unsloth/Llama-3.2-11B-Vision-Instruct"
load_in_4bit: true
finetune_vision_layers: false
finetune_language_layers: true

# LoRA parameters
lora_r: 16
lora_alpha: 16

# Training parameters
batch_size: 2
grad_accum: 4
epochs: 1
lr: 0.0002

# Dataset parameters
dataset: "AQUA"
external_knowledge: false  # Set to true to include QAs requiring external knowledge

# Validation settings
use_validation: true
evaluation_metrics:
  - "bleu"
  - "rouge"
  - "exact_match"
best_model_metric: "rougeL_fmeasure"
greater_is_better: true
```

### Start Training

Run the training script:

```bash
python train.py
```

Or specify a custom configuration file (Different from `config.yaml`):

```bash
python train.py custom_config.yaml
```

### Evaluate your Model

After training, you can evaluate the model on the test set:

```bash
python test.py
```

The results will be saved in a dedicated `results` folder with timestamp, including:
- Detailed predictions
- Comprehensive metrics
- Visual charts
- Summary report

## 🔧 Configuration Options

### Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_name` | Model identifier or path | `"unsloth/Llama-3.2-11B-Vision-Instruct"` |
| `load_in_4bit` | Use 4-bit quantization | `true` |
| `use_gradient_checkpointing` | Gradient checkpointing strategy | `"unsloth"` |
| `name_trained_model` | Name for saving the fine-tuned model | `"VizSage_final_model"` |
| `finetune_vision_layers` | Fine-tune vision encoder | `false` |
| `finetune_language_layers` | Fine-tune language model | `true` |
| `finetune_attention_modules` | Fine-tune attention modules | `true` |
| `finetune_mlp_modules` | Fine-tune MLP modules | `true` |

### LoRA Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `lora_r` | LoRA rank | `16` |
| `lora_alpha` | LoRA scaling factor | `16` |
| `lora_dropout` | LoRA dropout rate | `0` |
| `lora_bias` | LoRA bias type | `"none"` |
| `use_rslora` | Use rank-stabilized LoRA | `false` |
| `random_state` | Random seed for LoRA initialization | `3407` |

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch_size` | Batch size per device | `2` |
| `grad_accum` | Gradient accumulation steps | `4` |
| `epochs` | Number of training epochs | `1` |
| `lr` | Learning rate | `0.0002` |
| `warmup_steps` | Learning rate warmup steps | `5` |
| `weight_decay` | Weight decay for AdamW optimizer | `0.01` |
| `logging_steps` | Steps between logging updates | `1` |
| `optim` | Optimizer | `"adamw_8bit"` |
| `scheduler` | Learning rate scheduler | `"linear"` |
| `seed` | Random seed for training | `3407` |

### Validation and Evaluation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_validation` | Enable validation set during training | `true` |
| `eval_batch_size` | Batch size for evaluation | `4` |
| `max_val_samples` | Maximum validation samples to use | `500` |
| `evaluation_metrics` | Metrics to calculate during evaluation | `["bleu", "rouge", "exact_match"]` |
| `best_model_metric` | Metric for selecting best model | `"rougeL_fmeasure"` |
| `greater_is_better` | Whether higher metric is better | `true` |
| `n_evals` | Number of evaluations during streaming training | `10` |
| `save_total_limit` | Number of best checkpoints to keep | `3` |

### Inference Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `instruction` | System prompt for model inference | `"You are an expert art historian..."` |

### Hardware Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_bf16` | Use bfloat16 precision | `true` |

### Dataset and Output

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dataset` | Dataset name | `"AQUA"` |
| `base_path` | Base directory for data | `"data"` |
| `external_knowledge` | Include QAs requiring external knowledge | `true` |
| `external_knowledge_path` | Path to external knowledge file | `"data/semart.csv"` (combined SemArt dataset for retrieving descriptions of artworks) |
| `num_proc` | Number of processes for loading | `4` |
| `max_seq_length` | Maximum sequence length | `2048` |
| `output_dir` | Directory to save models | `"outputs"` |

### Weights & Biases (Optional)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_wandb` | Enable W&B logging | `true` |
| `wandb_project` | W&B project name | `"VizSage"` |
| `wandb_tags` | Tags for the run | `["llama3", "vision", "finetune"]` |

### Streaming Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_streaming` | Enable dataset streaming | `true` |
| `stream_buffer_size` | Buffer size for streaming | `1000` |
| `save_steps` | Number of steps between model saves | `100` |
| `n_saves` | Maximum number of checkpoint saves to keep | `5` |
| `test_samples_to_check` | Number of test samples to check during validation | `1` |

## 📊 Experiment Tracking

If you enable Weights & Biases integration, VizSage will log:

- Model configuration
- Training parameters
- Dataset statistics
- Training loss and metrics
- Validation metrics during training
- Model summary (without uploading model weights)

To view your experiments, visit [wandb.ai](https://wandb.ai).

## 📊 Results Analysis

After running the `test.py` script, a comprehensive results analysis is generated in the `results` folder:

```
results/
└── ModelName_DatasetName_20250509_123456/
    ├── test_output.json        # Detailed predictions for each sample
    ├── test_metrics.json       # All metrics in JSON format
    ├── test_metrics.csv        # Metrics in CSV format for easy import 
    ├── summary.md              # Human-readable summary report
    ├── test_config.yaml        # Configuration used for testing
    ├── bleu_scores.png         # Chart of BLEU scores
    └── rouge_scores.png        # Chart of ROUGE scores
```

This organized structure makes it easy to:
- Compare different model versions
- Track improvements across experiments
- Share results with collaborators
- Generate reports for presentations

## 📁 Project Structure

```
vizsage/
├── config.yaml           # Main configuration file
├── train.py              # Training script with config support
├── test.py               # Evaluation script with metrics
├── model.py              # Model definition and initialization
├── data_utils.py         # Dataset loading and processing
├── config_utils.py       # Configuration utilities
├── evaluate_metrics.py   # Metrics calculation module
├── .env.template         # Template for environment variables
├── requirements.txt      # Project dependencies
└── README.md             # This documentation
```

## 🔍 Troubleshooting

- **Out of memory errors**: Try reducing `batch_size`, enabling `load_in_4bit`, or setting `use_gradient_checkpointing: true`
- **Slow training**: Increase `batch_size` if memory allows, or try a smaller model
- **Poor results**: Adjust `lr`, `lora_r`, or increase `epochs`
- **API key errors**: Ensure your Hugging Face API key is correctly set in the `.env` file
- **Dataset path errors**: Check that the `base_path` in `config.yaml` points to your data directory
- **Metrics calculation errors**: Make sure you have installed the required packages with `pip install -r requirements.txt`

## 📜 License

[MIT License](LICENSE)

## 🙏 Acknowledgments

- [AQUA Dataset](https://github.com/noagarcia/ArtVQA/tree/master/AQUA) for the art visual question answering data
- [SemArt Dataset](https://noagarcia.github.io/SemArt/) for the artwork images
- [Unsloth](https://github.com/unslothai/unsloth) for the optimized training code
- [Hugging Face](https://huggingface.co) for the Transformers library
- [Weights & Biases](https://wandb.ai) for experiment tracking
- [NLTK](https://www.nltk.org/) and [ROUGE](https://github.com/google-research/google-research/tree/master/rouge) for evaluation metrics

## 📧 Contact

For questions or issues, please open an issue on GitHub or mail me at [gio.biancini@stud.uniroma3.it](mailto:gio.biancini@stud.uniroma3.it)