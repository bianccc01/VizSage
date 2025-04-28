# VizSage: Vision-Language Model Fine-tuning Framework

VizSage is a streamlined framework for fine-tuning vision-language models (VLMs) with a focus on simplicity, flexibility, and reproducibility. The project provides an easy way to customize model parameters and training configurations through a simple YAML file.

## 🌟 Features

- **Easy Configuration**: Fine-tune models by modifying a single YAML file without changing code
- **Vision Model Support**: Built specifically for vision-language models like Llama-3-Vision, Qwen2-VL, and others
- **LoRA Fine-tuning**: Efficient parameter-efficient fine-tuning through LoRA
- **Memory Optimization**: Process large datasets in manageable chunks
- **Experiment Tracking**: Optional Weights & Biases integration for tracking experiments
- **Reproducibility**: Save and reuse configurations for consistent results

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
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

3. Create your `.env` file from the template:
   ```bash
   cp .env.template .env
   ```

4. Edit the `.env` file to add your API keys:
   ```
   HUGGINGFACE_TOKEN=your_hf_token_here  # Required for downloading models
   WANDB_API_KEY=your_wandb_api_key_here  # Optional for experiment tracking
   ```

5. Download the AQUA dataset:
   - Visit the [AQUA GitHub repository](https://github.com/noagarcia/ArtVQA/tree/master/AQUA)
   - Download the dataset files (train.json, val.json, test.json)
   - Create a directory structure: `data/AQUA/` and place the files there

6. Download the SemArt images:
   - Download the image dataset from the [SemArt website](https://noagarcia.github.io/SemArt/)
   - Place the images in your desired directory (you'll configure this path in the next step)

7. Update the `base_path` in the configuration:
   - In the `train_with_config.py` file, find the line where `dp.get_dataset` is called
   - Update the `base_path` parameter to point to your data directory, for example:
     ```python
     train_dataset, val_dataset, test_dataset = dp.get_dataset(
         base_path="/path/to/your/data",  # Update this path
         dataset=config.get("dataset", "AQUA"),
         external_knowledge=config.get("external_knowledge", False)
     )
     ```

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
- `need_external_knowledge`: Whether the question requires external knowledge (`True` for QAs generated from comments and `False` for QAs generated from paintings)

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
```

### Start Training

Run the training script:

```bash
python train.py
```

Or specify a custom configuration file:

```bash
python train.py custom_config.yaml
```

## 🔧 Configuration Options

### Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_name` | Model identifier or path | `"unsloth/Llama-3.2-11B-Vision-Instruct"` |
| `load_in_4bit` | Use 4-bit quantization | `true` |
| `use_gradient_checkpointing` | Gradient checkpointing strategy | `"unsloth"` |
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

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `batch_size` | Batch size per device | `2` |
| `grad_accum` | Gradient accumulation steps | `4` |
| `epochs` | Number of training epochs | `1` |
| `lr` | Learning rate | `0.0002` |
| `warmup_steps` | Learning rate warmup steps | `5` |
| `optim` | Optimizer | `"adamw_8bit"` |
| `scheduler` | Learning rate scheduler | `"linear"` |

### Dataset and Output

| Parameter | Description | Default |
|-----------|-------------|---------|
| `dataset` | Dataset name | `"AQUA"` |
| `external_knowledge` | Include QAs requiring external knowledge | `false` |
| `num_proc` | Number of processes for loading | `4` |
| `max_seq_length` | Maximum sequence length | `2048` |
| `output_dir` | Directory to save models | `"outputs"` |

### Weights & Biases (Optional)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_wandb` | Enable W&B logging | `true` |
| `wandb_project` | W&B project name | `"VizSage"` |
| `wandb_tags` | Tags for the run | `[]` |

## 📊 Experiment Tracking

If you enable Weights & Biases integration, VizSage will log:

- Model configuration
- Training parameters
- Dataset statistics
- Training loss and metrics
- Model summary (without uploading model weights)

To view your experiments, visit [wandb.ai](https://wandb.ai).

## 📁 Project Structure

```
vizsage/
├── config.yaml           # Main configuration file
├── train.py  # Training script with config support
├── model.py              # Model definition and initialization
├── data_preprocessing.py # Dataset loading and processing
├── .env.template         # Template for environment variables
├── requirements.txt      # Project dependencies
└── README.md             # This documentation
```

## 🔍 Troubleshooting

- **Out of memory errors**: Try reducing `batch_size`, enabling `load_in_4bit`, or setting `use_gradient_checkpointing: true`
- **Slow training**: Increase `batch_size` if memory allows, or try a smaller model
- **Poor results**: Adjust `lr`, `lora_r`, or increase `epochs`
- **API key errors**: Ensure your Hugging Face API key is correctly set in the `.env` file
- **Dataset path errors**: Check that the `base_path` in `train_with_config.py` points to your data directory

## 📜 License

[MIT License](LICENSE)

## 🙏 Acknowledgments

- [AQUA Dataset](https://github.com/noagarcia/ArtVQA/tree/master/AQUA) for the art visual question answering data
- [SemArt Dataset](https://noagarcia.github.io/SemArt/) for the artwork images
- [Unsloth](https://github.com/unslothai/unsloth) for the optimized training code
- [Hugging Face](https://huggingface.co) for the Transformers library
- [Weights & Biases](https://wandb.ai) for experiment tracking

## 📧 Contact

For questions or issues, please open an issue on GitHub
