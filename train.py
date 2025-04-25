#!/usr/bin/env python
# train_with_config.py

import yaml
import sys
import os
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import model as m
import data_preprocessing as dp
from dotenv import load_dotenv
import wandb
import socket
from datetime import datetime

# Load environment variables from .env file
load_dotenv()


def load_config(config_file="config.yaml"):
    """Load training configuration from YAML file"""
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_wandb(config):
    """Setup Weights & Biases logging"""
    # Check if wandb logging is enabled in config
    if not config.get("use_wandb", False):
        return None

    # Get wandb API key from environment variable
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        print("Warning: WANDB_API_KEY not found in .env file. Wandb logging disabled.")
        return None

    # Login to wandb
    wandb.login(key=wandb_api_key)

    # Create a unique run name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hostname = socket.gethostname()
    run_name = f"VizSage_{config.get('model_name', 'model').split('/')[-1]}_{timestamp}_{hostname}"

    # Initialize wandb run
    wandb_run = wandb.init(
        project=config.get("wandb_project", "VizSage"),
        name=run_name,
        config={
            # Model parameters
            "model_name": config.get("model_name"),
            "finetune_vision": config.get("finetune_vision_layers"),
            "finetune_language": config.get("finetune_language_layers"),
            "lora_r": config.get("lora_r"),
            "lora_alpha": config.get("lora_alpha"),

            # Training parameters
            "batch_size": config.get("batch_size"),
            "grad_accum": config.get("grad_accum"),
            "effective_batch_size": config.get("batch_size") * config.get("grad_accum"),
            "learning_rate": config.get("lr"),
            "epochs": config.get("epochs"),
            "optimizer": config.get("optim"),
            "weight_decay": config.get("weight_decay"),
            "warmup_steps": config.get("warmup_steps"),

            # Dataset info
            "dataset": config.get("dataset"),
            "max_seq_length": config.get("max_seq_length"),
            "external_knowledge": config.get("external_knowledge", False),
        },
        tags=config.get("wandb_tags", []),
        # Disable model saving to wandb
        settings=wandb.Settings(
            _disable_stats=True,  # Disable system stats
            _disable_meta=True,  # Disable metadata collection
        )
    )

    # Disable model checkpoints sync to wandb
    os.environ["WANDB_LOG_MODEL"] = "false"

    return wandb_run


def train(model, tokenizer, converted_dataset, config, wandb_run=None):
    """Train the model with parameters from config"""

    # Enable model for training
    FastVisionModel.for_training(model)

    # Set BF16/FP16 based on configuration and hardware support
    use_bf16 = is_bf16_supported() and config.get("use_bf16", True)
    use_fp16 = not use_bf16

    # Set wandb logging
    report_to = "wandb" if wandb_run else "none"

    # Create output directory
    output_dir = config.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Configure trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted_dataset,
        args=SFTConfig(
            per_device_train_batch_size=config.get("batch_size", 2),
            gradient_accumulation_steps=config.get("grad_accum", 4),
            warmup_steps=config.get("warmup_steps", 5),
            num_train_epochs=config.get("epochs", 1),
            learning_rate=config.get("lr", 2e-4),
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=config.get("logging_steps", 1),
            optim=config.get("optim", "adamw_8bit"),
            weight_decay=config.get("weight_decay", 0.01),
            lr_scheduler_type=config.get("scheduler", "linear"),
            seed=config.get("seed", 3407),
            output_dir=output_dir,
            report_to=report_to,  # Use wandb if configured
            run_name=wandb_run.name if wandb_run else None,

            # Disable saving model weights to wandb
            hub_model_id=None,
            hub_strategy=None,
            push_to_hub=False,
            save_strategy="epoch",  # Only save locally at end of epoch

            # Required parameters for vision finetuning
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=config.get("num_proc", 4),
            max_seq_length=config.get("max_seq_length", 2048),
        ),
    )

    # Start training
    print(f"Starting training with batch size {config.get('batch_size', 2)} and learning rate {config.get('lr', 2e-4)}")
    trainer.train()

    # Save final model
    final_model_path = f"{output_dir}/VizSage_final_model"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model saved to {final_model_path}")

    # Log model metadata to wandb if enabled
    if wandb_run:
        wandb_run.summary.update({
            "train_samples": len(converted_dataset),
            "final_model_path": final_model_path,
            "training_completed": True,
            "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        # Create a text summary instead of saving the model card as an artifact
        model_card_text = f"""
        # VizSage Model Summary

        ## Model Information
        - Base model: {config.get('model_name')}
        - Fine-tuned on: {config.get('dataset')}
        - Date: {datetime.now().strftime("%Y-%m-%d")}

        ## Training Parameters
        - Batch size: {config.get('batch_size')}
        - Learning rate: {config.get('lr')}
        - Epochs: {config.get('epochs')}
        - LoRA rank: {config.get('lora_r')}

        ## Local Path
        - Saved to: {final_model_path}
        """

        # Log the text summary directly to wandb
        wandb.log({"model_card": wandb.Html(model_card_text.replace('\n', '<br>'))})

        # Also save the model card locally
        with open(f"{final_model_path}/model_card.md", "w") as f:
            f.write(model_card_text)

    return trainer


def get_model_from_config(config):
    """Get model and tokenizer using parameters from config"""
    model, tokenizer = m.get_model(
        model_name=config.get("model_name", "unsloth/Llama-3.2-11B-Vision-Instruct"),
        load_in_4bit=config.get("load_in_4bit", True),
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", "unsloth"),
        finetune_vision_layers=config.get("finetune_vision_layers", False),
        finetune_language_layers=config.get("finetune_language_layers", True),
        finetune_attention_modules=config.get("finetune_attention_modules", True),
        finetune_mlp_modules=config.get("finetune_mlp_modules", True),
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0),
        bias=config.get("lora_bias", "none"),
        random_state=config.get("random_state", 3407),
        use_rslora=config.get("use_rslora", False),
    )
    return model, tokenizer


if __name__ == "__main__":
    # Check if a config file was provided as an argument
    config_file = "config.yaml"
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    # Ensure the config file exists
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found.")
        sys.exit(1)

    # Load configuration
    config = load_config(config_file)
    print(f"Loaded configuration from {config_file}")

    # Setup wandb logging
    wandb_run = setup_wandb(config)
    if wandb_run:
        print(f"Wandb logging enabled: {wandb_run.name}")
    else:
        print("Wandb logging disabled")

    # Load model using config parameters
    model, tokenizer = get_model_from_config(config)

    # Load dataset using data_preprocessing module
    train_dataset, val_dataset, test_dataset = dp.get_dataset(
        base_path="data",
        dataset=config.get("dataset", "AQUA"),
        external_knowledge=config.get("external_knowledge", False)
    )

    # Convert train dataset to conversation format
    converted_dataset = [dp.convert_to_conversation(sample) for sample in train_dataset]

    # Log dataset information to wandb
    if wandb_run:
        wandb_run.log({
            "train_dataset_size": len(train_dataset),
            "val_dataset_size": len(val_dataset) if val_dataset else 0,
            "test_dataset_size": len(test_dataset) if test_dataset else 0,
        })

    # Train the model
    trainer = train(model, tokenizer, converted_dataset, config, wandb_run)

    # Close wandb run
    if wandb_run:
        wandb_run.finish()

    print("Training completed successfully!")