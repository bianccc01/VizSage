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
import random
from datasets import Dataset, load_from_disk
import tempfile
import json


# Load environment variables from .env file
load_dotenv()


def prepare_streaming_dataset(streaming_dataset, config):
    """
    Prepare the streaming dataset for training.

    Args:
        streaming_dataset (IterableDataset): Streaming dataset to be processed
        config (dict): Configuration dictionary containing training parameters

    Returns:
        Dataset: Processed dataset ready for training
    """
    if streaming_dataset is None:
        return None

    # Apply limit to the dataset if specified
    buffer_size = config.get("stream_buffer_size", 1000)
    dataset = streaming_dataset.shuffle(buffer_size=buffer_size)

    # Convert every example to a conversation format
    def convert_to_conversation_streaming(example):
        return dp.convert_to_conversation(example)

    # Apply the conversion function to the streaming dataset
    processed_dataset = dataset.map(convert_to_conversation_streaming)

    return processed_dataset


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
            hub_strategy="end",
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


def train_streaming(model, tokenizer, streaming_dataset, config, wandb_run=None):
    """Train the model with streaming dataset"""

    # Enable model for training
    FastVisionModel.for_training(model)

    # Set BF16/FP16 based on configuration and hardware support
    use_bf16 = is_bf16_supported() and config.get("use_bf16", True)
    use_fp16 = not use_bf16

    # Set wandb logging
    report_to = "wandb" if wandb_run else "none"

    # Set max steps for streaming
    max_steps = config.get("max_steps", 1000)
    if max_steps is None or max_steps <= 0:
        max_steps = 1000

    # Create output directory
    output_dir = config.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Configure trainer per lo streaming
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=streaming_dataset,
        args=SFTConfig(
            per_device_train_batch_size=config.get("batch_size", 2),
            gradient_accumulation_steps=config.get("grad_accum", 4),
            warmup_steps=config.get("warmup_steps", 5),
            # For streaming, set max_steps to a positive integer
            max_steps=max_steps,
            num_train_epochs=1,  # Set to 1 for streaming, it's not used
            learning_rate=config.get("lr", 2e-4),
            fp16=use_fp16,
            bf16=use_bf16,
            logging_steps=config.get("logging_steps", 1),
            optim=config.get("optim", "adamw_8bit"),
            weight_decay=config.get("weight_decay", 0.01),
            lr_scheduler_type=config.get("scheduler", "linear"),
            seed=config.get("seed", 3407),
            output_dir=output_dir,
            report_to=report_to,
            run_name=wandb_run.name if wandb_run else None,

            # Edit this to save the model every N steps
            save_strategy="steps",
            save_steps=config.get("save_steps", 100),

            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=config.get("num_proc", 4),
            max_seq_length=config.get("max_seq_length", 2048),
        ),
    )

    # Start training
    print(
        f"Starting streaming training with batch size {config.get('batch_size', 2)} and max steps {max_steps}")
    trainer.train()
    print("Training completed successfully!")

    # Save final model
    final_model_path = f"{output_dir}/VizSage_final_model"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model saved to {final_model_path}")

    # Log model metadata to wandb if enabled
    if wandb_run:
        # Adatta il logging wandb per lo streaming
        wandb_run.summary.update({
            "training_completed": True,
            "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "final_model_path": final_model_path,
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
        - Max steps: {max_steps}
        - LoRA rank: {config.get('lora_r')}
        - Training mode: Streaming

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

    # Check use streaming option
    use_streaming = config.get("use_streaming", False)

    # Load dataset
    train_dataset, val_dataset, test_dataset = dp.get_dataset(
        base_path="data",
        dataset=config.get("dataset", "AQUA"),
        external_knowledge=config.get("external_knowledge", False),
        use_streaming=use_streaming
    )

    # Seed random for reproducibility or different results each run
    import time

    random.seed(int(time.time()))  # Comment this line for reproducible results

    # Different setup for streaming vs non-streaming
    if use_streaming:
        print("Using streaming mode for dataset processing")

        # Prepare the streaming dataset
        stream_ready_dataset = prepare_streaming_dataset(train_dataset, config)

        # For the inference example - Select a random test sample
        if test_dataset:
            # Estimate the size of the test dataset
            test_size = 0
            temp_samples = []

            # Collect a few samples to estimate size and select from
            max_samples_to_collect = 10  # Limit number of samples to avoid memory issues
            for i, example in enumerate(test_dataset):
                test_size += 1
                if len(temp_samples) < max_samples_to_collect:
                    temp_samples.append(example)
                if i >= 100:  # Stop after 100 to avoid iterating the whole dataset
                    break

            if test_size > 0:
                # Select a random sample from the collected ones
                if temp_samples:
                    random_index = random.randint(0, len(temp_samples) - 1)
                    test_sample = temp_samples[random_index]
                    print(
                        f"Selected random test sample (index {random_index} of {len(temp_samples)} collected samples)")
                else:
                    # Fallback: take the first one if we couldn't collect any
                    for i, example in enumerate(test_dataset):
                        if i == 0:
                            test_sample = example
                            break
                    print("Using first test sample (couldn't collect samples)")

                image = test_sample["image"]
                question = test_sample["question"]
                ground_truth = test_sample["answer"]

                print("\n=== PRE-TRAINING INFERENCE ===")
                print(f"Question: {question}")
                print(f"Image path: {image}")
                print(f"Ground truth answer: {ground_truth}")
                print("Model prediction:")
                pre_training_output = m.make_inference(model=model, tokenizer=tokenizer, image=image, question=question)

                # Save the test sample for post-training inference
                post_training_test_sample = {
                    "image": image,
                    "question": question,
                    "answer": ground_truth
                }
            else:
                print("Could not extract a test sample from streaming dataset")
                post_training_test_sample = None
        else:
            print("No test dataset available")
            post_training_test_sample = None

        # Train the model
        print("\n=== STARTING TRAINING ===")
        trainer = train_streaming(model, tokenizer, stream_ready_dataset, config, wandb_run)

    else:
        # Non streaming mode
        converted_dataset = [dp.convert_to_conversation(sample) for sample in train_dataset]

        # Select a random sample for pre-training inference
        if test_dataset:
            # Select a truly random index from the entire test dataset
            index = random.randint(0, len(test_dataset) - 1)
            sample = test_dataset[index]
            image = sample["image"]
            question = sample["question"]
            ground_truth = sample["answer"]

            print("\n=== PRE-TRAINING INFERENCE ===")
            print(f"Test example index: {index} of {len(test_dataset)}")
            print(f"Question: {question}")
            print(f"Image path: {image}")
            print(f"Ground truth answer: {ground_truth}")
            print("Model prediction:")
            pre_training_output = m.make_inference(model=model, tokenizer=tokenizer, image=image, question=question)

            # Save the test sample for post-training inference
            post_training_test_sample = sample
        else:
            post_training_test_sample = None

        # Train the model
        print("\n=== STARTING TRAINING ===")
        trainer = train(model, tokenizer, converted_dataset, config, wandb_run)

    print("Training completed successfully!")

    # Inference post-training
    if post_training_test_sample:
        print("\n=== POST-TRAINING INFERENCE ===")
        print(f"Question: {post_training_test_sample['question']}")
        print(f"Ground truth answer: {post_training_test_sample['answer']}")
        print("Model prediction:")
        post_training_output = m.make_inference(
            model=model,
            tokenizer=tokenizer,
            image=post_training_test_sample['image'],
            question=post_training_test_sample['question']
        )

        # Compare the results
        print("\n=== COMPARISON ===")
        print(f"Ground truth: {post_training_test_sample['answer']}")
        print(f"Pre-training: {pre_training_output}")
        print(f"Post-training: {post_training_output}")

    # Save the model if configured to do so
    if config.get("save_model", False):
        save_path = config.get("save_path", "models/trained_model")
        print(f"\nSaving model to {save_path}")
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print("Model saved successfully!")

    # Close wandb run
    if wandb_run:
        wandb_run.finish()

    print("Training script completed successfully!")