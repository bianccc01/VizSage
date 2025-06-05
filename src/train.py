#!/usr/bin/env python3
"""
Unified VizSage training script
Run with: python train.py [config_file]
"""
import unsloth
from unsloth import FastVisionModel, UnslothVisionDataCollator
import os
import sys
import time
import random
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List

import wandb
from trl import SFTTrainer, SFTConfig

import config_utils
import data_utils
import model as model_utils

# Import utility modules
from evaluation_utils import create_compute_metrics_fn, preprocess_logits_for_metrics
from training_utils import (
    setup_training_config,
    print_training_config,
    MemoryManagementCallback,
    setup_early_stopping,
    save_model_and_tokenizer,
    update_wandb_summary,
    is_bf16_supported
)
from formatting_utils import create_formatting_function


# ================================================================================
# TRAINING FUNCTIONS
# ================================================================================

def train_streaming(
        model,
        tokenizer,
        streaming_dataset,
        config: Dict[str, Any],
        wandb_run=None,
        len_train_dataset: Optional[int] = None,
        val_dataset=None
):
    """Train the model with streaming dataset for vision tasks"""

    print("Setting up streaming training configuration...")

    # === CONFIGURATION SETUP ===
    enhanced_config = setup_training_config(config, len_train_dataset)
    print_training_config(enhanced_config)

    # === MODEL PREPARATION ===
    FastVisionModel.for_training(model)

    # Mixed precision setup
    use_bf16 = is_bf16_supported() and enhanced_config.get("use_bf16", True)
    use_fp16 = not use_bf16

    # === DATASET FORMATTING ===
    formatting_func = create_formatting_function(tokenizer)

    # === EVALUATION SETUP ===
    compute_metrics_fn = create_compute_metrics_fn(tokenizer, enhanced_config)

    # === OUTPUT DIRECTORY ===
    output_dir = enhanced_config.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # === LOGGING SETUP ===
    report_to = "wandb" if wandb_run else "none"

    # === TRAINER INITIALIZATION ===
    print("\nInitializing SFTTrainer for streaming...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=streaming_dataset,
        eval_dataset=val_dataset,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics_fn,
        formatting_func=formatting_func,
        args=SFTConfig(
            # Batch and gradient settings
            per_device_train_batch_size=enhanced_config.get("batch_size", 2),
            gradient_accumulation_steps=enhanced_config.get("grad_accum", 4),

            # Learning settings
            learning_rate=enhanced_config.get("lr", 2e-4),
            warmup_steps=enhanced_config.get("warmup_steps", 5),
            max_steps=enhanced_config["max_steps"],
            num_train_epochs=1,  # Use max_steps instead

            # Precision and optimization
            fp16=use_fp16,
            bf16=use_bf16,
            max_grad_norm=enhanced_config.get("max_grad_norm", 1.0),
            optim=enhanced_config.get("optim", "adamw_8bit"),
            weight_decay=enhanced_config.get("weight_decay", 0.01),
            lr_scheduler_type=enhanced_config.get("scheduler", "linear"),

            # Sequence and data settings
            max_seq_length=enhanced_config.get("max_seq_length", 2048),
            remove_unused_columns=False,
            dataset_text_field="",  # Use formatting_func instead
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=enhanced_config.get("num_proc", 4),

            # Logging and saving
            logging_steps=enhanced_config.get("logging_steps", 1),
            save_strategy="steps",
            save_steps=enhanced_config["save_steps"],
            output_dir=output_dir,
            report_to=report_to,
            run_name=wandb_run.name if wandb_run else None,

            # Evaluation settings
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=enhanced_config.get("eval_steps", enhanced_config["eval_steps"]) if val_dataset else None,
            metric_for_best_model="eval_loss" if val_dataset else None,
            greater_is_better=False if val_dataset else None,
            load_best_model_at_end=True if val_dataset else False,

            # System settings
            seed=enhanced_config.get("seed", 3407),

            # Logging settings
            logging_dir=os.path.join(output_dir, "logs"),
            logging_first_step=True,
            logging_nan_inf_filter=True,
            log_level="info",
        ),
    )

    # === CALLBACKS SETUP ===
    trainer.add_callback(MemoryManagementCallback())

    early_stopping_callback = setup_early_stopping(
        enhanced_config,
        enhanced_config["max_steps"],
        val_dataset
    )
    if early_stopping_callback:
        trainer.add_callback(early_stopping_callback)

    # === TRAINING ===
    print(f"\nStarting streaming training...")
    print(f"Dataset size: {len_train_dataset if len_train_dataset else 'Unknown'}")
    print(f"Validation dataset: {'Yes' if val_dataset else 'No'}")

    # Ensure Unsloth returns logits for evaluation
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    try:
        trainer.train()
        print("\nStreaming training completed successfully!")
    except Exception as e:
        print(f"\nStreaming training failed with error: {e}")
        raise

    # === MODEL SAVING ===
    final_model_path = save_model_and_tokenizer(model, tokenizer, enhanced_config)

    # === WANDB LOGGING ===
    update_wandb_summary(wandb_run, enhanced_config, trainer, final_model_path)

    print(f"\n🎉 Streaming training pipeline completed!")
    print(f"📁 Model saved at: {final_model_path}")

    return trainer


def train_regular(
        model,
        tokenizer,
        converted_dataset,
        config: Dict[str, Any],
        wandb_run=None,
        val_dataset=None
):
    """Train the model with regular dataset for vision tasks"""

    print("Setting up regular training configuration...")

    # === CONFIGURATION SETUP ===
    len_train_dataset = len(converted_dataset) if hasattr(converted_dataset, '__len__') else None
    enhanced_config = setup_training_config(config, len_train_dataset)
    print_training_config(enhanced_config)

    # === MODEL PREPARATION ===
    FastVisionModel.for_training(model)

    # Mixed precision setup
    use_bf16 = is_bf16_supported() and enhanced_config.get("use_bf16", True)
    use_fp16 = not use_bf16

    # === DATASET FORMATTING ===
    formatting_func = create_formatting_function(tokenizer)

    # === EVALUATION SETUP ===
    compute_metrics_fn = create_compute_metrics_fn(tokenizer, enhanced_config)

    # === OUTPUT DIRECTORY ===
    output_dir = enhanced_config.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # === LOGGING SETUP ===
    report_to = "wandb" if wandb_run else "none"

    # === TRAINER INITIALIZATION ===
    print(
        f"\nStarting training with batch size {enhanced_config.get('batch_size', 2)} and learning rate {enhanced_config.get('lr', 2e-4)}")
    print("Initializing SFTTrainer...")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted_dataset,
        eval_dataset=val_dataset,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics_fn,
        formatting_func=formatting_func,
        args=SFTConfig(
            # Batch and gradient settings
            per_device_train_batch_size=enhanced_config.get("batch_size", 2),
            gradient_accumulation_steps=enhanced_config.get("grad_accum", 4),

            # Learning settings
            learning_rate=enhanced_config.get("lr", 2e-4),
            warmup_steps=enhanced_config.get("warmup_steps", 5),
            max_steps=enhanced_config.get("max_steps", -1),  # Use epochs if max_steps not set
            num_train_epochs=enhanced_config.get("epochs", 1),

            # Precision and optimization
            fp16=use_fp16,
            bf16=use_bf16,
            max_grad_norm=enhanced_config.get("max_grad_norm", 1.0),
            optim=enhanced_config.get("optim", "adamw_8bit"),
            weight_decay=enhanced_config.get("weight_decay", 0.01),
            lr_scheduler_type=enhanced_config.get("scheduler", "linear"),

            # Sequence and data settings
            max_seq_length=enhanced_config.get("max_seq_length", 2048),
            remove_unused_columns=False,
            dataset_text_field="",  # Use formatting_func instead
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=enhanced_config.get("num_proc", 4),

            # Logging and saving
            logging_steps=enhanced_config.get("logging_steps", 1),
            save_strategy=enhanced_config.get("save_strategy", "epoch"),
            save_steps=enhanced_config.get("save_steps"),
            output_dir=output_dir,
            report_to=report_to,
            run_name=wandb_run.name if wandb_run else None,

            # Hub settings (disabled)
            hub_model_id=None,
            hub_strategy="end",
            push_to_hub=False,

            # Evaluation settings
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=enhanced_config.get("eval_steps") if val_dataset else None,
            metric_for_best_model="eval_exact_match" if val_dataset else None,
            greater_is_better=True if val_dataset else None,
            load_best_model_at_end=True if val_dataset else False,

            # System settings
            seed=enhanced_config.get("seed", 3407),
        ),
    )

    # === CALLBACKS SETUP ===
    trainer.add_callback(MemoryManagementCallback())

    early_stopping_callback = setup_early_stopping(
        enhanced_config,
        enhanced_config.get("max_steps",
                            len_train_dataset * enhanced_config.get("epochs", 1) if len_train_dataset else 1000),
        val_dataset
    )
    if early_stopping_callback:
        trainer.add_callback(early_stopping_callback)

    # === TRAINING ===
    print(f"\nStarting training...")
    print(f"Dataset size: {len_train_dataset if len_train_dataset else 'Unknown'}")
    print(f"Validation dataset: {'Yes' if val_dataset else 'No'}")

    # Ensure Unsloth returns logits for evaluation
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

    try:
        trainer.train()
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise

    # === MODEL SAVING ===
    final_model_path = save_model_and_tokenizer(model, tokenizer, enhanced_config)

    # === WANDB LOGGING ===
    if wandb_run:
        try:
            # Update run summary with training info
            wandb_run.summary.update({
                "train_samples": len_train_dataset if len_train_dataset else "Unknown",
                "final_model_path": final_model_path,
                "training_completed": True,
                "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_epochs": enhanced_config.get("epochs", 1),
                "final_loss": trainer.state.log_history[-1].get("train_loss",
                                                                "N/A") if trainer.state.log_history else "N/A"
            })

            # Create enhanced model card
            model_card_text = f"""
# VizSage Model Summary

## Model Information
- **Base model**: {enhanced_config.get('model_name', 'Unknown')}
- **Fine-tuned on**: {enhanced_config.get('dataset', 'Unknown')}
- **Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Model Path**: {final_model_path}

## Training Parameters
- **Batch Size**: {enhanced_config.get('batch_size', 2)}
- **Learning Rate**: {enhanced_config.get('lr', 2e-4)}
- **Epochs**: {enhanced_config.get('epochs', 1)}
- **Mixed Precision**: {'BF16' if use_bf16 else 'FP16'}

## Evaluation Configuration
- **Text Normalization**: {'Enabled' if enhanced_config.get('use_text_normalization', True) else 'Disabled'}
- **Debug Mode**: {'Enabled' if enhanced_config.get('debug_exact_match', False) else 'Disabled'}

## Performance
- **Training Samples**: {len_train_dataset if len_train_dataset else 'Unknown'}
- **Final Loss**: {trainer.state.log_history[-1].get('train_loss', 'N/A') if trainer.state.log_history else 'N/A'}

## Local Path
- **Saved to**: {final_model_path}
"""

            # Log to wandb
            wandb.log({"model_card": wandb.Html(model_card_text.replace('\n', '<br>'))})

            # Save locally
            with open(f"{final_model_path}/model_card.md", "w") as f:
                f.write(model_card_text)

            print("Model card saved and logged to wandb")

        except Exception as e:
            print(f"Warning: Error updating wandb: {e}")

    print(f"\n🎉 Training pipeline completed!")
    print(f"📁 Model saved at: {final_model_path}")

    return trainer


# ================================================================================
# MAIN PIPELINE
# ================================================================================

def setup_environment():
    """Setup environment variables"""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Fix for Unsloth 2024.11+ - Required for logits in evaluation
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"


def get_config_file():
    """Get config file from arguments or use default"""
    if len(sys.argv) > 1:
        return sys.argv[1]
    return "config/config.yaml"


def setup_wandb(config):
    """Setup wandb if configured"""
    try:
        if config.get("use_wandb", False):
            wandb_run = wandb.init(
                project=config.get("wandb_project", "vizsage-training"),
                name=config.get("wandb_run_name"),
                config=config
            )
            print(f"Wandb logging enabled: {wandb_run.name}")
            return wandb_run
        else:
            print("Wandb logging disabled")
            return None
    except Exception as e:
        print(f"Warning: Wandb setup failed: {e}")
        return None


def setup_reproducibility(config):
    """Setup random seeds"""
    if config.get("reproducible", False):
        seed = config.get("seed", 42)
        random.seed(seed)
        print(f"Using seed {seed} for reproducible results")
    else:
        random.seed(int(time.time()))
        print("Using random seed for different results each run")


def prepare_datasets(config):
    """Load and prepare datasets with separate streaming control for train and validation"""
    use_train_streaming = config.get("use_streaming", False)
    use_val_streaming = config.get("use_val_streaming", use_train_streaming)  # Default to same as training

    print(f"Loading datasets:")
    print(f"  - Training: {'streaming' if use_train_streaming else 'regular'} mode")
    print(f"  - Validation: {'streaming' if use_val_streaming else 'regular'} mode")

    # Load datasets with separate streaming settings
    train_dataset, val_dataset, test_dataset, len_train_dataset, len_test_dataset = data_utils.get_dataset_with_separate_streaming(
        base_path=config.get("base_path", "data"),
        dataset=config.get("dataset", "AQUA"),
        external_knowledge=config.get("external_knowledge", False),
        use_train_streaming=use_train_streaming,
        use_val_streaming=use_val_streaming
    )

    # Load external knowledge if needed
    semart_dataset = None
    if config.get("external_knowledge", False):
        try:
            semart_dataset = data_utils.load_external_knowledge(
                config.get("external_knowledge_path", "data/semart.csv")
            )
        except Exception as e:
            print(f"Error loading external knowledge: {e}")
            raise

    return train_dataset, val_dataset, test_dataset, len_train_dataset, len_test_dataset, semart_dataset


def select_test_sample(test_dataset, is_streaming):
    """Select a random test sample for inference comparison"""
    if not test_dataset:
        return None

    if is_streaming:
        # For streaming: collect a few samples and pick randomly
        temp_samples = []
        for i, example in enumerate(test_dataset):
            temp_samples.append(example)
            if i >= 9:  # Collect 10 samples
                break

        if temp_samples:
            return random.choice(temp_samples)
    else:
        # For regular dataset: pick random index
        index = random.randint(0, len(test_dataset) - 1)
        return test_dataset[index]

    return None


def run_pre_training_inference(model, tokenizer, test_sample, config, semart_dataset=None):
    """Run inference before training"""
    if not test_sample:
        return None

    print("\n=== PRE-TRAINING INFERENCE ===")

    image = test_sample["image"]
    question = test_sample["question"]
    ground_truth = test_sample["answer"]
    instruction = config.get("instruction", "")

    # Get description if using external knowledge
    description = None
    if (semart_dataset is not None and
            config.get("external_knowledge", False) and
            test_sample.get("need_external_knowledge", False)):
        try:
            desc_match = semart_dataset.loc[semart_dataset['image_file'] == image, 'description']
            if not desc_match.empty:
                description = desc_match.values[0]
        except:
            pass

    print(f"Question: {question}")
    print(f"Image: {image}")
    print(f"Ground truth: {ground_truth}")

    try:
        if hasattr(model_utils, 'make_inference'):
            if config.get("use_streaming", False):
                prediction, _ = model_utils.make_inference(
                    model=model, tokenizer=tokenizer, image_path=image,
                    question=question, instruction=instruction, description=description,
                    base_path=config.get("base_path", "data")
                )
            else:
                prediction = model_utils.make_inference(
                    model=model, tokenizer=tokenizer, image_path=image,
                    question=question, instruction=instruction, description=description,
                    base_path=config.get("base_path", "data")
                )
        else:
            prediction = "Inference function not available"

        print(f"Model prediction: {prediction}")
        return prediction
    except Exception as e:
        print(f"Inference error: {e}")
        return None


def prepare_training_data(train_dataset, val_dataset, config, semart_dataset):
    """Prepare data for training with separate streaming control"""
    use_train_streaming = config.get("use_streaming", False)
    use_val_streaming = config.get("use_val_streaming", use_train_streaming)

    print("Preparing training datasets...")
    print(f"  - Training streaming: {use_train_streaming}")
    print(f"  - Validation streaming: {use_val_streaming}")

    # Prepare training dataset
    if use_train_streaming:
        print("Preparing streaming training dataset...")
        prepared_train_dataset = data_utils.prepare_streaming_dataset(
            streaming_dataset=train_dataset,
            config=config,
            semart_dataset=semart_dataset,
            base_path=config.get("base_path", "data")
        )
    else:
        print("Converting training dataset to conversation format...")
        if semart_dataset is not None:
            prepared_train_dataset = [
                data_utils.convert_to_conversation(sample, semart_dataset=semart_dataset)
                for sample in train_dataset
            ]
        else:
            prepared_train_dataset = [
                data_utils.convert_to_conversation(sample)
                for sample in train_dataset
            ]

    # Prepare validation dataset
    prepared_val_dataset = None
    if val_dataset:
        if use_val_streaming:
            print("Preparing streaming validation dataset...")
            prepared_val_dataset = data_utils.prepare_streaming_dataset(
                streaming_dataset=val_dataset,
                config=config,
                semart_dataset=semart_dataset,
                base_path=config.get("base_path", "data")
            )
        else:
            print("Using regular validation dataset...")
            # For regular validation, we keep the original format
            prepared_val_dataset = val_dataset

    return prepared_train_dataset, prepared_val_dataset


def run_training(model, tokenizer, prepared_data, config, wandb_run, len_train_dataset):
    """Run the actual training with hybrid streaming support"""
    print("\n=== STARTING TRAINING ===")

    use_train_streaming = config.get("use_streaming", False)
    use_val_streaming = config.get("use_val_streaming", use_train_streaming)

    prepared_train_dataset, prepared_val_dataset = prepared_data

    print(f"Training mode: {'streaming' if use_train_streaming else 'regular'}")
    print(f"Validation mode: {'streaming' if use_val_streaming else 'regular'}")

    if use_train_streaming:
        # Use streaming training function
        trainer = train_streaming(
            model=model,
            tokenizer=tokenizer,
            streaming_dataset=prepared_train_dataset,
            config=config,
            wandb_run=wandb_run,
            len_train_dataset=len_train_dataset,
            val_dataset=prepared_val_dataset
        )
    else:
        # Use regular training function
        trainer = train_regular(
            model=model,
            tokenizer=tokenizer,
            converted_dataset=prepared_train_dataset,
            config=config,
            wandb_run=wandb_run,
            val_dataset=prepared_val_dataset
        )

    return trainer


def run_post_training_inference(model, tokenizer, test_sample, pre_training_output, config):
    """Run inference after training and compare"""
    if not test_sample:
        return

    print("\n=== POST-TRAINING INFERENCE ===")

    question = test_sample["question"]
    ground_truth = test_sample["answer"]
    instruction = config.get("instruction", "")

    print(f"Question: {question}")
    print(f"Ground truth: {ground_truth}")

    try:
        post_training_output = model_utils.make_inference(
            model=model, tokenizer=tokenizer, image_path=test_sample["image"],
            question=question, instruction=instruction,
            base_path=config.get("base_path", "data")
        )
        print(f"Model prediction: {post_training_output}")

        # Comparison
        print("\n=== COMPARISON ===")
        print(f"Ground truth:    {ground_truth}")
        print(f"Pre-training:    {pre_training_output}")
        print(f"Post-training:   {post_training_output}")

    except Exception as e:
        print(f"Post-training inference error: {e}")


def save_model_if_needed(model, tokenizer, config):
    """Save model if configured"""
    if config.get("save_model", False):
        save_path = config.get("save_path", "models/trained_model")
        print(f"\nSaving model to {save_path}")
        try:
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print("Model saved successfully!")
        except Exception as e:
            print(f"Error saving model: {e}")


def main():
    """Main training pipeline"""
    print("🚀 Starting VizSage training")

    try:
        # Setup
        setup_environment()
        config_file = get_config_file()
        config = config_utils.load_config(config_file)

        # Print basic config info
        print(f"Model: {config.get('model_name', 'Unknown')}")
        print(f"Dataset: {config.get('dataset', 'Unknown')}")
        print(f"Training streaming: {config.get('use_streaming', False)}")
        print(f"Validation streaming: {config.get('use_val_streaming', config.get('use_streaming', False))}")
        print(f"Text normalization: {config.get('use_text_normalization', True)}")
        print(f"Debug exact match: {config.get('debug_exact_match', False)}")

        # Setup
        wandb_run = setup_wandb(config)
        setup_reproducibility(config)

        # Load model
        print("Loading model...")
        model, tokenizer = model_utils.get_model_from_config(config)

        # Prepare datasets
        train_dataset, val_dataset, test_dataset, len_train_dataset, len_test_dataset, semart_dataset = prepare_datasets(
            config)

        # Select test sample and run pre-training inference
        # For test sample selection, we check if test dataset is streaming based on train streaming setting
        test_sample = select_test_sample(test_dataset, config.get("use_streaming", False))
        pre_training_output = run_pre_training_inference(model, tokenizer, test_sample, config, semart_dataset)

        # Prepare training data
        prepared_data = prepare_training_data(train_dataset, val_dataset, config, semart_dataset)

        # Train
        trainer = run_training(model, tokenizer, prepared_data, config, wandb_run, len_train_dataset)

        print("✅ Training completed successfully!")

        # Post-training inference
        run_post_training_inference(model, tokenizer, test_sample, pre_training_output, config)

        # Save model if needed
        save_model_if_needed(model, tokenizer, config)

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Cleanup
        if 'wandb_run' in locals() and wandb_run:
            wandb_run.finish()

    print("🎉 Training completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())