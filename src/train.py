import yaml
import sys
import os
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import model as m
import data_utils
import config_utils
from dotenv import load_dotenv
import wandb
import socket
import random
from datasets import Dataset, load_from_disk
import pandas as pd
from transformers import TrainerCallback, TrainerState, TrainerControl, EarlyStoppingCallback, EvalPrediction
import numpy as np
import re
import unicodedata
from datetime import datetime
import torch
import gc

# Load environment variables from .env file
load_dotenv()


def setup_wandb(config):
    """Setup Weights & Biases logging"""
    if not config.get("use_wandb", False):
        return None

    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        print("Warning: WANDB_API_KEY not found in .env file. Wandb logging disabled.")
        return None

    wandb.login(key=wandb_api_key)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hostname = socket.gethostname()
    run_name = f"VizSage_{config.get('model_name', 'model').split('/')[-1]}_{timestamp}_{hostname}"

    wandb_run = wandb.init(
        project=config.get("wandb_project", "VizSage"),
        name=run_name,
        config={
            "model_name": config.get("model_name"),
            "finetune_vision": config.get("finetune_vision_layers"),
            "finetune_language": config.get("finetune_language_layers"),
            "lora_r": config.get("lora_r"),
            "lora_alpha": config.get("lora_alpha"),
            "batch_size": config.get("batch_size"),
            "grad_accum": config.get("grad_accum"),
            "effective_batch_size": config.get("batch_size") * config.get("grad_accum"),
            "learning_rate": config.get("lr"),
            "epochs": config.get("epochs"),
            "optimizer": config.get("optim"),
            "weight_decay": config.get("weight_decay"),
            "warmup_steps": config.get("warmup_steps"),
            "dataset": config.get("dataset"),
            "max_seq_length": config.get("max_seq_length"),
            "external_knowledge": config.get("external_knowledge", False),
        },
        tags=config.get("wandb_tags", []),
        settings=wandb.Settings(
            _disable_stats=False,
            _disable_meta=False,
            _service_wait=300,
            start_method="thread",
        )
    )

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

    name_trained_model = config.get("name_trained_model", "VizSage_final_model")

    # Save final model
    final_model_path = f"{output_dir}/{name_trained_model}"
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


def train_streaming(model, tokenizer, streaming_dataset, config, wandb_run=None, len_train_dataset=None,
                    val_dataset=None):
    """
    Train the model with streaming dataset for vision tasks

    Args:
        model: The vision model to train
        tokenizer: Tokenizer for text processing
        streaming_dataset: Training dataset in streaming format
        config: Training configuration dictionary
        wandb_run: Optional wandb run for logging
        len_train_dataset: Length of training dataset
        val_dataset: Optional validation dataset

    Returns:
        trainer: The trained SFTTrainer object
    """

    def formatting_func(example):
        """
        Format the conversation for vision model training
        Converts multimodal conversations (text + images) to model format
        """
        try:
            messages = example.get("messages", [])

            if not messages:
                print("Warning: Empty messages in example")
                return ""

            # Debug counter for first few examples
            if not hasattr(formatting_func, 'debug_count'):
                formatting_func.debug_count = 0

            # Debug output for first 3 examples
            if formatting_func.debug_count < 3:
                #print(f"\n=== Formatting Debug {formatting_func.debug_count} ===")
                #print(f"Messages: {len(messages)} messages")

                for i, msg in enumerate(messages):
                    #print(f"  Message {i} - Role: {msg.get('role', 'NO_ROLE')}")
                    content = msg.get('content', [])

                    if isinstance(content, list):
                        for j, content_item in enumerate(content):
                            content_type = content_item.get('type', 'unknown')
                            if content_type == 'text':
                                text_preview = content_item.get('text', '')[:50]
                                #print(f"    Content {j} (text): {text_preview}...")
                            #else:
                                #print(f"    Content {j} ({content_type})")
                    #else:
                        #print(f"    Content: {str(content)[:50]}...")

            # Apply chat template to convert messages to model format
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            # Cleanup corrupted characters and control characters
            if formatted:
                formatted = formatted.replace('锦', '')  # Remove specific corrupted character
                formatted = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', formatted)  # Remove control chars
                formatted = formatted.strip()

            # Debug output
            if formatting_func.debug_count < 3:
                #print(f"Formatted length: {len(formatted) if formatted else 0}")
                #print(f"Formatted preview: {formatted[:200] if formatted else 'EMPTY'}...")
                formatting_func.debug_count += 1
                #print("=" * 50)

            return formatted or ""

        except Exception as e:
            print(f"Error in formatting_func: {e}")
            print(f"Example keys: {list(example.keys()) if isinstance(example, dict) else 'Not a dict'}")
            return ""

    def extract_assistant_response(text, tokenizer):
        """
        Extract only the assistant's response from the formatted text
        This focuses evaluation on what the model should actually predict
        """
        try:
            if not text or not isinstance(text, str):
                return ""

            # Multiple regex patterns to handle different chat template formats
            patterns = [
                # Llama format - most permissive
                r'<\|start_header_id\|>assistant<\|end_header_id\|>\s*\n*(.*?)(?:<\|eot_id\|>|$)',
                # ChatML format
                r'<\|assistant\|>\s*(.*?)(?:<\||$)',
                # Simple Assistant: format
                r'[Aa]ssistant:\s*(.*?)(?:\n\n|<\||$)',
                # Generic assistant format
                r'[Aa]ssistant[^a-zA-Z]+(.*?)(?:<\||$)',
                # Fallback - everything after last "assistant"
                r'.*[Aa]ssistant[^a-zA-Z]+(.*?)$'
            ]

            best_response = ""

            # Try each pattern
            for i, pattern in enumerate(patterns):
                matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
                if matches:
                    # Take the last assistant response (the one to predict)
                    response = matches[-1].strip()

                    # Basic cleanup
                    response = response.replace('锦', '')
                    response = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', response)

                    # Remove end-of-sequence tokens and artifacts
                    response = re.sub(r'<\|.*?\|>\s*$', '', response)
                    response = re.sub(r'!!+\s*$', '', response)  # Remove !!!!! artifacts
                    response = response.strip()

                    # Keep the longest valid response found
                    if len(response) > len(best_response):
                        best_response = response

                    if extract_assistant_response.debug_count <= 3:
                        print(f"Pattern {i} found: '{response[:100]}...'")

            # Final fallback: simple split on "assistant"
            if not best_response and 'assistant' in text.lower():
                text_lower = text.lower()
                last_assistant_pos = text_lower.rfind('assistant')
                if last_assistant_pos != -1:
                    after_assistant = text[last_assistant_pos:]
                    response = re.sub(r'^[Aa]ssistant[^a-zA-Z]*', '', after_assistant)
                    response = response.strip()

                    if response:
                        response = response.replace('锦', '')
                        response = re.sub(r'<\|.*?\|>\s*$', '', response)
                        response = re.sub(r'!!+\s*$', '', response)
                        best_response = response.strip()

                        if extract_assistant_response.debug_count <= 3:
                            print(f"Fallback split found: '{response[:100]}...'")

            return best_response

        except Exception as e:
            print(f"Error extracting assistant response: {e}")
            return ""

    def compute_exact_match(eval_prediction):
        """
        Compute exact match scores for assistant responses only
        Provides strict evaluation metric
        """
        pred_ids, label_ids = eval_prediction.predictions, eval_prediction.label_ids

        try:
            # Convert tensors to numpy arrays if needed
            if hasattr(pred_ids, 'cpu'):
                pred_ids = pred_ids.cpu().numpy()
            if hasattr(label_ids, 'cpu'):
                label_ids = label_ids.cpu().numpy()

            #print(f"\n=== EVALUATION START ===")
            #print(f"Prediction shape: {pred_ids.shape}, Label shape: {label_ids.shape}")

            def normalize_text(text):
                """
                Normalize text for robust comparison
                Handles case, punctuation, spacing, and common articles
                """
                if not text or not isinstance(text, str):
                    return ""

                # Basic normalization
                text = text.strip().lower()
                text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
                text = unicodedata.normalize("NFKC", text)  # Normalize unicode
                text = re.sub(r"\s+", " ", text)  # Multiple spaces to single

                # Remove common articles at start to avoid false negatives
                articles = [r'^the\s+', r'^a\s+', r'^an\s+']
                for article in articles:
                    text = re.sub(article, '', text)

                return text.strip()

            # Storage for results
            decoded_preds = []
            decoded_labels = []
            assistant_preds = []
            assistant_labels = []

            # Process each sample in the batch
            for i, (pred_row, label_row) in enumerate(zip(pred_ids, label_ids)):
                try:
                    # Filter out -100 tokens (ignore tokens in loss calculation)
                    valid_mask = label_row != -100
                    valid_labels = label_row[valid_mask]

                    # Align prediction length with valid labels
                    if len(pred_row) >= len(valid_labels):
                        valid_preds = pred_row[valid_mask]
                    else:
                        valid_preds = pred_row

                    # Ensure same length for both
                    min_len = min(len(valid_preds), len(valid_labels))
                    valid_preds = valid_preds[:min_len]
                    valid_labels = valid_labels[:min_len]

                    # Clamp token IDs to valid vocabulary range
                    vocab_size = getattr(tokenizer, 'vocab_size',
                                         len(tokenizer.get_vocab()) if hasattr(tokenizer, 'get_vocab') else 128256)
                    valid_preds = np.clip(valid_preds, 0, vocab_size - 1)
                    valid_labels = np.clip(valid_labels, 0, vocab_size - 1)

                    # Decode tokens to text
                    full_pred = tokenizer.decode(valid_preds, skip_special_tokens=True)
                    full_label = tokenizer.decode(valid_labels, skip_special_tokens=True)

                    decoded_preds.append(full_pred)
                    decoded_labels.append(full_label)

                    # Extract only assistant responses for evaluation
                    assistant_pred = extract_assistant_response(full_pred, tokenizer)
                    assistant_label = extract_assistant_response(full_label, tokenizer)

                    assistant_preds.append(assistant_pred)
                    assistant_labels.append(assistant_label)

                except Exception as sample_error:
                    print(f"Error processing sample {i}: {sample_error}")
                    assistant_preds.append("")
                    assistant_labels.append("")

            # Calculate exact match metric
            exact_matches = []

            for i, (pred, label) in enumerate(zip(assistant_preds, assistant_labels)):
                norm_pred = normalize_text(pred)
                norm_label = normalize_text(label)

                # Exact Match calculation
                if not norm_pred and not norm_label:
                    exact_match = True  # Both empty
                elif not norm_pred or not norm_label:
                    exact_match = False  # One empty, one not
                else:
                    exact_match = norm_pred == norm_label

                exact_matches.append(int(exact_match))

            # Final metrics
            exact_match_score = float(np.mean(exact_matches)) if exact_matches else 0.0

            print(f"\n=== EVALUATION RESULTS ===")
            print(f"Total samples: {len(exact_matches)}")
            print(f"Exact matches: {sum(exact_matches)}")
            print(f"Exact Match Score: {exact_match_score:.4f} ({100 * exact_match_score:.1f}%)")
            print(f"Match rate: {sum(exact_matches)}/{len(exact_matches)}")
            print("=" * 30)

            return {
                "exact_match": exact_match_score
            }

        except Exception as e:
            print(f"Critical error in compute_exact_match: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return {"exact_match": 0.0}

    def preprocess_logits_for_metrics(logits, labels):
        """
        Preprocess logits to extract predictions for metrics calculation
        Handles both single logits and tuple of logits
        """
        if isinstance(logits, tuple):
            logits = logits[0]

        # Get predictions from argmax of logits
        predictions = logits.argmax(dim=-1)

        # Ensure predictions match label shape
        if predictions.shape != labels.shape:
            print(f"Shape mismatch: predictions {predictions.shape} vs labels {labels.shape}")
            min_seq_len = min(predictions.shape[-1], labels.shape[-1])
            predictions = predictions[:, :min_seq_len]

        return predictions

    # === TRAINING SETUP ===

    print("Setting up training configuration...")

    # Prepare model for training
    FastVisionModel.for_training(model)

    # Mixed precision setup
    use_bf16 = is_bf16_supported() and config.get("use_bf16", True)
    use_fp16 = not use_bf16

    # Logging setup
    report_to = "wandb" if wandb_run else "none"

    # Output directory
    output_dir = config.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Training steps calculation
    epochs = config.get("epochs", 1)
    batch_size = config.get("batch_size", 2)
    grad_accum = config.get("grad_accum", 4)
    effective_batch_size = batch_size * grad_accum

    max_steps = epochs * (len_train_dataset // effective_batch_size) if len_train_dataset else 1000

    eval_steps = config.get("eval_steps", max(1, max_steps // 10))

    # Save checkpoints
    n_saves = config.get("n_saves", 5)
    raw_save_steps = max(1, max_steps // n_saves)
    save_steps = max(eval_steps,
                     eval_steps * (raw_save_steps // eval_steps + (1 if raw_save_steps % eval_steps else 0)))


    print(f"Training configuration:")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Gradient accumulation: {grad_accum}")
    print(f"  - Effective batch size: {effective_batch_size}")
    print(f"  - Max steps: {max_steps}")
    print(f"  - Save every: {save_steps} steps")
    print(f"  - Mixed precision: {'BF16' if use_bf16 else 'FP16'}")

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=streaming_dataset,
        eval_dataset=val_dataset,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_exact_match,
        formatting_func=formatting_func,
        args=SFTConfig(
            # Batch and gradient settings
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,

            # Learning settings
            learning_rate=config.get("lr", 2e-4),
            warmup_steps=config.get("warmup_steps", 5),
            max_steps=max_steps,
            num_train_epochs=1,  # Use max_steps instead

            # Precision and optimization
            fp16=use_fp16,
            bf16=use_bf16,
            max_grad_norm=config.get("max_grad_norm", 1.0),
            optim=config.get("optim", "adamw_8bit"),
            weight_decay=config.get("weight_decay", 0.01),
            lr_scheduler_type=config.get("scheduler", "linear"),

            # Sequence and data settings
            max_seq_length=config.get("max_seq_length", 2048),
            remove_unused_columns=False,
            dataset_text_field="",  # Use formatting_func instead
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=config.get("num_proc", 4),

            # Logging and saving
            logging_steps=config.get("logging_steps", 1),
            save_strategy="steps",
            save_steps=save_steps,
            output_dir=output_dir,
            report_to=report_to,
            run_name=wandb_run.name if wandb_run else None,

            # Evaluation settings
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=config.get("eval_steps", eval_steps) if val_dataset else None,
            metric_for_best_model="eval_exact_match" if val_dataset else None,
            greater_is_better=True if val_dataset else None,
            load_best_model_at_end=True if val_dataset else False,

            # System settings
            seed=config.get("seed", 3407),
        ),
    )

    # Add memory management callback
    class MemoryManagementCallback(TrainerCallback):
        """Callback to manage GPU memory during training"""

        def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            # Clear GPU cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            gc.collect()
            return control

        def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            # Clear cache before evaluation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return control

    trainer.add_callback(MemoryManagementCallback())

    # Add early stopping if requested
    if config.get("use_early_stopping", False) and val_dataset:
        patience_ratio = config.get("early_stopping_patience_ratio", 0.1)
        patience_steps = max(1, int(max_steps * patience_ratio))

        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=patience_steps,
            early_stopping_threshold=config.get("early_stopping_threshold", 0.001),
        )

        trainer.add_callback(early_stopping_callback)
        print(f"Early stopping enabled: patience={patience_steps} steps")

    # === TRAINING ===

    print(f"\nStarting training...")
    print(f"Dataset size: {len_train_dataset if len_train_dataset else 'Unknown'}")
    print(f"Validation dataset: {'Yes' if val_dataset else 'No'}")

    try:
        trainer.train()
        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise

    # === MODEL SAVING ===

    model_name = config.get("name_trained_model", "VizSage_final_model")
    final_model_path = os.path.join(output_dir, model_name)

    print(f"\nSaving model to: {final_model_path}")

    try:
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print("Model saved successfully!")

    except Exception as e:
        print(f"Error saving model: {e}")
        raise

    # === WANDB LOGGING ===

    if wandb_run:
        try:
            # Update run summary
            wandb_run.summary.update({
                "training_completed": True,
                "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "final_model_path": final_model_path,
                "total_steps": max_steps,
                "final_loss": trainer.state.log_history[-1].get("train_loss",
                                                                "N/A") if trainer.state.log_history else "N/A"
            })

            # Create and save model card
            model_card_text = f"""
# VizSage Model Training Summary

## Model Information
- **Base Model**: {config.get('model_name', 'Unknown')}
- **Dataset**: {config.get('dataset', 'Unknown')}
- **Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Model Path**: {final_model_path}

## Training Configuration
- **Batch Size**: {batch_size}
- **Gradient Accumulation**: {grad_accum}
- **Effective Batch Size**: {effective_batch_size}
- **Learning Rate**: {config.get('lr', 2e-4)}
- **Max Steps**: {max_steps}
- **Max Sequence Length**: {config.get('max_seq_length', 2048)}
- **Mixed Precision**: {'BF16' if use_bf16 else 'FP16'}

## LoRA Configuration
- **LoRA Rank**: {config.get('lora_r', 'N/A')}
- **LoRA Alpha**: {config.get('lora_alpha', 'N/A')}
- **LoRA Dropout**: {config.get('lora_dropout', 'N/A')}

## Training Mode
- **Mode**: Streaming Dataset
- **Evaluation**: {'Enabled' if val_dataset else 'Disabled'}
- **Early Stopping**: {'Enabled' if config.get('use_early_stopping', False) else 'Disabled'}

## Performance
- **Final Training Loss**: {trainer.state.log_history[-1].get('train_loss', 'N/A') if trainer.state.log_history else 'N/A'}
- **Best Validation Score**: {trainer.state.best_metric if hasattr(trainer.state, 'best_metric') else 'N/A'}
"""

            # Log model card to wandb
            wandb.log({"model_card": wandb.Html(model_card_text.replace('\n', '<br>'))})

            # Save model card locally
            with open(os.path.join(final_model_path, "model_card.md"), "w") as f:
                f.write(model_card_text)

            print("Model card saved and logged to wandb")

        except Exception as e:
            print(f"Warning: Error updating wandb: {e}")

    print(f"\n🎉 Training pipeline completed!")
    print(f"📁 Model saved at: {final_model_path}")

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
        finetune_norm_layers=config.get("finetune_norm_layers", False),
    )
    return model, tokenizer


if __name__ == "__main__":

    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Check if a config file was provided as an argument
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config/config.yaml"

    # Load configuration
    config = config_utils.load_config(config_file)

    # Setup wandb logging
    wandb_run = setup_wandb(config)
    if wandb_run:
        print(f"Wandb logging enabled: {wandb_run.name}")
    else:
        print("Wandb logging disabled")

    # Load model using config parameters
    model, tokenizer = get_model_from_config(config)

    # Load instruction
    instruction = config.get("instruction")

    # Check use streaming option
    use_streaming = config.get("use_streaming", False)

    # Load dataset
    if use_streaming:
        train_dataset, val_dataset, test_dataset, len_train_dataset, len_test_dataset = data_utils.get_dataset(
            base_path=config.get("base_path", "data"),
            dataset=config.get("dataset", "AQUA"),
            external_knowledge=config.get("external_knowledge", False),
            use_streaming=use_streaming
        )
    else:
        train_dataset, val_dataset, test_dataset, len_train_dataset = data_utils.get_dataset(
            base_path=config.get("base_path", "data"),
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

        if (config.get("external_knowledge", False)):
            # Load the external knowledge dataset
            semart_dataset = pd.read_csv(config.get("external_knowledge_path", "data/semart.csv"), sep= '\t', encoding='latin1',header=0)
            # Prepare the streaming dataset
            stream_ready_dataset = data_utils.prepare_streaming_dataset(
                streaming_dataset=train_dataset,
                config=config,
                semart_dataset=semart_dataset,
                base_path=config.get("base_path", "data")
            )

            val_streaming_dataset = data_utils.prepare_streaming_dataset(
                streaming_dataset=val_dataset,
                config=config,
                semart_dataset=semart_dataset,
                base_path=config.get("base_path", "data")
            )
        else:
            # Prepare the streaming dataset without external knowledge
            stream_ready_dataset = data_utils.prepare_streaming_dataset(
                streaming_dataset=train_dataset,
                config=config,
                base_path=config.get("base_path", "data")
            )

            val_streaming_dataset = data_utils.prepare_streaming_dataset(
                streaming_dataset=val_dataset,
                config=config,
                base_path=config.get("base_path", "data")
            )

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
                # If external knowledge is used, get the description
                if config.get("external_knowledge", False) and test_sample["need_external_knowledge"]:
                    description = semart_dataset.loc[semart_dataset['image_file'] == image, 'description'].values[0]
                else:
                    description = None

                print("\n=== PRE-TRAINING INFERENCE ===")
                print(f"Question: {question}")
                print(f"Image path: {image}")
                print(f"Ground truth answer: {ground_truth}")
                pre_training_output, description_passed = m.make_inference(model=model, tokenizer=tokenizer, image_path=image, question=question,
                                                       instruction=instruction, description=description, base_path=config.get("base_path", "data"))

                print(f"Description: {description_passed if description_passed else 'No description provided'}")
                print(f"Model prediction: {pre_training_output}")

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
        trainer = train_streaming(model, tokenizer, stream_ready_dataset, config, wandb_run, len_train_dataset, val_dataset=val_streaming_dataset)

    else:
        if config.get("external_knowledge", False):
            # Load the external knowledge dataset
            semart_dataset = pd.read_csv(config.get("external_knowledge_path", "data/semart.csv"), sep= '\t', encoding='latin1',header=0)
            # Convert the dataset to a conversation format
            train_dataset = [data_utils.convert_to_conversation(sample, semart_dataset=semart_dataset) for sample in train_dataset]  # Aggiornato per usare data_utils
        else:
            converted_dataset = [data_utils.convert_to_conversation(sample) for sample in train_dataset]  # Aggiornato per usare data_utils

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
            pre_training_output = m.make_inference(model=model, tokenizer=tokenizer, image_path=image, question=question,
                                                   instruction=instruction, base_path=config.get("base_path", "data"))

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
            image_path=post_training_test_sample['image'],
            question=post_training_test_sample['question'],
            instruction=instruction,
            base_path=config.get("base_path", "data")
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