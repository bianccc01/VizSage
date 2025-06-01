"""
Training utilities including callbacks and configuration setup
"""

import os
import gc
import torch
from datetime import datetime
from typing import Dict, Any, Optional
from transformers import TrainerCallback, TrainerState, TrainerControl, EarlyStoppingCallback
import wandb


def setup_training_config(config: Dict[str, Any], len_train_dataset: Optional[int] = None) -> Dict[str, Any]:
    """
    Setup and validate training configuration

    Args:
        config: Base configuration dictionary
        len_train_dataset: Length of training dataset for step calculation

    Returns:
        Enhanced configuration with calculated values
    """
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

    # Update config with calculated values
    enhanced_config = config.copy()
    enhanced_config.update({
        "effective_batch_size": effective_batch_size,
        "max_steps": max_steps,
        "eval_steps": eval_steps,
        "save_steps": save_steps
    })

    return enhanced_config


def print_training_config(config: Dict[str, Any]) -> None:
    """Print training configuration in a readable format"""
    print(f"Training configuration:")
    print(f"  - Epochs: {config.get('epochs', 1)}")
    print(f"  - Batch size: {config.get('batch_size', 2)}")
    print(f"  - Gradient accumulation: {config.get('grad_accum', 4)}")
    print(f"  - Effective batch size: {config.get('effective_batch_size', 'N/A')}")
    print(f"  - Max steps: {config.get('max_steps', 'N/A')}")
    print(f"  - Save every: {config.get('save_steps', 'N/A')} steps")
    print(f"  - Mixed precision: {'BF16' if config.get('use_bf16', True) else 'FP16'}")
    print(f"  - Text normalization: {'ON' if config.get('use_text_normalization', True) else 'OFF'}")
    print(f"  - Debug exact match: {'ON' if config.get('debug_exact_match', False) else 'OFF'}")


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


def setup_early_stopping(config: Dict[str, Any], max_steps: int, val_dataset) -> Optional[EarlyStoppingCallback]:
    """
    Setup early stopping callback if requested

    Args:
        config: Configuration dictionary
        max_steps: Maximum training steps
        val_dataset: Validation dataset (None if no validation)

    Returns:
        EarlyStoppingCallback if configured, None otherwise
    """
    if not config.get("use_early_stopping", False) or not val_dataset:
        return None

    patience_ratio = config.get("early_stopping_patience_ratio", 0.1)
    patience_steps = max(1, int(max_steps * patience_ratio))

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=patience_steps,
        early_stopping_threshold=config.get("early_stopping_threshold", 0.001),
    )

    print(f"Early stopping enabled: patience={patience_steps} steps")
    return early_stopping_callback


def save_model_and_tokenizer(model, tokenizer, config: Dict[str, Any]) -> str:
    """
    Save model and tokenizer to disk

    Args:
        model: Trained model
        tokenizer: Tokenizer
        config: Configuration dictionary

    Returns:
        Path where model was saved
    """
    output_dir = config.get("output_dir", "outputs")
    model_name = config.get("name_trained_model", "VizSage_final_model")
    final_model_path = os.path.join(output_dir, model_name)

    print(f"\nSaving model to: {final_model_path}")

    try:
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print("Model saved successfully!")
        return final_model_path

    except Exception as e:
        print(f"Error saving model: {e}")
        raise


def create_model_card(config: Dict[str, Any], trainer, final_model_path: str) -> str:
    """
    Create model card with training information

    Args:
        config: Configuration dictionary
        trainer: Trained SFTTrainer object
        final_model_path: Path where model was saved

    Returns:
        Model card text
    """
    model_card_text = f"""
# VizSage Model Training Summary

## Model Information
- **Base Model**: {config.get('model_name', 'Unknown')}
- **Dataset**: {config.get('dataset', 'Unknown')}
- **Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Model Path**: {final_model_path}

## Training Configuration
- **Batch Size**: {config.get('batch_size', 'N/A')}
- **Gradient Accumulation**: {config.get('grad_accum', 'N/A')}
- **Effective Batch Size**: {config.get('effective_batch_size', 'N/A')}
- **Learning Rate**: {config.get('lr', 2e-4)}
- **Max Steps**: {config.get('max_steps', 'N/A')}
- **Max Sequence Length**: {config.get('max_seq_length', 2048)}
- **Mixed Precision**: {'BF16' if config.get('use_bf16', True) else 'FP16'}

## Evaluation Configuration
- **Text Normalization**: {'Enabled' if config.get('use_text_normalization', True) else 'Disabled'}
- **Debug Mode**: {'Enabled' if config.get('debug_exact_match', False) else 'Disabled'}

## LoRA Configuration
- **LoRA Rank**: {config.get('lora_r', 'N/A')}
- **LoRA Alpha**: {config.get('lora_alpha', 'N/A')}
- **LoRA Dropout**: {config.get('lora_dropout', 'N/A')}

## Training Mode
- **Mode**: Streaming Dataset
- **Evaluation**: {'Enabled' if config.get('val_dataset') else 'Disabled'}
- **Early Stopping**: {'Enabled' if config.get('use_early_stopping', False) else 'Disabled'}

## Performance
- **Final Training Loss**: {trainer.state.log_history[-1].get('train_loss', 'N/A') if trainer.state.log_history else 'N/A'}
- **Best Validation Score**: {trainer.state.best_metric if hasattr(trainer.state, 'best_metric') else 'N/A'}
"""
    return model_card_text


def update_wandb_summary(wandb_run, config: Dict[str, Any], trainer, final_model_path: str) -> None:
    """
    Update wandb run summary with training results

    Args:
        wandb_run: Wandb run object
        config: Configuration dictionary
        trainer: Trained SFTTrainer object
        final_model_path: Path where model was saved
    """
    if not wandb_run:
        return

    try:
        # Update run summary
        wandb_run.summary.update({
            "training_completed": True,
            "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "final_model_path": final_model_path,
            "total_steps": config.get('max_steps', 'N/A'),
            "final_loss": trainer.state.log_history[-1].get("train_loss",
                                                            "N/A") if trainer.state.log_history else "N/A"
        })

        # Create and save model card
        model_card_text = create_model_card(config, trainer, final_model_path)

        # Log model card to wandb
        wandb.log({"model_card": wandb.Html(model_card_text.replace('\n', '<br>'))})

        # Save model card locally
        with open(os.path.join(final_model_path, "model_card.md"), "w") as f:
            f.write(model_card_text)

        print("Model card saved and logged to wandb")

    except Exception as e:
        print(f"Warning: Error updating wandb: {e}")


def is_bf16_supported() -> bool:
    """
    Check if BF16 is supported on current hardware

    Returns:
        True if BF16 is supported, False otherwise
    """
    if not torch.cuda.is_available():
        return False

    try:
        # Check if current GPU supports BF16
        return torch.cuda.get_device_capability()[0] >= 8
    except:
        return False