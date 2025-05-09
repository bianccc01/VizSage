import yaml
import sys
import os
import re
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import model as m
import data_utils
import config_utils
from dotenv import load_dotenv
import wandb
import socket
from datetime import datetime
import random
from datasets import Dataset, load_from_disk
import pandas as pd
import numpy as np
import torch
import evaluate_metrics

# Load environment variables from .env file
load_dotenv()


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
            "use_validation": config.get("use_validation", True),
            "evaluation_metrics": config.get("evaluation_metrics", ["bleu", "rouge"]),
            "best_model_metric": config.get("best_model_metric", "rougeL_fmeasure"),
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


def train(model, tokenizer, converted_dataset, config, wandb_run=None, validation_dataset=None):
    """Train the model with parameters from config and optional validation dataset"""

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

    # Determine if we should use validation set
    evaluation_strategy = "no"
    if validation_dataset is not None and len(validation_dataset) > 0:
        evaluation_strategy = "epoch"
        print(f"Using validation dataset with {len(validation_dataset)} samples")
        # Log to wandb if enabled
        if wandb_run:
            wandb_run.config.update({"validation_samples": len(validation_dataset)})
    else:
        print("No validation dataset provided, skipping evaluation")
        validation_dataset = None

    # Setup metrics computation function
    metric_choices = config.get("evaluation_metrics", ["bleu", "rouge", "exact_match"])
    print(f"Using evaluation metrics: {metric_choices}")
    compute_metrics_fn = evaluate_metrics.compute_metrics_factory(tokenizer, metric_choices)

    # Get the best model metric configuration
    best_model_metric = config.get("best_model_metric", "rougeL_fmeasure")
    greater_is_better = config.get("greater_is_better", True)
    print(f"Using {best_model_metric} as the metric for best model selection (greater_is_better={greater_is_better})")

    # Configure trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted_dataset,
        eval_dataset=validation_dataset,  # Add validation dataset here
        compute_metrics=compute_metrics_fn if validation_dataset is not None else None,  # Add metrics function
        args=SFTConfig(
            per_device_train_batch_size=config.get("batch_size", 2),
            per_device_eval_batch_size=config.get("eval_batch_size", config.get("batch_size", 2)),
            # Batch size for validation
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

            # Evaluation strategy based on validation dataset availability
            evaluation_strategy=evaluation_strategy,
            eval_steps=config.get("eval_steps", None),  # Evaluate every n steps if specified

            # Save strategy
            save_strategy="epoch",  # Save at the end of each epoch
            save_total_limit=config.get("save_total_limit", 3),  # Keep only the 3 best models
            load_best_model_at_end=validation_dataset is not None,
            # Load the best model at the end if we have validation data
            metric_for_best_model=best_model_metric if validation_dataset is not None else None,
            greater_is_better=greater_is_better,  # If True, higher metric is better

            # Disable saving model weights to wandb
            hub_model_id=None,
            hub_strategy="end",
            push_to_hub=False,

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
        # Get validation metrics if available
        eval_metrics = {}
        if validation_dataset is not None and hasattr(trainer, "state"):
            if hasattr(trainer.state, "best_metric") and trainer.state.best_metric is not None:
                eval_metrics["best_" + best_model_metric] = trainer.state.best_metric

            # Log all training history metrics
            if hasattr(trainer.state, "log_history") and trainer.state.log_history:
                for log_entry in trainer.state.log_history:
                    if "eval_" in str(log_entry):
                        for key, value in log_entry.items():
                            if key.startswith("eval_"):
                                eval_metrics[key] = value

        wandb_run.summary.update({
            "train_samples": len(converted_dataset),
            "validation_samples": len(validation_dataset) if validation_dataset is not None else 0,
            "final_model_path": final_model_path,
            "training_completed": True,
            "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **eval_metrics  # Add all evaluation metrics to summary
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
        - Validation: {"Yes" if validation_dataset is not None else "No"}
        """

        # Add evaluation metrics to model card if available
        if eval_metrics:
            model_card_text += "\n\n## Evaluation Metrics\n"
            for key, value in eval_metrics.items():
                model_card_text += f"- {key}: {value:.4f}\n"

        model_card_text += f"\n## Local Path\n- Saved to: {final_model_path}\n"

        # Log the text summary directly to wandb
        wandb.log({"model_card": wandb.Html(model_card_text.replace('\n', '<br>'))})

        # Also save the model card locally
        with open(f"{final_model_path}/model_card.md", "w") as f:
            f.write(model_card_text)

    return trainer


def train_streaming(model, tokenizer, streaming_dataset, config, wandb_run=None, len_train_dataset=None,
                    validation_dataset=None):
    """Train the model with streaming dataset and optional validation dataset"""

    # Enable model for training
    FastVisionModel.for_training(model)

    # Set BF16/FP16 based on configuration and hardware support
    use_bf16 = is_bf16_supported() and config.get("use_bf16", True)
    use_fp16 = not use_bf16

    # Set wandb logging
    report_to = "wandb" if wandb_run else "none"

    # Set max steps for streaming con correzione dei tipi
    try:
        max_steps = int(config.get("max_steps", 1000))
        if max_steps <= 0:
            max_steps = 1000
    except (ValueError, TypeError):
        print(f"Warning: Invalid max_steps value, using default 1000")
        max_steps = 1000

    # Create output directory
    output_dir = config.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Calculate max steps based on dataset size con correzione dei tipi
    try:
        batch_size = int(config.get("batch_size", 2))
        grad_accum = int(config.get("grad_accum", 4))
        max_steps = int(len_train_dataset / (batch_size * grad_accum))
        # Assicurati che max_steps sia almeno 1
        max_steps = max(1, max_steps)
    except (ValueError, TypeError, ZeroDivisionError):
        print("Warning: Error calculating max_steps, using default 1000")
        max_steps = 1000

    # how many times to save the model con correzione dei tipi
    try:
        n_saves = int(config.get("n_saves", 5))
        if n_saves <= 0:
            n_saves = 5
        save_steps = max(1, max_steps // n_saves)
    except (ValueError, TypeError, ZeroDivisionError):
        print("Warning: Error calculating save_steps, using default value")
        save_steps = max(1, max_steps // 5)

    # Determine validation settings
    evaluation_strategy = "no"
    eval_steps = None
    if validation_dataset is not None and len(validation_dataset) > 0:
        evaluation_strategy = "steps"
        try:
            n_evals = int(config.get("n_evals", 5))
            if n_evals <= 0:
                n_evals = 5
            eval_steps = max(1, max_steps // n_evals)
        except (ValueError, TypeError, ZeroDivisionError):
            print("Warning: Error calculating eval_steps, using default value")
            eval_steps = max(1, max_steps // 5)

        print(f"Using validation dataset with {len(validation_dataset)} samples, evaluating every {eval_steps} steps")
        # Log to wandb if enabled
        if wandb_run:
            wandb_run.config.update({"validation_samples": len(validation_dataset), "eval_steps": eval_steps})
    else:
        print("No validation dataset provided, skipping evaluation")
        validation_dataset = None

    # Setup metrics computation function
    metric_choices = config.get("evaluation_metrics", ["bleu", "rouge", "exact_match"])
    print(f"Using evaluation metrics: {metric_choices}")
    compute_metrics_fn = evaluate_metrics.compute_metrics_factory(tokenizer, metric_choices)

    # Get the best model metric configuration
    best_model_metric = config.get("best_model_metric", "rougeL_fmeasure")
    greater_is_better_value = config.get("greater_is_better", True)
    # Assicurati che sia un booleano
    if isinstance(greater_is_better_value, str):
        greater_is_better = greater_is_better_value.lower() in ('true', 'yes', '1', 't', 'y')
    else:
        greater_is_better = bool(greater_is_better_value)

    print(f"Using {best_model_metric} as the metric for best model selection (greater_is_better={greater_is_better})")

    # Configura TrainingArguments invece di SFTConfig
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.get("batch_size", 2),
        per_device_eval_batch_size=config.get("eval_batch_size", config.get("batch_size", 2)),
        gradient_accumulation_steps=config.get("grad_accum", 4),
        warmup_steps=config.get("warmup_steps", 5),
        max_steps=max_steps,
        num_train_epochs=1,
        learning_rate=float(config.get("lr", 2e-4)),
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=config.get("logging_steps", 1),
        optim=config.get("optim", "adamw_8bit"),
        weight_decay=float(config.get("weight_decay", 0.01)),
        lr_scheduler_type=config.get("scheduler", "linear"),
        seed=config.get("seed", 3407),
        report_to=report_to,
        run_name=wandb_run.name if wandb_run else None,

        # Evaluation strategy
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,

        # Save strategy
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=config.get("save_total_limit", 3),
        load_best_model_at_end=validation_dataset is not None,
        metric_for_best_model=best_model_metric if validation_dataset is not None else None,
        greater_is_better=greater_is_better,

        remove_unused_columns=False,
        # Non includere dataset_text_field nei TrainingArguments
        # dataset_text_field="",
        # Non includere dataset_kwargs nei TrainingArguments
        # dataset_kwargs={"skip_prepare_dataset": True},
        # Rimossa max_seq_length da TrainingArguments poiché non è supportata
        # max_seq_length=config.get("max_seq_length", 2048),
    )

    # Configure trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=streaming_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics_fn if validation_dataset is not None else None,
        args=training_args,  # Usa training_args invece di SFTConfig
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
    )

    # Start training
    print(
        f"Starting streaming training with batch size {config.get('batch_size', 2)} and max steps {max_steps}")
    trainer.train()
    print("Training completed successfully!")

    name_trained_model = config.get("name_trained_model", "VizSage_final_model")

    # Save final model
    final_model_path = f"{output_dir}/{name_trained_model}"
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model saved to {final_model_path}")

    # Log model metadata to wandb if enabled
    if wandb_run:
        # Get validation metrics if available
        eval_metrics = {}
        if validation_dataset is not None and hasattr(trainer, "state"):
            if hasattr(trainer.state, "best_metric") and trainer.state.best_metric is not None:
                eval_metrics["best_" + best_model_metric] = trainer.state.best_metric

            # Log all training history metrics
            if hasattr(trainer.state, "log_history") and trainer.state.log_history:
                for log_entry in trainer.state.log_history:
                    if "eval_" in str(log_entry):
                        for key, value in log_entry.items():
                            if key.startswith("eval_"):
                                eval_metrics[key] = value

        # Aggiungi i risultati della valutazione al riepilogo di wandb
        wandb_run.summary.update({
            "training_completed": True,
            "completion_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "final_model_path": final_model_path,
            "validation_samples": len(validation_dataset) if validation_dataset is not None else 0,
            **eval_metrics  # Aggiungi tutte le metriche di valutazione al riepilogo
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
        - Validation: {"Yes" if validation_dataset is not None else "No"}
        """

        # Add evaluation metrics to model card if available
        if eval_metrics:
            model_card_text += "\n\n## Evaluation Metrics\n"
            for key, value in eval_metrics.items():
                model_card_text += f"- {key}: {value:.4f}\n"

        model_card_text += f"\n## Local Path\n- Saved to: {final_model_path}\n"

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
    # Check if validation should be used
    use_validation = config.get("use_validation", True)

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

    # Load external knowledge dataset if needed
    semart_dataset = None
    if config.get("external_knowledge", False):
        semart_dataset = pd.read_csv(config.get("external_knowledge_path", "data/semart.csv"), sep='\t',
                                     encoding='latin1', header=0)
        print(f"Loaded SemArt dataset with {len(semart_dataset)} entries")

    # Prepare validation dataset if available and requested
    val_ready_dataset = None
    if val_dataset and use_validation:
        print(f"Preparing validation dataset...")
        if use_streaming:
            # For streaming validation, process it first to convert to a standard dataset
            val_samples = []
            # Collect a limited number of validation samples to avoid memory issues
            max_val_samples = config.get("max_val_samples", 500)  # Limit the validation set size
            val_count = 0

            for sample in val_dataset:
                # Convert sample to conversation format
                if config.get("external_knowledge", False) and sample.get("need_external_knowledge", False):
                    converted = data_utils.convert_to_conversation(sample, semart_dataset=semart_dataset,
                                                                   base_path=config.get("base_path", "data"))
                else:
                    converted = data_utils.convert_to_conversation(sample, base_path=config.get("base_path", "data"))

                val_samples.append(converted)
                val_count += 1

                if val_count >= max_val_samples:
                    break

            if val_samples:
                val_ready_dataset = val_samples
                print(f"Created validation dataset with {len(val_ready_dataset)} samples")
            else:
                print("Could not create validation dataset from streaming source")
        else:
            # For regular validation dataset, convert all samples
            if config.get("external_knowledge", False):
                val_ready_dataset = [data_utils.convert_to_conversation(sample, semart_dataset=semart_dataset,
                                                                        base_path=config.get("base_path", "data")) for
                                     sample in val_dataset]
            else:
                val_ready_dataset = [
                    data_utils.convert_to_conversation(sample, base_path=config.get("base_path", "data")) for sample in
                    val_dataset]

            print(f"Created validation dataset with {len(val_ready_dataset)} samples")

    # Different setup for streaming vs non-streaming
    if use_streaming:
        print("Using streaming mode for dataset processing")

        # Prepare the streaming dataset
        if config.get("external_knowledge", False):
            stream_ready_dataset = data_utils.prepare_streaming_dataset(
                streaming_dataset=train_dataset,
                config=config,
                semart_dataset=semart_dataset,
                base_path=config.get("base_path", "data")
            )
        else:
            stream_ready_dataset = data_utils.prepare_streaming_dataset(
                streaming_dataset=train_dataset,
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
                if config.get("external_knowledge", False) and test_sample.get("need_external_knowledge", False):
                    description = semart_dataset.loc[semart_dataset['image'] == image, 'description'].values[0]
                else:
                    description = None

                print("\n=== PRE-TRAINING INFERENCE ===")
                print(f"Question: {question}")
                print(f"Image path: {image}")
                print(f"Ground truth answer: {ground_truth}")
                print("Model prediction:")
                pre_training_output = m.make_inference(model=model, tokenizer=tokenizer, image_path=image,
                                                       question=question,
                                                       instruction=instruction, description=description,
                                                       base_path=config.get("base_path", "data"))

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

        # Train the model with validation
        print("\n=== STARTING TRAINING ===")
        trainer = train_streaming(model, tokenizer, stream_ready_dataset, config, wandb_run, len_train_dataset,
                                  validation_dataset=val_ready_dataset)

    else:
        # Prepare training dataset
        if config.get("external_knowledge", False):
            # Convert the dataset to a conversation format with external knowledge
            converted_dataset = [data_utils.convert_to_conversation(sample, semart_dataset=semart_dataset,
                                                                    base_path=config.get("base_path", "data")) for
                                 sample in train_dataset]
        else:
            # Convert the dataset to a conversation format without external knowledge
            converted_dataset = [data_utils.convert_to_conversation(sample, base_path=config.get("base_path", "data"))
                                 for sample in train_dataset]

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
            pre_training_output = m.make_inference(model=model, tokenizer=tokenizer, image_path=image,
                                                   question=question,
                                                   instruction=instruction, base_path=config.get("base_path", "data"))

            # Save the test sample for post-training inference
            post_training_test_sample = sample
        else:
            post_training_test_sample = None

        # Train the model with validation
        print("\n=== STARTING TRAINING ===")
        trainer = train(model, tokenizer, converted_dataset, config, wandb_run, validation_dataset=val_ready_dataset)

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

        # Calculate metrics on the test sample
        try:
            metric_choices = config.get("evaluation_metrics", ["bleu", "rouge", "exact_match"])
            ref = [clean_text(post_training_test_sample['answer'])]
            pred = [clean_text(post_training_output)]

            print("\n=== TEST SAMPLE METRICS ===")
            if "bleu" in metric_choices:
                bleu_scores = evaluate_metrics.compute_bleu(ref, pred)
                for key, value in bleu_scores.items():
                    print(f"{key}: {value:.4f}")

            if "rouge" in metric_choices:
                rouge_scores = evaluate_metrics.compute_rouge(ref, pred)
                for key, value in rouge_scores.items():
                    print(f"{key}: {value:.4f}")

            if "exact_match" in metric_choices:
                exact_match = evaluate_metrics.compute_exact_match(ref, pred)
                print(f"exact_match: {exact_match['exact_match']:.4f}")

        except Exception as e:
            print(f"Error calculating metrics: {e}")

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


# Helper function to clean text for evaluation metrics
def clean_text(text):
    """Clean the text for evaluation metrics calculation."""
    if text is None:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert to lowercase
    text = text.lower()
    return text