"""
Enhanced configuration utilities with validation and defaults
"""

import os
import sys
import yaml
from typing import Dict, Any, List, Tuple


def load_config(config_file: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load training configuration from YAML file with validation

    Args:
        config_file: Path to YAML configuration file

    Returns:
        Configuration dictionary with defaults applied
    """

    # Ensure the config file exists
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found.")
        sys.exit(1)

    try:
        # Load the configuration file
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

        print(f"Loaded configuration from {config_file}")

        # Apply defaults
        config = apply_config_defaults(config)

        # Validate configuration
        is_valid, errors = validate_config(config)
        if not is_valid:
            print("❌ Configuration validation failed:")
            for error in errors:
                print(f"  • {error}")
            sys.exit(1)

        return config

    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)


def apply_config_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply default values to configuration

    Args:
        config: Input configuration dictionary

    Returns:
        Configuration with defaults applied
    """

    defaults = {
        # Basic training parameters
        "batch_size": 2,
        "epochs": 1,
        "lr": 2e-4,
        "warmup_steps": 5,
        "grad_accum": 4,
        "max_seq_length": 2048,

        # Optimization parameters
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "scheduler": "linear",
        "use_bf16": True,

        # LoRA parameters
        "lora_r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_rslora": False,
        "loftq_config": None,

        # Evaluation parameters (NEW)
        "use_text_normalization": True,
        "debug_exact_match": False,

        # Early stopping parameters
        "use_early_stopping": False,
        "early_stopping_patience_ratio": 0.1,
        "early_stopping_threshold": 0.001,

        # System parameters
        "seed": 3407,
        "num_proc": 4,
        "reproducible": False,

        # Paths and directories
        "base_path": "data",
        "output_dir": "outputs",
        "name_trained_model": "VizSage_final_model",

        # Logging parameters
        "logging_steps": 1,
        "n_saves": 5,
        "save_model": False,
        "save_path": "models/trained_model",

        # Wandb parameters
        "use_wandb": False,
        "wandb_project": "vizsage-training",
        "wandb_run_name": None,
        "wandb_tags": [],

        # Dataset parameters
        "use_streaming": False,
        "stream_buffer_size": 1000,
        "external_knowledge": False,
        "external_knowledge_path": "data/semart.csv",

        # Model parameters
        "load_in_4bit": True,
        "use_gradient_checkpointing": "unsloth",
        "finetune_vision_layers": False,
        "finetune_language_layers": True,
        "finetune_attention_modules": True,
        "finetune_mlp_modules": True,
        "finetune_norm_layers": False,

        # Inference parameters
        "instruction": "You are an expert art historian. Answer the questions you will be asked about the image.",
        "max_new_tokens": 128,
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 50,
        "num_beams": 1,
        "do_sample": True,
    }

    # Apply defaults for missing keys
    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value

    return config


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate configuration for common issues

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Required fields
    required_fields = ["model_name", "dataset"]
    for field in required_fields:
        if field not in config or not config[field]:
            errors.append(f"Missing required field: {field}")

    # Validate paths
    if "base_path" in config:
        base_path = config["base_path"]
        if not os.path.exists(base_path):
            errors.append(f"Base path does not exist: {base_path}")

    if config.get("external_knowledge", False):
        ext_path = config.get("external_knowledge_path")
        if not ext_path:
            errors.append("external_knowledge is True but external_knowledge_path not specified")
        elif not os.path.exists(ext_path):
            errors.append(f"External knowledge path does not exist: {ext_path}")

    # Validate numeric parameters
    numeric_validations = {
        "batch_size": (1, 128, int),
        "epochs": (1, 100, int),
        "lr": (1e-6, 1e-1, (int, float)),
        "max_seq_length": (128, 8192, int),
        "grad_accum": (1, 32, int),
        "warmup_steps": (0, 1000, int),
        "lora_r": (1, 256, int),
        "lora_alpha": (1, 512, int),
        "max_new_tokens": (1, 2048, int),
        "temperature": (0.0, 2.0, (int, float)),
        "top_p": (0.0, 1.0, (int, float)),
        "top_k": (1, 1000, int),
    }

    for param, (min_val, max_val, expected_type) in numeric_validations.items():
        if param in config:
            value = config[param]
            if not isinstance(value, expected_type):
                errors.append(f"{param} should be {expected_type}, got {type(value)}")
            elif value < min_val or value > max_val:
                errors.append(f"{param} should be between {min_val} and {max_val}, got {value}")

    # Validate boolean parameters
    bool_params = [
        "use_streaming", "external_knowledge", "save_model", "use_wandb",
        "use_text_normalization", "debug_exact_match", "use_early_stopping",
        "reproducible", "load_in_4bit", "use_bf16", "finetune_vision_layers",
        "finetune_language_layers", "finetune_attention_modules",
        "finetune_mlp_modules", "finetune_norm_layers", "do_sample", "use_rslora"
    ]

    for param in bool_params:
        if param in config and not isinstance(config[param], bool):
            errors.append(f"{param} should be boolean, got {type(config[param])}")

    # Validate string parameters
    string_params = [
        "model_name", "dataset", "base_path", "output_dir", "name_trained_model",
        "instruction", "optim", "scheduler", "bias"
    ]

    for param in string_params:
        if param in config and config[param] is not None and not isinstance(config[param], str):
            errors.append(f"{param} should be string, got {type(config[param])}")

    # Validate dataset path
    if "dataset" in config and "base_path" in config:
        dataset_path = os.path.join(config["base_path"], config["dataset"])
        if not os.path.exists(dataset_path):
            errors.append(f"Dataset path does not exist: {dataset_path}")

    return len(errors) == 0, errors


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a formatted summary of the configuration

    Args:
        config: Configuration dictionary
    """
    print("\n" + "=" * 60)
    print("📋 CONFIGURATION SUMMARY")
    print("=" * 60)

    # Model and dataset info
    print(f"🤖 Model: {config.get('model_name', 'Not specified')}")
    print(f"📊 Dataset: {config.get('dataset', 'Not specified')}")
    print(f"🗂️  Base path: {config.get('base_path', 'Not specified')}")

    # Training parameters
    print(f"\n⚙️  Training Parameters:")
    print(f"   • Batch size: {config.get('batch_size')}")
    print(f"   • Gradient accumulation: {config.get('grad_accum')}")
    print(f"   • Effective batch size: {config.get('batch_size') * config.get('grad_accum')}")
    print(f"   • Epochs: {config.get('epochs')}")
    print(f"   • Learning rate: {config.get('lr')}")
    print(f"   • Max sequence length: {config.get('max_seq_length')}")
    print(f"   • Mixed precision: {'BF16' if config.get('use_bf16') else 'FP16'}")

    # LoRA parameters
    print(f"\n🔧 LoRA Parameters:")
    print(f"   • Rank (r): {config.get('lora_r')}")
    print(f"   • Alpha: {config.get('lora_alpha')}")
    print(f"   • Dropout: {config.get('lora_dropout')}")
    print(f"   • Bias: {config.get('bias')}")

    # Evaluation parameters
    print(f"\n📈 Evaluation Parameters:")
    print(f"   • Text normalization: {'✅' if config.get('use_text_normalization') else '❌'}")
    print(f"   • Debug mode: {'✅' if config.get('debug_exact_match') else '❌'}")
    print(f"   • Early stopping: {'✅' if config.get('use_early_stopping') else '❌'}")

    # Dataset settings
    print(f"\n📂 Dataset Settings:")
    print(f"   • Streaming mode: {'✅' if config.get('use_streaming') else '❌'}")
    print(f"   • External knowledge: {'✅' if config.get('external_knowledge') else '❌'}")
    if config.get('external_knowledge'):
        print(f"   • Knowledge path: {config.get('external_knowledge_path')}")

    # Logging
    print(f"\n📊 Logging:")
    print(f"   • Wandb: {'✅' if config.get('use_wandb') else '❌'}")
    if config.get('use_wandb'):
        print(f"   • Project: {config.get('wandb_project')}")
        if config.get('wandb_run_name'):
            print(f"   • Run name: {config.get('wandb_run_name')}")

    # Output settings
    print(f"\n💾 Output:")
    print(f"   • Output dir: {config.get('output_dir')}")
    print(f"   • Model name: {config.get('name_trained_model')}")
    print(f"   • Save model: {'✅' if config.get('save_model') else '❌'}")

    # System settings
    print(f"\n🖥️  System:")
    print(f"   • Reproducible: {'✅' if config.get('reproducible') else '❌'}")
    if config.get('reproducible'):
        print(f"   • Seed: {config.get('seed')}")

    print("=" * 60)


def save_config(config: Dict[str, Any], filepath: str) -> None:
    """
    Save configuration to YAML file

    Args:
        config: Configuration dictionary
        filepath: Output file path
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        print(f"Configuration saved to {filepath}")

    except Exception as e:
        print(f"Error saving config: {e}")


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract model-specific configuration parameters

    Args:
        config: Full configuration dictionary

    Returns:
        Model configuration subset
    """
    model_keys = [
        "model_name", "load_in_4bit", "use_gradient_checkpointing",
        "finetune_vision_layers", "finetune_language_layers",
        "finetune_attention_modules", "finetune_mlp_modules",
        "finetune_norm_layers", "lora_r", "lora_alpha", "lora_dropout",
        "bias", "use_rslora", "loftq_config", "seed"
    ]

    return {key: config[key] for key in model_keys if key in config}


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract training-specific configuration parameters

    Args:
        config: Full configuration dictionary

    Returns:
        Training configuration subset
    """
    training_keys = [
        "batch_size", "epochs", "lr", "warmup_steps", "grad_accum",
        "max_seq_length", "optim", "weight_decay", "max_grad_norm",
        "scheduler", "use_bf16", "logging_steps", "n_saves",
        "use_early_stopping", "early_stopping_patience_ratio",
        "early_stopping_threshold", "seed", "num_proc"
    ]

    return {key: config[key] for key in training_keys if key in config}


def get_evaluation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract evaluation-specific configuration parameters

    Args:
        config: Full configuration dictionary

    Returns:
        Evaluation configuration subset
    """
    eval_keys = [
        "use_text_normalization", "debug_exact_match"
    ]

    return {key: config[key] for key in eval_keys if key in config}