"""
Enhanced model utilities with better configuration handling and error management
"""

import torch
from unsloth import FastVisionModel
import data_utils
from transformers import AutoTokenizer, TextStreamer
from typing import Dict, Any, Tuple, Optional, Union, List
import os


def get_model_from_config(config: Dict[str, Any]) -> Tuple[FastVisionModel, AutoTokenizer]:
    """
    Load model and tokenizer based on configuration parameters

    Args:
        config: Configuration dictionary containing model parameters

    Returns:
        Tuple of (model, tokenizer)
    """
    return get_model(
        model_name=config.get("model_name", "unsloth/Llama-3.2-11B-Vision-Instruct"),
        load_in_4bit=config.get("load_in_4bit", True),
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", "unsloth"),
        finetune_vision_layers=config.get("finetune_vision_layers", False),
        finetune_language_layers=config.get("finetune_language_layers", True),
        finetune_attention_modules=config.get("finetune_attention_modules", True),
        finetune_mlp_modules=config.get("finetune_mlp_modules", True),
        finetune_norm_layers=config.get("finetune_norm_layers", False),
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0),
        bias=config.get("bias", "none"),
        random_state=config.get("seed", 3407),
        use_rslora=config.get("use_rslora", False),
        loftq_config=config.get("loftq_config", None)
    )


def get_model(
    model_name: str = "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit: bool = True,
    use_gradient_checkpointing: Union[bool, str] = "unsloth",
    finetune_vision_layers: bool = False,
    finetune_language_layers: bool = True,
    finetune_attention_modules: bool = True,
    finetune_mlp_modules: bool = True,
    finetune_norm_layers: bool = False,
    r: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0,
    bias: str = "none",
    random_state: int = 3407,
    use_rslora: bool = False,
    loftq_config: Optional[Dict] = None
) -> Tuple[FastVisionModel, AutoTokenizer]:
    """
    Load and configure a vision model with LoRA for fine-tuning

    Args:
        model_name: Name/path of the model to load
        load_in_4bit: Whether to load model in 4-bit quantization
        use_gradient_checkpointing: Gradient checkpointing configuration
        finetune_vision_layers: Whether to fine-tune vision layers
        finetune_language_layers: Whether to fine-tune language layers
        finetune_attention_modules: Whether to fine-tune attention modules
        finetune_mlp_modules: Whether to fine-tune MLP modules
        finetune_norm_layers: Whether to fine-tune normalization layers
        r: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        bias: LoRA bias configuration
        random_state: Random seed for reproducibility
        use_rslora: Whether to use rank-stabilized LoRA
        loftq_config: LoftQ configuration

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ValueError: If model name is not supported
        RuntimeError: If model loading fails
    """

    print(f"Loading model: {model_name}")

    # List of supported 4-bit models
    fourbit_models = [
        "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
        "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit",
        "unsloth/Llama-3.2-90B-Vision-bnb-4bit",
        "unsloth/Pixtral-12B-2409-bnb-4bit",
        "unsloth/Pixtral-12B-Base-2409-bnb-4bit",
        "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
        "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",
        "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",
        "unsloth/llava-1.5-7b-hf-bnb-4bit",
    ]

    # Check if model supports 4-bit quantization
    if load_in_4bit and model_name not in fourbit_models and not os.path.exists(model_name):
        print(f"⚠️  Warning: {model_name} may not support 4-bit quantization")
        print(f"Supported 4-bit models: {fourbit_models}")

    try:
        # Load base model and tokenizer
        print("Loading base model and tokenizer...")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing=use_gradient_checkpointing
        )

        print("Configuring LoRA layers...")

        # Configure LoRA
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=finetune_vision_layers,
            finetune_language_layers=finetune_language_layers,
            finetune_attention_modules=finetune_attention_modules,
            finetune_mlp_modules=finetune_mlp_modules,
            finetune_norm_layers=finetune_norm_layers,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            random_state=random_state,
            use_rslora=use_rslora,
            loftq_config=loftq_config,
        )

        print("✅ Model loaded and configured successfully")

        # Print model configuration summary
        print_model_config_summary(
            model_name, load_in_4bit, r, lora_alpha, lora_dropout,
            finetune_vision_layers, finetune_language_layers
        )

        return model, tokenizer

    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name}: {e}")


def print_model_config_summary(
    model_name: str,
    load_in_4bit: bool,
    r: int,
    lora_alpha: int,
    lora_dropout: float,
    finetune_vision_layers: bool,
    finetune_language_layers: bool
) -> None:
    """Print a summary of model configuration"""
    print("\n" + "="*50)
    print("🤖 MODEL CONFIGURATION")
    print("="*50)
    print(f"Model: {model_name}")
    print(f"Quantization: {'4-bit' if load_in_4bit else '16-bit'}")
    print(f"LoRA Rank (r): {r}")
    print(f"LoRA Alpha: {lora_alpha}")
    print(f"LoRA Dropout: {lora_dropout}")
    print(f"Fine-tune Vision Layers: {'✅' if finetune_vision_layers else '❌'}")
    print(f"Fine-tune Language Layers: {'✅' if finetune_language_layers else '❌'}")
    print("="*50)


def make_inference(
    model: FastVisionModel,
    tokenizer: AutoTokenizer,
    image_path: str,
    question: str,
    instruction: str,
    max_new_tokens: int = 128,
    temperature: float = 0.1,
    top_p: float = 0.95,
    top_k: int = 50,
    num_beams: int = 1,
    do_sample: bool = True,
    description: Optional[str] = None,
    base_path: str = "data"
) -> Union[str, Tuple[str, str]]:
    """
    Make an inference with the model and tokenizer with enhanced error handling

    Args:
        model: Vision model for inference
        tokenizer: Tokenizer for text processing
        image_path: Path to the image file
        question: Question to ask about the image
        instruction: System instruction for the model
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        num_beams: Number of beams for beam search
        do_sample: Whether to use sampling
        description: Optional description for external knowledge
        base_path: Base path for image files

    Returns:
        Generated assistant response, or tuple of (response, description) if description provided

    Raises:
        FileNotFoundError: If image file not found
        RuntimeError: If inference fails
    """

    try:
        # Ensure Unsloth returns logits if needed
        os.environ["UNSLOTH_RETURN_LOGITS"] = "1"

        # Enable inference mode
        FastVisionModel.for_inference(model)

        # Load and validate image
        try:
            image = data_utils.extract_image(image_path, base_path=base_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {image_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")

        # Build message content
        content = [
            {"type": "text", "text": instruction},
            {"type": "image", "image": image}
        ]

        # Add description if provided
        if description:
            content.append({"type": "text", "text": description})

        # Add question
        content.append({"type": "text", "text": question})

        messages = [{"role": "user", "content": content}]

        # Apply chat template
        try:
            input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        except Exception as e:
            raise RuntimeError(f"Error applying chat template: {e}")

        # Tokenize input
        try:
            inputs = tokenizer(
                text=input_text,
                images=image,
                add_special_tokens=True,
                return_tensors="pt",
            ).to(model.device)
        except Exception as e:
            raise RuntimeError(f"Error tokenizing input: {e}")

        # Generate response
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_beams=num_beams,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id
                )
        except Exception as e:
            raise RuntimeError(f"Error during generation: {e}")

        # Decode response
        try:
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            raise RuntimeError(f"Error decoding output: {e}")

        # Extract assistant's response
        assistant_response = extract_assistant_response(generated_text)

        # Return response or tuple based on whether description was provided
        if description is not None:
            return assistant_response, description
        else:
            return assistant_response

    except Exception as e:
        print(f"Inference failed: {e}")
        raise


def extract_assistant_response(generated_text: str) -> str:
    """
    Extract the assistant's response from generated text

    Args:
        generated_text: Full generated text from model

    Returns:
        Cleaned assistant response
    """
    try:
        # Look for assistant response marker
        if "assistant" in generated_text.lower():
            # Split on assistant and take the last part
            parts = generated_text.lower().split("assistant")
            if len(parts) > 1:
                # Get the response after the last "assistant"
                assistant_part = generated_text[generated_text.lower().rfind("assistant"):]
                # Remove the "assistant" prefix
                response = assistant_part[assistant_part.find("assistant") + len("assistant"):].strip()
                # Clean up any remaining formatting
                response = response.lstrip(":").strip()
                return response

        # Fallback: return the generated text as-is
        return generated_text.strip()

    except Exception as e:
        print(f"Error extracting assistant response: {e}")
        return generated_text.strip()


def load_model(
    output_dir: str,
    load_in_4bit: bool = True,
    device_map: str = "auto"
) -> Tuple[FastVisionModel, AutoTokenizer]:
    """
    Load a previously saved model and tokenizer from directory

    Args:
        output_dir: Directory containing saved model
        load_in_4bit: Whether to load in 4-bit quantization
        device_map: Device mapping strategy

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        FileNotFoundError: If model directory doesn't exist
        RuntimeError: If model loading fails
    """

    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Model directory not found: {output_dir}")

    try:
        print(f"Loading model from: {output_dir}")

        model, tokenizer = FastVisionModel.from_pretrained(
            output_dir,
            load_in_4bit=load_in_4bit,
            device_map=device_map,
        )

        print("✅ Model loaded successfully from checkpoint")
        return model, tokenizer

    except Exception as e:
        raise RuntimeError(f"Failed to load model from {output_dir}: {e}")


def get_model_memory_usage() -> Dict[str, float]:
    """
    Get current GPU memory usage information

    Returns:
        Dictionary with memory usage statistics (in GB)
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    try:
        current_device = torch.cuda.current_device()

        # Get memory info
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1e9
        memory_reserved = torch.cuda.memory_reserved(current_device) / 1e9
        max_memory_allocated = torch.cuda.max_memory_allocated(current_device) / 1e9

        # Get device properties
        props = torch.cuda.get_device_properties(current_device)
        total_memory = props.total_memory / 1e9

        return {
            "device": current_device,
            "device_name": props.name,
            "total_memory_gb": total_memory,
            "allocated_gb": memory_allocated,
            "reserved_gb": memory_reserved,
            "max_allocated_gb": max_memory_allocated,
            "free_gb": total_memory - memory_reserved,
            "utilization_percent": (memory_reserved / total_memory) * 100
        }

    except Exception as e:
        return {"error": f"Error getting memory info: {e}"}


def print_model_memory_usage() -> None:
    """Print formatted model memory usage information"""
    memory_info = get_model_memory_usage()

    if "error" in memory_info:
        print(f"❌ {memory_info['error']}")
        return

    print("\n" + "="*40)
    print("💾 GPU MEMORY USAGE")
    print("="*40)
    print(f"Device: {memory_info['device']} ({memory_info['device_name']})")
    print(f"Total Memory: {memory_info['total_memory_gb']:.1f} GB")
    print(f"Allocated: {memory_info['allocated_gb']:.1f} GB")
    print(f"Reserved: {memory_info['reserved_gb']:.1f} GB")
    print(f"Free: {memory_info['free_gb']:.1f} GB")
    print(f"Utilization: {memory_info['utilization_percent']:.1f}%")
    print("="*40)


def clear_model_cache() -> None:
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        print("✅ GPU memory cache cleared")
    else:
        print("❌ CUDA not available, cannot clear cache")


def validate_generation_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate generation configuration parameters

    Args:
        config: Generation configuration

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    validations = {
        "max_new_tokens": (1, 2048, int),
        "temperature": (0.0, 2.0, (int, float)),
        "top_p": (0.0, 1.0, (int, float)),
        "top_k": (1, 1000, int),
        "num_beams": (1, 10, int),
    }

    for param, (min_val, max_val, expected_type) in validations.items():
        if param in config:
            value = config[param]
            if not isinstance(value, expected_type):
                errors.append(f"{param} should be {expected_type}, got {type(value)}")
            elif value < min_val or value > max_val:
                errors.append(f"{param} should be between {min_val} and {max_val}, got {value}")

    if "do_sample" in config and not isinstance(config["do_sample"], bool):
        errors.append("do_sample should be boolean")

    return len(errors) == 0, errors