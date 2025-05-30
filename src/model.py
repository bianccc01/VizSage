from unsloth import FastVisionModel # FastLanguageModel for LLMs
import data_utils  # Importa dal nuovo file combinato invece di data_preprocessing
from transformers import AutoTokenizer, TextStreamer


def get_model(model_name = "unsloth/Llama-3.2-11B-Vision-Instruct", load_in_4bit = True,
              use_gradient_checkpointing = "unsloth", finetune_vision_layers = False, finetune_language_layers = True,
              finetune_attention_modules = True, finetune_mlp_modules = True, r = 16, lora_alpha = 16, lora_dropout = 0,
              bias = "none", random_state = 3407, use_rslora = False, loftq_config = None, finetune_norm_layers = False):

    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
        "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
        "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
        "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

        "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
        "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

        "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
        "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

        "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
        "unsloth/llava-1.5-7b-hf-bnb-4bit",
    ] # More models at https://huggingface.co/unsloth

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit = load_in_4bit, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = use_gradient_checkpointing # True or "unsloth" for long context
    )


    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = finetune_vision_layers, # False if not finetuning vision layers
        finetune_language_layers   = finetune_language_layers, # False if not finetuning language layers
        finetune_attention_modules = finetune_attention_modules, # False if not finetuning attention modules
        finetune_mlp_modules       = finetune_mlp_modules, # False if not finetuning MLP modules
        finetune_norm_layers       = finetune_norm_layers, # False if not finetuning norm layers

        r = r,           # The larger, the higher the accuracy, but might overfit
        lora_alpha = lora_alpha,  # Recommended alpha == r at least
        lora_dropout = lora_dropout,
        bias = bias, # "none" or "all" or "lora_only"
        random_state = random_state, # Random state for reproducibility
        use_rslora = use_rslora,  # We support rank stabilized LoRA
        loftq_config = loftq_config, # And LoftQ
        # target_modules = "all-linear", # Optional now! Can specify a list if needed
    )

    return model, tokenizer


def make_inference(model, tokenizer, image_path, question, instruction, max_length=512,
                   temperature=0.1, top_p=0.95, top_k=50, num_beams=1,
                   do_sample=True, return_dict=True, description=None, base_path="data"):
    """
    Make an inference with the model and tokenizer.
    """
    FastVisionModel.for_inference(model)  # Enable for inference!

    image = data_utils.extract_image(image_path, base_path=base_path)

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": instruction},
            {"type": "image", "image": image},
            {"type": "text", "text": description if description is not None else ""},
            {"type": "text", "text": question}
        ]}
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        text=input_text,
        images = image,
        add_special_tokens=True,
        return_tensors="pt",
    ).to("cuda")

    # Generate without streaming
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=1,
        do_sample=do_sample
    )

    # Get the full text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant's response
    if "assistant" in generated_text:
        assistant_response = generated_text.split("assistant", 1)[1].strip()
    else:
        assistant_response = generated_text

    return assistant_response, description if description else None


def load_model(output_dir, load_in_4bit=True):
    """
    Load the model and tokenizer from the output directory.
    """
    model, tokenizer = FastVisionModel.from_pretrained(
        output_dir,
        load_in_4bit=load_in_4bit,
        device_map="auto",
    )
    return model, tokenizer