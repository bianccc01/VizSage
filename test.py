import model
import config
import os
import sys
import yaml
from transformers import FastVisionModel
from transformers import TextStreamer
import data_preprocessing as dp
import tqdm

import utils

if __name__ == "__main__":
    # Check if a config file was provided as an argument
    config_file = "config.yaml"

    # Load configuration
    config = utils.load_config(config_file)
    print(f"Loaded configuration from {config_file}")
    path = config.get("output")+ "/" +config.get("model_name")
    load_in_4bit = config.get("load_in_4bit")
    model, tokenizer = model.load_model(
        output_dir=path,
        load_in_4bit=load_in_4bit
    )

    FastVisionModel.for_inference(model) # Enable for inference!
    # Load the test dataset

    use_streaming = config.get("use_streaming", True)

    train_dataset, val_dataset, test_dataset = dp.get_dataset(
        base_path="data",
        dataset=config.get("dataset", "AQUA"),
        external_knowledge=config.get("external_knowledge", False),
        use_streaming=use_streaming
    )

    # create a json with image_path, question, response for each sample of test_dataset
    output_json = []
    with tqdm (total=len(test_dataset), desc="Processing samples") as pbar:
        for sample in test_dataset:
            image_path = sample.get("image")
            question = sample.get("question")
            ground_truth = sample.get("answer")

            # Check if the image path is valid
            if not os.path.exists(image_path):
                print(f"Image path {image_path} does not exist.")
                continue

            instruction = config.get("instruction")

            # Make inference
            response = model.make_inference(
                model = model,
                tokenizer = tokenizer,
                image_path = image_path,
                question = question,
                instruction = instruction
            )

            # Append to output json
            output_json.append({
                "image_path": image_path,
                "question": question,
                "response": response,
                "ground_truth": ground_truth
            })

            pbar.update(1)


    # Save the output json to a file
    output_file = os.path.join(path, "test_output.json")
    with open(output_file, "w") as f:
        yaml.dump(output_json, f)
    print(f"Output saved to {output_file}")
