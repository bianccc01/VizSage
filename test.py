import model as model_utils
import config
import os
import sys
import yaml
from transformers import FastVisionModel
from transformers import TextStreamer
import data_preprocessing as dp
from tqdm import tqdm
import torch
import pandas as pd

import utils

if __name__ == "__main__":
    # Check if a config file was provided as an argument
    config_file = "config.yaml"

    # Load configuration
    config = utils.load_config(config_file)
    print(f"Loaded configuration from {config_file}")
    path = config.get("output_dir")+ "/" + config.get("name_trained_model")
    load_in_4bit = config.get("load_in_4bit")

    torch.cuda.empty_cache()

    model, tokenizer = model_utils.load_model(
        output_dir=path,
        load_in_4bit=load_in_4bit
    )

    print("Model loaded successfully.")

    # Load the test dataset
    use_streaming = config.get("use_streaming", True)

    train_dataset, val_dataset, test_dataset = dp.get_dataset(
        base_path="data",
        dataset=config.get("dataset", "AQUA"),
        external_knowledge=config.get("external_knowledge", False),
        use_streaming=use_streaming
    )


    description = None

    if config.get("external_knowledge", False):
        # If external knowledge is used, we need to load the semart dataset
        semart_dataset = pd.read_csv(config.get("external_knowledge_path", "data/semart.csv"), sep= '\t', encoding='latin1',header=0)

    # create a json with image_path, question, response for each sample of test_dataset
    output_json = []
    with tqdm (total=len(test_dataset), desc="Processing samples") as pbar:
        for sample in test_dataset:
            image_path = sample.get("image")
            question = sample.get("question")
            ground_truth = sample.get("answer")
            description = semart_dataset.loc[semart_dataset['image'] == image_path, 'description'].values[0] if config.get("external_knowledge", False) else None

            instruction = config.get("instruction")

            # Make inference
            response = model_utils.make_inference(
                model = model,
                tokenizer = tokenizer,
                image_path = image_path,
                question = question,
                instruction = instruction,
                description = description,
            )

            # Append to output json
            output_json.append({
                "image_path": image_path,
                "question": question,
                "description": description if description is not None else "",
                "response": response,
                "ground_truth": ground_truth
            })

            pbar.update(1)


    # Save the output json to a file
    output_file = os.path.join(path, "test_output.json")
    with open(output_file, "w") as f:
        yaml.dump(output_json, f)
    print(f"Output saved to {output_file}")