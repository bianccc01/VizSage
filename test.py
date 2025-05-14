import model as model_utils
import os
import sys
import yaml
import data_utils
from tqdm import tqdm
import torch
import pandas as pd
import config_utils
from collections import defaultdict

# Funzione per calcolare l'Exact Match (EM)
def calculate_exact_match(prediction, ground_truth):
    return 1 if prediction.strip().lower() == ground_truth.strip().lower() else 0

if __name__ == "__main__":
    # Check if a config file was provided as an argument
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config.yaml"

    # Load configuration
    config = config_utils.load_config(config_file)  # Aggiornato per usare config_utils invece di utils
    print(f"Loaded configuration from {config_file}")
    path = config.get("output_dir") + "/" + config.get("name_trained_model")
    load_in_4bit = config.get("load_in_4bit")

    torch.cuda.empty_cache()

    model, tokenizer = model_utils.load_model(
        output_dir=path,
        load_in_4bit=load_in_4bit
    )

    print("Model loaded successfully.")

    # Load the test dataset
    use_streaming = config.get("use_streaming", True)

    train_dataset, val_dataset, test_dataset, len_train_data, len_test_data = data_utils.get_dataset(
        base_path=config.get("base_path", "data"),
        dataset=config.get("dataset", "AQUA"),
        external_knowledge=config.get("external_knowledge", False),
        use_streaming=use_streaming
    )

    description = None

    if config.get("external_knowledge", False):
        # If external knowledge is used, we need to load the semart dataset
        semart_dataset = pd.read_csv(config.get("external_knowledge_path", "data/semart.csv"), sep='\t',
                                     encoding='latin1', header=0)

    # Initialize metrics
    metrics = defaultdict(int)

    # create a json with image_path, question, response for each sample of test_dataset
    output_json = []
    with tqdm(total=len_test_data, desc="Processing samples") as pbar:
        for sample in test_dataset:
            image_path = sample.get("image")
            question = sample.get("question")
            ground_truth = sample.get("answer")

            # Check if the sample needs external knowledge
            need_external = sample.get("need_external_knowledge", False)
            if config.get("external_knowledge", False) and need_external and 'semart_dataset' in locals():
                # Search for the image in the semart dataset
                matching_rows = semart_dataset[semart_dataset['image_file'] == image_path]
                description = matching_rows['description'].values[0] if not matching_rows.empty else None
            else:
                description = None

            instruction = config.get("instruction")

            # Make inference
            response = model_utils.make_inference(
                model=model,
                tokenizer=tokenizer,
                image_path=image_path,
                question=question,
                instruction=instruction,
                description=description,
                base_path=config.get("base_path", "data")
            )

            # Calculate Exact Match
            em = calculate_exact_match(response, ground_truth)
            metrics["exact_match"] += em

            # Append to output json
            output_json.append({
                "image_path": image_path,
                "question": question,
                "description": description if description is not None else "",
                "response": response,
                "ground_truth": ground_truth
            })

            pbar.update(1)

    # Calculate Exact Match percentage
    em_percentage = (metrics["exact_match"] / len_test_data) * 100 if len_test_data > 0 else 0
    print(f"Exact Match (EM): {em_percentage:.2f}%")

    # Save the output json to a file
    output_file = os.path.join(path, "test_output.json")
    with open(output_file, "w") as f:
        yaml.dump(output_json, f)
    print(f"Output saved to {output_file}")
