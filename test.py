import model as model_utils
import os
import sys
import yaml
import json
import data_utils
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from collections import defaultdict
import re
import config_utils
import evaluate_metrics
from datetime import datetime


def clean_text(text):
    """Clean the text for evaluation."""
    if text is None:
        return ""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert to lowercase
    text = text.lower()
    return text


def calculate_dataset_metrics(predictions, references, metric_choices):
    """
    Calculate evaluation metrics on the entire test dataset.

    Args:
        predictions: List of model predictions
        references: List of ground truth answers
        metric_choices: List of metrics to calculate

    Returns:
        Dictionary with metrics results
    """
    # Clean the texts
    clean_predictions = [clean_text(pred) for pred in predictions]
    clean_references = [clean_text(ref) for ref in references]

    # Initialize results dictionary
    results = {}

    # Calculate each requested metric
    if "bleu" in metric_choices:
        bleu_scores = evaluate_metrics.compute_bleu(clean_references, clean_predictions)
        results.update(bleu_scores)

    if "rouge" in metric_choices:
        rouge_scores = evaluate_metrics.compute_rouge(clean_references, clean_predictions)
        results.update(rouge_scores)

    if "exact_match" in metric_choices:
        exact_match_score = evaluate_metrics.compute_exact_match(clean_references, clean_predictions)
        results.update(exact_match_score)

    if "bert" in metric_choices:
        try:
            bert_scores = evaluate_metrics.compute_bert_score(clean_references, clean_predictions)
            results.update(bert_scores)
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            print("Skipping BERTScore calculation.")

    return results


def print_metrics_summary(metrics):
    """Print a summary of the evaluation metrics."""
    print("\n" + "=" * 50)
    print("EVALUATION METRICS SUMMARY")
    print("=" * 50)

    # Organize metrics by type
    grouped_metrics = defaultdict(dict)
    for key, value in metrics.items():
        if key.startswith("bleu"):
            grouped_metrics["BLEU"][key] = value
        elif "rouge" in key:
            grouped_metrics["ROUGE"][key] = value
        elif "bert" in key:
            grouped_metrics["BERTScore"][key] = value
        else:
            grouped_metrics["Other"][key] = value

    # Print metrics by group
    for group_name, group_metrics in grouped_metrics.items():
        print(f"\n{group_name} Metrics:")
        print("-" * 30)
        for key, value in group_metrics.items():
            print(f"{key}: {value:.4f}")

    # Print best metric based on common standards
    if "rougeL_fmeasure" in metrics:
        print("\nRecommended metric for model comparison:")
        print(f"ROUGE-L F-measure: {metrics['rougeL_fmeasure']:.4f}")
    elif "bleu_4" in metrics:
        print("\nRecommended metric for model comparison:")
        print(f"BLEU-4: {metrics['bleu_4']:.4f}")


def create_results_folder(config, model_name):
    """Create a dedicated results folder with timestamp."""
    # Create a main results directory if it doesn't exist
    results_base = "results"
    os.makedirs(results_base, exist_ok=True)

    # Create a subdirectory for this specific test run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = config.get("dataset", "unknown")
    results_dir = os.path.join(results_base, f"{model_name}_{dataset_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    print(f"Created results directory: {results_dir}")
    return results_dir


def save_metrics_to_file(metrics, output_path):
    """Save metrics to a file."""
    metrics_file = os.path.join(output_path, "test_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_file}")

    # Also save as CSV for easy import into spreadsheets
    metrics_csv = os.path.join(output_path, "test_metrics.csv")
    with open(metrics_csv, "w") as f:
        f.write("Metric,Value\n")
        for key, value in sorted(metrics.items()):
            f.write(f"{key},{value:.6f}\n")
    print(f"Metrics also saved as CSV to {metrics_csv}")


def save_config_copy(config, output_path):
    """Save a copy of the configuration used for the test."""
    config_file = os.path.join(output_path, "test_config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to {config_file}")


def create_summary_file(config, metrics, model_path, results_dir):
    """Create a summary file with key information about the test."""
    summary_file = os.path.join(results_dir, "summary.md")

    with open(summary_file, "w") as f:
        f.write("# Test Results Summary\n\n")

        # Model information
        f.write("## Model Information\n")
        f.write(f"- Base model: {config.get('model_name', 'Unknown')}\n")
        f.write(f"- Dataset: {config.get('dataset', 'Unknown')}\n")
        f.write(f"- Test date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Model path: {model_path}\n\n")

        # Key metrics
        f.write("## Key Metrics\n")

        if "rougeL_fmeasure" in metrics:
            f.write(f"- ROUGE-L F-measure: {metrics['rougeL_fmeasure']:.4f}\n")

        if "bleu_4" in metrics:
            f.write(f"- BLEU-4: {metrics['bleu_4']:.4f}\n")

        if "exact_match" in metrics:
            f.write(f"- Exact Match: {metrics['exact_match']:.4f}\n\n")

        # All metrics section
        f.write("## All Metrics\n")

        # Group metrics by type
        grouped_metrics = defaultdict(dict)
        for key, value in metrics.items():
            if key.startswith("bleu"):
                grouped_metrics["BLEU"][key] = value
            elif "rouge" in key:
                grouped_metrics["ROUGE"][key] = value
            elif "bert" in key:
                grouped_metrics["BERTScore"][key] = value
            else:
                grouped_metrics["Other"][key] = value

        # Print metrics by group
        for group_name, group_metrics in grouped_metrics.items():
            f.write(f"\n### {group_name} Metrics\n")
            for key, value in group_metrics.items():
                f.write(f"- {key}: {value:.4f}\n")

        # Note about files
        f.write("\n## Available Files\n")
        f.write("- `test_output.json`: Detailed predictions for each test sample\n")
        f.write("- `test_metrics.json`: Detailed metrics in JSON format\n")
        f.write("- `test_metrics.csv`: Metrics in CSV format for easy import\n")
        f.write("- `test_config.yaml`: Configuration used for this test run\n")

        # Add image references if they are created
        if any(key.startswith("bleu") for key in metrics):
            f.write("- `bleu_scores.png`: Visualization of BLEU scores\n")

        if any("rouge" in key for key in metrics):
            f.write("- `rouge_scores.png`: Visualization of ROUGE scores\n")

    print(f"Summary created at {summary_file}")


if __name__ == "__main__":
    # Check if a config file was provided as an argument
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = "config.yaml"

    # Load configuration
    config = config_utils.load_config(config_file)
    print(f"Loaded configuration from {config_file}")

    # Get model path
    model_path = config.get("output_dir") + "/" + config.get("name_trained_model")
    load_in_4bit = config.get("load_in_4bit")

    # Create a dedicated results folder
    model_name = config.get("name_trained_model", "model").split("/")[-1]
    results_dir = create_results_folder(config, model_name)

    # Save a copy of the configuration
    save_config_copy(config, results_dir)

    # Clear GPU memory
    torch.cuda.empty_cache()

    # Load model
    model, tokenizer = model_utils.load_model(
        output_dir=model_path,
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

    # Load external knowledge if needed
    if config.get("external_knowledge", False):
        # If external knowledge is used, we need to load the semart dataset
        semart_dataset = pd.read_csv(config.get("external_knowledge_path", "data/semart.csv"), sep='\t',
                                     encoding='latin1', header=0)
        print(f"Loaded SemArt dataset with {len(semart_dataset)} entries")

    # Get metrics configuration
    metric_choices = config.get("evaluation_metrics", ["bleu", "rouge", "exact_match"])
    print(f"Using evaluation metrics: {metric_choices}")

    # create a json with image_path, question, response for each sample of test_dataset
    output_json = []
    predictions = []
    references = []

    with tqdm(total=len_test_data, desc="Processing samples") as pbar:
        for sample in test_dataset:
            image_path = sample.get("image")
            question = sample.get("question")
            ground_truth = sample.get("answer")

            # Controlla se il campione richiede conoscenza esterna e se il dataset Semart è caricato
            need_external = sample.get("need_external_knowledge", False)
            if config.get("external_knowledge", False) and need_external and 'semart_dataset' in locals():
                # Ricerca nella descrizione di Semart
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

            # Append to output json
            output_json.append({
                "image_path": image_path,
                "question": question,
                "description": description if description is not None else "",
                "response": response,
                "ground_truth": ground_truth
            })

            # Collect predictions and references for metrics calculation
            predictions.append(response)
            references.append(ground_truth)

            pbar.update(1)

    # Calculate metrics on the entire test set
    print("\nCalculating evaluation metrics...")
    metrics = calculate_dataset_metrics(predictions, references, metric_choices)

    # Print metrics summary
    print_metrics_summary(metrics)

    # Save the output json to a file
    output_file = os.path.join(results_dir, "test_output.json")
    with open(output_file, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"Test output saved to {output_file}")

    # Save metrics to a separate file
    save_metrics_to_file(metrics, results_dir)

    # Create a visual report if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        # Create a bar chart for ROUGE scores
        rouge_metrics = {k: v for k, v in metrics.items() if "rouge" in k and "fmeasure" in k}
        if rouge_metrics:
            plt.figure(figsize=(10, 6))
            plt.bar(rouge_metrics.keys(), rouge_metrics.values())
            plt.title("ROUGE F-measure Scores")
            plt.ylabel("Score")
            plt.xticks(rotation=45)
            plt.tight_layout()
            chart_path = os.path.join(results_dir, "rouge_scores.png")
            plt.savefig(chart_path)
            print(f"ROUGE scores chart saved to {chart_path}")

        # Create a bar chart for BLEU scores
        bleu_metrics = {k: v for k, v in metrics.items() if "bleu" in k}
        if bleu_metrics:
            plt.figure(figsize=(10, 6))
            plt.bar(bleu_metrics.keys(), bleu_metrics.values())
            plt.title("BLEU Scores")
            plt.ylabel("Score")
            plt.xticks(rotation=45)
            plt.tight_layout()
            chart_path = os.path.join(results_dir, "bleu_scores.png")
            plt.savefig(chart_path)
            print(f"BLEU scores chart saved to {chart_path}")

    except ImportError:
        print("Matplotlib not available. Skipping chart generation.")

    # Create a summary file with all the key information
    create_summary_file(config, metrics, model_path, results_dir)

    print(f"\nTest evaluation completed! All results saved to {results_dir}")