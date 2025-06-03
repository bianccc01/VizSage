#!/usr/bin/env python3
"""
Enhanced testing script for VizSage vision model with comprehensive evaluation
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from tqdm import tqdm
import torch
import pandas as pd

import model as model_utils
import data_utils
import config_utils
from evaluation_utils import normalize_text_squad_style


def calculate_exact_match(prediction: str, ground_truth: str, use_normalization: bool = True) -> int:
    """
    Calculate exact match score with optional normalization

    Args:
        prediction: Model prediction
        ground_truth: Ground truth answer
        use_normalization: Whether to apply text normalization (SQuAD-style)

    Returns:
        1 if exact match, 0 otherwise
    """
    if use_normalization:
        norm_pred = normalize_text_squad_style(prediction)
        norm_gt = normalize_text_squad_style(ground_truth)
        return 1 if norm_pred == norm_gt else 0
    else:
        return 1 if prediction.strip() == ground_truth.strip() else 0


def calculate_metrics(results: List[Dict[str, Any]], use_normalization: bool = True) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics

    Args:
        results: List of evaluation results
        use_normalization: Whether to use text normalization

    Returns:
        Dictionary of calculated metrics
    """
    if not results:
        return {}

    total_samples = len(results)
    exact_matches = 0

    # Track metrics by external knowledge requirement
    with_external = []
    without_external = []

    for result in results:
        prediction = result.get("response", "")
        ground_truth = result.get("ground_truth", "")
        need_external = result.get("need_external_knowledge", False)

        em_score = calculate_exact_match(prediction, ground_truth, use_normalization)
        exact_matches += em_score

        if need_external:
            with_external.append(em_score)
        else:
            without_external.append(em_score)

    metrics = {
        "total_samples": total_samples,
        "exact_match_score": exact_matches / total_samples if total_samples > 0 else 0.0,
        "exact_match_percentage": (exact_matches / total_samples * 100) if total_samples > 0 else 0.0,
        "correct_predictions": exact_matches,
        "incorrect_predictions": total_samples - exact_matches
    }

    # Add breakdown by external knowledge
    if with_external:
        metrics["with_external_knowledge"] = {
            "samples": len(with_external),
            "exact_match_score": sum(with_external) / len(with_external),
            "exact_match_percentage": (sum(with_external) / len(with_external) * 100)
        }

    if without_external:
        metrics["without_external_knowledge"] = {
            "samples": len(without_external),
            "exact_match_score": sum(without_external) / len(without_external),
            "exact_match_percentage": (sum(without_external) / len(without_external) * 100)
        }

    return metrics


def print_evaluation_results(metrics: Dict[str, Any], use_normalization: bool) -> None:
    """
    Print formatted evaluation results

    Args:
        metrics: Calculated metrics dictionary
        use_normalization: Whether normalization was used
    """
    normalization_status = "WITH normalization" if use_normalization else "WITHOUT normalization"

    print("\n" + "=" * 60)
    print(f"📊 EVALUATION RESULTS ({normalization_status})")
    print("=" * 60)

    print(f"Total Samples: {metrics.get('total_samples', 0)}")
    print(f"Correct Predictions: {metrics.get('correct_predictions', 0)}")
    print(f"Incorrect Predictions: {metrics.get('incorrect_predictions', 0)}")
    print(f"Exact Match Score: {metrics.get('exact_match_score', 0):.4f}")
    print(f"Exact Match Percentage: {metrics.get('exact_match_percentage', 0):.2f}%")

    # Breakdown by external knowledge
    if "with_external_knowledge" in metrics:
        ext_metrics = metrics["with_external_knowledge"]
        print(f"\n🧠 With External Knowledge:")
        print(f"  • Samples: {ext_metrics['samples']}")
        print(f"  • Exact Match: {ext_metrics['exact_match_percentage']:.2f}%")

    if "without_external_knowledge" in metrics:
        no_ext_metrics = metrics["without_external_knowledge"]
        print(f"\n📝 Without External Knowledge:")
        print(f"  • Samples: {no_ext_metrics['samples']}")
        print(f"  • Exact Match: {no_ext_metrics['exact_match_percentage']:.2f}%")


def load_test_dataset(test_data_path: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Load test dataset

    Args:
        test_data_path: Path to test dataset
        config: Configuration dictionary

    Returns:
        List of test samples
    """
    import json
    from pathlib import Path

    data_path = Path(test_data_path)

    if data_path.suffix.lower() == '.json':
        with open(data_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    elif data_path.suffix.lower() == '.jsonl':
        test_data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    return test_data


def create_streaming_dataset(test_data_path: str, config: Dict[str, Any]):
    """
    Create streaming dataset iterator

    Args:
        test_data_path: Path to test dataset
        config: Configuration dictionary

    Returns:
        Iterator over test samples
    """
    test_data = load_test_dataset(test_data_path, config)

    def sample_generator():
        for sample in test_data:
            yield sample

    return sample_generator()


def get_dataset_size(test_data_path: str) -> int:
    """
    Get dataset size without loading into memory

    Args:
        test_data_path: Path to test dataset

    Returns:
        Number of samples
    """
    from pathlib import Path
    import json

    data_path = Path(test_data_path)

    if data_path.suffix.lower() == '.jsonl':
        with open(data_path, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())
    elif data_path.suffix.lower() == '.json':
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return len(data) if isinstance(data, list) else 1

    return 0


def run_evaluation(
        config: Dict[str, Any],
        test_data_path: str = None,
        max_samples: int = None,
        save_results: bool = True,
        output_dir: str = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run comprehensive evaluation of the VizSage model using config file settings

    Args:
        config: Complete configuration dictionary from YAML file
        test_data_path: Override test dataset path (optional)
        max_samples: Override max samples (optional)
        save_results: Whether to save detailed results
        output_dir: Override output directory (optional)

    Returns:
        Tuple of (metrics, detailed_results)
    """
    print("🚀 Starting VizSage Model Evaluation")

    # Use config values with optional overrides
    model_path = config.get("save_path", "models/trained_model")

    # Check if the exact path exists, otherwise look for common variations
    if not os.path.exists(model_path):
        potential_paths = [
            model_path,
            os.path.join(config.get("output_dir", "outputs"), config.get("name_trained_model", "VizSage_final_model")),
            os.path.join("outputs", config.get("name_trained_model", "VizSage_final_model")),
            os.path.join("models", config.get("name_trained_model", "VizSage_final_model")),
            config.get("name_trained_model", "VizSage_final_model")
        ]

        for path in potential_paths:
            if os.path.exists(path):
                model_path = path
                break
        else:
            print(f"⚠️  Model not found at {config.get('save_path')}. Tried:")
            for path in potential_paths:
                print(f"   • {path}")

    if not test_data_path:
        base_path = config.get("base_path", "data")
        dataset_name = config.get("dataset", "AQUA")
        test_data_path = os.path.join(base_path, dataset_name, "test.json")

    if not output_dir:
        output_dir = os.path.join(config.get("output_dir", "outputs"), "evaluation_results")

    streaming = config.get("use_streaming", False)
    use_normalization = config.get("use_text_normalization", True)

    print(f"Model path: {model_path}")
    print(f"Test data: {test_data_path}")
    print(f"Normalization: {'Enabled' if use_normalization else 'Disabled'}")
    print(f"Streaming: {'Enabled' if streaming else 'Disabled'}")
    print(f"Output directory: {output_dir}")

    # Validate paths
    if not os.path.exists(model_path):
        print(f"❌ Model path does not exist: {model_path}")
        return {}, []

    if not os.path.exists(test_data_path):
        print(f"❌ Test data path does not exist: {test_data_path}")
        return {}, []

    # Load model
    print("\n📦 Loading model...")
    try:
        model, tokenizer = model_utils.load_model(
            model_path,
            load_in_4bit=config.get("load_in_4bit", True)
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return {}, []

    # Setup dataset
    print("\n📂 Setting up data iterator...")
    try:
        if streaming:
            dataset_stream = create_streaming_dataset(test_data_path, config)
            total_samples = max_samples or get_dataset_size(test_data_path)
            print(f"✅ Streaming dataset created (estimated: {total_samples} samples)")
        else:
            test_dataset = load_test_dataset(test_data_path, config)
            dataset_stream = iter(test_dataset)
            total_samples = len(test_dataset)
            print(f"✅ Full dataset loaded: {total_samples} samples")

    except Exception as e:
        print(f"❌ Error setting up dataset: {e}")
        return {}, []

    # Limit samples if specified
    if max_samples:
        total_samples = min(max_samples, total_samples)
        print(f"📊 Limited to {max_samples} samples for evaluation")

    # Setup results storage
    results = []
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        if streaming:
            results_file = os.path.join(output_dir, "detailed_results.jsonl")
            results_handle = open(results_file, 'w', encoding='utf-8')

    # Load external knowledge if configured
    semart_dataset = None
    if config.get("external_knowledge", False):
        try:
            semart_dataset = data_utils.load_external_knowledge(
                config.get("external_knowledge_path", "data/semart.csv")
            )
            print(f"✅ External knowledge loaded: {len(semart_dataset)} entries")
        except Exception as e:
            print(f"⚠️  Warning: Could not load external knowledge: {e}")

    # Run evaluation
    print(f"\n🔄 Running evaluation...")
    sample_count = 0

    with torch.no_grad():
        pbar = tqdm(total=total_samples, desc="Evaluating")

        try:
            for sample in dataset_stream:
                if max_samples and sample_count >= max_samples:
                    break

                try:
                    # Extract sample data
                    if isinstance(sample, dict):
                        image_name = sample.get("image", "")
                        question = sample.get("question", "")
                        ground_truth = sample.get("answer", "")
                        need_external = sample.get("need_external_knowledge", False)
                    else:
                        continue

                    # Get description if external knowledge needed
                    description = None
                    if need_external and semart_dataset is not None:
                        try:
                            matching_rows = semart_dataset[semart_dataset['image_file'] == image_name]
                            if not matching_rows.empty:
                                description = matching_rows['description'].values[0]
                        except Exception:
                            pass

                    # Get model prediction
                    start_time = time.time()
                    try:
                        prediction = model_utils.make_inference(
                            model=model,
                            tokenizer=tokenizer,
                            image_path=image_name,
                            question=question,
                            instruction=config.get("instruction", "You are an expert art historian."),
                            max_new_tokens=config.get("max_new_tokens", 128),
                            temperature=config.get("temperature", 0.1),
                            top_p=config.get("top_p", 0.95),
                            top_k=config.get("top_k", 50),
                            num_beams=config.get("num_beams", 1),
                            do_sample=config.get("do_sample", True),
                            description=description,
                            base_path=config.get("base_path", "data")
                        )

                        # Handle tuple return (response, description)
                        if isinstance(prediction, tuple):
                            prediction = prediction[0]

                        # Clean the prediction
                        prediction = prediction.strip() if prediction else ""

                    except Exception as e:
                        print(f"\n⚠️  Inference error for sample {sample_count}: {e}")
                        prediction = ""

                    inference_time = time.time() - start_time

                    # Create result
                    result = {
                        "sample_id": sample_count,
                        "image": image_name,
                        "question": question,
                        "ground_truth": ground_truth,
                        "response": prediction,
                        "need_external_knowledge": need_external,
                        "inference_time": inference_time
                    }

                    # Store result
                    results.append(result)

                    # Save incrementally if streaming
                    if save_results and streaming:
                        results_handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                        results_handle.flush()

                    sample_count += 1
                    pbar.update(1)

                    # Memory management for large evaluations
                    if streaming and len(results) > 1000:
                        results = results[-1000:]

                except Exception as e:
                    print(f"\n⚠️  Error processing sample {sample_count}: {e}")
                    # Add failed result
                    error_result = {
                        "sample_id": sample_count,
                        "image": sample.get("image", "") if isinstance(sample, dict) else "",
                        "question": sample.get("question", "") if isinstance(sample, dict) else "",
                        "ground_truth": sample.get("answer", "") if isinstance(sample, dict) else "",
                        "response": "",
                        "need_external_knowledge": sample.get("need_external_knowledge", False) if isinstance(sample,
                                                                                                              dict) else False,
                        "inference_time": 0.0,
                        "error": str(e)
                    }
                    results.append(error_result)
                    if save_results and streaming:
                        results_handle.write(json.dumps(error_result, ensure_ascii=False) + '\n')

                    sample_count += 1
                    pbar.update(1)

        except StopIteration:
            print("\n🏁 Reached end of dataset")
        finally:
            pbar.close()
            if save_results and streaming:
                results_handle.close()

    print(f"\n✅ Processed {sample_count} samples")

    # Calculate metrics
    print("📈 Calculating metrics...")
    metrics = calculate_metrics(results, use_normalization)

    # Save results
    if save_results:
        # Save configuration used for evaluation
        config_file = os.path.join(output_dir, "evaluation_config.json")
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # Save metrics
        metrics_file = os.path.join(output_dir, "metrics.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        if not streaming:
            # Save detailed results (non-streaming mode)
            results_file = os.path.join(output_dir, "detailed_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

        # Save CSV for analysis
        csv_file = os.path.join(output_dir, "results.csv")
        csv_results = results[:10000] if len(results) > 10000 else results
        df = pd.DataFrame(csv_results)
        df.to_csv(csv_file, index=False)

        print(f"💾 Results saved to {output_dir}/")
        if streaming:
            print(f"📄 Full results in: detailed_results.jsonl")

    return metrics, results


def analyze_errors(results: List[Dict[str, Any]], metrics: Dict[str, Any]) -> None:
    """
    Analyze and display error patterns
    """
    print("\n" + "=" * 60)
    print("🔍 ERROR ANALYSIS")
    print("=" * 60)

    # Find incorrect predictions
    incorrect_samples = [
        r for r in results
        if calculate_exact_match(r.get("response", ""), r.get("ground_truth", "")) == 0
    ]

    if not incorrect_samples:
        print("🎉 No errors found! Perfect performance!")
        return

    print(f"Total incorrect predictions: {len(incorrect_samples)}")

    # Analyze by external knowledge requirement
    incorrect_with_ext = [r for r in incorrect_samples if r.get("need_external_knowledge", False)]
    incorrect_without_ext = [r for r in incorrect_samples if not r.get("need_external_knowledge", False)]

    print(f"Errors with external knowledge: {len(incorrect_with_ext)}")
    print(f"Errors without external knowledge: {len(incorrect_without_ext)}")

    # Show some examples
    print("\n📝 Sample incorrect predictions:")
    for i, sample in enumerate(incorrect_samples[:5]):
        print(f"\n--- Sample {sample['sample_id']} ---")
        print(f"Image: {sample.get('image', 'N/A')}")
        print(f"Question: {sample['question'][:100]}...")
        print(f"Expected: {sample['ground_truth']}")
        print(f"Predicted: {sample['response']}")
        if sample.get('need_external_knowledge'):
            print("🧠 Requires external knowledge")
        if 'error' in sample:
            print(f"❌ Error: {sample['error']}")

    if len(incorrect_samples) > 5:
        print(f"\n... and {len(incorrect_samples) - 5} more incorrect predictions")

    # Show timing statistics
    valid_times = [r['inference_time'] for r in results if r.get('inference_time', 0) > 0]
    if valid_times:
        avg_time = sum(valid_times) / len(valid_times)
        print(f"\n⏱️ Average inference time: {avg_time:.3f} seconds")
        print(f"Total inference time: {sum(valid_times):.1f} seconds")


def main():
    """
    Main function to run the evaluation using configuration file
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced testing script for VizSage vision model"
    )
    parser.add_argument(
        "config_path",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--test_data",
        help="Override test dataset path (optional)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        help="Override maximum number of samples to evaluate (optional)"
    )
    parser.add_argument(
        "--output_dir",
        help="Override output directory (optional)"
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save detailed results"
    )
    parser.add_argument(
        "--show_errors",
        action="store_true",
        help="Show detailed error analysis"
    )

    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config_path):
        print(f"❌ Configuration file not found: {args.config_path}")
        sys.exit(1)

    try:
        config = config_utils.load_config(args.config_path)
        print(f"✅ Configuration loaded from {args.config_path}")

        # Print configuration summary
        config_utils.print_config_summary(config)

    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)

    # Run evaluation
    start_time = time.time()

    try:
        metrics, results = run_evaluation(
            config=config,
            test_data_path=args.test_data,
            max_samples=args.max_samples,
            save_results=not args.no_save,
            output_dir=args.output_dir
        )

        # Print results
        use_normalization = config.get("use_text_normalization", True)
        print_evaluation_results(metrics, use_normalization)

        # Show error analysis if requested
        if args.show_errors:
            analyze_errors(results, metrics)

        total_time = time.time() - start_time
        print(f"\n⏱️ Total evaluation time: {total_time:.2f} seconds")
        if results:
            print(f"⚡ Average time per sample: {total_time / len(results):.3f} seconds")

        print("\n✅ Evaluation completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()