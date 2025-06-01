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