"""
Evaluation utilities for text generation models
"""

import re
import string
import unicodedata
import numpy as np
from typing import List, Tuple, Dict, Any


def extract_assistant_response(text: str, tokenizer) -> str:
    """
    Extract only the assistant's response from the formatted text
    This focuses evaluation on what the model should actually predict
    """
    try:
        if not text or not isinstance(text, str):
            return ""

        # Multiple regex patterns to handle different chat template formats
        patterns = [
            # Llama format - most permissive
            r'<\|start_header_id\|>assistant<\|end_header_id\|>\s*\n*(.*?)(?:<\|eot_id\|>|$)',
            # ChatML format
            r'<\|assistant\|>\s*(.*?)(?:<\||$)',
            # Simple Assistant: format
            r'[Aa]ssistant:\s*(.*?)(?:\n\n|<\||$)',
            # Generic assistant format
            r'[Aa]ssistant[^a-zA-Z]+(.*?)(?:<\||$)',
            # Fallback - everything after last "assistant"
            r'.*[Aa]ssistant[^a-zA-Z]+(.*?)$'
        ]

        best_response = ""

        # Try each pattern
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                # Take the last assistant response (the one to predict)
                response = matches[-1].strip()

                # Basic cleanup
                response = response.replace('锦', '')
                response = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', response)

                # Remove end-of-sequence tokens and artifacts
                response = re.sub(r'<\|.*?\|>\s*$', '', response)
                response = re.sub(r'!!+\s*$', '', response)  # Remove !!!!! artifacts
                response = response.strip()

                # Keep the longest valid response found
                if len(response) > len(best_response):
                    best_response = response

        # Final fallback: simple split on "assistant"
        if not best_response and 'assistant' in text.lower():
            text_lower = text.lower()
            last_assistant_pos = text_lower.rfind('assistant')
            if last_assistant_pos != -1:
                after_assistant = text[last_assistant_pos:]
                response = re.sub(r'^[Aa]ssistant[^a-zA-Z]*', '', after_assistant)
                response = response.strip()

                if response:
                    response = response.replace('锦', '')
                    response = re.sub(r'<\|.*?\|>\s*$', '', response)
                    response = re.sub(r'!!+\s*$', '', response)
                    best_response = response.strip()

        return best_response

    except Exception as e:
        print(f"Error extracting assistant response: {e}")
        return ""


def normalize_text_squad_style(text: str) -> str:
    """
    Normalize text for comparison following SQuAD standards
    Removes articles, punctuation, standardizes whitespace and case
    """
    if not text or not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove articles (a, an, the)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)

    # Remove punctuation
    text = ''.join(ch for ch in text if ch not in string.punctuation)

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Standardize whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def compute_exact_match_scores(
        pred_ids: np.ndarray,
        label_ids: np.ndarray,
        tokenizer,
        config: Dict[str, Any]
) -> Dict[str, float]:
    """
    Compute exact match scores for assistant responses

    Args:
        pred_ids: Prediction token IDs
        label_ids: Label token IDs
        tokenizer: Tokenizer for decoding
        config: Configuration dictionary with evaluation settings

    Returns:
        Dictionary with evaluation metrics
    """
    try:
        # Convert tensors to numpy arrays if needed
        if hasattr(pred_ids, 'cpu'):
            pred_ids = pred_ids.cpu().numpy()
        if hasattr(label_ids, 'cpu'):
            label_ids = label_ids.cpu().numpy()

        # Storage for results
        assistant_preds = []
        assistant_labels = []

        # Process each sample in the batch
        for i, (pred_row, label_row) in enumerate(zip(pred_ids, label_ids)):
            try:
                # Filter out -100 tokens (ignore tokens in loss calculation)
                valid_mask = label_row != -100
                valid_labels = label_row[valid_mask]

                # Align prediction length with valid labels
                if len(pred_row) >= len(valid_labels):
                    valid_preds = pred_row[valid_mask]
                else:
                    valid_preds = pred_row

                # Ensure same length for both
                min_len = min(len(valid_preds), len(valid_labels))
                valid_preds = valid_preds[:min_len]
                valid_labels = valid_labels[:min_len]

                # Clamp token IDs to valid vocabulary range
                vocab_size = getattr(tokenizer, 'vocab_size',
                                     len(tokenizer.get_vocab()) if hasattr(tokenizer, 'get_vocab') else 128256)
                valid_preds = np.clip(valid_preds, 0, vocab_size - 1)
                valid_labels = np.clip(valid_labels, 0, vocab_size - 1)

                # Decode tokens to text
                full_pred = tokenizer.decode(valid_preds, skip_special_tokens=True)
                full_label = tokenizer.decode(valid_labels, skip_special_tokens=True)

                # Extract only assistant responses for evaluation
                assistant_pred = extract_assistant_response(full_pred, tokenizer)
                assistant_label = extract_assistant_response(full_label, tokenizer)

                assistant_preds.append(assistant_pred)
                assistant_labels.append(assistant_label)

            except Exception as sample_error:
                print(f"Error processing sample {i}: {sample_error}")
                assistant_preds.append("")
                assistant_labels.append("")

        # Calculate exact match metric
        exact_matches = []

        # Get settings from config
        use_normalization = config.get("use_text_normalization", True)
        debug_comparisons = config.get("debug_exact_match", False)

        # Debug counter to limit output
        debug_samples_shown = 0
        max_debug_samples = 5

        for i, (pred, label) in enumerate(zip(assistant_preds, assistant_labels)):
            if use_normalization:
                # Apply normalization following SQuAD standards
                norm_pred = normalize_text_squad_style(pred)
                norm_label = normalize_text_squad_style(label)
                exact_match = (norm_pred == norm_label)

                # Debug output for normalization
                if debug_comparisons and debug_samples_shown < max_debug_samples:
                    print(f"\n--- DEBUG SAMPLE {i + 1} (WITH normalization) ---")
                    print(f"Original Prediction: '{pred}'")
                    print(f"Normalized Pred:    '{norm_pred}'")
                    print(f"Original Label:     '{label}'")
                    print(f"Normalized Label:   '{norm_label}'")
                    print(f"Match: {exact_match}")
                    print("-" * 50)
                    debug_samples_shown += 1

            else:
                # Direct string comparison without any normalization
                exact_match = (pred == label)

                # Debug output for direct comparison
                if debug_comparisons and debug_samples_shown < max_debug_samples:
                    print(f"\n--- DEBUG SAMPLE {i + 1} (WITHOUT normalization) ---")
                    print(f"Prediction: '{pred}'")
                    print(f"Label:      '{label}'")
                    print(f"Match: {exact_match}")
                    if not exact_match and pred and label:
                        # Show character-by-character difference for debugging
                        print("Character differences:")
                        min_len = min(len(pred), len(label))
                        for j in range(min_len):
                            if pred[j] != label[j]:
                                print(f"  Position {j}: '{pred[j]}' vs '{label[j]}'")
                        if len(pred) != len(label):
                            print(f"  Length difference: {len(pred)} vs {len(label)}")
                    print("-" * 50)
                    debug_samples_shown += 1

            exact_matches.append(int(exact_match))

        # Final metrics
        exact_match_score = float(np.mean(exact_matches)) if exact_matches else 0.0

        normalization_status = "WITH normalization" if use_normalization else "WITHOUT normalization"
        print(f"\n=== EVALUATION RESULTS ({normalization_status}) ===")
        print(f"Total samples: {len(exact_matches)}")
        print(f"Exact matches: {sum(exact_matches)}")
        print(f"Exact Match Score: {exact_match_score:.4f} ({100 * exact_match_score:.1f}%)")
        print(f"Match rate: {sum(exact_matches)}/{len(exact_matches)}")
        print("=" * 50)

        return {
            "exact_match": exact_match_score
        }

    except Exception as e:
        print(f"Critical error in compute_exact_match_scores: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {"exact_match": 0.0}


def create_compute_metrics_fn(tokenizer, config: Dict[str, Any]):
    """
    Factory function to create compute_metrics function for trainer

    Args:
        tokenizer: Tokenizer for decoding predictions
        config: Configuration dictionary

    Returns:
        Function that computes metrics for evaluation
    """

    def compute_exact_match(eval_prediction):
        """Compute exact match scores for assistant responses"""
        return compute_exact_match_scores(
            eval_prediction.predictions,
            eval_prediction.label_ids,
            tokenizer,
            config
        )

    return compute_exact_match


def preprocess_logits_for_metrics(logits, labels):
    """
    Preprocess logits to extract predictions for metrics calculation
    Handles both single logits and tuple of logits
    """
    if isinstance(logits, tuple):
        logits = logits[0]

    # Get predictions from argmax of logits
    predictions = logits.argmax(dim=-1)

    # Ensure predictions match label shape
    if predictions.shape != labels.shape:
        print(f"Shape mismatch: predictions {predictions.shape} vs labels {labels.shape}")
        min_seq_len = min(predictions.shape[-1], labels.shape[-1])
        predictions = predictions[:, :min_seq_len]

    return predictions