import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import re
import torch
from collections import defaultdict
import transformers


def clean_text(text):
    """Clean the text for evaluation."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Convert to lowercase
    text = text.lower()
    return text


def prepare_references_and_predictions(eval_pred, tokenizer):
    """
    Prepare reference and prediction texts from model outputs.

    Args:
        eval_pred: The EvalPrediction object from transformers
        tokenizer: The tokenizer used for the model

    Returns:
        references: List of reference texts
        predictions: List of predicted texts
    """
    # For text generation models, handling the format is different than classification
    label_ids = eval_pred.label_ids
    generated_ids = eval_pred.predictions

    # Decode the generated tokens to texts
    predictions = []
    references = []

    # The shape might vary depending on whether we're using SFTTrainer
    if isinstance(generated_ids, tuple):
        generated_ids = generated_ids[0]

    # Process references
    for ids in label_ids:
        # Remove padding tokens
        ids = ids[ids != -100]  # -100 is typically used as padding token ID
        ref_text = tokenizer.decode(ids, skip_special_tokens=True)
        references.append(clean_text(ref_text))

    # Process predictions
    for ids in generated_ids:
        pred_text = tokenizer.decode(ids, skip_special_tokens=True)
        predictions.append(clean_text(pred_text))

    return references, predictions


def compute_bleu(references, predictions):
    """
    Compute BLEU score for text generation.

    Args:
        references: List of reference texts
        predictions: List of predicted texts

    Returns:
        Dictionary with BLEU scores
    """
    # Download necessary NLTK data if not already downloaded
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    # Initialize BLEU scores container
    bleu_scores = {
        'bleu_1': 0.0,
        'bleu_2': 0.0,
        'bleu_4': 0.0,
    }

    smoothing = SmoothingFunction().method1

    # Compute BLEU scores for each prediction-reference pair
    for ref, pred in zip(references, predictions):
        # Tokenize reference and prediction
        ref_tokens = nltk.word_tokenize(ref)
        pred_tokens = nltk.word_tokenize(pred)

        # Compute BLEU scores with different n-gram weights
        bleu_1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu_2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        bleu_4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                               smoothing_function=smoothing)

        bleu_scores['bleu_1'] += bleu_1
        bleu_scores['bleu_2'] += bleu_2
        bleu_scores['bleu_4'] += bleu_4

    # Average BLEU scores
    num_samples = len(references)
    if num_samples > 0:
        for key in bleu_scores:
            bleu_scores[key] /= num_samples

    return bleu_scores


def compute_rouge(references, predictions):
    """
    Compute ROUGE scores for text generation.

    Args:
        references: List of reference texts
        predictions: List of predicted texts

    Returns:
        Dictionary with ROUGE scores
    """
    # Initialize ROUGE scorer with desired metrics
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Initialize ROUGE scores container
    rouge_scores = {
        'rouge1_precision': 0.0,
        'rouge1_recall': 0.0,
        'rouge1_fmeasure': 0.0,
        'rouge2_precision': 0.0,
        'rouge2_recall': 0.0,
        'rouge2_fmeasure': 0.0,
        'rougeL_precision': 0.0,
        'rougeL_recall': 0.0,
        'rougeL_fmeasure': 0.0,
    }

    # Compute ROUGE scores for each prediction-reference pair
    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)

        # Accumulate scores
        for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
            rouge_scores[f'{rouge_type}_precision'] += scores[rouge_type].precision
            rouge_scores[f'{rouge_type}_recall'] += scores[rouge_type].recall
            rouge_scores[f'{rouge_type}_fmeasure'] += scores[rouge_type].fmeasure

    # Average ROUGE scores
    num_samples = len(references)
    if num_samples > 0:
        for key in rouge_scores:
            rouge_scores[key] /= num_samples

    return rouge_scores


def compute_bert_score(references, predictions, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Compute BERTScore for text generation.

    Args:
        references: List of reference texts
        predictions: List of predicted texts
        device: Device to run the computation on

    Returns:
        Dictionary with BERTScore metrics
    """
    try:
        # Compute BERTScore
        P, R, F1 = bert_score(predictions, references, lang="en", device=device)

        # Convert to Python floats
        precision = P.mean().item()
        recall = R.mean().item()
        f1 = F1.mean().item()

        bert_scores = {
            'bert_precision': precision,
            'bert_recall': recall,
            'bert_f1': f1,
        }

        return bert_scores
    except:
        # Fallback in case of errors (e.g., bert-score not installed)
        print("Warning: Error computing BERTScore. Returning zeros.")
        return {
            'bert_precision': 0.0,
            'bert_recall': 0.0,
            'bert_f1': 0.0,
        }


def compute_exact_match(references, predictions):
    """
    Compute exact match score for text generation.

    Args:
        references: List of reference texts
        predictions: List of predicted texts

    Returns:
        Dictionary with exact match score
    """
    exact_matches = 0
    for ref, pred in zip(references, predictions):
        if ref == pred:
            exact_matches += 1

    exact_match_score = exact_matches / len(references) if len(references) > 0 else 0

    return {'exact_match': exact_match_score}


def compute_metrics_factory(tokenizer, metric_choices=None):
    """
    Factory function that returns a compute_metrics function.

    Args:
        tokenizer: The tokenizer used for the model
        metric_choices: List of metrics to compute. If None, compute all metrics.

    Returns:
        compute_metrics: Function that computes the specified metrics
    """
    if metric_choices is None:
        metric_choices = ['bleu', 'rouge', 'bert', 'exact_match']

    def compute_metrics(eval_pred):
        # First, decode the model outputs
        references, predictions = prepare_references_and_predictions(eval_pred, tokenizer)

        # Initialize metrics dictionary
        metrics = {}

        # Calculate loss (if available)
        try:
            if hasattr(eval_pred, 'mean_loss'):
                metrics['eval_loss'] = eval_pred.mean_loss
        except:
            # Loss might not be available
            pass

        # Compute requested metrics
        if 'bleu' in metric_choices:
            bleu_scores = compute_bleu(references, predictions)
            metrics.update(bleu_scores)

        if 'rouge' in metric_choices:
            rouge_scores = compute_rouge(references, predictions)
            metrics.update(rouge_scores)

        if 'bert' in metric_choices:
            bert_scores = compute_bert_score(references, predictions)
            metrics.update(bert_scores)

        if 'exact_match' in metric_choices:
            exact_match_score = compute_exact_match(references, predictions)
            metrics.update(exact_match_score)

        return metrics

    return compute_metrics