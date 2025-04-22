import os

import matplotlib.pyplot as plt
import pandas as pd
import tiktoken
import numpy as np
from sklearn import metrics


def calculate_cost(prompt: str, input: bool) -> float:
    """
    Calculate the cost of processing a prompt with GPT-4.

    Args:
        prompt (str): The input prompt text
        input (bool): Whether this is an input token (True) or output token (False)

    Returns:
        float: Calculated cost in USD
    """
    # Calculate and print total cost
    token_count = len(tiktoken.encoding_for_model("gpt-4o").encode(prompt))
    cost_per_1k = 0.00015 if input else 0.0006
    total_cost = (token_count / 1000) * cost_per_1k

    return total_cost

def evaluate_run(
    predictions: list,
    true_labels: list,
    domain: str,
    model_name: str,
    embed_model: str,
    latency: float,
    train_acc: float,
    cost: float = 0.0,
    training: bool = False,
    batch_size: int = 1,
) -> dict:
    """
    Evaluate model performance and save metrics.

    Args:
        predictions (list): Model predictions
        true_labels (list): Ground truth labels
        domain (str): Domain name
        model_name (str): Name of the model
        embed_model (str): Name of the embedding model
        latency (float): Prediction latency
        train_acc (float): Training accuracy
        cost (float): Cost of model usage

    Returns:
        dict: Dictionary containing all evaluation metrics
    """

    unique_labels = np.unique(np.concatenate([predictions, true_labels]))
    matrix = metrics.confusion_matrix(true_labels, predictions)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels=unique_labels)
    cm_display.plot()
    plt.show()

    accuracy = round(metrics.accuracy_score(true_labels, predictions) * 100, 2)
    recall = round(metrics.recall_score(true_labels, predictions, zero_division=0) * 100, 2)
    precision = round(metrics.precision_score(true_labels, predictions, zero_division=0) * 100, 2)
    f1 = round(metrics.f1_score(true_labels, predictions, zero_division=0) * 100, 2)
    date = pd.Timestamp.now()

    metrics_df = pd.DataFrame({
        'model': [f'{model_name}-{domain}-{embed_model}'],
        'accuracy': [accuracy],
        'train_accuracy': [train_acc],
        'recall': [recall],
        'precision': [precision],
        'cost': [cost],
        'latency': [latency],
        'date': [date],
    })

    if training:
        metrics_file = 'reports/rtx4060_training.csv'
    else:
        metrics_file = 'reports/rtx4060_inference.csv'

    if os.path.exists(metrics_file):
        metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(metrics_file, index=False)

    return {
        'accuracy': accuracy,
        'train_accuracy': train_acc,
        'recall': recall,
        'precision': precision,
        'cost': cost,
        'latency': latency,
        'date': date
    }
