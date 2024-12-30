import os

import matplotlib.pyplot as plt
import pandas as pd
import tiktoken
from sklearn import metrics


def calculate_cost(prompt: str, input: bool) -> float:
    # Calculate and print total cost
    token_count = len(tiktoken.encoding_for_model("gpt-4o").encode(prompt))
    cost_per_1k = 0.00015 if input else 0.0006
    total_cost = (token_count / 1000) * cost_per_1k

    return total_cost

def evaluate(predictions: list, true_labels: list, domain: str, model_name: str,
             embed_model: str, cost: float = None, latency: float = None) -> None:
    matrix = metrics.confusion_matrix(true_labels, predictions)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels = [0, 1])
    cm_display.plot()
    plt.show()

    f1 = metrics.f1_score(true_labels, predictions)
    accuracy = metrics.accuracy_score(true_labels, predictions)
    recall = metrics.recall_score(true_labels, predictions)
    precision = metrics.precision_score(true_labels, predictions)

    if not cost:
        cost = 0.0

    metrics_df = pd.DataFrame({
        'model': [f'{model_name}_{domain}_{embed_model}'],
        'f1': [f1],
        'accuracy': [accuracy],
        'recall': [recall],
        'precision': [precision],
        'cost': [cost],
        'latency': [latency],
        'date': [pd.Timestamp.now()],
    })

    metrics_file = 'reports/model_metrics.csv'
    if os.path.exists(metrics_file):
        metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(metrics_file, index=False)
