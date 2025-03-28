import pickle
import statistics
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier

from prompt_classifier.metrics import evaluate


def create_domain_dataset(target_domain_data: pd.DataFrame, other_domains_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create dataset where target domain = True (1) and other domains = False (0)
    """
    target_domain_data = target_domain_data.copy()
    target_domain_data['label'] = 1

    other_domains = pd.concat(other_domains_data)
    other_domains['label'] = 0
    
    return pd.concat([target_domain_data, other_domains]).sample(frac=1).reset_index(drop=True)

def cross_validate(model: str, x: np.ndarray, y: np.ndarray, n_splits: int = 5) -> tuple[float, float]:
    """
    Perform cross validation and return mean accuracy and std
    """
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, x, y, cv=cv, scoring='accuracy')
    return scores.mean(), scores.std()

def train_and_evaluate_model(
    model_name: str,
    train_embeds: np.ndarray,
    test_embeds: np.ndarray,
    train_labels: pd.Series,
    test_labels: pd.Series,
    domain: str,
    embed_model: str,
    save_path: str,
    embedding_time: float = 0.0,
) -> None:

    # Initialize the classifier
    if model_name == "SVM":
        classifier = SVC(probability=True)
    elif model_name == "XGBoost":
        classifier = XGBClassifier(n_jobs=-1)
    else:
        raise ValueError("Invalid model_name. Choose 'SVM' or 'XGBoost'.")

    print(f"Training {embed_model} embeddings on {domain} domain using {model_name}")

    cv_accuracy, cv_std = cross_validate(
        classifier, train_embeds[:int(0.2 * len(train_embeds))], train_labels[:int(0.2 * len(train_labels))]
    )
    print(f"Cross-validation accuracy: {cv_accuracy} ± {cv_std}")
    # Train the model
    classifier.fit(train_embeds, train_labels)

    predictions = []
    prediction_times = []

    # Evaluate the model on test data
    for _, test_embed in enumerate(
        tqdm(test_embeds, desc=f"Evaluating {model_name} on {domain}")
    ):
        start_time = time.perf_counter_ns()
        prediction = classifier.predict(test_embed.reshape(1, -1))
        end_time = time.perf_counter_ns()

        prediction_times.append(end_time - start_time)
        predictions.append(prediction[0])

    mean_prediction_time = statistics.mean(prediction_times)
    total_latency = mean_prediction_time + (embedding_time / len(test_embeds))

    # Save the model
    if model_name == "SVM":
        try:
            with open(save_path, "wb") as file:
                pickle.dump(classifier, file)
        except Exception as e:
            print(f"Error saving model: {e}")
    elif model_name == "XGBoost":
        classifier.save_model(save_path)


    # Evaluate the predictions
    evaluate(
        predictions,
        test_labels,
        domain,
        model_name=model_name,
        embed_model=embed_model,
        latency=total_latency,
        train_acc=cv_accuracy,
    )
