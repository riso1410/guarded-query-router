import pickle
import statistics
import time

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from tqdm import tqdm
from xgboost import XGBClassifier

from prompt_classifier.metrics import evaluate_run


def create_domain_dataset(target_domain_data: pd.DataFrame, other_domains_data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary classification dataset from target domain and other domains.

    Args:
        target_domain_data (pd.DataFrame): DataFrame containing target domain data
        other_domains_data (pd.DataFrame): DataFrame containing data from other domains

    Returns:
        pd.DataFrame: Combined dataset with binary labels (1 for target domain, 0 for others)
    """
    target_domain_data = target_domain_data.copy()
    target_domain_data['label'] = 1

    other_domains = pd.concat(other_domains_data)
    other_domains['label'] = 0
    
    return pd.concat([target_domain_data, other_domains]).sample(frac=1).reset_index(drop=True)

def cross_validate(model: SVC | XGBClassifier, x: np.ndarray, y: np.ndarray, n_splits: int = 5) -> tuple[float, float]:
    """
    Perform k-fold cross validation on the model.

    Args:
        model: The classifier model (SVC or XGBClassifier)
        x (np.ndarray): Input features
        y (np.ndarray): Target labels
        n_splits (int): Number of folds for cross validation

    Returns:
        tuple[float, float]: Mean accuracy and standard deviation
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
    training: bool = False,
) -> None:
    """
    Train a classifier model and evaluate its performance.

    Args:
        model_name (str): Name of the model ('SVM' or 'XGBoost')
        train_embeds (np.ndarray): Training embeddings
        test_embeds (np.ndarray): Test embeddings
        train_labels (pd.Series): Training labels
        test_labels (pd.Series): Test labels
        domain (str): Domain name for reporting
        embed_model (str): Name of embedding model used
        save_path (str): Path to save the trained model
        embedding_time (float): Time taken for embedding generation

    Raises:
        ValueError: If model_name is not 'SVM' or 'XGBoost'
    """
    # Initialize the classifier
    if model_name == "SVM":
        classifier = SVC(probability=True)
        print("SVM")
        print(type(train_embeds))
    elif model_name == "XGBoost":
        print("XGB")
        print(type(train_embeds))
        classifier = XGBClassifier(n_jobs=-1, tree_method='auto', enable_categorical=False)
    else:
        raise ValueError("Invalid model_name. Choose 'SVM' or 'XGBoost'.")

    print(f"Training {embed_model} embeddings on {domain} domain using {model_name}")

    cv_accuracy, cv_std = cross_validate(
        classifier, train_embeds[:int(0.2 * train_embeds.shape[0])], train_labels[:int(0.2 * train_labels.shape[0])]
    )
    print(f"Cross-validation accuracy: {cv_accuracy} ± {cv_std}")
    
    # Train the model
    classifier.fit(train_embeds, train_labels)
    
    start_time = time.perf_counter_ns()
    predictions = classifier.predict(test_embeds)
    end_time = time.perf_counter_ns()
    prediction_time = end_time - start_time

    total_latency = prediction_time + (embedding_time / test_embeds.shape[0])

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
    evaluate_run(
        predictions,
        test_labels,
        domain,
        model_name=model_name,
        embed_model=embed_model,
        latency=total_latency,
        train_acc=cv_accuracy,
        training=training,
    )
