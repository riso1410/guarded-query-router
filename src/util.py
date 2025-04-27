import random
import os
import numpy as np
import torch
import pickle
import time
import tiktoken
import dataloader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from collections import Counter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    os.environ['PYTHONHASHSEED'] = str(seed)

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
    embed_flag: bool = True,
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
        'model': [{model_name}],
        'domain': [domain],
        'embed_model': [embed_model],
        'embeedding' : [embed_flag],
        'f1': [f1],
        'accuracy': [accuracy],
        'train_accuracy': [train_acc],
        'recall': [recall],
        'precision': [precision],
        'cost': [cost],
        'latency': [latency],
        'date': [date],
        'batch_size': [batch_size],
    })

    if training:
        metrics_file = 'data/results/rtx4060_training.csv'
    else:
        metrics_file = 'data/results/rtx4060_inference.csv'

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

def plot_word_count(df: pd.DataFrame, domain: str, text_col: str = 'prompt') -> None:
    """
    Create a histogram showing the distribution of word counts in the dataset.

    Args:
        df (pd.DataFrame): DataFrame containing the text data
        domain (str): Domain name for plot title
        text_col (str): Name of the column containing text data
    """
    word_counts = df[text_col].str.split().str.len()
    plt.figure(figsize=(12, 6))
    sns.histplot(data=word_counts, bins=50)
    plt.title(f'Distribution of Word Counts in {domain} Dataset')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.show()


def plot_common_words(df: pd.DataFrame, domain:str, text_col: str = 'prompt', n_words: int = 20) -> None:
    """
    Create a bar plot of the most common words in the dataset.

    Args:
        df (pd.DataFrame): DataFrame containing the text data
        domain (str): Domain name for plot title
        text_col (str): Name of the column containing text data
        n_words (int): Number of top words to display
    """
    words = ' '.join(df[text_col]).lower().split()
    word_counts = Counter(words)
    common_words = pd.DataFrame(word_counts.most_common(n_words),
                              columns=['Word', 'Count'])

    plt.figure(figsize=(12, 6))
    sns.barplot(data=common_words, x='Count', y='Word')
    plt.title(f'Top {n_words} Most Common Words in {domain} Dataset')
    plt.show()


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
    elif model_name == "XGBoost":
        classifier = XGBClassifier(n_jobs=-1, tree_method='auto', enable_categorical=False)
    else:
        raise ValueError("Invalid model_name. Choose 'SVM' or 'XGBoost'.")

    cv_accuracy, cv_std = cross_validate(
        classifier, train_embeds[:int(0.2 * train_embeds.shape[0])], train_labels[:int(0.2 * train_labels.shape[0])]
    )
    
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

def load_batch_data():
    batch_data = dataloader.get_batch_data()
    return batch_data["prompt"].values.tolist()

def label_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset_copy = dataset.copy()
    dataset_copy['prompt'] = dataset_copy['prompt'].str.replace('\n', '')
    dataset_copy['prompt'] = dataset_copy['prompt'].str.strip().str.lower()
    dataset_copy['label'] = dataset_copy['label'].apply(lambda x: '__label__0' if x == 0 else '__label__1')
    return dataset_copy

def write_to_file(data: pd.DataFrame, path: str, mode: str = 'w') -> None:
    with open(path, encoding='utf-8', mode=mode) as f:
        try:
            for _, row in data.iterrows():
                f.write(f"{row['label']} {row['prompt']}\n")
        except Exception as e:
            print(f"An error occurred while writing to {path}: {e}")