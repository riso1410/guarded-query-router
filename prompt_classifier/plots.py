from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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
