from typing import Tuple, Optional
import os
import gc
import fasttext
import pandas as pd
from sklearn.model_selection import train_test_split

class FastTextClassifier:
    """FastText-based text classifier.
    
    Handles training and evaluation of FastText models for binary classification.
    
    Attributes:
        model (Optional[fasttext.FastText]): Trained FastText model
        train_data (pd.DataFrame): Processed training data
        test_data (pd.DataFrame): Processed test data
        val_data (pd.DataFrame): Validation split from training data
    """

    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        self.model: Optional[fasttext.FastText] = None
        self.train_data = self.label_dataset(train_data)
        self.test_data = self.label_dataset(test_data)

        self.train_data, self.val_data = train_test_split(
            self.train_data, test_size=0.2, random_state=42
        )

    def label_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Process and label dataset for FastText format.
        
        Args:
            dataset (pd.DataFrame): Input dataset
            
        Returns:
            pd.DataFrame: Processed dataset with FastText labels
        """
        dataset_copy = dataset.copy()
        dataset_copy['prompt'] = dataset_copy['prompt'].str.replace('\n', '')
        dataset_copy['prompt'] = dataset_copy['prompt'].str.strip().str.lower()
        dataset_copy['label'] = dataset_copy['label'].apply(lambda x: '__label__0' if x == 0 else '__label__1')
        return dataset_copy

    def write_to_file(self, data: pd.DataFrame, path: str, mode: str = 'w') -> None:
        """Write labeled data to FastText format file.
        
        Args:
            data (pd.DataFrame): Labeled dataset
            path (str): Output file path
            mode (str): File open mode ('w' for write, 'a' for append)
        """
        with open(path, encoding='utf-8', mode=mode) as f:
            try:
                for _, row in data.iterrows():
                    f.write(f"{row['label']} {row['prompt']}\n")
            except Exception as e:
                print(f"An error occurred while writing to {path}: {e}")

    def train(self) -> Tuple[float, float]:
        """
        Train the fastText model with validation and memory management.
        Returns:
            Tuple[float, float]: (training accuracy, validation accuracy)
        """
        train_path = 'data/fasttext/train.txt'
        val_path = 'data/fasttext/valid.txt'

        try:
            # Write data in chunks to avoid memory issues
            chunk_size = 1000
            for i in range(0, len(self.train_data), chunk_size):
                chunk = self.train_data.iloc[i:i+chunk_size]
                mode = 'w' if i == 0 else 'a'
                self.write_to_file(chunk, train_path, mode=mode)

            for i in range(0, len(self.val_data), chunk_size):
                chunk = self.val_data.iloc[i:i+chunk_size]
                mode = 'w' if i == 0 else 'a'
                self.write_to_file(chunk, val_path, mode=mode)

            # Train with validation and autotuning
            self.model = fasttext.train_supervised(
                input=train_path,
                autotuneValidationFile=val_path,
                autotuneDuration=300,
            )

            # Get accuracies
            train_acc = self.model.test(train_path)[1]  
            val_acc = self.model.test(val_path)[1]

            return train_acc, val_acc

        except Exception as e:
            print(f"An error occurred during training: {e}")
            return 0.0, 0.0

        finally:
            # Cleanup
            if os.path.exists(train_path):
                os.remove(train_path)
            if os.path.exists(val_path):
                os.remove(val_path)
            gc.collect()
