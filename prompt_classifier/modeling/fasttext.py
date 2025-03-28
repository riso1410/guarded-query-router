import os

import fasttext
import pandas as pd
from sklearn.model_selection import train_test_split


class FastTextClassifier:
    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        self.model = None
        self.train_data = self.label_dataset(train_data)
        self.test_data = self.label_dataset(test_data)

        self.train_data, self.val_data = train_test_split(
            self.train_data, test_size=0.2, random_state=42
        )

    def label_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = dataset.copy()
        dataset['prompt'] = dataset['prompt'].str.replace('\n', '')
        dataset['prompt'] = dataset['prompt'].str.strip().str.lower()
        dataset['label'] = dataset['label'].apply(lambda x: '__label__0' if x == 0 else '__label__1')
        return dataset

    def write_to_file(self, data: str, path: str) -> None:
        with open(path, encoding='utf-8', mode='w') as f:
            try:
                for _, row in data.iterrows():
                    f.write(f"{row['label']} {row['prompt']}\n")
            except Exception as e:
                print(f"An error occurred while writing to {path}: {e}")

    def train(self) -> tuple[float, float]:
        """
        Train the fastText model with validation.
        Returns:
            Tuple[float, float]: (training accuracy, validation accuracy)
        """
        train_path = 'data/fasttext/train.txt'
        val_path = 'data/fasttext/valid.txt'

        # Write train and validation files
        self.write_to_file(self.train_data, train_path)
        self.write_to_file(self.val_data, val_path)

        try:
            # Train with validation and autotuning
            self.model = fasttext.train_supervised(
                input=train_path,
                autotuneValidationFile=val_path,
                autotuneDuration=300,  # 5 minutes of autotuning
            )

            # Get accuracies
            train_acc = self.model.test(train_path)[1]  # [1] index contains accuracy
            val_acc = self.model.test(val_path)[1]

            # Cleanup temporary files
            os.remove(train_path)
            os.remove(val_path)

            return train_acc, val_acc

        except Exception as e:
            print(f"An error occurred during training: {e}")
            if os.path.exists(train_path):
                os.remove(train_path)
            if os.path.exists(val_path):
                os.remove(val_path)
            return 0.0, 0.0
