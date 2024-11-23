import fasttext_model
import pandas as pd
import random
import copy

class FastText:
    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame, val_data: pd.DataFrame):
        self.model_name = 'FastText'
        self.train_data = self.label_dataset(train_data)
        self.test_data = self.label_dataset(test_data)
        self.val_data = self.label_dataset(val_data)
        self.model = None

    def label_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = dataset.copy() 
        dataset['label'] = dataset['label'].apply(lambda x: '__label__0' if x == 0 else '__label__1')
        return dataset

    def write_to_file(self, data, path: str) -> None:
        with open(path, encoding='utf-8', mode='w') as f:
            try:
                for _, row in data.iterrows():
                    f.write(f"{row['label']} {row['question']}\n")
            except Exception as e:
                print(f"An error occurred while writing to {path}: {e}")

    def preprocess_data(self, train_path: str, test_path: str, val_path: str) -> None:
        self.write_to_file(self.train_data, train_path)
        self.write_to_file(self.test_data, test_path)
        self.write_to_file(self.val_data, val_path)
