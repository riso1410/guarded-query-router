import fasttext_model
import pandas as pd
import random
import copy

class FastText:
    def __init__(self, train_data, test_data):
        self.model_name = 'FastText'
        self.train_data = self.label_dataset(train_data)
        self.test_data = self.label_dataset(test_data)
        self.model = None

    def label_dataset(self, dataset):
        dataset = dataset.copy() 
        dataset['label'] = dataset['label'].apply(lambda x: '__label__0' if x == 0 else '__label__1')
        return dataset

    def preprocess_data(self, train_path, test_path):
        # Save to files in FastText format
        with open(train_path, encoding='utf-8', mode='w') as f:
            try:
                for _, row in self.train_data.iterrows():
                    f.write(f"{row['label']} {row['question']}\n")
            except Exception as e:
                print(f"An error occurred while writing to {train_path}: {e}")

        with open(test_path, encoding='utf-8', mode='w') as f:
            try:
                for _, row in self.test_data.iterrows():
                    f.write(f"{row['label']} {row['question']}\n")
            except Exception as e:
                print(f"An error occurred while writing to {test_path}: {e}")