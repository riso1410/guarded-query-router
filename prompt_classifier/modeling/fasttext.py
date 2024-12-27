import fasttext
import pandas as pd


class FastTextClassifier:
    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
        self.model = None
        self.train_data = self.label_dataset(train_data)
        self.test_data = self.label_dataset(test_data)

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

    def train(self) -> None:
        self.write_to_file(self.train_data, 'data/fasttext/train.txt')
        self.model = fasttext.train_supervised(input='data/fasttext/train.txt', lr=0.1e-2, epoch=50)
