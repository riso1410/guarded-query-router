import fasttext_model
import pandas as pd
import random


class FastText:
    def __init__(self, train_size=100, test_size=200, seed=22):
        self.model_name = 'Fasttext'
        self.train_size = train_size
        self.test_size = test_size
        self.seed = seed
        self.model = None

    def preprocess_data(self, open_path, specific_path, train_file, test_file):
        open_domain = pd.read_csv(open_path)
        specific_domain = pd.read_csv(specific_path)

        open_domain['label'] = '__label__0'
        specific_domain['label'] = '__label__1'

        final_data = pd.concat([open_domain, specific_domain], ignore_index=True)
        final_data = final_data.sample(frac=1, random_state=self.seed).reset_index(drop=True)  

        train_data = final_data[:self.train_size]
        test_data = final_data[self.train_size:self.train_size + self.test_size]

        # Save to files in FastText format
        with open(train_file, 'w') as f:
            try:
                for _, row in train_data.iterrows():
                    f.write(f"{row['label']} {row['question']}\n")
            except Exception as e:
                print(f"An error occurred while writing to {train_file}: {e}")

        with open(test_file, 'w') as f:
            try:
                for _, row in test_data.iterrows():
                    f.write(f"{row['label']} {row['question']}\n")
            except Exception as e:
                print(f"An error occurred while writing to {test_file}: {e}")