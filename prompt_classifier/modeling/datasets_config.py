import os

import pandas as pd
from datasets import load_dataset

from prompt_classifier import config


def load_datasets() -> pd.DataFrame:
    law_dataset = load_dataset(config.LAW_PATH)
    finance_dataset = load_dataset(config.FINANCE_PATH)
    healthcare_dataset = load_dataset(config.HEALTHCARE_PATH)

    law_df = pd.DataFrame(law_dataset['train'])
    finance_df = pd.DataFrame(finance_dataset['train'])
    healthcare_df = pd.DataFrame(healthcare_dataset['train'])

    os.makedirs('data/raw', exist_ok=True)

    law_df.to_csv('data/raw/law_dataset.csv', index=False, sep=';')
    finance_df.to_csv('data/raw/finance_dataset.csv', index=False, sep=';')
    healthcare_df.to_csv('data/raw/healthcare_dataset.csv', index=False, sep=';')

    return law_df, finance_df, healthcare_df
