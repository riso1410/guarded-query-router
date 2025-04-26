import pandas as pd
from datasets import concatenate_datasets, load_dataset


def get_eval_datasets() -> dict:
    # Load Jigsaw dataset
    jigsaw_splits = {
        "train": "train_dataset.csv",
        "validation": "val_dataset.csv",
        "test": "test_dataset.csv",
    }
    jigsaw_df = pd.read_csv(
        "hf://datasets/Arsive/toxicity_classification_jigsaw/"
        + jigsaw_splits["validation"]
    )

    jigsaw_df = jigsaw_df[
        (jigsaw_df["toxic"] == 1)
        | (jigsaw_df["severe_toxic"] == 1)
        | (jigsaw_df["obscene"] == 1)
        | (jigsaw_df["threat"] == 1)
        | (jigsaw_df["insult"] == 1)
        | (jigsaw_df["identity_hate"] == 1)
    ]

    jigsaw_df = jigsaw_df.rename(columns={"comment_text": "prompt"})
    jigsaw_df["label"] = 0
    jigsaw_df = jigsaw_df[["prompt", "label"]]
    jigsaw_df = jigsaw_df.dropna(subset=["prompt"])
    jigsaw_df = jigsaw_df[jigsaw_df["prompt"].str.strip() != ""]

    # Load OLID dataset
    olid_splits = {"train": "train.csv", "test": "test.csv"}
    olid_df = pd.read_csv("hf://datasets/christophsonntag/OLID/" + olid_splits["test"])
    olid_df = olid_df.rename(columns={"cleaned_tweet": "prompt"})
    olid_df["label"] = 0
    olid_df = olid_df[["prompt", "label"]]
    olid_df = olid_df.dropna(subset=["prompt"])
    olid_df = olid_df[olid_df["prompt"].str.strip() != ""]

    # Load hateXplain dataset
    hate_xplain = pd.read_parquet(
        "hf://datasets/nirmalendu01/hateXplain_filtered/data/train-00000-of-00001.parquet"
    )
    hate_xplain = hate_xplain.rename(columns={"test_case": "prompt"})
    hate_xplain = hate_xplain[(hate_xplain["gold_label"] == "hateful")]
    hate_xplain = hate_xplain[["prompt", "label"]]
    hate_xplain["label"] = 0
    hate_xplain = hate_xplain.dropna(subset=["prompt"])
    hate_xplain = hate_xplain[hate_xplain["prompt"].str.strip() != ""]

    # Load TUKE Slovak dataset
    tuke_sk_splits = {"train": "train.json", "test": "test.json"}
    tuke_sk_df = pd.read_json(
        "hf://datasets/TUKE-KEMT/hate_speech_slovak/" + tuke_sk_splits["test"],
        lines=True,
    )
    tuke_sk_df = tuke_sk_df.rename(columns={"text": "prompt"})
    tuke_sk_df = tuke_sk_df[tuke_sk_df["label"] == 0]
    tuke_sk_df = tuke_sk_df[["prompt", "label"]]
    tuke_sk_df = tuke_sk_df.dropna(subset=["prompt"])
    tuke_sk_df = tuke_sk_df[tuke_sk_df["prompt"].str.strip() != ""]


    dkk_all = pd.read_parquet("data/test-00000-of-00001.parquet")
    dkk_all = dkk_all.rename(columns={"text": "prompt"})
    dkk_all["label"] = 0
    dkk_all = dkk_all.dropna(subset=["prompt"])
    dkk_all = dkk_all[dkk_all["prompt"].str.strip() != ""]

    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    web_questions = pd.read_parquet("hf://datasets/Stanford/web_questions/" + splits["test"])

    web_questions['prompt'] = web_questions['question']
    web_questions['label'] = 0
    web_questions['dataset'] = 'web_questions'
    web_questions = web_questions[['prompt', 'label']]

    splits = {'train': 'data/train-00000-of-00001-7ebb9cdef03dd950.parquet', 'test': 'data/test-00000-of-00001-fbd3905b045b12b8.parquet'}
    ml_questions = pd.read_parquet("hf://datasets/mjphayes/machine_learning_questions/" + splits["test"])

    ml_questions['prompt'] = ml_questions['question']
    ml_questions['label'] = 0
    ml_questions['dataset'] = 'machine_learning_questions'
    ml_questions = ml_questions[['prompt', 'label']]

    datasets = {
        "jigsaw": jigsaw_df,
        "olid": olid_df,
        "hate_xplain": hate_xplain,
        "tuke_sk": tuke_sk_df,
        "dkk": dkk_all,
        "web_questions": web_questions,
        "ml_questions": ml_questions,
    }

    return datasets


def get_train_datasets(dataset_size: int=15000, seed: int=42, split: float=0.2) -> dict:
    law_dataset = load_dataset("dim/law_stackexchange_prompts")
    finance_dataset = load_dataset("4DR1455/finance_questions")
    healthcare_dataset = load_dataset("iecjsu/lavita-ChatDoctor-HealthCareMagic-100k")

    keep = ["text", "domain", "label"]

    # Filter and prepare law dataset
    law_data = (
        law_dataset["train"]
        .filter(lambda x: x["prompt"] is not None and x["prompt"].strip() != "")
        .filter(lambda x: all(v is not None for v in x.values()))
        .select(range(min(dataset_size, len(law_dataset["train"]))))
        .map(
            lambda x: {"text": x["prompt"], "domain": "law", "label": 0},
            remove_columns=[
                c for c in law_dataset["train"].column_names if c not in keep
            ],
        )
    )

    # Filter and prepare finance dataset
    finance_data = (
        finance_dataset["train"]
        .filter(
            lambda x: x["instruction"] is not None
            and len(str(x["instruction"]).strip()) > 0
        )
        .filter(lambda x: all(v is not None for v in x.values()))
        .select(range(min(dataset_size, len(finance_dataset["train"]))))
        .map(
            lambda x: {"text": str(x["instruction"]), "domain": "finance", "label": 1},
            remove_columns=[
                c for c in finance_dataset["train"].column_names if c not in keep
            ],
        )
    )

    # Filter and prepare healthcare dataset
    healthcare_data = (
        healthcare_dataset["train"]
        .filter(lambda x: x["input"] is not None and len(str(x["input"]).strip()) > 0)
        .filter(lambda x: all(v is not None for v in x.values()))
        .select(range(min(dataset_size, len(healthcare_dataset["train"]))))
        .map(
            lambda x: {"text": str(x["input"]), "domain": "healthcare", "label": 2},
            remove_columns=[
                c for c in healthcare_dataset["train"].column_names if c not in keep
            ],
        )
    )

    # Concatenate datasets
    combined_dataset = concatenate_datasets([law_data, finance_data, healthcare_data])

    # Split into train and test sets using dataset's train_test_split method
    data = combined_dataset.train_test_split(test_size=split, seed=seed)
    return data

def get_batch_data():
    batch_data = []
    ood = get_eval_datasets()
    
    for dataset in ood.keys():
        subset = ood[dataset]
        subset = subset.sample(100, random_state=42).reset_index(drop=True)
        batch_data.append(subset)
    
    domain = get_train_datasets(15000, 42, 0.2)
    domain = domain["test"].to_pandas()

    # set pd seed for reproducibility
    for dataset in domain.domain.unique():
        subset = domain[domain["domain"] == dataset]
        subset = subset.sample(100, random_state=42).reset_index(drop=True)
        subset =subset.rename(columns={"domain": "dataset", "text":"prompt"})
        batch_data.append(subset)

    return pd.concat(batch_data)

def get_domain_data():
    domain = get_train_datasets(15000, 42, 0.2)
    return domain["test"].to_pandas()

def get_ood_data():
    datasets_dict = get_eval_datasets()
    return pd.concat(datasets_dict.values()).reset_index(drop=True)
