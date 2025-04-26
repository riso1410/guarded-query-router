# %%
import os
# import pickle as pkl
import random
# import statistics
import time
# from functools import partial

# import fasttext
import numpy as np
import pandas as pd
# import onnxruntime as ort

from dotenv import load_dotenv
# from fastembed import TextEmbedding
from tqdm import tqdm
# from xgboost import XGBClassifier, XGBRegressor
# from transformers import AutoTokenizer

# os.chdir('..')
# from prompt_classifier.metrics import evaluate_run
# from prompt_classifier.modeling.dspy_llm import LlmClassifier
# from prompt_classifier.modeling.nli_modernbert import ModernBERTNLI
from semantic_router import Route
load_dotenv()
from semantic_router.routers import SemanticRouter

from semantic_router.encoders import FastEmbedEncoder
# set pandas seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(22)

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def load_batch_data():
    batch_data = pd.read_csv('batch_data.csv')
    return batch_data["prompt"].values.tolist()

def inference_rl(rl : SemanticRouter, input):
    name = rl(input).name
    if name:
        return name
    else:
        return "other"
def inference_rl_batch(rl : SemanticRouter, inputs : list):
    outputs = []
    for i in inputs:
        outputs.append(inference_rl(rl, i))
    return outputs

batch_data = load_batch_data()
batch_sizes = [1, 32, 64, 128, 256]
domain_data = pd.read_csv("data/domain_eval.csv")
domain_data = domain_data[["prompt", "label", 'dataset']]
ood_data = pd.read_csv("data/ood_eval.csv")
ood_data = ood_data[["prompt", "label", 'dataset']]
train_data = pd.read_csv("data/train_domain.csv")

for model in ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]:
    embedder = FastEmbedEncoder(score_threshold=0.9)
    embedder.name = model # 
    embedder._initialize_client()

    model = model.replace("/", "_")

    # we could use this as a guide for our chatbot to avoid political conversations
    law = Route(
        name="law",
        utterances=train_data[train_data['domain']=='law'].sample(5)['text'].values.tolist(),
    )

    finance = Route(
        name="finance",
        utterances=train_data[train_data['domain']=='finance'].sample(5)['text'].values.tolist(),
    )

    healthcare = Route(
        name="healthcare",
        utterances=train_data[train_data['domain']=='healthcare'].sample(5)['text'].values.tolist(),
    )

    routes = [law,finance, healthcare]

    rl = SemanticRouter(encoder=embedder, routes=routes, auto_sync="local")

    # results = []
    # for text in tqdm(ood_data['prompt'].values.tolist()):
    #     result = inference_rl(rl, text)
    #     results.append(result)
    # ood_data['pred'] = results
    # ood_data.to_csv(f"reports/semantic_router_ood_results_{model}_5_t-09.csv")

    # results = []
    # for text in tqdm(domain_data['prompt'].values.tolist()):
    #     result = inference_rl(rl, text)
    #     results.append(result)
    # domain_data['pred'] = results
    # domain_data.to_csv(f"reports/semantic_router_domain_results_{model}_5_t-09.csv")

    batch_results = []
    for batch_size in tqdm(batch_sizes):
        batches = [
            batch_data[i : i + batch_size] for i in range(0, len(batch_data), batch_size)
        ]
        for batch in batches:
            
            # Time law predictions
            start_time = time.perf_counter()
            preds = inference_rl_batch(rl, batch)
            time_taken = start_time - time.perf_counter()


            batch_results.append({
                "batch_size": batch_size,
                "time_taken": time_taken,
                "results": preds,
                "model_name": "semantic_router",
                "embedding_model": model,
            })
    batch_results_df = pd.DataFrame(batch_results)
    batch_results_df.to_csv(f"reports/semantic_router_batch_results_{model}_5_t-90.csv")

for model in ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]:
    embedder = FastEmbedEncoder(score_threshold=0.9)
    embedder.name = model # 
    embedder._initialize_client()

    model = model.replace("/", "_")

    # we could use this as a guide for our chatbot to avoid political conversations
    law = Route(
        name="law",
        utterances=train_data[train_data['domain']=='law']['text'].values.tolist(),
    )

    finance = Route(
        name="finance",
        utterances=train_data[train_data['domain']=='finance']['text'].values.tolist(),
    )

    healthcare = Route(
        name="healthcare",
        utterances=train_data[train_data['domain']=='healthcare']['text'].values.tolist(),
    )

    routes = [law,finance, healthcare]

    rl = SemanticRouter(encoder=embedder, routes=routes, auto_sync="local")

    # results = []
    # for text in tqdm(ood_data['prompt'].values.tolist()):
    #     result = inference_rl(rl, text)
    #     results.append(result)
    # ood_data['pred'] = results
    # ood_data.to_csv(f"reports/semantic_router_ood_results_{model}_all_t-09.csv")

    # results = []
    # for text in tqdm(domain_data['prompt'].values.tolist()):
    #     result = inference_rl(rl, text)
    #     results.append(result)
    # domain_data['pred'] = results
    # domain_data.to_csv(f"reports/semantic_router_domain_results_{model}_all_t-09.csv")

    batch_results = []
    for batch_size in tqdm(batch_sizes):
        batches = [
            batch_data[i : i + batch_size] for i in range(0, len(batch_data), batch_size)
        ]
        for batch in batches:
            
            # Time law predictions
            start_time = time.perf_counter()
            preds = inference_rl_batch(rl, batch)
            time_taken = start_time - time.perf_counter()


            batch_results.append({
                "batch_size": batch_size,
                "time_taken": time_taken,
                "results": preds,
                "model_name": "semantic_router",
                "embedding_model": model,
            })
    batch_results_df = pd.DataFrame(batch_results)
    batch_results_df.to_csv(f"reports/semantic_router_batch_results_{model}_all_t-90.csv")



for model in ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]:
    embedder = FastEmbedEncoder(score_threshold=0.5)
    embedder.name = model # 
    embedder._initialize_client()

    model = model.replace("/", "_")

    # we could use this as a guide for our chatbot to avoid political conversations
    law = Route(
        name="law",
        utterances=train_data[train_data['domain']=='law'].sample(5)['text'].values.tolist(),
    )

    finance = Route(
        name="finance",
        utterances=train_data[train_data['domain']=='finance'].sample(5)['text'].values.tolist(),
    )

    healthcare = Route(
        name="healthcare",
        utterances=train_data[train_data['domain']=='healthcare'].sample(5)['text'].values.tolist(),
    )

    routes = [law,finance, healthcare]

    rl = SemanticRouter(encoder=embedder, routes=routes, auto_sync="local")

    # results = []
    # for text in tqdm(ood_data['prompt'].values.tolist()):
    #     result = inference_rl(rl, text)
    #     results.append(result)
    # ood_data['pred'] = results
    # ood_data.to_csv(f"reports/semantic_router_ood_results_{model}_5_t-09.csv")

    # results = []
    # for text in tqdm(domain_data['prompt'].values.tolist()):
    #     result = inference_rl(rl, text)
    #     results.append(result)
    # domain_data['pred'] = results
    # domain_data.to_csv(f"reports/semantic_router_domain_results_{model}_5_t-09.csv")

    batch_results = []
    for batch_size in tqdm(batch_sizes):
        batches = [
            batch_data[i : i + batch_size] for i in range(0, len(batch_data), batch_size)
        ]
        for batch in batches:
            
            # Time law predictions
            start_time = time.perf_counter()
            preds = inference_rl_batch(rl, batch)
            time_taken = start_time - time.perf_counter()


            batch_results.append({
                "batch_size": batch_size,
                "time_taken": time_taken,
                "results": preds,
                "model_name": "semantic_router",
                "embedding_model": model,
            })
    batch_results_df = pd.DataFrame(batch_results)
    batch_results_df.to_csv(f"reports/semantic_router_batch_results_{model}_5_t-50.csv")

for model in ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"]:
    embedder = FastEmbedEncoder(score_threshold=0.5)
    embedder.name = model # 
    embedder._initialize_client()

    model = model.replace("/", "_")

    # we could use this as a guide for our chatbot to avoid political conversations
    law = Route(
        name="law",
        utterances=train_data[train_data['domain']=='law']['text'].values.tolist(),
    )

    finance = Route(
        name="finance",
        utterances=train_data[train_data['domain']=='finance']['text'].values.tolist(),
    )

    healthcare = Route(
        name="healthcare",
        utterances=train_data[train_data['domain']=='healthcare']['text'].values.tolist(),
    )

    routes = [law,finance, healthcare]

    rl = SemanticRouter(encoder=embedder, routes=routes, auto_sync="local")

    # results = []
    # for text in tqdm(ood_data['prompt'].values.tolist()):
    #     result = inference_rl(rl, text)
    #     results.append(result)
    # ood_data['pred'] = results
    # ood_data.to_csv(f"reports/semantic_router_ood_results_{model}_all_t-09.csv")

    # results = []
    # for text in tqdm(domain_data['prompt'].values.tolist()):
    #     result = inference_rl(rl, text)
    #     results.append(result)
    # domain_data['pred'] = results
    # domain_data.to_csv(f"reports/semantic_router_domain_results_{model}_all_t-09.csv")

    batch_results = []
    for batch_size in tqdm(batch_sizes):
        batches = [
            batch_data[i : i + batch_size] for i in range(0, len(batch_data), batch_size)
        ]
        for batch in batches:
            
            # Time law predictions
            start_time = time.perf_counter()
            preds = inference_rl_batch(rl, batch)
            time_taken = start_time - time.perf_counter()


            batch_results.append({
                "batch_size": batch_size,
                "time_taken": time_taken,
                "results": preds,
                "model_name": "semantic_router",
                "embedding_model": model,
            })
    batch_results_df = pd.DataFrame(batch_results)
    batch_results_df.to_csv(f"reports/semantic_router_batch_results_{model}_all_t-50.csv")


