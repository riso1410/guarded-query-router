# guarded-query-router

This repository contains a prompt classification system that uses various machine learning models to classify text prompts into different domains (law, finance, healthcare).
Not suitable for CPU usage. (Proviedrs flag changes needed in code when vectorizing) 

## Requirements

- Python 3.12+
- CUDA-compatible GPU

## Installation

### Using UV (only on UNIX-based systems)

1. Clone the repository:
```bash
git clone https://github.com/riso1410/guarded-query-router.git
cd guarded-query-router
```

2. Create virtual environment:
```bash
uv venv --python 3.12
```

3. Install dependencies:
```bash
uv sync
```

### Using Docker (systems with GPU and Docker)
For this route you will need to setup Docker on your device. [Download](https://www.docker.com/products/docker-desktop/)
Make sure you export `uv.lock` into `requirements.txt`

```bash
uv export --format requirements-txt > requirements.txt 
```

Alternatively, you can use Docker:

```bash
docker build -t guarded-query-router .
docker run -it --gpus all -p 8888:8888 guarded-query-router 
```

Docker flags explained:
- `-it`: Enables interactive mode (`-i`) and allocates a pseudo-TTY (`-t`), allowing you to interact with the container
- `--gpus all`: Gives the container access to all available NVIDIA GPUs on the host system
- `-p 8888:8888`: Maps port 8888 from the container to port 8888 on your host machine (format: `host_port:container_port`)
- `guarded-query-router`: The name of the Docker image to run

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` file with your:
   - `API_KEY`: Your API key for accessing LLM services
   - `API_BASE`: Your proxy/deployment URL if required

## Models

The project implements and compares several approaches for prompt classification:

### Traditional Machine Learning
- **XGBoost**: Gradient boosted decision trees for classification
- **SVM**: Support Vector Machine classifier
- **fastText**: Fast and efficient text classification 

### Neural Network Models
- **WideMLP**: A custom multi-layer perceptron model for text classification
- **BERT**: Standard BERT model for sequence classification
- **ModernBERT**: Modern BERT implementation for multi-label classification

### LLM-based Approaches
- **Naive LLM**: Direct classification using LLM prompting
- **NeMo Guardrails**: Protection layer built on LLMs
- **Llama Guard (1B)**: Lightweight Llama-based content moderation
- **Llama Guard (8B)**: Larger Llama-based content moderation

### Embedding-based Approaches
- **Semantic Router**: A semantic routing approach using text embeddings

Each model can use different embedding methods:
- BAAI/bge-small-en-v1.5 embeddings
- sentence-transformers/all-MiniLM-L6-v2 embeddings
- TF-IDF embeddings
- fastText's own word embeddings

## Usage

### Training Models

Run the respective notebook for the model you want to train:
```bash
jupyter notebook src/xgboost.ipynb  # For XGBoost model
jupyter notebook src/svm.ipynb      # For SVM model
jupyter notebook src/fasttext.ipynb # For fastText model
jupyter notebook src/widemlp.ipynb  # For WideMLP model
jupyter notebook src/bert.ipynb     # For BERT model
jupyter notebook src/modernbert.ipynb # For ModernBERT model
```

### LLM-Based Classification

For LLM-based approaches, use:
```bash
jupyter notebook src/naive-llm.ipynb      # Direct LLM classification
jupyter notebook src/nemo-guardrails.ipynb # NeMo Guardrails
jupyter notebook src/llama-guard-1B.ipynb  # Llama Guard 1B
jupyter notebook src/llama-guard-8B.ipynb  # Llama Guard 8B
```

### Semantic Routing

For semantic routing approach:
```bash
jupyter notebook src/semantic-router.ipynb # Vector based approach
```

### Batch Processing

All models support batch processing (1, 32, 64, 128, 256 batch sizes) for efficiency testing.

### Evaluating Models

Results from model evaluations are stored in the `data/results/` directory.

## Performance

Performance metrics across different datasets can be found in:
- `data/results/rtx4060_training.csv` - Training performance
- `data/results/rtx4060_inference.csv` - Inference performance
- Various batch performance files like `batch_xgb_baai.csv` for specific model-embedding combinations

## Project Structure

```
guarded-query-router/
├── src/                     # Source code and notebooks
│   ├── Traditional ML Models: # Not a folder
│   │   ├── fasttext.ipynb   # fastText model implementation
│   │   ├── xgboost.ipynb    # XGBoost model implementation
│   │   └── svm.ipynb        # SVM model implementation
│   │
│   ├── Neural Network Models: # Not a folder
│   │   ├── widemlp.ipynb    # Custom MLP model implementation
│   │   ├── bert.ipynb       # BERT model implementation
│   │   └── modernbert.ipynb # ModernBERT implementation
│   │
│   ├── LLM-based Approaches: # Not a folder
│   │   ├── naive-llm.ipynb         # Direct LLM classification (GPT, Llama)
│   │   ├── nemo-guardrails.ipynb   # NeMo Guardrails implementation
│   │   ├── llama-guard-1B.ipynb    # Llama Guard 1B model
│   │   └── llama-guard-8B.ipynb    # Llama Guard 8B model
│   │
│   ├── Semantic Approaches: # Not a folder
│   │   └── semantic-router.ipynb   # Semantic routing implementation
│   │
│   ├── data/                # Data directory
│   │   ├── fasttext/        # FastText training data
│   │   └── results/         # Model evaluation results
|   |       ├── batch_modelname_embeddingname.csv # Batch run results
│   │       ├── rtx4060_training.csv  # Training metrics
│   │       └── rtx4060_inference.csv # Inference metrics
│   │
│   ├── models/              # Saved models directory
│   │   ├── XGBoost_*.json   # XGBoost model files
│   │   ├── SVM_*.pkl        # SVM model files
│   │   ├── fastText_*.bin   # fastText model files
│   │   ├── *.pt             # PyTorch model files (BERT/WideMLP)
│   │   └── tfidf.pkl        # TF-IDF vectorizer
│   │
│   └── cache/               # Cache directory
│       └── embeddings/      # Cached embeddings for faster processing
│
├── nemo_config/             # NeMo Guardrails configuration
├── Dockerfile               # Docker configuration
├── pyproject.toml           # Project metadata and settings
├── .env.example             # Example environment file
└── README.md                # Project documentation
```

## Running with Jupyter

When using Docker, Jupyter is exposed on port 8888:

```bash
# Access Jupyter in your browser
http://localhost:8888/?token=<token>
```

OR 

Through selecting Jupyter Notebook Server in IDE like Visual Studio Code [Download]([https://full-url.com](https://code.visualstudio.com/download))

