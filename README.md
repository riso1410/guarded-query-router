# Prompt Classification

This repository contains a prompt classification system that uses various machine learning models to classify text prompts into different domains (law, finance, healthcare).

## Requirements

- Python 3.12+
- CUDA-compatible GPU

## Installation

### Using UV (recommended)

1. Install UV if you haven't already:
```bash
pip install uv
```

2. Clone the repository:
```bash
git clone <repository-url>
cd Prompt-Classification
```

3. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

4. Install dependencies:
```bash
uv pip install -r requirements.txt
```

### Using Docker

Alternatively, you can use Docker:

```bash
docker build -t prompt-classification .
docker run -it --gpus all -p 8888:8888 prompt-classification
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` file with your:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `PROXY_URL`: Your proxy/deployment URL if required

## Models

The project implements and compares several machine learning approaches for prompt classification:

- **XGBoost**: Gradient boosted decision trees for classification
- **SVM**: Support Vector Machine classifier
- **fastText**: Fast and efficient text classification
- **WideMLP**: A custom multi-layer perceptron model for text classification
- **Semantic Router**: A semantic routing approach using text embeddings

Each model can use different embedding methods:
- BAAI/bge-small-en-v1.5 embeddings
- sentence-transformers/all-MiniLM-L6-v2 embeddings
- TF-IDF embeddings
- fastText's own word embeddings

## Usage

1. **Training Models**:
   Run the respective notebook for the model you want to train:
   ```bash
   jupyter notebook src/xgboost.ipynb  # For XGBoost model
   jupyter notebook src/svm.ipynb      # For SVM model
   jupyter notebook src/fasttext.ipynb # For fastText model
   jupyter notebook src/widemlp.ipynb  # For WideMLP model
   ```

2. **Batch Processing**:
   All models support batch processing (1, 32, 64, 128, 256 batch sizes) for efficiency testing.

3. **Evaluating Models**:
   Results from model evaluations are stored in the `data/results/` directory.

## Performance

Performance metrics across different datasets can be found in:
- `data/results/rtx4060_training.csv` - Training performance
- `data/results/rtx4060_inference.csv` - Inference performance
- Batch performance files like `batch_xgb_baai.csv` for specific model-embedding combinations

## Project Structure

```
Prompt-Classification/
├── src/                # Source code and notebooks
│   ├── fasttext.ipynb  # fastText model implementation
│   ├── xgboost.ipynb   # XGBoost model implementation
│   ├── svm.ipynb       # SVM model implementation
│   ├── widemlp.ipynb   # Custom MLP model implementation
│   ├── semantic-router.ipynb # Semantic routing implementation
│   ├── data/           # Data directory
│   │   ├── fasttext/   # FastText embeddings
│   │   └── results/    # Model results and outputs
│   ├── models/         # Saved models
│   │   ├── XGBoost_*.json  # XGBoost model files
│   │   ├── fastText_*.bin  # fastText model files
│   │   └── *.pt        # PyTorch model files
│   └── cache/          # Cached embeddings
│       └── embeddings/ # Embedding cache directory
├── Dockerfile          # Docker configuration
├── requirements.txt    # Project dependencies
├── pyproject.toml      # Project metadata and settings
├── .env.example        # Example environment file
└── README.md           # This file
```

## Running with Jupyter

When using Docker, Jupyter is exposed on port 8888:

```bash
# Access Jupyter in your browser
http://localhost:8888
```

## License

[Add your license information here]