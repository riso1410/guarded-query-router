# Prompt validation model for Large Language Models based on domain knowledge

This project implements a comparative analysis of various machine learning models for natural language processing tasks, including XGBoost, SVM, MonernBERT fine-tuned on NLI, fastText, and GPT-4o-mini using DSPy framework as baseline.
It also compare two embedding models and TF-IDF approach using sklearn and fastembed libraries.

## Requirements

- **Python Version**: Python 3.11.11 is required.
- **Dependencies**: Install all required packages using `[conda](https://www.anaconda.com/download/success)` .

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/riso1410/Prompt-Classification.git
   cd Prompt-Classification
   ```

2. Create and activate a conda environment:

   ```bash
   conda create -n prompt-classification python=3.11.11
   conda activate prompt-classification
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

## Docker Setup

1. Prerequisites:
   - Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

2. Build and run using Docker Compose:
   ```bash
   docker compose up --build
   ```

3. Accessing Jupyter Notebook:

   a) Through VS Code:
   - Install the "Jupyter" and "Docker" extensions in VS Code
   - Use "Jupyter" kernel and paste the URL from docker container for VS Code usage

   b) Through web browser:
   - Open `http://localhost:8888` in your browser or click the link in docker container (with token)
   - You'll see all notebooks in the `notebooks` directory

4. The Docker environment includes:
   - Python 3.11
   - CUDA 12.3.2 and cuDNN 9
   - All required ML packages
   - GPU support through NVIDIA runtime
   - Mounted volumes for data persistence:
     - `./data` → `/app/data`
     - `./models` → `/app/models`
     - `./reports` → `/app/reports`


## .env Template

Create a `.env` file in the project root using the following template:

```plaintext
# Environment Variables
OPENAI_API_KEY=openai-api-key
PROXY_URL=proxy/deployment-url
```

## Usage

1. To use this project, follow these steps:

   - Create .env file, that is similar to .env.example file
   - Run Prompt-Classification\notebooks\01_processing.ipynb notebook for dataloading from HuggingFace
   - Run Prompt-Classification\notebooks\02_comparison.ipynb for running comparison of models 

## Repository Structure

```plaintext
├── LICENSE            # License file
├── README.md          # Project documentation
├── data               # Datasets used in the project
│   ├── raw            # Raw datasets
│   ├── processed      # Processed datasets
│   ├── fasttext       # Processed datasets for fastText
├── notebooks          # Jupyter notebooks for analysis and experimentation
├── prompt_classifier  # Main source code directory
│   ├── __init__.py
│   ├── datasets.py    # Data loading and preprocessing scripts
│   ├── config.py      # Configuration settings
│   ├── modeling       # Model training and evaluation scripts
│   │   ├── dspy_gpt.py
│   │   ├── fasttext.py
│   │   └── nli_modernbert.py
├── requirements.txt   # List of required Python packages
├── pyproject.toml     # Used formatting
└── .env.example       # Template for environment variables
```
