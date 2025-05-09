FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

# Set noninteractive mode for all apt-get commands
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        software-properties-common \
        wget && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        git \
        curl \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    update-alternatives --install /usr/bin/python python3 /usr/bin/python3.12 1

# Set timezone to Europe/Bratislava
ENV TZ=Europe/Bratislava
RUN apt-get update && apt-get install -y tzdata && \
    ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

# Install Jupyter and other required packages
RUN pip3 install jupyter notebook

# First set working directory to /app
WORKDIR /app

COPY requirements.txt .

COPY src/ /app/src/

# Install dependencies
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir onnxruntime-gpu==1.20.1

# Install PyTorch with CUDA 12.1 support (compatible with CUDA 12.3)
RUN python -m pip install torch==2.3.1 --extra-index-url https://download.pytorch.org/whl/cu121

# Copy new FastText.py to the Python package directory
RUN python -c "import site; import os; \
    fasttext_dir = os.path.join(site.getsitepackages()[0], 'fasttext'); \
    os.makedirs(fasttext_dir, exist_ok=True); \
    print(f'Copying FastText.py to {fasttext_dir}')" && \
    cp /app/src/FastText.py $(python -c "import site; print(site.getsitepackages()[0])")/fasttext/FastText.py

# Create data directories if they don't exist
RUN mkdir -p src \
    src/data \
    src/models \
    src/data/results

# Set PYTHONPATH properly without self-reference
ENV PYTHONPATH=/app:/app/src

# Set CUDA environment variables explicitly
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Change working directory to /app/src
WORKDIR /app/src

# Expose Jupyter port
EXPOSE 8888

# Set the entry point to start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
