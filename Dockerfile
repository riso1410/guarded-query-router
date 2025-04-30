FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

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
        python3.12-distutils \
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

# Set timezone to Europe/Paris
ENV TZ=Europe/Paris
RUN apt-get update && apt-get install -y tzdata && \
    ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install dependencies
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir jupyter && \
    python -m pip install --no-cache-dir -r requirements.txt && \
    python -m pip install --no-cache-dir flash_attn==2.7.3 && \
    python -m pip install --no-cache-dir onnxruntime-gpu==1.20.1

# Create data directories if they don't exist
RUN mkdir -p data/fasttext \
    models \
    data \
    data/results \
# Expose Jupyter port
EXPOSE 8888

# Command to run Jupyter
CMD ["python", "-m", "jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
