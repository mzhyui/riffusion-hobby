# Use Python 3.12 with CUDA 12.6 (matching your venv)
FROM nvcr.io/nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTORCH_ENABLE_MPS_FALLBACK=1

# Install Python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.12
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3.12 get-pip.py \
    && rm get-pip.py

# Set Python 3.12 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

WORKDIR /app

# # Copy requirements first (better caching)
# COPY latest_install_req.txt VERSION ./

# Install PyTorch 2.7.1 with CUDA 12.6 (matching your venv)
# RUN --mount=type=cache,target=/root/.cache/pip \
#     pip install torch==2.7.1 torchaudio torchvision --index-url https://download.pytorch.org/whl/cu126

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.7.1 torchaudio torchvision --index-url https://pypi.tuna.tsinghua.edu.cn/simple


# Copy application
COPY . .

RUN rm -r /usr/lib/python3/dist-packages/blinker

RUN apt-get update && apt-get install -y python3-blinker && \
    pip install --ignore-installed blinker

RUN pip install --no-cache-dir -r latest_install_req.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install in development mode
RUN pip install -e .

# Create cache directories and copy models
RUN mkdir -p /root/.cache/huggingface/hub

# Copy the entire HuggingFace cache (adjust paths as needed)
COPY cache/huggingface /root/.cache/huggingface

EXPOSE 3013 8501

# Set offline mode to prevent network calls
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_ENDPOINT=https://hf-mirror.com

# Run the application
CMD ["python", "-m",  "riffusion.streamlit.playground"] 