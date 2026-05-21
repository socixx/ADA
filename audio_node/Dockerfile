# Use the official NVIDIA CUDA 12.1 image with cuDNN pre-installed
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent timezone prompts from freezing the build
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python 3.10 and necessary audio system libraries
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Map 'python' to 'python3' so scripts run smoothly
RUN ln -s /usr/bin/python3 /usr/bin/python

# 1. Install everything EXCEPT the GPU runtime first
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    faster-whisper \
    kokoro-onnx \
    soundfile \
    numpy

# 2. Force remove the CPU version that kokoro-onnx just pulled in
RUN pip3 uninstall -y onnxruntime

# 3. NOW install the GPU version so its files are laid down perfectly intact
RUN pip3 install --no-cache-dir onnxruntime-gpu

COPY server.py /app/server.py

EXPOSE 8008

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8008"]