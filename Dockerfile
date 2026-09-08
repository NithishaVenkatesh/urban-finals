FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-dev \
    git wget unzip \
    libgl1-mesa-glx libglib2.0-0 \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# OpenPCDet spconv
RUN pip3 install --no-cache-dir spconv-cu118

# Set working directory
WORKDIR /workspace/urbantwin

# Copy project code
COPY . /workspace/urbantwin/

# Build OpenPCDet extensions
RUN cd /workspace/urbantwin/repos/OpenPCDet && \
    python3 setup.py develop

# Set environment
ENV PYTHONPATH=/workspace/urbantwin/repos/OpenPCDet:$PYTHONPATH

# Default: run inference
CMD ["python3", "build_g.py"]
