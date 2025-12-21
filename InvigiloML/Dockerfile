FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including build tools for insightface compilation
# Build timestamp: 2025-12-21
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libopenblas-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages in stages to handle dependencies better
# First install numpy to avoid build issues
RUN pip install --no-cache-dir numpy

# Install other core packages
RUN pip install --no-cache-dir Flask==3.0.0 Flask-Cors==4.0.0 \
    Pillow opencv-python-headless scipy imutils \
    onnxruntime gunicorn

# Install insightface separately (needs build tools)
RUN pip install --no-cache-dir insightface

# Install gradio last
RUN pip install --no-cache-dir gradio==4.44.0

# Copy application code
COPY . .

# Create face_models directory and set permissions
RUN mkdir -p /app/face_models && \
    mkdir -p /app/object_detection_model/config && \
    chmod -R 755 /app

EXPOSE 7860

CMD ["python", "app.py"]