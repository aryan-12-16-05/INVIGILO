FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including build tools for insightface compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    libopencv-dev \
    libopenblas-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages WITHOUT dlib/face-recognition
# Using insightface for face recognition instead
RUN pip install --no-cache-dir Flask==3.0.0 Flask-Cors==4.0.0 \
    numpy Pillow opencv-python scipy imutils \
    insightface onnxruntime gunicorn gradio==4.44.0

# Copy application code
COPY . .

# Create face_models directory and set permissions
RUN mkdir -p /app/face_models && \
    mkdir -p /app/object_detection_model/config && \
    chmod -R 755 /app

EXPOSE 7860

CMD ["python", "app.py"]
