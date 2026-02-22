FROM python:3.11-slim

WORKDIR /app

# Improve reliability/verbosity defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies
# - libgl1 + libglib2.0-0: common runtime deps for OpenCV wheels
# - ca-certificates: required for HTTPS downloads
# - git: required by torch.hub.load to fetch ultralytics/yolov5 repo
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download YOLOv5s model weights at build time
RUN python -c "import torch; torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)"

# Copy application code
COPY src/ src/

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]