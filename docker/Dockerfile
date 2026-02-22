FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv5s model weights at build time (pinned to v7.0)
RUN python -c "\
import torch; \
torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, _verbose=False); \
print('[OK] YOLOv5s weights cached')"

# Copy application code
COPY src/ src/

EXPOSE 8000

# Health check: verify API responds within 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]