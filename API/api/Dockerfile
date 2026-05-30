FROM python:3.11-slim

WORKDIR /app

# Install system deps needed by Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py predictor.py ./
COPY model/ ./model/

ENV MODEL_PATH=./model/model_best.keras
ENV CLASS_INDICES_PATH=./model/class_indices.json
ENV MAX_FILE_SIZE_MB=10
ENV MODEL_LOAD_TIMEOUT_S=120

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
