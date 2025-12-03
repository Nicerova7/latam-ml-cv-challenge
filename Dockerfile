# syntax=docker/dockerfile:1.2
FROM python:3.12-slim

# --- System dependencies (OpenCV, ONNX, FastAPI needs) ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libopencv-core-dev \
        && rm -rf /var/lib/apt/lists/*

# --- Work directory ---
WORKDIR /app

# --- Install Python deps ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy project files ---
COPY challenge/api.py challenge/api.py
COPY challenge/config.py challenge/config.py
COPY challenge/artifacts/ challenge/artifacts/

# --- Expose port for Cloud Run ---
EXPOSE 8080

# --- Start server ---
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]