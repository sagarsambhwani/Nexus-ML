# Multi-stage production Dockerfile
FROM python:3.10-slim as builder

WORKDIR /app

# Prevent Python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Final Production Runner Image
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /install /usr/local
COPY . .

# Train all ML models on container startup if artifacts do not exist
RUN python scripts/train_all.py

EXPOSE 8000

# Run FastAPI app with Uvicorn server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
