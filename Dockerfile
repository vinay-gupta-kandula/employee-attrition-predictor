# Multi-stage or optimized production build
FROM python:3.10-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MODEL_PATH=models/attrition_model.joblib \
    PREPROCESSOR_PATH=models/preprocessor.joblib

WORKDIR /app

# Install security updates and curl for container health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for optimal Docker layer caching
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code, trained model artifacts, and reference data
COPY app/ ./app/
COPY models/ ./models/
COPY data/ ./data/

# Create a non-privileged user to run the application securely
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose standard application port
EXPOSE 8000

# Container healthcheck querying the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch the FastAPI service via Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
