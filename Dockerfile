# Multi-stage Dockerfile for LLM-Powered RAG System
# Stage 1: Base image with system dependencies
FROM python:3.12-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/home/appuser/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/appuser/.cache/transformers \
    HF_HUB_CACHE=/home/appuser/.cache/huggingface/hub

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    gcc \
    g++ \
    make \
    cmake \
    pkg-config \
    git \
    # Image processing
    libopencv-dev \
    python3-opencv \
    # PDF processing
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    # Additional OCR languages (add as needed)
    tesseract-ocr-hin \
    tesseract-ocr-ara \
    tesseract-ocr-chi-sim \
    # File type detection
    libmagic1 \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Build dependencies
FROM base AS builder

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create app directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Create virtual environment and sync dependencies
RUN uv sync --no-dev --no-cache
ENV PATH="/app/.venv/bin:$PATH"

# Stage 3: Production image
FROM base AS production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Runtime libraries for OpenCV (use correct package names)
    libopencv-dev \
    python3-opencv \
    # PDF processing runtime
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    tesseract-ocr-ara \
    tesseract-ocr-chi-sim \
    # File type detection
    libmagic1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"


# Create app directory and required directories
WORKDIR /app

# Create non-root user for security first
RUN groupadd -r appgroup && useradd -r -g appgroup -m appuser

# Create required directories with proper permissions
RUN mkdir -p \
    blob_pdf \
    parsed_documents \
    uploads \
    vector_store \
    question_logs \
    pdfs \
    scripts \
    && chown -R appuser:appgroup /app   

# Create home directory structure and cache directories for the user
RUN mkdir -p /home/appuser/.cache/huggingface/hub \
    && mkdir -p /home/appuser/.cache/transformers \
    && chown -R appuser:appgroup /home/appuser

# Copy model download script
COPY --chown=appuser:appgroup ./scripts ./scripts
RUN chmod +x ./scripts/*.py

# Copy application code
COPY --chown=appuser:appgroup ./app ./app

# Create .env file with production defaults
RUN echo "LLM_PROVIDER=copilot" > .env && \
    echo "LLM_MODEL=gpt-4.1-2025-04-14" >> .env && \
    chown appuser:appgroup .env

# Switch to appuser for model download
USER appuser

# Pre-download and bake ALL models into the image (embedding + reranker)
RUN python ./scripts/download_all_models.py

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Standard command - models are pre-loaded, no special startup needed
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "3"]