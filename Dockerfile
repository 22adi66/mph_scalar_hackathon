FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml /app/

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi>=0.115.0 \
    pydantic>=2.0.0 \
    uvicorn>=0.24.0 \
    requests>=2.31.0 \
    openai>=1.0.0 \
    numpy>=1.24.0 \
    websockets>=12.0

# Copy environment code
COPY . /app/env/

# Set Python path
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the FastAPI server
CMD ["sh", "-c", "cd /app/env && python -m uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
