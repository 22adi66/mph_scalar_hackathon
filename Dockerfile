FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project definition first for better layer caching
COPY pyproject.toml /app/
COPY requirements.txt /app/

# Install Python dependencies from requirements.txt (lightweight, no torch/gymnasium)
RUN pip install --no-cache-dir -r requirements.txt

# Copy all environment code
COPY . /app/env/

# Set Python path so server/ can import sibling modules (models, simulation, graders)
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV PORT=7860

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the FastAPI server
CMD ["sh", "-c", "cd /app/env && python -m uvicorn server.app:app --host 0.0.0.0 --port ${PORT}"]
