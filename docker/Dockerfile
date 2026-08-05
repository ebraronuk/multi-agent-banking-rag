FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for chromadb (sqlite) build chain
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

RUN pip install .

COPY data/sample_docs ./data/sample_docs

# Non-root runtime user. The vectorstore directory is created and chowned here
# (not left for the volume mount to sort out) because Docker initializes a
# fresh named volume from whatever already exists at the mount point in the
# image — without this, the volume comes up root-owned and Chroma can't open
# its sqlite file as `appuser`.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /data/vectorstore \
    && chown -R appuser:appuser /app /data/vectorstore
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
