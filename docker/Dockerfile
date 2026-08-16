FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# chromadb'nin (sqlite) build zinciri için sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src ./src

RUN pip install .

COPY data/sample_docs ./data/sample_docs
COPY scripts ./scripts

# Root olmayan bir runtime kullanıcısı. vectorstore dizini burada oluşturulup
# chown ediliyor (volume mount'un halletmesine bırakılmıyor) çünkü Docker
# taze bir named volume'ü, imajdaki mount noktasında zaten var olan şeyden
# başlatıyor — bu olmasaydı volume root-owned gelirdi ve Chroma sqlite
# dosyasını `appuser` olarak açamazdı.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /data/vectorstore \
    && chown -R appuser:appuser /app /data/vectorstore
USER appuser

EXPOSE 8000

# Burada bilinçli olarak HEALTHCHECK yok: bu tek imaj iki farklı servisin
# arkasında duruyor (api :8000'de, mcp docker-compose.yml'deki bir command
# override ile :8765'te) — tek, gömülü bir kontrol ikisi için de doğru
# olamaz. Her servis kendi `healthcheck:`ini docker-compose.yml'de tanımlıyor;
# buraya :8000'e sabitlenmiş bir kontrol koymak mcp konteynerini sonsuza dek
# sessizce unhealthy gösterirdi.

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
