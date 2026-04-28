# ─── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies (needed for asyncpg, cryptography, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─── Stage 2: runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: non-root user
RUN groupadd --gid 1001 algotrader \
    && useradd  --uid 1001 --gid algotrader --no-create-home --shell /sbin/nologin algotrader

# Runtime libpq for asyncpg
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy application code
COPY --chown=algotrader:algotrader . .

# Create log directory
RUN mkdir -p /app/logs && chown algotrader:algotrader /app/logs

USER algotrader

# Health check: ping the FastAPI control plane
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
    || exit 1

# Default: run the trading bot
CMD ["python", "-m", "main"]

EXPOSE 8000
