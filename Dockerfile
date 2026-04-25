# =============================================================================
# EnlightenLM 多阶段 Dockerfile
# 优化镜像体积，支持GPU加速，适用于生产环境
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - 依赖安装
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

# -----------------------------------------------------------------------------
# Stage 2: Development - 开发环境
# -----------------------------------------------------------------------------
FROM python:3.11-slim as development

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY . .

ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

EXPOSE 8000

CMD ["python", "-m", "enlighten.api_server", "--host", "0.0.0.0", "--port", "8000"]

# -----------------------------------------------------------------------------
# Stage 3: Production - 生产环境 (CPU)
# -----------------------------------------------------------------------------
FROM python:3.11-slim as production

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 appuser

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser enlighten/ ./enlighten/

ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV PYTHONDONTWRITEBYTECODE=1

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "enlighten.api_server", "--host", "0.0.0.0", "--port", "8000"]

# -----------------------------------------------------------------------------
# Stage 4: Production-GPU - 生产环境 (NVIDIA GPU)
# -----------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 as production-gpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 appuser

COPY requirements.txt .

RUN python3.11 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY --from=builder --chown=appuser:appuser /root/.local /root/.local

COPY --chown=appuser:appuser configs/ ./configs/
COPY --chown=appuser:appuser enlighten/ ./enlighten/

ENV PATH=/opt/venv/bin:/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/opt/venv/bin/python", "-m", "enlighten.api_server", "--host", "0.0.0.0", "--port", "8000"]
