# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files needed to resolve dependencies
COPY pyproject.toml .
COPY app/ ./app/

# Install dependencies into an isolated venv (not editable — no symlinks needed)
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python . --no-cache

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# System deps: wget + tar for Piper download
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget tar && \
    rm -rf /var/lib/apt/lists/*

# Copy venv from builder (includes all Python deps)
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Download Piper Linux binary.
# The tar extracts as: piper/piper (binary) + piper/espeak-ng-data/ etc.
# We extract into /app/piper/ so the binary lands at /app/piper/piper/piper
RUN wget -q https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz \
    -O /tmp/piper.tar.gz && \
    mkdir -p /app/piper && \
    tar -xzf /tmp/piper.tar.gz -C /app/piper/ && \
    chmod +x /app/piper/piper/piper && \
    rm /tmp/piper.tar.gz && \
    echo "Piper binary check:" && ls -la /app/piper/piper/

# Download Piper Spanish voice model
RUN wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx" \
        -O /app/piper/es_ES-sharvard-medium.onnx && \
    wget -q "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx.json" \
        -O /app/piper/es_ES-sharvard-medium.onnx.json

# Copy application code and static files
COPY --from=builder /app/app/ ./app/
COPY static/ ./static/

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

ENV WHISPER_MODEL=base
ENV WHISPER_DEVICE=cpu
ENV TTS_ENGINE=piper

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
