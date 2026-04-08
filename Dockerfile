# SpecGuardian++ — OpenEnv Docker Space for Hugging Face
# SDK: docker  |  Tag: openenv
# Starts a FastAPI HTTP server on port 7860

# ---- Stage 1: dependency builder ----
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ---- Stage 2: runtime image ----
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="SpecGuardian++"
LABEL org.opencontainers.image.description="OpenEnv agent safety evaluation environment"
LABEL org.opencontainers.image.version="2.0"

# Non-root user (HF Spaces requirement)
RUN useradd --create-home --shell /bin/bash --uid 1000 specguardian
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy source (owned by non-root user)
COPY --chown=specguardian:specguardian . .

USER specguardian

# Expose the standard HF Spaces port
EXPOSE 7860

# Health check via the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" || exit 1

# Start the FastAPI server
CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
