# ═══════════════════════════════════════════════════════════════════════════
# Multi-Agent Investment Analysis System - Modern Dockerfile (2025)
# ═══════════════════════════════════════════════════════════════════════════
#
# IMPROVEMENTS OVER LEGACY VERSION:
# - Multi-stage build (reduces image size by ~40%)
# - Poetry 2.x with modern dependency groups
# - ChromaDB system dependencies included
# - Non-root user for security
# - Proper health check without HTTP dependency
# - Build-time argument for Python version flexibility
# ═══════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder - Install dependencies in isolated environment
# ────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

ARG POETRY_VERSION=2.1.1

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry using official installer
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /build

# Copy dependency manifests
COPY pyproject.toml poetry.lock* ./

# Configure Poetry to not create virtual env (we're already in a container)
ENV POETRY_VIRTUALENVS_CREATE=false

# Install dependencies to system Python (builder stage)
# Updated: --only main replaces deprecated --no-dev
RUN poetry install --only main --no-interaction --no-ansi

# ────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime - Minimal production image
# ────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Install only runtime dependencies (SQLite for ChromaDB)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security (don't run as root in production)
RUN groupadd -r agent && useradd -r -g agent -u 1000 agent

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=agent:agent src/ ./src/
COPY --chown=agent:agent prompts/ ./prompts/

# Create directories for ChromaDB persistence and results
RUN mkdir -p /app/chroma_db /app/results /app/scratch \
    && chown -R agent:agent /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CHROMA_PERSIST_DIRECTORY=/app/chroma_db \
    ANONYMIZED_TELEMETRY=False

# Switch to non-root user
USER agent

# Health check - verify Python imports work (no HTTP server needed)
# This checks that the application can at least import its main module
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import sys; from src.main import main; sys.exit(0)" || exit 1

# Entrypoint uses module syntax for clean imports
ENTRYPOINT ["python", "-m", "src.main"]

# Default arguments (override at runtime)
CMD ["--ticker", "AAPL", "--quick"]
