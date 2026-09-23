# SenseCast image: one image, three roles (pipeline / api / dashboard) chosen by command.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    MPLCONFIGDIR=/tmp/mpl

# libgomp1: OpenMP runtime needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY dashboard ./dashboard
RUN pip install --no-deps -e . \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /app/data /app/artifacts /app/mlruns /app/docs/03-evaluation \
    && chown -R app:app /app

USER app
EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "sensecast.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
