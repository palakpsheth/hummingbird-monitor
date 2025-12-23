FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only packaging metadata first for better docker caching
COPY pyproject.toml README.md /app/
COPY src /app/src

ARG GIT_COMMIT=unknown
ENV HBMON_GIT_COMMIT=${GIT_COMMIT}

# Optionally copy lightweight git metadata when available (no-op outside git repos)
RUN --mount=type=bind,source=.,target=/tmp/src,ro \
    if [ -f /tmp/src/.git/HEAD ]; then \
        mkdir -p /app/.git && cp /tmp/src/.git/HEAD /app/.git/HEAD; \
        if [ -d /tmp/src/.git/refs ]; then mkdir -p /app/.git && cp -r /tmp/src/.git/refs /app/.git/ 2>/dev/null || true; fi; \
        if [ -f /tmp/src/.git/packed-refs ]; then cp /tmp/src/.git/packed-refs /app/.git/packed-refs; fi; \
    fi

RUN pip install --no-cache-dir -e .

EXPOSE 8000
