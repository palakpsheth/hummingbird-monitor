FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    procps \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip

WORKDIR /app

# Copy only packaging metadata first for better docker caching
COPY pyproject.toml README.md /app/
COPY src /app/src

ARG GIT_COMMIT=unknown
ENV HBMON_GIT_COMMIT=${GIT_COMMIT}
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# Optionally copy lightweight git metadata when available (no-op outside git repos)
RUN --mount=type=bind,source=.,target=/tmp/src,ro \
    if [ -d /tmp/src/.git ] || [ -f /tmp/src/.git ]; then \
        git_dir="/tmp/src/.git"; \
        if [ -f "$git_dir" ]; then \
            git_dir_path=$(sed -n 's/^gitdir: //p' "$git_dir"); \
            case "$git_dir_path" in \
                /*) git_dir="$git_dir_path" ;; \
                *) git_dir="/tmp/src/$git_dir_path" ;; \
            esac; \
        fi; \
        if [ -f "$git_dir/HEAD" ]; then \
            mkdir -p /app/.git && cp "$git_dir/HEAD" /app/.git/HEAD; \
            if [ -d "$git_dir/refs" ]; then mkdir -p /app/.git && cp -r "$git_dir/refs" /app/.git/ 2>/dev/null || true; fi; \
            if [ -f "$git_dir/packed-refs" ]; then cp "$git_dir/packed-refs" /app/.git/packed-refs; fi; \
        fi; \
    fi

RUN pip install --no-cache-dir --index-url ${PYTORCH_INDEX_URL} --extra-index-url https://pypi.org/simple -e .

EXPOSE 8000
