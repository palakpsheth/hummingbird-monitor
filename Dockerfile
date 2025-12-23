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

RUN pip install --no-cache-dir -e .

EXPOSE 8000
