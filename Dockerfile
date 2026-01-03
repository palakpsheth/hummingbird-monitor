
# Pin to bookworm for stability
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get upgrade -y && apt-get dist-upgrade -y && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    procps \
    ca-certificates \
    curl \
    wget \
    gnupg \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Install static FFmpeg to avoid library-level vulnerabilities (e.g. mbedtls, libmfx1)
# Note: we use x86_64/amd64 build; if needed for other archs, this would need detection.
RUN mkdir -p /tmp/ffmpeg && \
    curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz | tar -xJ -C /tmp/ffmpeg --strip-components=1 && \
    cp /tmp/ffmpeg/ffmpeg /tmp/ffmpeg/ffprobe /usr/local/bin/ && \
    rm -rf /tmp/ffmpeg


# Install Intel GPU compute runtime (required for OpenVINO GPU plugin)
# This enables HBMON_INFERENCE_BACKEND=openvino-gpu on Intel Iris Xe / Arc GPUs
# Only installed if INSTALL_OPENVINO=1 is passed as build argument
ARG INSTALL_OPENVINO=0
RUN if [ "$INSTALL_OPENVINO" = "1" ]; then \
    # Note: We use the Ubuntu Jammy repository as Intel does not provide a dedicated 
    # Debian Bookworm repository for these packages. The packages are largely 
    # self-contained and compatible with the Debian Bookworm base.
    wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" \
    > /etc/apt/sources.list.d/intel-gpu-jammy.list \
    # Pin Intel packages to prefer the Intel repository over Debian's
    && echo "Package: intel-* level-zero libigc* intel-igc-*\nPin: origin repositories.intel.com\nPin-Priority: 1000" > /etc/apt/preferences.d/intel-gpu-pin \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    intel-level-zero-gpu \
    level-zero \
    intel-opencl-icd \
    intel-gpu-tools \
    && rm -rf /var/lib/apt/lists/*; \
    fi

RUN pip install --upgrade pip

WORKDIR /app

# Copy only packaging metadata first for better docker caching
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY scripts /app/scripts

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
