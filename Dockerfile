# syntax=docker/dockerfile:1

# Use Python 3.13 as base image
FROM python:3.13-slim

# uv is pinned: an unpinned `pip install uv` makes every rebuild a different image.
ARG UV_VERSION=0.7.20
# The runtime user's UID/GID. 1000 is the first non-system id on a typical Linux
# desktop, so the bind-mounted ./data and ./results directories (see
# docker-compose.yml) stay writable without a chown on the host. Override with
# `docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) .` if your host
# user differs. On Docker Desktop (Windows/macOS) bind mounts ignore this anyway.
ARG UID=1000
ARG GID=1000

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Install uv for dependency management
RUN pip install --no-cache-dir "uv==${UV_VERSION}"

# Create an unprivileged user to run the application - containers must not run as
# root, and this one mounts host directories.
RUN groupadd --gid "${GID}" niffler \
    && useradd --uid "${UID}" --gid "${GID}" --create-home --shell /bin/bash niffler

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv. --locked fails loudly if uv.lock is out of date
# with pyproject.toml instead of silently building something unreproducible.
# --no-dev keeps the linter/type checker out of the runtime image.
RUN uv sync --locked --no-dev

# Copy the application (see .dockerignore - .git/, .venv*/, data/ and results/ are
# excluded so the build context stays small and local market data is never baked
# into the image).
COPY . .

# Mount points for the compose bind mounts; created up front so they are owned by
# the runtime user rather than by root.
RUN mkdir -p /app/data /app/results \
    && chown -R niffler:niffler /app

USER niffler

# Default command (can be overridden in docker-compose)
CMD ["python", "scripts/backtest.py", "--help"]
