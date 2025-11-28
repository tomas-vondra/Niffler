# Use Python 3.13 as base image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install uv for dependency management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen

# Copy the entire application
COPY . .

# Set Python path to include the application directory
ENV PYTHONPATH=/app

# Default command (can be overridden in docker-compose)
CMD ["uv", "run", "python", "scripts/backtest.py", "--help"]