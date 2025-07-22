# Installation Guide

## Requirements

- Python ≥3.13
- `uv` package manager

## Installing uv

This project uses `uv` for dependency management. To install `uv`, follow the instructions [here](https://github.com/astral-sh/uv).

## Project Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Niffler
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Verify installation by running tests:
   ```bash
   python -m unittest discover -s tests -p "test_*.py"
   ```

## Core Dependencies

- **pandas** (≥2.3.1) - Data manipulation and analysis
- **ccxt** - Cryptocurrency exchange data access
- **yfinance** - Traditional financial market data
- **numpy** - Numerical computations and statistical analysis
- **python-dateutil** - Advanced date handling

## Development Dependencies

The project includes comprehensive testing and development tools:
- **unittest** - Built-in Python testing framework
- **mock** - Testing utilities for external dependencies