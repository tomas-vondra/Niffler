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
   The expected count is quoted in one place only:
   [README → Testing](../README.md#testing).

## Core Dependencies

Every runtime dependency is pinned with both a lower **and an upper** bound, so a major
release upstream cannot silently change behaviour on `uv sync`:

| Package | Constraint | Purpose |
|---------|-----------|---------|
| pandas | `>=2.3.1,<3.0.0` | Data manipulation and analysis |
| ccxt | `>=4.4.94,<5.0.0` | Cryptocurrency exchange data access |
| yfinance | `>=0.2.65,<0.3.0` | Traditional financial market data |
| python-dateutil | `>=2.9.0,<3.0.0` | Advanced date handling |
| elasticsearch | `>=8.0.0,<9.0.0` | Optional export destination |

`numpy` comes in transitively via pandas.

**yfinance is the dangerous one.** Its `auto_adjust` default flipped in 0.2.51, silently
changing whether downloaded prices were split/dividend adjusted. Niffler now passes
`auto_adjust` explicitly on every call (see
[Data Management](data-management.md#price-adjustment-convention-important)), but the upper
bound stays as a second line of defence.

## Development Dependencies

`uv sync` also installs a dev group:

- **ruff** - Linter; `ruff check .` must pass and is enforced in CI
- **mypy** - Type checker; scoped to `niffler/`, non-strict, and **advisory only**
  (`continue-on-error` in CI)
- **types-python-dateutil** - Type stubs
- **unittest** (standard library) - The test framework. There is no pytest

## Lockfile

`uv.lock` is committed and must stay in sync with `pyproject.toml`. Docker builds run
`uv sync --locked --no-dev`, which **fails loudly** if the lockfile has drifted (the older
`--frozen` did not) and keeps the linter and type checker out of the runtime image.

After changing a dependency, run:

```bash
uv lock
uv lock --check   # exits 0 when the lockfile matches the manifest
```

## Continuous Integration

`.github/workflows/ci.yml` runs on pushes to `main`/`master` and on every pull request:
checkout → Python 3.13 → pinned `uv` (with a cache keyed on `uv.lock`) → `uv sync --locked`
→ `ruff check .` → the unittest discovery command → an advisory `mypy` pass.

## Docker (optional)

The visualization stack (Elasticsearch, Grafana, optionally Kibana) runs via Docker Compose.
See [visualization/README.md](../visualization/README.md). Two things worth knowing before
you build:

- The container runs as an unprivileged user (UID/GID 1000 by default). On a Linux host
  whose user is not UID 1000, writes to the bind-mounted `./data` and `./results` will fail
  until you rebuild with `--build-arg UID=$(id -u) --build-arg GID=$(id -g)` or enable the
  commented-out `user:` line in `docker-compose.yml`. Docker Desktop on Windows/macOS is
  unaffected.
- `.dockerignore` keeps `.git/`, `.venv*/`, `data/`, `results/` and tool caches out of the
  build context, so images no longer bake in local market data or a host-specific
  virtualenv.
- **These Dockerfile changes have not been verified at runtime** — no machine with a working
  Docker daemon has run `docker compose build && up` against them yet.