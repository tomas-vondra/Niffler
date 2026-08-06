# Roadmap

Forward-looking notes on where Niffler is going. This file replaces the informal `todo`
file that used to sit in the repository root; the items below were carried over and each
one was re-checked against the code before being kept, struck through, or reworded.

Nothing here is a commitment or a schedule. It is a record of what is known to be missing,
so that the gaps are visible rather than implied.

## Delivered

Kept struck through rather than deleted, so the list stays honest about what moved.

- ~~Compare with a buy-and-hold strategy~~ — shipped. `niffler/backtesting/benchmark.py`
  computes the benchmark over the same bars, charged the same commission and the same cost
  model, and it is on by default (`--benchmark buy_and_hold`).
- ~~Consolidate duplicate code~~ — largely done. There is one FIFO trade-pairing routine
  (`niffler/backtesting/round_trip.py`), one equity-metrics module
  (`niffler/backtesting/metrics.py`), and one OHLCV CSV loader plus one shared
  transaction-cost CLI (`scripts/common.py`). The two remaining "duplicates"
  (`config/logging.py`, `niffler/exporters/json_utils.py`) are deliberate
  backwards-compatible re-export shims, not copies.
- ~~Create the logger once at start~~ — done. `niffler/config/logging.py` holds the only
  `logging.basicConfig` call; every other module just calls `logging.getLogger(__name__)`,
  and all five CLI scripts configure it inside `main()` from `--log-level`, so importing a
  script has no logging side effect.
- ~~Unify the `__init__.py` files~~ — done. Every package `__init__.py` now declares
  `__all__` with explicit re-exports.
- ~~Work out why `__pycache__` is everywhere~~ — resolved. `__pycache__/` and `*.py[cod]`
  are gitignored and nothing matching them is tracked.
- ~~The preprocessor's default output is not under `data/`~~ — the premise no longer
  holds. `scripts/preprocessor.py` has no fixed default output path: it writes next to its
  input, so cleaning `data/x.csv` produces `data/x_cleaned.csv`.

## Framework and usability

- **Unify the factory pattern.** Three shapes coexist today: module-level dicts plus free
  functions (`niffler/optimization/optimizer_factory.py`), a class attribute plus instance
  methods (`niffler/exporters/exporter_manager.py`), and no factory at all — `scripts/backtest.py`
  builds the risk manager and the strategy with inline `if` chains, duplicating what
  `get_strategy_class()` already does. Pick one and converge.
- **Let the user supply the parameter space.** `PARAMETER_SPACES` is hardcoded in
  `niffler/optimization/parameter_space.py` and selected by strategy name;
  `scripts/optimize.py` has no way to override or extend it. A JSON parameter-space file
  would make the optimizer usable on a strategy without a code change.
- **Progress reporting during optimization.** A grid search currently prints nothing until
  it finishes. There is no per-combination feedback and no ETA.
- **More strategies.** There is exactly one concrete strategy
  (`niffler/strategies/simple_ma_strategy.py`). A second, structurally different one is the
  real test of whether `BaseStrategy` is the right seam — and it needs its own record of
  out-of-sample performance, not just a backtest.

## Observability

- **Ship logs to Elasticsearch or a database.** Distinct from the existing Elasticsearch
  *results* exporter, which writes backtest/trade/position documents. Application logging
  still goes only to a stream and a file handler; there is no log shipping.
- **Prometheus.** None exists. `docker-compose.yml` runs Elasticsearch, Grafana and
  (behind a `debug` profile) Kibana, and Grafana is provisioned against Elasticsearch —
  so dashboards are real, but metrics are not.
- **Alerting.** No alert rules, contact points or notification policies are provisioned.
  `config/grafana/README.md` documents the manual UI steps only.
- **Static analysis beyond ruff.** The old note said "update sonar"; there is no SonarQube
  or SonarCloud configuration in the repository at all. CI runs `ruff check` and the test
  suite, plus an advisory mypy pass. Whether Sonar is worth adding on top is an open
  question, not a decided one.

## Longer term

These are the genuinely large items, and none of them is started.

- **Paper trading**, then live execution, against Binance / Bybit / IBKR. `ccxt` is already
  a dependency, but only for historical downloads — there is no order routing, no position
  reconciliation and no broker abstraction. This is the single biggest gap between the
  project as it stands and something that touches real money.
- **Monitoring and alerting for live trading**, which is the item above made useful:
  dashboards and alerts are only worth building once there is a live position to watch.

## Deliberately out of scope

Recorded here so they are not mistaken for oversights. Niffler is long-only, single-strategy
and has no live trading. The Kelly risk manager is a stub — the class exists and every
method raises. There is no multiple-testing correction and no deflated Sharpe ratio, so a
p-value from a run whose parameters were fitted on the same data overstates the evidence;
the documentation and the console output say so rather than pretending otherwise.
