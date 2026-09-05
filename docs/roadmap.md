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
  and all five CLI scripts configure it inside `main()`, so importing a script has no
  logging side effect. (`analyze.py` still derives its level from `--verbose` rather than
  `--log-level`; converging that flag is a separate, purely cosmetic change.)
- ~~Unify the `__init__.py` files~~ — done. Every package `__init__.py` now declares
  `__all__` with explicit re-exports.
- ~~Work out why `__pycache__` is everywhere~~ — resolved. `__pycache__/` and `*.py[cod]`
  are gitignored and nothing matching them is tracked.
- ~~The preprocessor's default output is not under `data/`~~ — the premise no longer
  holds. `scripts/preprocessor.py` has no fixed default output path: it writes next to its
  input, so cleaning `data/x.csv` produces `data/x_cleaned.csv`.
- ~~More strategies~~ — shipped in #9. `rsi` (mean reversion on Wilder's RSI) and
  `breakout` (Donchian channel) join `simple_ma`, picked as structurally different
  families rather than variations on a crossover. Neither has an out-of-sample record
  worth calling an edge yet; see **Research rigor** below.
- ~~The strategy half of the factory problem~~ — resolved in #9. There is now one
  registry (`niffler/strategies/registry.py`); every CLI's `--strategy` choices derive
  from it and `scripts/backtest.py` constructs generically, so adding a strategy is one
  class plus one registry line. This also killed a real bug: `analyze.py` defined its own
  shadowing `get_strategy_class`, so a strategy `optimize.py` accepted was rejected there.

## Framework and usability

- **Unify the remaining factory shapes.** #9 removed the strategy-construction `if` chain
  from `scripts/backtest.py`, but two shapes still coexist: module-level dicts plus free
  functions (`niffler/optimization/optimizer_factory.py`) and a class attribute plus
  instance methods (`niffler/exporters/exporter_manager.py`). The risk manager in
  `backtest.py` is still built inline. Pick one and converge.
- **Let the user supply the parameter space.** #9 moved the search space onto the strategy
  as a `PARAMETER_SPEC` class attribute, which removed the second definition that could
  drift — but it is still a code constant. `scripts/optimize.py` has no way to override or
  widen it for a single run, which the plateau analysis asks for by name whenever a plateau
  runs into the edge of the searched range. A JSON parameter-space file would close that
  loop.
- **Progress reporting during optimization.** A grid search currently prints nothing until
  it finishes. There is no per-combination feedback and no ETA.

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

## Experiment tracking

The gap between what the platform computes and what a person can actually see. Agreed as
the direction on 2026-09-05, deferred rather than started.

Today only `scripts/backtest.py` exports to Elasticsearch. `optimize.py` and `analyze.py`
write local JSON, so the 396-1632 trials of a grid search and every walk-forward fold and
Monte Carlo simulation — the actual research record — never reach Kibana or Grafana. The
single provisioned dashboard (`config/grafana/dashboards/backtest-detailed-analysis.json`)
is a one-run drill-down: pick a `backtest_id`, read eleven gauges. There is no
cross-strategy comparison, and nothing joins an optimization to the validation and the
final backtest that came out of it.

In dependency order:

- **An experiment id.** Minted once at the CLI beside provenance — same invariant:
  collected once per run, never raises — threaded through `optimize.py`, `analyze.py` and
  `backtest.py`, stamped on every exported document. This is the join key that does not
  exist today.
- **Export optimization and analysis results.** Two indices (`optimizations`: one document
  per trial plus a run summary; `analyses`: per-fold / per-simulation plus a summary),
  reusing `ExporterManager` and the existing provenance block. The `SELECTION_TRUNCATED`
  discipline has to carry over: a truncated result set exports what it has, flagged, never
  a grid statistic computed from score-biased survivors.
- **A cross-strategy dashboard.** Leaderboard by strategy and experiment, in-sample versus
  out-of-sample scatter from the walk-forward efficiency ratio, and performance over time.

Two traps found while scoping this, both worth handling in the export work rather than
discovering later:

- `strategy_name` is `strategy.name`, the **display** string ("RSI Mean Reversion"), not
  the registry key (`rsi`). A Kibana group-by therefore keys on prose, and editing a
  display name silently splits a strategy's history into two buckets. Stamp the registry
  key alongside it.
- `strategy_params` is dynamic-mapped `{"type": "object"}`, so the first document to
  arrive locks each field's type — and RSI's `oversold` is an `int` in `PARAMETER_SPEC`
  but a `float` from the constructor default. Use `flattened`, or a JSON keyword sidecar.

## Research rigor

What stands between "the optimizer returned a winner" and "this strategy has an edge".
The platform is already honest about each of these in its console output; none is corrected
for.

- **No multiple-testing correction, no deflated Sharpe ratio.** A grid search maximises over
  hundreds of estimates and the significance test then judges the winner as if it were the
  only trial. Worked example, `breakout` on BTCUSDT 2019-07 to 2024-07: the winner returns
  935.64% against a 540.48% buy-and-hold baseline, but the *median* of the 396 combinations
  returns 438.42% — below the baseline — and only 24.7% of the grid beats holding at all.
  The inputs for a deflated Sharpe are all already in hand (trial count, the spread of
  scores across the complete result set, the winner's return moments, the bar count).
- **One asset, one window.** Every CLI takes a single `--data` CSV, so a result is an
  anecdote until it is repeated. A batch runner that fans one strategy across N datasets
  and reports the cross-sectional distribution is what turns it into evidence.
- **Walk-forward folds overlap by default.** `test_window=6` with `step=3` means
  consecutive out-of-sample windows share half their bars. The run reports
  `oos_overlap_pct` and the pooled metrics deduplicate, but 15 folds are not 15
  independent observations, and the per-fold counters still treat them as such.

## Longer term

These are the genuinely large items, and none of them is started.

- **Paper trading**, then live execution, against Binance / Bybit / IBKR. `ccxt` is already
  a dependency, but only for historical downloads — there is no order routing, no position
  reconciliation and no broker abstraction. This is the single biggest gap between the
  project as it stands and something that touches real money — but it is **not** the
  current priority. The stated goal is a strategy *research* platform; nothing here is
  trading yet, so the items under **Experiment tracking** and **Research rigor** come
  first.
- **Monitoring and alerting for live trading**, which is the item above made useful:
  dashboards and alerts are only worth building once there is a live position to watch.

## Deliberately out of scope

Recorded here so they are not mistaken for oversights. Niffler is long-only and has no live
trading. The Kelly risk manager is a stub — the class exists and every
method raises. There is no multiple-testing correction and no deflated Sharpe ratio, so a
p-value from a run whose parameters were fitted on the same data overstates the evidence;
the documentation and the console output say so rather than pretending otherwise.
