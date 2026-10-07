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
  logging side effect. `analyze.py` now declares `--log-level` like the others and keeps
  `--verbose` as a shorthand for `DEBUG`, so the two spellings cannot disagree.
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
- ~~Let the user supply the parameter space~~ — shipped. `[optimize.parameter_space.<strategy>]`
  in `niffler.toml` replaces the entry for each parameter it names and leaves the rest of the
  strategy's `PARAMETER_SPEC` alone, so a plateau that has run into the edge of the searched
  range can be widened for one project without editing the strategy. The override is validated
  against the strategy's own constructor signature and against `ParameterSpace`, so a parameter
  the strategy does not accept is an error rather than a silently ignored line.
- ~~Stop retyping every parameter on every run~~ — shipped. `scripts/config_file.py` folds a
  `niffler.toml` into each CLI's argparse defaults: shared `[common]`, `[costs]`, `[engine]`
  and `[risk]` sections, a per-script section, and named `[profile.NAME]` overlays, with the
  command line always winning. It exists for the cost model above all: a friction assumption
  typed into one script and forgotten in another compares two different markets, and sharing
  the flag *definitions* in `common.py` could never prevent that on its own.

## Framework and usability

- **Unify the remaining factory shapes.** #9 removed the strategy-construction `if` chain
  from `scripts/backtest.py`, but two shapes still coexist: module-level dicts plus free
  functions (`niffler/optimization/optimizer_factory.py`) and a class attribute plus
  instance methods (`niffler/exporters/exporter_manager.py`). The risk manager in
  `backtest.py` is still built inline. Pick one and converge.
- ~~**Progress reporting during optimization.**~~ — shipped, and the original note was only
  half right. With `--jobs` above 1 a search did print nothing until it finished, because
  spawned workers do not inherit the logging configuration. The parent process now logs
  `Progress: done/total | elapsed | ETA` at most every 10 seconds, on every evaluation path.
- ~~**A sequential search floods the log.**~~ — found while measuring the item above, and
  fixed with it. With `--jobs 1` the search was the opposite of silent: the engine logs every
  fill and five lines per backtest at `INFO`, so the default 396-combination `breakout` grid
  on five years of daily BTCUSDT wrote about 41,700 lines, 39,524 of them `BUY:`/`SELL:`. The
  same run now writes 175. `quiet_backtests()` holds the backtesting package below `WARNING`
  while the optimizer, walk-forward or Monte Carlo runs backtests in a loop; a single
  `backtest.py` run still logs its fills, and `--log-level DEBUG` restores the full log.

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
the direction on 2026-09-05; the data side is now delivered and the dashboards are not.

Until the two items struck through below, only `scripts/backtest.py` exported to
Elasticsearch: the 396-1632 trials of a grid search and every walk-forward fold and Monte
Carlo simulation — the actual research record — stayed in local JSON, and nothing joined
an optimization to the validation and the final backtest that came out of it. What is left
is the view. The single provisioned dashboard
(`config/grafana/dashboards/backtest-detailed-analysis.json`) is still a one-run
drill-down: pick a `run_id`, read eleven gauges. There is no cross-strategy comparison yet.

In dependency order:

- ~~**An experiment id.**~~ — shipped, in a different shape than first written. Every run
  of every script gets a minted `run_id`; a run fed by `--params-file` records the run that
  produced the file as `parent_run_id`; and the **experiment is a name the user chooses**
  (`--experiment`, or `experiment` in a `[profile.<name>]`), never a minted id — a minted
  one would make a forgotten flag look like a deliberate one-run experiment. A file-fed run
  inherits an unset experiment and a different one is an error. The identity is written
  into every saved JSON result and onto the `niffler-runs` summary document.
- ~~**Export optimization and analysis results.**~~ — shipped, with a different index
  layout than first written. Every script takes the same `--exporters` flags; every run of
  any kind writes one summary to `niffler-runs`, and its trials, folds, simulations or
  comparison rows go to `niffler-trials` / `-folds` / `-simulations` / `-comparisons`. One
  summary index rather than one per kind, so a leaderboard reads a single place. Every
  document carries the run's header, because Elasticsearch has no joins. The
  `SELECTION_TRUNCATED` discipline carried over: a truncated result set exports flagged,
  with its grid statistics null.
- **Nothing has been run against a live Elasticsearch.** Every exporter test mocks the
  client, so nobody has watched the indices fill or the dashboard render. Two things to
  settle in that first run: `elasticsearch_exporter.py` annualises rolling Sharpe and
  volatility with a hardcoded 252, against this project's own invariant; and the dashboard's
  `now-5y` range with a `created_at` datasource time field either cuts off old bars or
  filters on export time. Indices created before #24-#26 need recreating for the new fields.
- **A cross-strategy dashboard.** Not started; the data it needs is now in Elasticsearch.
  Three levels: a leaderboard over `niffler-runs` (one row per experiment, sorted by
  out-of-sample result, with the walk-forward efficiency ratio, the share of the grid
  beating buy-and-hold and plateau retention beside it); an experiment view filtered on
  `experiment` (trial-score distribution with the winner and the baseline marked, in-sample
  versus out-of-sample per fold, the Monte Carlo return distribution); and the existing
  single-run drill-down. Missing must render as missing, never as zero, and a truncated
  grid must show no distribution.

Two traps found while scoping this, both worth handling in the export work rather than
discovering later:

- ~~`strategy_name` is `strategy.name`, the **display** string ("RSI Mean Reversion"), not
  the registry key (`rsi`).~~ — handled: `strategy_key` is recorded beside it.
- ~~`strategy_params` is dynamic-mapped `{"type": "object"}`, so the first document to
  arrive locks each field's type.~~ — handled: the `niffler-runs` mapping uses `flattened`.

## Research rigor

What stands between "the optimizer returned a winner" and "this strategy has an edge".
Three of these were delivered on 2026-10-07; what each one still does not do is listed under
it, because a correction that is read as complete is worse than none.

- ~~**No multiple-testing correction, no deflated Sharpe ratio.**~~ — shipped in #26 for the
  optimizer's winner. `optimize.py` prints a `SEARCH LUCK` block with two figures: the
  grid-relative one (is the winner better than the best of N equally good combinations),
  which leads and drives the verdict, and the published deflated Sharpe ratio (null: no
  edge at all), which a long-only strategy on a rising asset clears from market exposure
  alone. Worked example, `breakout` on BTCUSDT 2019-07 to 2024-07, 396 combinations: the
  winner's Sharpe is 1.263 against a grid-relative luck line of 1.373, a 40.1% probability of
  being above it - no evidence for those parameters - while the published figure reads
  97.9%. Still open:
  - **Effective trial count.** Every combination counts as independent, which over-corrects
    because neighbouring parameter sets are near-duplicates. `--effective-trials` is a manual
    override; nothing estimates it.
  - **One search only.** Trying three strategies and keeping the best is the same selection
    one level up, and nothing counts it, although every run's `experiment` is now recorded.
  - ~~**Not a gate.**~~ — `screen.py` stops at stage 2 on the grid-relative probability
    since #29 (`--min-grid-relative-probability`, default 0.5, a judgment call). A search
    whose luck cannot be assessed stops too, and says it is not a pass. The gate always
    counts every combination, so the BTCUSDT example above (40.1%) now stops there.
- ~~**No untouched data.**~~ — shipped in #24. Walk-forward is out-of-sample per fold, but the
  loop around it is not: a strategy adjusted until it passes has been fitted by whoever was
  adjusting. `screen.py --holdout-data` runs one backtest of the winning parameters on data
  no stage saw, gated on completed round trips and on excess over buy-and-hold; `--force`
  does not spend it, and every look exports the file's hash so the looks can be counted. How
  the files are made is in [data-management.md](data-management.md#research-and-holdout-files).
  Since #29 `--holdout-data` takes several files and pools them - round trips summed, excess
  the median across files, each file its own exported row - and every other script warns
  when a data file is named like a holdout. Still open: the warning goes by file name only
  and refuses nothing, and no cap on looks per holdout file is enforced.
- ~~**One asset, one window.**~~ — shipped in #11 and #14. `scripts/compare.py` takes several
  `--data` files, runs the same walk-forward over each and reports excess over buy-and-hold
  per dataset; `scripts/screen.py` makes that cross-asset comparison the last gate of its
  funnel. `backtest.py`, `optimize.py` and `analyze.py` still take one CSV each, by design.
- ~~**Walk-forward folds overlap by default.**~~ — fixed in #25. `analyze.py --step` and
  `WalkForwardAnalyzer` now default to the test window, as `compare.py` and `screen.py`
  already did: `simple_ma` on BTCUSDT research data goes from 15 folds with 46.4% of
  out-of-sample bars repeated to 8 independent ones, and the pooled Sharpe from 1.17 to 0.91.
  Explicit overlap is still allowed; its per-fold figures are then labelled as
  non-independent - labelled, not corrected - and the run exports `folds_independent`.

## A system for new strategies

Recorded as a direction on 2026-10-07. Nothing is built and the design is not agreed.

The goal: a scheduled agent finds strategies published on the internet, writes each one as
a Niffler strategy, runs the whole experiment on it, and the results appear somewhere they
can be compared - a leaderboard, with the detail behind each row. That needs a way to
document a strategy first, by hand or by the agent, because today a strategy is only a class
and a registry line: where the idea came from, what its rules are and what has already been
tried on it are written down nowhere.

What a design has to answer before any of it is built:

- **A strategy catalog.** One reviewed record per strategy - source, rules, parameters,
  status - that survives independently of Elasticsearch, and tells a new idea from one
  already tested under another name.
- **One fixed test protocol.** The same datasets, costs, folds and gates for every strategy,
  pinned and versioned, or the leaderboard compares runs that were never comparable.
- **Selection across strategies.** Testing fifty strategies and reading the top of the list
  is the grid-search problem one level up (see **Research rigor**); the leaderboard has to
  show "best of N" beside the winner.
- **Who spends the holdout.** An agent that screens every night would use the holdout up in
  a week. It should stop before that stage and leave the look to a person.
- **Generated code is untrusted.** Rules read from a web page are reimplemented and
  reviewed, never executed as found, and nothing merges itself.
- **Where it shows.** This is the cross-strategy dashboard under **Experiment tracking**,
  which therefore comes first, as does the first run against a live Elasticsearch.

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
method raises. A single backtest's p-value is not corrected for the search that found its
parameters: the correction exists for the optimizer's winner only (see **Research rigor**),
and the documentation and the console output say so rather than pretending otherwise.
