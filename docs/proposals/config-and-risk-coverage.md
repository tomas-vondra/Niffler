# Proposal: config files + risk-manager coverage

Analysed against `origin/master` @ `c57f379` ("Merge pull request #13 from
feat/risk-manager-seam").

> **Read this first.** Your working checkout `D:/repositories/niffler` is on branch
> `exporters` (`7409ca6`), which is the **merge base** — `origin/master` is 46 commits
> ahead of it, 0 behind. Everything below describes master. On `exporters` both problems
> look much larger than they actually are, because master has already built most of the
> machinery each task needs.

Both tasks land in the same two files (`scripts/common.py`,
`niffler/backtesting/run_config.py`) and are best done together.

---

# Task 2 first — it is smaller than you think, and half-built already

Taking this one first because master's answer changed it completely.

## The gap is real and still open

`--risk-manager` exists in exactly one place: `scripts/backtest.py:321` (flag),
`:359` (construction), `:370` (injection into `create_strategy`).

`optimize.py`, `analyze.py`, `compare.py` and `screen.py` never construct one. The three
engine-driving consumers build strategies bypassing the registry seam entirely:

| Site | Call |
|---|---|
| `base_optimizer.py:361, 389` | `strategy_class(**parameters)` |
| `walk_forward_analyzer.py:177, 218, 670` | `strategy_class(**parameters)` |
| `monte_carlo_analyzer.py:81, 101` | `strategy_class(**parameters)` |
| `screen.py:241` | `create_strategy(strategy_name, {})` — no `risk_manager=` |

`BaseStrategy.__init__` defaults `risk_manager=None`, so **optimize, walk-forward, Monte
Carlo and screen all run with the risk layer switched off**, unconditionally.

## Why it matters

The pipeline is *optimize → validate → trade*. Today:

- **Optimization** selects parameters for an unconstrained, full-size-position system.
- **Walk-forward** then certifies those parameters as temporally robust — for a system
  with no stop-loss and no position cap.
- **Monte Carlo** reports the drawdown and return distribution. That output *is* a risk
  measurement, produced with the risk layer off. Stop-losses are precisely what truncates
  the left tail, so the distribution does not describe the system you would deploy.
- **screen.py** ranks candidate strategies under one risk regime (none) and you then trade
  them under another.

This is the same class of defect `RunConfig` was built to eliminate. From
`run_config.py`'s own docstring:

> "the engine was constructed at six places and five of them passed three […] Every other
> knob silently reverted to its default somewhere inside a walk-forward fold"

The risk manager is the **one remaining knob that still does exactly that**. `RunConfig`
has ten fields; none of them is risk.

## The historical blocker is gone

`niffler/risk/contract.py` states why risk was never threaded into validation:

> "A risk manager used to keep its own `Dict[str, PositionInfo]` […] reusing one across two
> runs let the first run's open position veto the second run's first entry through
> `max_positions`. Walk-forward runs its folds in parallel and every fold is an independent
> hypothetical history, so a shared manager could not be threaded into validation at all —
> **which is why it never was.**"

PR #13 fixed that. Risk managers are now **stateless**: `BaseRiskManager` holds only
`self.config`, and portfolio state arrives per-call as a frozen `PortfolioSnapshot`
(`contract.py`). `_positions` / `update_position_state` / `clear_position` are gone.

A stateless, config-only manager is safely shareable across folds and picklable across the
spawn boundary. **The seam was built for this task and then not used.** Finishing it is
the natural next commit, not a new project.

## Recommendation: one field on `RunConfig`, mirroring `cost_model`

```python
# niffler/backtesting/run_config.py
from niffler.risk.contract import RiskManager   # backtesting -> risk, the allowed direction

@dataclass(frozen=True)
class RunConfig:
    ...
    cost_model: Optional[CostModel] = None
    risk_manager: Optional[RiskManager] = None   # None = no risk management
```

This is the right carrier for four reasons:

1. **It is what `RunConfig` is for.** Its docstring promises "a knob added to `RunConfig` is
   reachable from every script that calls `build_run_config`". Five CLIs call it.
2. **It already crosses the spawn boundary.** `RunConfig` is passed into workers at
   `base_optimizer.py:379/392`, `walk_forward_analyzer.py:574/605`,
   `monte_carlo_analyzer.py:366`. Risk rides along for free — no new plumbing.
3. **`cost_model` is the exact precedent.** Also an `Optional[...]` object on a frozen
   config, also kept as `None` rather than normalised so "not configured" stays visible in
   metadata. Follow it verbatim.
4. **Stateless makes sharing safe.** The one thing that made this impossible before is the
   thing PR #13 removed.

### CLI: one edit reaches every script

Move the flag out of `backtest.py` into `scripts/common.py`, beside
`add_cost_model_arguments`:

```python
def add_risk_manager_arguments(parser):
    group = parser.add_argument_group('risk management')
    group.add_argument('--risk-manager', choices=get_available_risk_managers(),
                       default=NO_RISK_MANAGER)
    group.add_argument('--max-position-size', type=float, default=None)
    group.add_argument('--stop-loss-pct',     type=float, default=None)
    group.add_argument('--max-positions',     type=int,   default=None)
    group.add_argument('--max-risk-per-trade', type=float, default=None)
```

and construct it inside `build_run_config` via the existing
`niffler.risk.create_risk_manager`. `create_risk_manager` already rejects a parameter the
chosen manager does not accept (`registry.py`), so `--stop-loss-pct` with a future manager
that ignores it fails loudly — mirror `build_cost_model`'s `_COST_MODEL_FLAGS` rejection
and use `None` defaults so "not supplied" stays distinguishable from "supplied at the
default".

> **Flag name ≠ constructor kwarg.** `--max-position-size` maps to `FixedRiskManager`'s
> `position_size_pct`, and `create_risk_manager` rejects unknown kwargs by design. The
> builder must translate, exactly as `backtest.py:359` does today. Without that, the
> `[risk]` TOML example below raises `ValueError` on first use.

**No import cycle.** `run_config.py` importing `niffler.risk.contract` is safe: nothing
under `niffler/risk/` imports `niffler.backtesting` (verified — only docstring references),
and `risk/__init__.py:9` states that as a rule. A `TYPE_CHECKING` guard is optional
insurance, since `RunConfig` needs the name only for the annotation.

### Engine: prefer the config's manager

`BacktestEngine` reaches its manager through `strategy.risk_manager`
(`backtest_engine.py:500, 572, 1127`). Give the engine its own, set by `from_config`, and
fall back to `strategy.risk_manager` when unset. Keep the existing Protocol check at
`:1128` — it now guards both paths. The strategy-attached route can stay for a release
with a deprecation warning; nothing needs to break.

### Effort

Roughly half a day, most of it tests. One field, one `add_*_arguments` helper, one branch
in the engine, five one-line CLI edits.

## Three things to decide while doing it

**1. `position_size` becomes a half-dead search dimension.** When a manager is active,
`backtest_engine.py:527` returns `risk_decision.position_size` for **entries**, discarding
the strategy's value. Exits keep it (`:516-520`, deliberately — an exit is a fraction of the
open position, not of portfolio value). So the six values in `PARAMETER_SPEC`
(`simple_ma_strategy.py:23`, and the same in the RSI and breakout specs) still differentiate
runs, but only through exit sizing — the entry half of the dimension is inert.

That is a much weaker case than "drop it", so **measure before acting**: run one grid with
and one without a manager and compare how much of the spread in results `position_size`
still explains. If it collapses, drop it from the space when a manager is configured and
take the 6× speedup; if exits carry real signal, keep it and document what it now means.
Either way it should not stay ambiguous, because the parameter's meaning silently changes
depending on whether `--risk-manager` was passed.

**2. Record the manager in the results JSON.** `base_optimizer.py:427` writes
`strategy_class` into the metadata. Add the risk configuration, so
`analyze.py --params_file` inherits it. Without this you optimize with a manager and
validate without one — the exact inconsistency the task exists to fix.
`niffler/utils/provenance.py` is the natural home for the serialisation.

**3. Kelly stays out of the registry.** `RISK_MANAGER_CLASSES` correctly contains only
`fixed`, and `KellyRiskManager` still raises `NotImplementedError`. Leave it out until
implemented — `--risk-manager kelly` should keep failing at argument parsing.

---

# Task 1 — Configuration files

## What master already solved

A lot. `scripts/common.py` is exactly the right structure and already carries:

- `load_ohlcv_csv` — one loader, used by all 7 CLIs. The three divergent `load_data`
  implementations are gone.
- `add_cost_model_arguments` / `build_cost_model`
- `add_engine_arguments` / `build_run_config` / `report_run_config`

So **shared flag *definitions* are solved.** What is not solved is persisting the *values*.

## What is still missing

There is no configuration file of any kind. `grep -rn "tomllib|yaml|configparser|--config"`
over `niffler/` and `scripts/` hits exactly one match: `utils/provenance.py:52`, reading
`pyproject.toml` for the version string. Every run still types every value.

122 `add_argument` calls across 8 files:

| Script | Flags |
|---|---|
| `analyze.py` | 22 |
| `backtest.py` | 22 |
| `screen.py` | 22 |
| `optimize.py` | 19 |
| `compare.py` | 14 |
| `common.py` (shared) | 11 |
| `download_data.py` | 8 |
| `preprocessor.py` | 4 |

`--data`, `--capital`, `--commission`, `--cost-model`, `--half-spread-bps`, `--benchmark`,
`--periods-per-year` get retyped on every backtest, every optimization, every analysis —
and a cost-model flag mistyped in one of the three silently invalidates the comparison
between them. That is the risk `common.py` was written to close, and a config file is what
actually closes it.

### Residual naming drift — `optimize.py` is the odd one out

| Concept | `analyze` / `backtest` / `screen` | `optimize` |
|---|---|---|
| Capital | `--capital` | `--initial-capital` |
| Parallel jobs | `--n_jobs` | `--jobs` |
| Random seed | `--random_seed` (analyze) / `--seed` (screen) | `--seed` |

`dest='initial_capital'` is already unified (`common.py` documents this deliberately), so
this is spelling only — but config keys are dests, so it wants tidying in the same pass.
`analyze.py` also carries both `--verbose` and `--log-level`, and `screen.py` defaults to
`WARNING` where every other script defaults to `INFO`.

## Recommendation: layered TOML, stdlib only

**`tomllib`.** Python ≥3.13 is already required, `provenance.py` already imports it, it
supports comments (JSON does not) and matches `pyproject.toml`. YAML would add PyYAML for
no gain.

**Precedence (lowest → highest):**

```
argparse defaults  <  niffler.toml  <  [profile.<name>]  <  CLI flags
```

CLI always wins, so every existing command line keeps working byte-for-byte.

**Location: `scripts/common.py`.** It is already the shared-CLI module and every script
imports it. Do not add a second config module — `niffler/config/` exists but is the logging
package, and this is CLI concern, not library concern.

### Mechanism: two-pass argparse

```python
def apply_config(parser, argv=None, section=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', default='niffler.toml')
    pre.add_argument('--profile')
    known, _ = pre.parse_known_args(argv)
    parser.set_defaults(**load_toml(known.config, section, known.profile))
    return parser
```

Called after the `add_*_arguments` helpers and before `parse_args()`. `--help` then prints
the *effective* defaults, which is a genuine win over merging after the parse.

> **Gotcha:** `required=True` still errors even when the file supplies the value. Drop
> `required` from `--data` / `--strategy` / `--analysis` and validate after parsing with
> `parser.error(...)`.

> **Second gotcha:** `build_cost_model` distinguishes "flag not supplied" (`None`) from
> "supplied" to reject flags the chosen model ignores. `set_defaults` makes a config-file
> value indistinguishable from a CLI one — intended, but it means a `[costs]` section with
> `impact_coefficient` plus `--cost-model fixed` on the CLI will now raise. That is the
> correct behaviour; just make the error message mention the config file.

### File layout

```toml
[common]
data       = "data/BTCUSDT_binance_1d_20200101_20251231.csv"
symbol     = "BTCUSDT"
capital    = 10000.0
commission = 0.001
clean      = true
log_level  = "INFO"
n_jobs     = 8

[costs]
cost_model      = "volume"
half_spread_bps = 1.0

[engine]
benchmark                   = "buy_and_hold"
periods_per_year            = 365
min_trades_for_significance = 30

[risk]                       # Task 2's spec — one section, five scripts
risk_manager       = "fixed"
max_position_size  = 0.1
stop_loss_pct      = 0.05
max_positions      = 5

[backtest]
strategy     = "simple_ma"
exporters    = ["console", "csv"]

[optimize]
method  = "grid"
sort_by = "sharpe_ratio"

[analyze]
test_window = 6
simulations = 1000

[profile.quick]              # niffler.toml + --profile quick
simulations = 100
n_jobs      = 2
```

Two things fall out for free:

- **`[risk]` is Task 2's configuration.** One section, and it reaches backtest, optimize,
  analyze, compare and screen at once. This is why the tasks belong together.
- **`[costs]` + `[engine]` in one file is the real prize.** It makes "optimised and
  validated under the same market assumption" the default rather than something you have
  to retype correctly three times.

### Also worth doing in the same pass

- **`[optimize.parameter_space.<strategy>]`** overriding `PARAMETER_SPEC`. This closes the
  standing todo *"PARAMETER_SPACES to be included as user input"*. The spec moved from a
  module constant to a class attribute (`simple_ma_strategy.py:20`) and is reachable via
  `get_parameter_spec`, but it is still code — `optimize.py` has no flag to override it.
- Add the missing `--config`-aware spelling aliases (`--initial-capital` ↔ `--capital`,
  `--jobs` ↔ `--n_jobs`) as extra option strings on the *same* argument so no existing
  command line or test breaks.

### Effort

~1 day: `apply_config` + `load_toml` in `common.py` (~120 lines), one call per CLI, a
`niffler.toml.example`, and tests for precedence and profile overlay. No behavioural change
to any existing invocation.

---

# Suggested order

1. **`RunConfig.risk_manager` + `add_risk_manager_arguments`** — finishes the seam PR #13
   built. Standalone value, ~half a day.
2. **Drop `position_size` from the search space when a manager is active** — immediate 6×
   grid-search speedup.
3. **Risk config into the results JSON** (`provenance.py`) so `--params_file` inherits it.
4. **`apply_config` / TOML loader** in `common.py`.
5. **`[optimize.parameter_space.*]`** override.
6. **Alias pass** for the `optimize.py` spelling drift.

Step 1 is the highest value per hour in the list and does not depend on any of the others.

---

# Before anything else

Your checkout is 46 commits behind on an already-merged branch. Rebase or switch to master
first — otherwise this work will be written against `exporters` and collide with
`common.py`, `run_config.py`, the strategy/risk registries and the `Portfolio` refactor,
all of which already exist upstream.

---

# Noted, not proposed

- `analyze.py` declares both `--verbose` and `--log-level`; `screen.py` defaults to
  `WARNING` where the others use `INFO`.
- `optimize.py` and `analyze.py` have no exporters, so optimization and validation runs
  never reach Elasticsearch/Grafana — only `backtest.py` does.
