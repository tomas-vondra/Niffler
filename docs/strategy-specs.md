# Strategy specs

Every strategy is documented by a spec: one TOML file in
[`niffler/strategies/specs/`](../niffler/strategies/specs/), named after its registry key.
A class says *how* a strategy computes its signals; the spec says what the strategy is
supposed to be: where the idea came from, its rules in prose, the parameters the source
published, what it needs from the engine, and when it was found and implemented.

This is step 3 of the build order in [roadmap.md](roadmap.md#build-order). The spec is the
input the rest of that plan reads: the scaffold turns it into code, and the rule test and the
algorithm are each written from it alone, by different hands.

## The format

The three shipped specs are the worked examples:
[`simple_ma.toml`](../niffler/strategies/specs/simple_ma.toml),
[`rsi.toml`](../niffler/strategies/specs/rsi.toml) and
[`breakout.toml`](../niffler/strategies/specs/breakout.toml).

```toml
spec_version = 1
key = "breakout"                   # registry key; must match the file name
class_name = "BreakoutStrategy"    # CamelCase, ends in Strategy
name = "Donchian Breakout"         # display name the class passes to BaseStrategy
family = "breakout"                # see below
summary = "One sentence."

[source]
kind = "book"                      # textbook | book | paper | web | original
reference = "Author, title, year - or what the idea is when no one owns it."
# url = "https://..."              # required when kind = "web"
# retrieved = 2026-10-10           # required whenever url is given: a page changes

[dates]
found = 2026-08-07                 # TOML dates, unquoted
implemented = 2026-08-07           # set in the change that registers the class

[rules]
entry = """Prose. When exactly a buy fires, judged at a bar's close."""
exit = """Prose. When exactly a sell fires."""
inputs = ["high", "low", "close"]  # the OHLCV columns the rules read
notes = """Optional: warm-up, edge cases, constraints between parameters."""
deviations = """Optional: where this strategy departs from its source, and why."""

[parameters.entry_window]
type = "int"                       # int | float | choice
default = 20
published = 20                     # optional: the value the source states
description = "What it means."
search = { min = 10, max = 60, step = 5 }   # optional; choice: { choices = [...] }

[requires]
capabilities = ["long_entry", "signal_exit", "fractional_position", "single_instrument_bars"]
unlisted = []                      # needs the capability list does not name
```

`family` is one of `trend_following`, `mean_reversion`, `breakout`, `momentum`,
`volatility`, `seasonal` or `other`.

Loading is strict, and every problem in a file is reported at once:

- **An unknown key is an error.** A misspelt `publshed` would otherwise disappear.
- **`position_size` and `risk_manager` are never declared.** Every strategy takes both;
  `position_size` defaults to 1.0 and is searched over 0.5-1.0 in steps of 0.1, as in every
  shipped strategy.
- **A default lies inside its search range**, `step` is required and positive, and a float
  parameter accepts a TOML integer.
- **Dates are TOML dates**, not strings, and `implemented` is not before `found`. The
  implementation date matters later: bars after it are data the strategy cannot have been
  fitted to (the forward holdout under **Later** in the roadmap).

## What the engine can express

[`niffler/strategies/capabilities.py`](../niffler/strategies/capabilities.py) declares every
capability a spec may name, and whether the engine supports it today. Run
`python scripts/scaffold_strategy.py --list` for every spec's status.

| Supported | Not supported (yet, or ever) |
|-----------|------------------------------|
| `long_entry`, `signal_exit`, `fractional_position`, `scale_in`, `fixed_pct_stop`, `single_instrument_bars` | `strategy_stop`, `take_profit`, `trailing_stop`, `time_exit`, `short_entry`, `limit_entry`, `stop_entry`, `same_bar_close_fill`, `intraday_bars`, `multiple_instruments`, `external_series`, `leverage` |

A spec that names an unsupported capability, or lists anything under `unlisted`, is
**unsupported**: it stays in the directory as a record of the idea and of why it cannot be
built, and the scaffold refuses it. It is never approximated into something the source did
not describe. When a capability lands, its entry flips to supported and every spec waiting on
it becomes buildable. `same_bar_close_fill` will never flip: it is look-ahead.

A spec has one of three statuses, derived rather than written:

| Status | Meaning |
|--------|---------|
| `unsupported` | Needs something the engine cannot do; the reasons are listed |
| `specified` | Buildable, not built: no `dates.implemented` |
| `implemented` | Registered, with `dates.implemented` set |

## The scaffold

```bash
python scripts/scaffold_strategy.py my_idea              # niffler/strategies/specs/my_idea.toml
python scripts/scaffold_strategy.py my_idea --dry-run    # what would be written
python scripts/scaffold_strategy.py --list               # every spec and its status
```

It writes three things and never overwrites a file:

- `niffler/strategies/<key>_strategy.py` - the class: constructor, defaults,
  `PARAMETER_SPEC`, display name, description, and the spec's rules in the docstring.
  `generate_signals` raises `NotImplementedError`.
- `tests/test_strategies/test_<key>_strategy.py` - the rule-test template: a `frame()`
  builder, an `assert_signals_exactly(result, buys, sells)` helper, and failing stubs for the
  entry rule and the exit rule. A spec whose `inputs` include `high` or `low` also gets a
  stub for the intra-bar check, the one leak the shared look-ahead test cannot see.
- one import and one `STRATEGY_CLASSES` line in `niffler/strategies/registry.py`.

A freshly scaffolded strategy makes the suite **fail** on purpose - the stubs fail and the
registry contract cannot generate signals - so it cannot merge as a strategy that silently
never trades. What is left, each from the spec alone:

1. Write the rule tests: a short synthetic series on which each rule must fire on known bars
   and nowhere else.
2. Write `generate_signals`, without reading those tests.
3. Set `dates.implemented` in the spec, in the same change.

## What keeps a spec honest

[`tests/test_strategies/test_specs.py`](../tests/test_strategies/test_specs.py) iterates the
registry and the spec directory, so a new strategy is covered without editing it:

- every registered strategy has a spec, and a spec has `dates.implemented` exactly when its
  key is registered;
- the class name, module, display name, constructor parameters and defaults, and
  `PARAMETER_SPEC` all match the spec;
- a default that differs from the published value is explained in `rules.deviations`.

[`tests/test_strategies/test_scaffold.py`](../tests/test_strategies/test_scaffold.py)
regenerates each shipped strategy from its spec and checks the result has the shipped
class's interface, so the scaffold and the hand-written classes cannot drift apart.

What none of this checks is whether a spec reads its source correctly. A spec that misreads
its page passes every test, and the strategy is then honestly tested under the wrong name.
The roadmap's answer is to keep the three writers apart and to flag results that look too
good, not to pretend a test can read the page.
