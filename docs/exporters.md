# Exporters

The export system writes backtest results to one or more destinations. Every backtest is
assigned a UUID, and every exporter stamps that id onto everything it writes, so results
correlate across formats.

## Quick Start

```bash
# Console only (default)
python scripts/backtest.py --data data/BTCUSDT_binance_1d.csv --strategy simple_ma

# CSV files into results/
python scripts/backtest.py --data data/BTCUSDT_binance_1d.csv --strategy simple_ma \
  --exporters csv --csv-output-dir results/

# Console + CSV + Elasticsearch
python scripts/backtest.py --data data/BTCUSDT_binance_1d.csv --strategy simple_ma \
  --exporters console,csv,elasticsearch --csv-output-dir results/
```

`--exporters` takes a comma-separated list. Available names: `console`, `csv`,
`elasticsearch`.

## Failure Reporting and Exit Codes

Export failures used to be invisible: exporters logged "skipping export" and returned
normally, the manager recorded them as successes, and the script exited 0 having written
nothing. That is fixed, and the contract is now explicit.

`ExporterManager.export_backtest_result(...)` returns an **`ExportSummary`**:

```python
from niffler.exporters import ExporterManager, ExportSummary

summary = manager.export_backtest_result(result)

summary.successes    # List[str]             - exporter class names that succeeded
summary.failures     # List[Tuple[str, str]] - (exporter class name, error message)
summary.backtest_id  # str                   - the id used for this run
summary.ok           # bool                  - True when failures is empty
```

- It returns an `ExportSummary`, **not** a bare backtest id string. Consumers must use
  `summary.backtest_id`.
- Exporters raise `ExportError` (from `niffler.exporters`) on a precondition failure — an
  invalid result, or an unreachable Elasticsearch cluster (the message carries the URL).
- Exporters whose **constructor** is rejected (for example an invalid
  `ELASTICSEARCH_SCHEME`) are recorded too, on `ExporterManager.creation_failures`, and are
  seeded into `ExportSummary.failures`. They are no longer silently dropped.
- Per-exporter isolation is retained: one exporter failing does not prevent the others from
  running.

`scripts/backtest.py` prints a per-exporter report and **exits 1** if any exporter failed or
could not be created:

```
Export report:
  OK     CSVExporter
  FAILED ElasticsearchExporter: Cannot connect to Elasticsearch at http://127.0.0.1:9
```

## Run Provenance

Every export carries a `provenance` block on its metadata, answering "what produced this
number?" — git SHA, branch, **dirty flag**, a SHA-256 of the input data file, and the
Python and library versions. Without it an Elasticsearch index full of Sharpe ratios is a
graveyard of numbers nobody can reproduce.

The record is produced by `collect_provenance()` in **`niffler/utils/provenance.py`**
(standard library only, so it respects the same layering rule as `json_utils.py`). Its full
shape is documented in the [README](../README.md#run-provenance).

**It is collected once per run, at the CLI**, and passed down:

```python
from niffler.utils.provenance import collect_provenance

provenance = collect_provenance(args.data)          # hashes the CSV once
exporter_manager.export_backtest_result(..., provenance=provenance)
```

Collecting it inside each exporter instead would re-hash the input file once per
destination and shell out to `git` once per exported result. `BaseExporter.create_metadata`
and `ExporterManager._create_metadata` take an optional `provenance` argument and add the
key **only when a record is supplied**, so an opted-out caller gets the previous metadata
shape unchanged.

How it degrades:

| Situation | Result |
|-----------|--------|
| Not a git repository, or `git` not installed | `code` fields `null`, including `dirty` |
| `git` call hangs | Bounded by a 5 s timeout, then `null` |
| Data file missing or unreadable | `sha256`/`size_bytes`/`modified_utc` `null`, `path` kept |
| Package not installed | That package's version is `null` |

Provenance collection **never raises and never blocks a run** — it is metadata about a
backtest, not part of one. Note that `dirty: null` means "not determined"; it is
deliberately not `false`, which would assert a cleanliness nobody checked.

## Console Exporter

Human-readable summary printed to stdout: strategy and symbol, date range, initial and
final capital, total return, max drawdown, Sharpe ratio, win rate and trade count, plus the
backtest id.

Provenance is condensed into a single line under the backtest id:

```
Backtest ID: 76d666a5-e4be-4d85-a2d7-7cb2f105fdf5
Provenance: code ff71eba19999 (feat/provenance, DIRTY) | data a9e9a1efe089
```

`DIRTY` means the working tree had uncommitted changes, so the recorded SHA alone will not
reproduce the result. `dirty-unknown` means the question could not be answered at all.

## CSV Exporter

Writes four files per backtest into `--csv-output-dir`:

| File | Contents |
|------|----------|
| `<base>_portfolio.csv` | Portfolio value time series |
| `<base>_trades.csv` | One row per executed trade |
| `<base>_metadata.json` | Strategy parameters, metrics and run metadata |
| `<base>_provenance.json` | Run provenance plus the `backtest_id` |

### Filename sanitisation

The base name is `{symbol}_{strategy}_{start}_{end}_{short_backtest_id}`, and each
user-derived component is slugified by `sanitize_path_component()` before it reaches the
filesystem:

- path separators, the Windows-illegal characters `<>:"|?*` and control characters → `_`
- whitespace runs collapsed, leading/trailing dots and spaces stripped
- Windows reserved device names (`CON`, `PRN`, `AUX`, `NUL`, `COM1`…) and empty results →
  the caller's fallback (`unknown_symbol` / `unknown_strategy` for filenames; the function's
  own default is `unknown`)
- truncated to 64 characters

So `BTC/USDT` + `Simple MA Crossover` now yields
`BTC_USDT_Simple_MA_Crossover_20240101_20240320_3e2a1c2f`. **Previously such an export
wrote nothing at all** — the slash was interpreted as a directory that did not exist.
Already-safe names such as `BTC-USD` and `Simple_MA_Strategy` are unchanged.

### Trade columns

The trades CSV carries a **`commission`** column (between `value` and `backtest_id`),
populated from the `Trade.commission` field the engine now fills in on both buys and sells.

### JSON metadata

`<base>_metadata.json` is written with `safe_json_dump`, so it is valid RFC 8259:

- `inf`, `-inf` and `NaN` are serialised as **`null`**, not the non-standard `Infinity` /
  `NaN` literals that `json.dump` emits by default and that most parsers reject
- numpy scalars are serialised as plain numbers rather than stringified
- `allow_nan=False` is forced, so a non-finite value that slipped past sanitisation raises
  instead of producing an unparseable file

The same helper is used by `niffler/optimization/base_optimizer.py` when saving optimization
results, because degenerate parameter combinations legitimately produce an infinite profit
factor or a NaN Sharpe ratio.

The provenance sidecar is written with the same helper and the same sanitised
`<base>` name, so `BTC/USDT` cannot escape into a directory separator. It duplicates the
`provenance` key already inside `<base>_metadata.json` on purpose: a directory of CSVs is
read by scripts that only want to know which code and which data produced the rows, and a
few hundred duplicated bytes is cheaper than making each of them parse the full metadata
document. No file is written when a run carries no provenance record.

`sanitize_numeric_values`, `safe_json_dumps` and `safe_json_dump` live in
**`niffler/utils/json_utils.py`** — deliberately outside the exporters package, so that
importing a generic JSON helper does not drag in the optional Elasticsearch client.
`niffler/exporters/json_utils.py` re-exports them for backwards compatibility.

## Elasticsearch Exporter

Bulk-indexes results into four indices (prefix configurable, default `niffler`):

| Index | One document per |
|-------|------------------|
| `niffler-backtests` | Backtest, with metadata and metrics |
| `niffler-portfolio-values` | Portfolio value observation |
| `niffler-trades` | Executed trade (includes `commission`) |
| `niffler-positions` | Completed **round trip** |

Mappings live in `config/elasticsearch/mappings/`.

### The provenance mapping

`niffler-backtests` maps the `provenance` object explicitly, because a mapping is the one
thing that cannot be fixed after the fact — once documents are indexed, correcting a field
type requires a reindex:

| Field | Type | Why |
|-------|------|-----|
| `provenance.run_timestamp_utc` | `date` | Range queries, Grafana time filters |
| `provenance.code.git_sha`, `git_sha_short`, `branch`, `niffler_version` | `keyword` | Exact match and terms aggregations; `text` would analyse them into useless tokens |
| `provenance.code.dirty` | `boolean` | Filter out unreproducible runs in one clause |
| `provenance.data.path`, `data.sha256` | `keyword` | "Show me every run over *this exact* file" |
| `provenance.data.size_bytes` | `long` | Sizes exceed `integer` for large datasets |
| `provenance.data.modified_utc` | `date` | |
| `provenance.environment.*` and `packages.*` | `keyword` | Group results by pandas/numpy version |

A `dynamic_templates` entry maps `provenance.environment.packages.*` to `keyword`, so a
package added to `TRACKED_PACKAGES` later cannot silently land as analysed `text`.

Fields with a `null` value are simply not indexed by Elasticsearch, so a run collected
without git available costs nothing.

### Positions reconcile with the metrics

`niffler-positions` used to be built by a second, hand-rolled pairing loop that matched one
buy to exactly one sell (dropping later sells), computed P&L as a partial exit's notional
against the full entry's, and ignored commission. Its documents contradicted the `win_rate`
and `total_return` reported for the same `backtest_id`.

It now emits one document per `RoundTrip` from the shared `pair_trades()` routine, carrying
`quantity`, `entry_price`, `exit_price`, `pnl`, `is_win`, plus `gross_pnl`,
`entry_commission` and `exit_commission`. Position documents and backtest metrics now agree.

### Configuration

Configured by environment variables, typically via a `.env` file. **An empty string counts
as unset** for all of them.

| Variable | Default | Purpose |
|----------|---------|---------|
| `ELASTICSEARCH_HOST` | `localhost` | Server hostname |
| `ELASTICSEARCH_PORT` | `9200` | Server port |
| `ELASTICSEARCH_INDEX_PREFIX` | `niffler` | Index name prefix |
| `ELASTICSEARCH_SCHEME` | `http` | `http` or `https`; anything else raises `ValueError` |
| `ELASTICSEARCH_API_KEY` | — | API key auth; **takes precedence over basic auth** |
| `ELASTICSEARCH_USERNAME` | — | Basic auth user (both user and password required) |
| `ELASTICSEARCH_PASSWORD` | — | Basic auth password |
| `ELASTICSEARCH_TIMEOUT` | `30` | Request timeout in seconds |
| `ELASTICSEARCH_VERIFY_CERTS` | `true` | TLS certificate verification (https only) |

Notes:

- A half-configured basic auth (user without password, or vice versa) logs a warning and
  sends **no** credentials rather than failing obscurely.
- Credentials are held in private attributes and are **never logged**. Log lines show only
  the URL and an auth mode of `api_key`, `basic` or `none`.
- The client is always constructed with an explicit `request_timeout` instead of the
  library default, and with `verify_certs` only when the scheme is `https`.
- `--es-host`, `--es-port` and `--es-index-prefix` on `backtest.py` override the
  corresponding variables. The auth and TLS options are environment-only; there is no CLI
  flag for them, which also keeps secrets out of shell history.
- `list_indices()` raises `ExportError` when the cluster is unreachable. An empty list now
  means "no indices", not "could not connect".

The `elasticsearch` Python package is optional. Both `Elasticsearch` and
`elasticsearch.helpers.bulk` are imported at module level behind their own
`try`/`except ImportError`, so the module always imports; attempting an export without the
package installed raises a clear `RuntimeError`.

## Adding an Exporter

1. Subclass `BaseExporter` and implement `export_backtest_result(result, backtest_id, ...)`.
2. **Raise on failure.** Do not log and return — the manager cannot distinguish that from
   success, which is exactly the bug this contract exists to prevent. Call the inherited
   `self.require_valid_result(result, destination)` for the standard precondition check;
   it raises `ExportError`.
3. Register it in `ExporterManager.EXPORTER_TYPES`, keyed by the name users pass to
   `--exporters`.

## See Also

- [Visualization setup](../visualization/README.md) — Grafana, Kibana and the Docker stack
- [Backtesting](backtesting.md) — what produces the results being exported
