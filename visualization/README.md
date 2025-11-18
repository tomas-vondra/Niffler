# Niffler Visualization Setup

Complete guide for setting up and managing Niffler's visualization stack (Grafana, Kibana, Elasticsearch).

## Overview

Niffler provides two visualization tools:
- **Grafana** (Primary) - Beautiful dashboards for trading analytics
- **Kibana** (Debug) - Raw data exploration and debugging

## Quick Start

### 1. Start Services

Start Elasticsearch and Grafana (default):
```bash
docker-compose up -d
```

Start with Kibana for debugging:
```bash
docker-compose --profile debug up -d
```

### 2. Run a Backtest

```bash
python scripts/backtest.py \
  --data data/BTCUSD_yahoo_1d_20240101_20241231_cleaned.csv \
  --strategy simple_ma \
  --capital 10000 \
  --commission 0.001 \
  --exporters elasticsearch
```

### 3. Access Dashboards

**Grafana (Primary):**
- URL: http://localhost:3000
- Login: admin / admin
- Dashboards are auto-provisioned

**Kibana (Optional, for debugging):**
- URL: http://localhost:5601
- Setup required (see below)

---

## Visualization Scripts

All visualization management scripts are in the `visualization/` directory.

### Clean Elasticsearch Indices

Remove all Niffler data for fresh starts.

**Dry run (preview what would be deleted):**
```bash
uv run python visualization/clean_elasticsearch.py --dry-run
```

**Delete with confirmation:**
```bash
uv run python visualization/clean_elasticsearch.py
```

**Force delete (no confirmation):**
```bash
uv run python visualization/clean_elasticsearch.py --force
```

**What gets deleted:**
- `niffler-backtests` - Backtest metadata
- `niffler-portfolio-values` - Portfolio time-series
- `niffler-trades` - Trade records

**Note:** Indices auto-recreate on next backtest run.

---

### Setup Kibana Data Views

Create Kibana data views automatically for data exploration.

**Prerequisites:**
- Kibana running: `docker-compose --profile debug up -d kibana`
- Indices exist (run a backtest first)

**Run setup:**
```bash
uv run python visualization/setup_kibana.py
```

**What gets created:**
- `Niffler Backtests` (niffler-backtests, time: created_at)
- `Niffler Portfolio Values` (niffler-portfolio-values, time: timestamp)
- `Niffler Trades` (niffler-trades, time: timestamp)

---

## Complete Workflow

### Development / Testing Workflow

```bash
# 1. Start services
docker-compose up -d

# 2. Clean old data (optional, for fresh start)
uv run python visualization/clean_elasticsearch.py --force

# 3. Run backtest
python scripts/backtest.py \
  --data data/your_data.csv \
  --strategy simple_ma \
  --exporters elasticsearch

# 4. View in Grafana
# Open http://localhost:3000 (admin/admin)
```

### Debugging Workflow

When you need to see raw data in Kibana:

```bash
# 1. Start services with Kibana
docker-compose --profile debug up -d

# 2. Run backtest (if needed)
python scripts/backtest.py \
  --data data/your_data.csv \
  --strategy simple_ma \
  --exporters elasticsearch

# 3. Setup Kibana data views
uv run python visualization/setup_kibana.py

# 4. Explore data in Kibana
# Open http://localhost:5601
# Go to: Analytics -> Discover
# Select data view, adjust time range (Last 90 days)
```

---

## Service Management

### Docker Compose Commands

```bash
# Start all services (ES + Grafana)
docker-compose up -d

# Start with Kibana (debug mode)
docker-compose --profile debug up -d

# Stop all services
docker-compose down

# View logs
docker logs niffler-elasticsearch
docker logs niffler-grafana
docker logs niffler-kibana

# Restart specific service
docker-compose restart grafana
docker-compose restart elasticsearch

# Stop Kibana only
docker-compose stop kibana
```

### Health Checks

```bash
# Elasticsearch
curl http://localhost:9200/_cluster/health

# Grafana
curl http://localhost:3000/api/health

# Kibana (if running)
curl http://localhost:5601/api/status

# List Niffler indices
curl http://localhost:9200/_cat/indices?v | grep niffler

# Count documents
curl http://localhost:9200/niffler-*/_count
```

---

## Grafana Dashboards

### Automatic Setup

Grafana dashboards and datasources are **automatically provisioned** when you start the container:

- **Datasource**: `Niffler Elasticsearch` (auto-configured, connects to Elasticsearch)
- **Dashboards**: 3 pre-configured dashboards loaded from `config/grafana/dashboards/`
- **No manual setup required**: Everything works on first start

### Available Dashboards

Grafana includes 3 pre-configured dashboards:

1. **Niffler Trading Analytics - Overview**
   - Key metrics: Return %, Sharpe Ratio, Win Rate
   - Portfolio evolution over time
   - Trade distribution (Buy vs Sell)
   - Activity timeline

2. **Niffler - Backtest Performance**
   - Strategy comparison
   - Performance metrics
   - Backtest heatmap

3. **Niffler - Trade Analysis**
   - Trade statistics
   - Price evolution
   - Symbol analysis

### Accessing Dashboards

1. Open http://localhost:3000
2. Login: `admin` / `admin`
3. Click "Dashboards" icon (four squares) in left sidebar
4. Select any Niffler dashboard

**Note:** Dashboards load correctly on first start. No additional configuration needed.

### Customizing Dashboards

Grafana dashboards can be edited directly in the UI. Changes are saved to the Docker volume.

For permanent changes:
1. Edit dashboard in Grafana
2. Export JSON: Settings → JSON Model → Copy
3. Save to `config/grafana/dashboards/`
4. Restart Grafana: `docker-compose restart grafana`

---

## Kibana Data Exploration

### When to Use Kibana

Use Kibana for:
- 🔍 Raw data inspection
- 🐛 Debugging data format issues
- 📊 Ad-hoc queries
- 🧪 Verifying data export

### Using Kibana

1. **Start Kibana:**
   ```bash
   docker-compose --profile debug up -d kibana
   ```

2. **Setup data views:**
   ```bash
   python visualization/setup_kibana.py
   ```

3. **Explore data:**
   - Open http://localhost:5601
   - Go to: Analytics → Discover
   - Select data view
   - Adjust time range (Last 90 days)
   - Use KQL for filtering: `strategy_name:"Simple MA Crossover"`

### Common Queries

**Filter by strategy:**
```
strategy_name:"Simple MA Crossover"
```

**Filter by trade side:**
```
side:"buy"
```

**Filter by date range:**
Use the time picker (top-right) to select date range

---

## Data Flow

```
Backtest Script
     ↓
ElasticsearchExporter
     ↓
Elasticsearch (3 indices)
     ↓
   ┌─┴─┐
   ↓   ↓
Grafana  Kibana
(Always) (Debug)
```

### Elasticsearch Indices

- **niffler-backtests** - One document per backtest with metadata and metrics
- **niffler-portfolio-values** - Time-series data of portfolio value evolution
- **niffler-trades** - Individual trade records with timestamps

---

## Troubleshooting

### Grafana Shows "No Data"

**Check:**
1. Elasticsearch has data: `curl http://localhost:9200/niffler-*/_count`
2. Time range in Grafana (top-right, set to "Last 90 days")
3. Datasource working: Configuration → Data sources → Test

**Solution:**
Run a backtest with `--exporters elasticsearch`

### Kibana Won't Start

**Check logs:**
```bash
docker logs niffler-kibana
```

**Common issues:**
- Elasticsearch not ready → Wait 30 seconds
- Port 5601 already in use → Stop other Kibana instances
- Not started with profile → Use `--profile debug`

**Solution:**
```bash
docker-compose stop kibana
docker-compose --profile debug up -d kibana
```

### Elasticsearch Connection Refused

**Check:**
```bash
docker ps | grep elasticsearch
```

**Solution:**
```bash
docker-compose restart elasticsearch
sleep 30
curl http://localhost:9200/_cluster/health
```

---

## Best Practices

### Development

1. **Start services:** `docker-compose up -d`
2. **Run backtests:** Export to Elasticsearch
3. **View in Grafana:** Primary analysis
4. **Use Kibana:** When debugging issues
5. **Clean data:** For fresh experiments

### Production

1. **Keep Grafana running:** Always on for dashboards
2. **Don't run Kibana:** Only for debugging
3. **Regular cleanup:** Delete old test data
4. **Monitor ES size:** Check disk usage

### Data Management

1. **Test data:** Clean frequently with cleanup script
2. **Important results:** Export to CSV for backup
3. **Indices:** Let them auto-recreate
4. **Mappings:** Keep in Git (config/elasticsearch/mappings/)

---

## Summary

**Daily Use:**
- ✅ Use Grafana (http://localhost:3000)
- ✅ Beautiful dashboards, always running

**Debugging:**
- 🔍 Use Kibana (http://localhost:5601)
- 🔍 Start with `--profile debug`
- 🔍 Raw data exploration

**Cleanup:**
- 🧹 Use `python visualization/clean_elasticsearch.py`
- 🧹 Safe and reversible

**Architecture:**
```
Backtest → Elasticsearch → Grafana (primary) + Kibana (debug)
```

---

## Quick Reference

```bash
# Start everything
docker-compose up -d

# Start with debugging
docker-compose --profile debug up -d

# Clean data
uv run python visualization/clean_elasticsearch.py --force

# Run backtest
python scripts/backtest.py --data <file> --strategy <strategy> --exporters elasticsearch

# Setup Kibana
uv run python visualization/setup_kibana.py

# View dashboards
# Grafana: http://localhost:3000 (admin/admin)
# Kibana:  http://localhost:5601

# Health checks
curl http://localhost:9200/_cluster/health  # Elasticsearch
curl http://localhost:3000/api/health       # Grafana
curl http://localhost:5601/api/status       # Kibana

# Stop everything
docker-compose down
```

---

## See Also

- [Main README](../README.md) - Project overview
- [Grafana Configuration](../config/grafana/README.md) - Grafana details
- [Docker Compose](../docker-compose.yml) - Service definitions
