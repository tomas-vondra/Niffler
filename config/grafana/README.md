# Grafana Dashboards for Niffler

Professional trading analytics dashboards powered by Grafana and Elasticsearch.

## Quick Start

### 1. Start the Stack

```bash
docker-compose down  # Stop old containers if running
docker-compose up -d
```

Wait 30-60 seconds for services to start.

### 2. Run a Backtest

```bash
python scripts/backtest.py \
  --data data/BTCUSD_yahoo_1d_20240101_20241231_cleaned.csv \
  --strategy simple_ma \
  --capital 10000 \
  --commission 0.001 \
  --exporters elasticsearch
```

### 3. Access Grafana

Open your browser: **http://localhost:3000**

**Login Credentials:**
- Username: `admin`
- Password: `admin`

### 4. View Dashboards

Click **Dashboards** (four squares icon) in the left sidebar.

You'll see 3 pre-configured dashboards:
- **Niffler Trading Analytics - Overview** ⭐ (Start here!)
- **Niffler - Backtest Performance**
- **Niffler - Trade Analysis**

---

## Dashboard Overview

### 1. Niffler Trading Analytics - Overview
**Best for:** Quick comprehensive view

**Panels:**
- **Gauges**: Avg Return %, Sharpe Ratio, Win Rate
- **Line Chart**: Backtests over time
- **Time Series**: Portfolio value evolution (multi-backtest comparison)
- **Donut Chart**: Buy vs Sell distribution
- **Bar Chart**: Trade activity timeline

### 2. Niffler - Backtest Performance
**Best for:** Strategy comparison and performance analysis

**Panels:**
- **Stats**: Return, Sharpe, Win Rate, Drawdown, Trades, Total Backtests
- **Bar Charts**: Strategy performance and Sharpe ratio comparison
- **Heatmap**: Backtest activity (Strategy × Time)

### 3. Niffler - Trade Analysis
**Best for:** Individual trade patterns

**Panels:**
- **Stats**: Total trades, Avg price, Avg quantity, Total value
- **Time Series**: Trade activity and price over time
- **Donut Chart**: Buy vs Sell distribution
- **Bar Chart**: Trade count by symbol

---

## Architecture

### Data Flow

```
Backtest → Elasticsearch → Grafana Dashboards
```

**Elasticsearch Indices:**
- `niffler-backtests` - Backtest metadata and metrics
- `niffler-portfolio-values` - Time-series portfolio data
- `niffler-trades` - Individual trade records

**Grafana Setup:**
- **Datasource**: Auto-provisioned Elasticsearch connection
- **Dashboards**: Auto-loaded from `config/grafana/dashboards/`
- **Port**: 3000 (http://localhost:3000)

---

## Directory Structure

```
config/grafana/
├── provisioning/
│   ├── datasources/
│   │   └── elasticsearch.yml        # Auto-configured datasource
│   └── dashboards/
│       └── dashboards.yml           # Dashboard provider config
├── dashboards/
│   ├── niffler-overview.json        # Main overview dashboard
│   ├── backtest-performance.json   # Performance metrics
│   └── trade-analysis.json          # Trade patterns
└── README.md                        # This file
```

---

## Configuration

### Docker Compose

Grafana is configured in `docker-compose.yml`:

```yaml
grafana:
  image: grafana/grafana:11.4.0
  container_name: niffler-grafana
  environment:
    - GF_SECURITY_ADMIN_PASSWORD=admin
    - GF_SECURITY_ADMIN_USER=admin
    - GF_INSTALL_PLUGINS=grafana-elasticsearch-datasource
  ports:
    - "3000:3000"
  volumes:
    - grafana_data:/var/lib/grafana
    - ./config/grafana/provisioning:/etc/grafana/provisioning
    - ./config/grafana/dashboards:/var/lib/grafana/dashboards
```

### Datasource Configuration

File: `config/grafana/provisioning/datasources/elasticsearch.yml`

- **Name**: Niffler Elasticsearch
- **Type**: elasticsearch
- **URL**: http://elasticsearch:9200
- **Version**: 8.0.0
- **Time Field**: created_at

### Dashboard Provisioning

File: `config/grafana/provisioning/dashboards/dashboards.yml`

Dashboards are automatically loaded from `/var/lib/grafana/dashboards` on startup.

---

## Customization

### Editing Dashboards

1. Open dashboard in Grafana
2. Click **Dashboard settings** (gear icon, top-right)
3. Make changes
4. Click **Save**
5. Export JSON (Settings → JSON Model → Copy to clipboard)
6. Save to `config/grafana/dashboards/`

### Creating New Panels

1. Open dashboard
2. Click **Add** → **Visualization**
3. Select **Niffler Elasticsearch** datasource
4. Configure query:
   - **Index**: niffler-backtests, niffler-portfolio-values, or niffler-trades
   - **Metrics**: avg, sum, count, min, max
   - **Group by**: Date histogram, Terms
5. Choose visualization type (Time series, Bar chart, Stat, etc.)
6. Save panel

### Common Queries

**Average Return by Strategy:**
```
Metric: Avg(total_return_pct)
Group by: Terms(strategy_name.keyword)
```

**Portfolio Evolution:**
```
Metric: Avg(portfolio_value)
Group by: Terms(backtest_id.keyword), Date histogram(timestamp)
```

**Trade Count Over Time:**
```
Metric: Count()
Group by: Date histogram(timestamp)
Index: niffler-trades
```

---

## Troubleshooting

### Dashboards Not Showing Data

**1. Check Elasticsearch has data:**
```bash
curl http://localhost:9200/niffler-*/_count
```

Should show documents > 0.

**2. Adjust time range:**
- Click time picker (top-right)
- Set to "Last 90 days" or "Last year"

**3. Verify datasource:**
- Go to **Configuration** → **Data sources**
- Click **Niffler Elasticsearch**
- Click **Save & test**
- Should show "Data source is working"

### Connection Errors

**Error: "Cannot connect to Elasticsearch"**

Check containers are running:
```bash
docker ps
```

Should see:
- niffler-elasticsearch
- niffler-grafana
- niffler-app

Restart if needed:
```bash
docker-compose restart grafana
```

### Panels Showing "No Data"

**Possible causes:**
1. No backtests have been run
2. Time range doesn't match your data
3. Index pattern mismatch

**Solutions:**
1. Run a backtest with `--exporters elasticsearch`
2. Expand time range to cover your backtest dates
3. Check index in panel query matches your indices

### Dashboard Not Loading

Clear Grafana cache:
```bash
docker-compose restart grafana
```

Or rebuild:
```bash
docker-compose down
docker volume rm niffler_grafana_data
docker-compose up -d
```

---

## Advanced Features

### Variables

Add dashboard variables for dynamic filtering:

1. Dashboard settings → Variables → Add variable
2. **Name**: `strategy`
3. **Type**: Query
4. **Data source**: Niffler Elasticsearch
5. **Query**: `{"find":"terms", "field":"strategy_name.keyword"}`
6. Use in panels: `strategy_name.keyword:$strategy`

### Alerts

Set up alerts on performance metrics:

1. Edit panel
2. Click **Alert** tab
3. Create alert rule (e.g., "Return drops below 0%")
4. Configure notification channel (Email, Slack, etc.)

### Annotations

Mark important events:

1. Dashboard settings → Annotations → Add annotation query
2. **Data source**: Niffler Elasticsearch
3. **Query**: Filter for specific events
4. Events appear as vertical lines on charts

---

## Tips & Best Practices

1. **Use the Overview dashboard first** - Best starting point
2. **Adjust time range** - Match your backtest dates
3. **Compare strategies** - Use performance dashboard to compare
4. **Monitor portfolio evolution** - Track capital growth over time
5. **Analyze trade patterns** - Use trade analysis for insights
6. **Export dashboards** - Settings → JSON Model → Save for backup
7. **Create copies** - Duplicate dashboards for experiments
8. **Use variables** - Filter by strategy, symbol, date range
9. **Set up alerts** - Get notified of performance changes
10. **Regular data cleanup** - Delete old test data from Elasticsearch

---

## Performance Optimization

### Large Datasets

For many backtests (100s+):

1. **Index lifecycle management** - Delete old data
2. **Time range limits** - Use shorter ranges
3. **Aggregation limits** - Reduce bucket sizes
4. **Dashboard refresh** - Disable auto-refresh

### Elasticsearch Queries

Optimize queries:
- Use `terms` aggregation for grouping
- Limit bucket sizes (default 10-20)
- Use date histograms with appropriate intervals
- Add query filters to reduce data

---

## Docker Services

### Elasticsearch
- **Port**: 9200
- **Container**: niffler-elasticsearch
- **Health**: `curl http://localhost:9200/_cluster/health`

### Grafana
- **Port**: 3000
- **Container**: niffler-grafana
- **Health**: `curl http://localhost:3000/api/health`

### Management Commands

```bash
# View logs
docker logs niffler-grafana
docker logs niffler-elasticsearch

# Restart services
docker-compose restart grafana
docker-compose restart elasticsearch

# Stop all
docker-compose down

# Clean restart
docker-compose down
docker volume rm niffler_grafana_data niffler_elasticsearch_data
docker-compose up -d
```

---

## Next Steps

1. ✅ Run multiple backtests with different parameters
2. ✅ Use Performance dashboard to compare strategies
3. ✅ Analyze portfolio evolution patterns
4. ✅ Study trade patterns in Trade Analysis
5. ✅ Create custom dashboards for specific needs
6. ✅ Set up alerts for key metrics
7. ✅ Export dashboards as backup
8. ✅ Share dashboards with team

---

## Support

**For issues:**
1. Check logs: `docker logs niffler-grafana`
2. Verify Elasticsearch: `curl localhost:9200/_cat/indices?v`
3. Test datasource in Grafana: Configuration → Data sources → Test
4. Check this README for troubleshooting

**Grafana Documentation:**
- Official docs: https://grafana.com/docs/
- Elasticsearch datasource: https://grafana.com/docs/grafana/latest/datasources/elasticsearch/

---

Happy Trading! 📊📈
