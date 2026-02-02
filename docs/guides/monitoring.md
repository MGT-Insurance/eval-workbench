# Online Monitoring

Online monitoring module with pluggable data sources and deduplication. Supports
multiple data sources (Langfuse, Slack, etc.) and tracks scored items to avoid
re-processing. Can be scheduled via APScheduler (local/dev) or GitHub Actions
cron (production).

## Scored Items Store

The `ScoredItemsStore` abstract base class provides deduplication to avoid re-processing
items that have already been evaluated. Two implementations are available:

- **CSVScoredItemsStore**: File-based storage using CSV (pandas for reads, raw CSV for appends)
- **DBScoredItemsStore**: Database-backed storage using the `evaluation_dataset` table

### CSV-backed deduplication

```python
from shared.monitoring import OnlineMonitor, CSVScoredItemsStore

store = CSVScoredItemsStore("data/scored_items.csv")
monitor = OnlineMonitor.from_yaml("config/monitoring.yaml", scored_store=store)
results = monitor.run(publish=True)
```

You can also configure a scored store directly in YAML:

```yaml
scored_store:
  type: csv
  file_path: "data/scored_items.csv"
```

### Database-backed deduplication

```python
from shared.monitoring import OnlineMonitor, DBScoredItemsStore

# Uses DATABASE_URL environment variable
store = DBScoredItemsStore()

# Or with explicit connection string
store = DBScoredItemsStore(connection_string="postgresql://...")

monitor = OnlineMonitor.from_yaml("config/monitoring.yaml", scored_store=store)
results = monitor.run(publish=True)
```

YAML version:

```yaml
scored_store:
  type: db
  connection_string: "postgresql://..."
```

### Scheduling via YAML

If you use `MonitoringScheduler`, you can define the interval or cron in YAML:

```yaml
schedule:
  interval_minutes: 10
  # cron: "*/10 * * * *"
```



### No deduplication

```python
from shared.monitoring import OnlineMonitor

monitor = OnlineMonitor.from_yaml("config/monitoring.yaml")  # scored_store=None
results = monitor.run(deduplicate=False, publish=True)
```

## Example - Programmatic setup

```python
from shared.monitoring import OnlineMonitor, LangfuseDataSource, DBScoredItemsStore

source = LangfuseDataSource(
    name="athena",
    extractor=extract_fn,
    limit=100,
    hours_back=2,
)
store = DBScoredItemsStore()
monitor = OnlineMonitor(
    name="athena_monitor",
    source=source,
    metrics_config={"Metric": {"class": "metric_class"}},
    scored_store=store,
)
results = monitor.run()
```

## Example - Scheduled monitoring

```python
from shared.monitoring import MonitoringScheduler, DBScoredItemsStore

store = DBScoredItemsStore()
scheduler = MonitoringScheduler(scored_store=store)

scheduler.add_monitor("config/monitoring.yaml", interval_minutes=60)
scheduler.start()
```

## Example - Config access

```python
from shared import config

config.load("config/monitoring.yaml")
limit = config.get("trace_loader.limit")

# Temporary config overrides
with config.set({"trace_loader.limit": 10}):
    pass
```
