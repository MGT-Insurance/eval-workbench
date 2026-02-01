# Online Monitoring

Online monitoring module with pluggable data sources and deduplication. Supports
multiple data sources (Langfuse, Slack, etc.) and tracks scored items to avoid
re-processing. Can be scheduled via APScheduler (local/dev) or GitHub Actions
cron (production).

## Example - Simple monitoring

```
from shared.monitoring import OnlineMonitor
from shared.monitoring.scored_items import ScoredItemsStore

store = ScoredItemsStore("data/scored_items.csv")
monitor = OnlineMonitor.from_yaml("config/monitoring.yaml", scored_store=store)
results = monitor.run()
```

## Example - Programmatic setup

```
from shared.monitoring import OnlineMonitor, LangfuseDataSource

source = LangfuseDataSource(
    name="athena",
    extractor=extract_fn,
    limit=100,
    hours_back=2,
)
monitor = OnlineMonitor(
    name="athena_monitor",
    source=source,
    metrics_config={"Metric": {"class": "metric_class"}},
)
results = monitor.run()
```

## Example - Scheduled monitoring

```
from shared.monitoring import MonitoringScheduler, ScoredItemsStore

store = ScoredItemsStore("data/scored_items.csv")
scheduler = MonitoringScheduler(scored_store=store)

scheduler.add_monitor("config/monitoring.yaml", interval_minutes=60)
scheduler.start()
```

## Example - Config access

```
from shared import config

config.load("config/monitoring.yaml")
limit = config.get("trace_loader.limit")

# Temporary config overrides
with config.set({"trace_loader.limit": 10}):
    pass
```
