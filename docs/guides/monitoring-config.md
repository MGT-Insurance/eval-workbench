# Monitoring Configuration (YAML)

This guide explains the `OnlineMonitor` YAML configuration format. Start from
`src/shared/monitoring/monitoring.yaml.example` and customize it for your
implementation.

---

## 1) Top-Level Fields

```yaml
version: "1.0"
name: "my_monitor"
```

- `version`: config schema version
- `name`: monitor name (used for dedup + experiment naming)

### Optional scored items store

```yaml
scored_store:
  # type: none | csv | db
  type: csv
  # file_path: "data/scored_items.csv"
  # connection_string: "postgresql://user:pass@host:5432/dbname"
```

- `type`: `csv` uses file-based dedup, `db` uses `evaluation_dataset`, `none` disables
- `file_path`: only for `csv`
- `connection_string`: only for `db`

### Optional schedule (for MonitoringScheduler)

```yaml
schedule:
  # Use ONE of these
  interval_minutes: 10
  # cron: "*/10 * * * *"
```

- `interval_minutes`: run every N minutes
- `cron`: cron expression for aligned schedules

---

## 2) Source Configuration

### Langfuse source

```yaml
source:
  type: langfuse
  name: "my_agent"
  component: "my_component"
  extractor: "implementations.my_agent.extractors.extract_trace"
  # prompt_patterns: "implementations.my_agent.langfuse.prompt_patterns.MyPromptPatterns"
  limit: 100
  days_back: 7
  tags: ["production"]
  timeout: 60
  fetch_full_traces: true
  show_progress: true
```

Key fields:

- `extractor`: python path to a `(Trace) -> DatasetItem` function
- `limit`: max traces to fetch
- `days_back` / `hours_back` / `minutes_back`: time window (days_back wins if both set; hours_back wins over minutes_back)
- `tags`: filter by Langfuse tags
- `fetch_full_traces`: include observations/scores (slower but richer)

### Slack source (alternative)

```yaml
source:
  type: slack
  name: "my_slack_channels"
  channel_ids:
    - "C0XXXXXXXXX"
  limit: 50
  scrape_threads: true
  bot_name: "MyBot"
  workspace_domain: "myworkspace"
  drop_if_first_is_user: true
  drop_if_all_ai: true
  max_concurrent: 2
  # filter_sender: "U1234567890"
```

---

## 3) Sampling Configuration

```yaml
sampling:
  strategy: random
  n: 10
  # seed: 42
```

Strategies:

- `all` (default)
- `random`
- `most_recent`
- `oldest`

Best practice: set `source.limit` much larger than `sampling.n`.

---

## 4) Metrics Configuration

```yaml
metrics_config:
  MyMetric:
    class: "my_metric_class"
    llm_provider: "openai"
    model_name: "gpt-4o"
```

Each metric key is a label in results; `class` must match a registered metric.

---

## 5) Publishing Configuration (Optional)

```yaml
publishing:
  push_to_db: false
  push_to_langfuse: false
  trace_experiment: false

  database:
    on_conflict: do_nothing
    chunk_size: 1000

  experiment:
    enabled: true
    dataset_name: "development"
    run_name: "online-${ENVIRONMENT:-preview}"
    link_to_traces: true
```

---

## Usage Examples

```python
from shared.monitoring import OnlineMonitor, ScoredItemsStore

# Basic usage
monitor = OnlineMonitor.from_yaml("config/monitoring.yaml")
results = monitor.run()

# With dedup store
store = ScoredItemsStore("data/scored_items.csv")
monitor = OnlineMonitor.from_yaml("config/monitoring.yaml", scored_store=store)
results = monitor.run(deduplicate=True)

# With overrides
monitor = OnlineMonitor.from_yaml(
    "config/monitoring.yaml",
    overrides={
        "source.limit": 50,
        "sampling.strategy": "most_recent",
        "sampling.n": 5,
    },
)
```
