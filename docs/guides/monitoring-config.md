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

### Slack source (`type: slack`)

```yaml
source:
  type: slack
  name: "my_slack_channels"
  channel_ids:
    - "C0XXXXXXXXX"
  limit: 50
  scrape_threads: true
  bot_name: "MyBot"
  bot_names: ["MyBot", "MyBotV2"]
  workspace_domain: "myworkspace"
  drop_if_first_is_user: true
  drop_if_all_ai: true
  max_concurrent: 2
  # filter_sender: "U1234567890"

  # Time windows (mutually exclusive with oldest_ts/latest_ts)
  # window_days: 7
  # window_hours: 24
  # window_minutes: 60
  # oldest_ts: 1700000000.0
  # latest_ts: 1700100000.0

  # Filtering
  exclude_senders: ["Prometheus"]
  drop_message_regexes:
    - "^Street View"
    - "google\\.com/maps"
  strip_citation_block: true

  # Display name mapping
  member_id_to_display_name:
    U09JKDU4RE2: "Athena assistant"
  human_mention_token: "@human"
```

#### Slack source field reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | Required | Source identifier |
| `channel_ids` | list[str] | Required | Slack channel IDs to scrape |
| `limit` | int | `10` | Max messages per channel |
| `scrape_threads` | bool | `true` | Include thread replies |
| `filter_sender` | str | — | Only include messages from this sender |
| `bot_name` | str | `"Athena"` | Name identifying AI messages |
| `bot_names` | list[str] | — | Multiple names identifying AI messages |
| `workspace_domain` | str | `"mgtinsurance"` | Slack workspace subdomain |
| `drop_if_first_is_user` | bool | `false` | Drop if first message is from user |
| `drop_if_all_ai` | bool | `false` | Drop if all messages are AI |
| `max_concurrent` | int | `2` | Max concurrent channel scrapes |
| `oldest_ts` | float | — | Inclusive lower timestamp bound |
| `latest_ts` | float | — | Inclusive upper timestamp bound |
| `window_days` | float | — | Relative lookback in days |
| `window_hours` | float | — | Relative lookback in hours |
| `window_minutes` | float | — | Relative lookback in minutes |
| `exclude_senders` | list[str] | — | Sender names to omit from conversations |
| `drop_message_regexes` | list[str] | — | Regex patterns — matching messages are dropped |
| `strip_citation_block` | bool | `false` | Remove trailing citation blocks like `[1] ...` |
| `member_id_to_display_name` | dict | — | Slack user ID → display name mapping |
| `human_mention_token` | str | `"@human"` | Token for identifying human mentions |

!!! note "Time window precedence"
    `oldest_ts`/`latest_ts` take priority over relative windows. Among relative windows: `window_days` > `window_hours` > `window_minutes`.

---

### Slack-Neon Join source (`type: slack_neon_join`)

Fetches Slack threads and Neon database rows, then joins them on configurable columns. Useful when conversation context needs to be enriched with database metadata (e.g., case IDs, trace IDs).

```yaml
source:
  type: slack_neon_join
  name: "athena"
  channel_ids:
    - "C09MAP9HR9D"
    - "C09JE5SSP43"
  limit: 100
  scrape_threads: true
  bot_names: ["Athena"]
  workspace_domain: "mgtinsurance"
  drop_if_first_is_user: true
  drop_if_all_ai: true
  max_concurrent: 2
  exclude_senders: ["Prometheus"]
  drop_message_regexes:
    - "^Street View"
    - "google\\.com/maps"
  strip_citation_block: true
  member_id_to_display_name:
    U09JKDU4RE2: "Athena assistant"
  human_mention_token: "@human"
  use_slack_thread_dataset_id: true
  window_minutes: 10000

  # Neon join configuration
  neon_query: |
    SELECT
      slack_thread_ts,
      slack_channel_id,
      quote_locator,
      langfuse_trace_id AS trace_id,
      created_at
    FROM athena_cases
  slack_join_columns: ["channel_id", "thread_ts"]
  neon_join_columns: ["slack_channel_id", "slack_thread_ts"]
  neon_time_column: "created_at"
  buffer_minutes: 30
  connection_string: "${DATABASE_URL}"
```

#### Join-specific fields

All Slack source fields above are also supported. These additional fields control the Neon join:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `neon_query` | str | Required | SQL query to fetch Neon rows |
| `slack_join_columns` | list[str] | Required | Column names in Slack dataset for join keys |
| `neon_join_columns` | list[str] | Required | Column names in Neon results for join keys (must match length) |
| `dataset_id_column` | str | — | Use this column from merged data as item ID |
| `use_slack_thread_dataset_id` | bool | `false` | Construct ID as `slack-{channel_id}-{thread_ts}` |
| `neon_time_column` | str | `"created_at"` | Timestamp column for time bounds |
| `buffer_minutes` | float | `0.0` | Time buffer (minutes) to expand Neon query bounds |
| `neon_connection_string` | str | `DATABASE_URL` env | Database URL for Neon |
| `neon_chunk_size` | int | `1000` | Rows per batch when streaming |
| `neon_timeout_seconds` | float | — | Query timeout override |

---

### Neon source (`type: neon`)

Direct PostgreSQL queries with a custom extractor function. Use when your data source is a database table rather than Slack or Langfuse.

```yaml
source:
  type: neon
  name: "athena_cases"
  query: |
    SELECT
      id,
      quote_locator,
      recommendation_entries,
      langfuse_trace_id,
      created_at
    FROM athena_cases
    WHERE created_at >= NOW() - INTERVAL '1 hour'
    ORDER BY created_at DESC
  extractor: "eval_workbench.implementations.athena.extractors.extract_recommendation_from_row"
  limit: 100
```

#### Neon source field reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | str | Required | Source identifier |
| `query` | str | Required | SQL query to execute |
| `extractor` | str | Required | Dotted Python path to `(row_dict) -> DatasetItem` function |
| `connection_string` | str | `DATABASE_URL` env | Database URL |
| `params` | dict/tuple | — | Query parameters for parameterized SQL |
| `limit` | int | — | Append `LIMIT` clause if query lacks one |
| `has_field` | str | — | Skip rows where this field is empty/falsy |
| `chunk_size` | int | `1000` | Rows per batch when streaming |
| `timeout_seconds` | float | — | Per-query statement timeout override |

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
  # metric_names:
  #   - "MyMetric"

  database:
    on_conflict: do_nothing
    chunk_size: 1000

  experiment:
    enabled: true
    dataset_name: "development"
    run_name: "online-${ENVIRONMENT:-preview}"
    link_to_traces: true
    # run_metadata: {}
    # tags: ["monitoring", "athena"]
    # flush: true
    # score_on_runtime_traces: false
    # metrics:  # legacy: prefer publishing.metric_names
    #   - "MyMetric"
```

---

## Usage Examples

```python
from eval_workbench.shared.monitoring import OnlineMonitor, ScoredItemsStore

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
