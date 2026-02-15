# Database Integration

<div style="background: linear-gradient(135deg, #1E3A5F 0%, #0F2440 100%); padding: 24px; border-radius: 12px; color: white; margin: 20px 0;">

<p style="margin: 0; font-size: 16px; line-height: 1.6;">
<strong>Neon/PostgreSQL with connection pooling, DataFrame I/O, and concurrent task execution.</strong> The shared database module provides synchronous and asynchronous interfaces for all persistence needs.
</p>
</div>

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     NeonSettings                                │
│  Configuration from environment variables                       │
│  • DATABASE_URL, DB_POOL_*, DB_STATEMENT_TIMEOUT_MS            │
└─────────────────────────────────────────────────────────────────┘
                             │
            ┌─────────────────┴─────────────────┐
            │                                   │
            ▼                                   ▼
┌───────────────────────┐         ┌───────────────────────────┐
│  NeonConnection       │         │  AsyncNeonConnection      │
│  (Synchronous)        │         │  (Asynchronous)           │
│                       │         │                           │
│  • fetch_all()        │         │  • await fetch_all()      │
│  • fetch_one()        │         │  • await fetch_one()      │
│  • execute_commit()   │         │  • await execute_commit() │
│  • fetch_dataframe()  │         │  • await fetch_dataframe()│
│  • upload_dataframe() │         │  • await upload_dataframe()│
│  • create_table()     │         │  • await create_table()   │
│  • check_health()     │         │  • await check_health()   │
└───────────────────────┘         └───────────────────────────┘
            │                                   │
            ▼                                   ▼
┌───────────────────────┐         ┌───────────────────────────┐
│   ConnectionPool      │         │   AsyncConnectionPool     │
│   (psycopg3)          │         │   (psycopg3)              │
└───────────────────────┘         └───────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     QueueExecutor                               │
│  Concurrent task execution with asyncio.Queue                   │
│  • Worker pool pattern                                          │
│  • Supports sync and async functions                            │
│  • Configurable backpressure                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### NeonSettings

```python
class NeonSettings(RepoSettingsBase):
    database_url: str | None              # PostgreSQL connection URL
    db_pool_min_size: int                 # Minimum pool connections
    db_pool_max_size: int                 # Maximum pool connections
    db_connect_timeout_seconds: int       # Connection timeout
    db_statement_timeout_ms: int          # Query timeout (0 = disabled)
    db_use_startup_statement_timeout: bool
    db_application_name: str | None       # For PostgreSQL monitoring
    db_upload_chunk_size: int             # Rows per batch for uploads
```

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DATABASE_URL` | str | Required | Postgres connection URL |
| `DB_POOL_MIN_SIZE` | int | `0` | Minimum connections in pool |
| `DB_POOL_MAX_SIZE` | int | `20` | Maximum connections in pool |
| `DB_CONNECT_TIMEOUT_SECONDS` | int | `10` | Connection timeout |
| `DB_STATEMENT_TIMEOUT_MS` | int | `60000` | Statement timeout (0 = disabled) |
| `DB_USE_STARTUP_STATEMENT_TIMEOUT` | bool | `False` | Use startup options (unsupported by Neon pooler) |
| `DB_APPLICATION_NAME` | str | None | PostgreSQL application_name |
| `DB_UPLOAD_CHUNK_SIZE` | int | `1000` | Rows per batch for DataFrame uploads |

### Settings Access

```python
from eval_workbench.shared.database.neon import get_neon_settings, reset_neon_settings_cache

# Get cached settings
settings = get_neon_settings()

# Clear cache when environment changes
reset_neon_settings_cache()
```

---

## Connection Managers

<table>
<tr>
<td width="50%" valign="top">

<h3><strong>NeonConnection</strong></h3>
<strong>Synchronous — scripts, data analysis, web apps</strong>

```python
from eval_workbench.shared.database.neon import NeonConnection

# Using environment variable
db = NeonConnection()

# Using explicit connection string
db = NeonConnection(
    connection_string="postgresql://user:pass@host/db"
)
```

Context manager (recommended):

```python
with NeonConnection() as db:
    users = db.fetch_all("SELECT * FROM users")
    # Pool automatically closed on exit
```

</td>
<td width="50%" valign="top">

<h3><strong>AsyncNeonConnection</strong></h3>
<strong>Asynchronous — FastAPI, high-concurrency, AI agents</strong>

```python
from eval_workbench.shared.database.neon import AsyncNeonConnection

db = AsyncNeonConnection()  # Pool opens lazily
```

Async context manager:

```python
async with AsyncNeonConnection() as db:
    users = await db.fetch_all("SELECT * FROM users")
    # Pool automatically closed on exit
```

Concurrent queries:

```python
async with AsyncNeonConnection() as db:
    users, orders, products = await asyncio.gather(
        db.fetch_all("SELECT * FROM users"),
        db.fetch_all("SELECT * FROM orders"),
        db.fetch_all("SELECT * FROM products"),
    )
```

</td>
</tr>
</table>

---

## Query Methods

All methods are available on both `NeonConnection` (sync) and `AsyncNeonConnection` (async with `await`).

### fetch_all / fetch_one

```python
# Positional parameters
results = db.fetch_all(
    "SELECT * FROM users WHERE age > %s", (18,)
)

# Named parameters
results = db.fetch_all(
    "SELECT * FROM users WHERE email = %(email)s",
    {"email": "user@example.com"}
)

# Single row
user = db.fetch_one("SELECT * FROM users WHERE id = %s", (1,))
if user:
    print(user['name'])
```

### execute_commit

```python
db.execute_commit(
    "INSERT INTO users (name, email) VALUES (%s, %s)",
    ("Alice", "alice@example.com")
)
```

### DataFrame I/O

```python
# Query to DataFrame
df = db.fetch_dataframe(
    "SELECT id, name, score FROM results WHERE score > %s", (75,)
)

# Upload DataFrame (batched by db_upload_chunk_size)
db.upload_dataframe(df, 'measurements')
```

### Table Management

```python
db.create_table(
    table_name='users',
    columns=[
        ('id', 'SERIAL PRIMARY KEY'),
        ('name', 'VARCHAR(255) NOT NULL'),
        ('email', 'VARCHAR(255) UNIQUE'),
        ('created_at', 'TIMESTAMP DEFAULT CURRENT_TIMESTAMP'),
    ]
)
```

!!! note "SQL Type Validation"
    Only allows: letters, digits, underscore, space, parentheses, comma, period. Prevents SQL injection attacks.

### Health Check

```python
if db.check_health():
    print("Database is healthy")
```

---

## QueueExecutor

<div style="background: rgba(30, 58, 95, 0.1); border-left: 3px solid #1E3A5F; padding: 16px; margin: 20px 0;">

Worker pool pattern using `asyncio.Queue` for concurrent task execution. Supports both sync and async functions with configurable backpressure.
</div>

```python
from eval_workbench.shared.database.neon import QueueExecutor

async def process_item(item_id: int):
    await asyncio.sleep(1)
    return f"Processed {item_id}"

async def main():
    async with QueueExecutor(num_workers=10) as executor:
        futures = [
            executor.submit(process_item, i)
            for i in range(100)
        ]
        results = await asyncio.gather(*futures)
        return results
```

```python
# With backpressure control — blocks when queue is full
async with QueueExecutor(num_workers=5, maxsize=100) as executor:
    future = executor.submit(slow_function)
```

---

## Usage Patterns

<div class="rule-grid">
  <div class="rule-card">
    <span class="rule-card__number">1</span>
    <p class="rule-card__title">Simple Queries</p>
    <p class="rule-card__desc">Direct <code>fetch_one</code> / <code>fetch_all</code> calls with <code>NeonConnection()</code>. Best for scripts and one-off data exploration.</p>
  </div>
  <div class="rule-card">
    <span class="rule-card__number">2</span>
    <p class="rule-card__title">DataFrame Workflow</p>
    <p class="rule-card__desc">Query to DataFrame, transform in pandas, upload back. Great for ETL pipelines and evaluation result processing.</p>
  </div>
  <div class="rule-card">
    <span class="rule-card__number">3</span>
    <p class="rule-card__title">Async + FastAPI</p>
    <p class="rule-card__desc"><code>AsyncNeonConnection</code> with lifecycle events. Pool opens at startup, closes on shutdown. Concurrent queries with <code>asyncio.gather</code>.</p>
  </div>
  <div class="rule-card">
    <span class="rule-card__number">4</span>
    <p class="rule-card__title">Batch Processing</p>
    <p class="rule-card__desc"><code>QueueExecutor</code> for parallel processing with backpressure. Combined with <code>upload_dataframe</code> for large-scale data operations.</p>
  </div>
</div>

---

## EvaluationUploader

<div markdown style="background: rgba(30, 58, 95, 0.1); border-left: 3px solid #1E3A5F; padding: 16px; margin: 20px 0;">

**Source:** `src/eval_workbench/shared/database/evaluation_upload.py`

High-level upload interface for evaluation data. Handles column subsetting, JSONB normalization, conflict resolution, and batched inserts.

</div>

### Column Definitions

**`EVALUATION_DATASET_COLUMNS`** (29 columns) — `dataset_id`, `query`, `expected_output`, `actual_output`, `additional_input`, `acceptance_criteria`, `dataset_metadata`, `user_tags`, `conversation`, `tools_called`, `expected_tools`, `judgment`, `critique`, `trace`, `additional_output`, `source_type`, `environment`, `source_name`, `source_component`, `created_at`, `retrieved_content`, `document_text`, `actual_reference`, `expected_reference`, `latency`, `trace_id`, `observation_id`, `has_errors`

**`EVALUATION_RESULTS_COLUMNS`** (23 columns) — `run_id`, `dataset_id`, `metric_name`, `metric_score`, `passed`, `explanation`, `metric_type`, `metric_category`, `threshold`, `signals`, `metric_id`, `parent`, `weight`, `evaluation_name`, `eval_mode`, `cost_estimate`, `model_name`, `llm_provider`, `version`, `timestamp`, `source`, `metric_metadata`, `evaluation_metadata`

### JSONB Normalization

Columns marked as JSONB are automatically handled:

- `dict` and `list` values are serialized to JSON strings
- JSON string values are validated for correctness
- `NaN`, `Inf`, and `NaT` values are sanitized to `None`
- Non-string dict keys raise `EvaluationUploadError`

### Conflict Resolution

| Mode | Behavior |
|------|----------|
| `error` | Raise on conflict (default) |
| `do_nothing` | Skip conflicting rows (`ON CONFLICT DO NOTHING`) |
| `upsert` | Update existing rows (`ON CONFLICT DO UPDATE`) |

### EvaluationUploader Class

```python
from eval_workbench.shared.database.evaluation_upload import EvaluationUploader
from eval_workbench.shared.database.neon import NeonConnection

with NeonConnection() as db:
    uploader = EvaluationUploader(
        db=db,
        on_conflict="do_nothing",   # error | do_nothing | upsert
        chunk_size=1000,
        include_missing_columns=False,
        dataset_id_source="dataset_id",  # dataset_id | id | metric_id
    )

    # Upload dataset items
    uploaded_df = uploader.upload_dataset(dataset_df)

    # Upload evaluation results
    uploaded_df = uploader.upload_results(results_df)

    # Generic upload (specify table)
    uploaded_df = uploader.upload(df, table="evaluation_dataset")
```

### Standalone Functions

```python
from eval_workbench.shared.database.evaluation_upload import (
    subset_evaluation_dataset_df_for_upload,
    subset_evaluation_results_df_for_upload,
    upload_evaluation_dataset_df,
    upload_evaluation_results_df,
)

# Subset DataFrame to only valid columns (no upload)
clean_df = subset_evaluation_dataset_df_for_upload(df, include_missing_columns=True)

# Full upload pipeline
with NeonConnection() as db:
    upload_evaluation_dataset_df(db, dataset_df, on_conflict="do_nothing")
    upload_evaluation_results_df(db, results_df, on_conflict="upsert")
```

---

## Error Handling

```python
try:
    result = db.fetch_all(query, params)
except psycopg.Error as e:
    logger.error(f"Database query failed: {e}")
    raise
```

| Exception | Description |
|-----------|-------------|
| `psycopg.Error` | Base exception for all psycopg errors |
| `psycopg.DatabaseError` | Database-specific errors |
| `psycopg.IntegrityError` | Constraint violations |
| `psycopg.OperationalError` | Connection/operational issues |
| `ValueError` | Invalid configuration or parameters |

---

## Connection Pool Management

=== "Synchronous"

    ```python
    self.pool = ConnectionPool(
        conninfo=self.dsn,
        min_size=0,          # Pre-allocated connections
        max_size=20,         # Maximum concurrent connections
        kwargs={
            "row_factory": dict_row,
            "connect_timeout": 10,
        },
    )
    ```

=== "Asynchronous"

    ```python
    self.pool = AsyncConnectionPool(
        conninfo=self.dsn,
        min_size=0,
        max_size=20,
        open=False,          # Lazy initialization
    )
    ```

!!! info "Statement Timeout"
    Applied per-connection when not using startup options: `SET statement_timeout = 60000;` (60 seconds).

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `psycopg[binary]` | >=3.0 | PostgreSQL adapter |
| `psycopg-pool` | >=3.0 | Connection pooling |
| `pandas` | >=1.0 | DataFrame support |
| `pydantic` | >=2.0 | Settings validation |
| `pydantic-settings` | >=2.0 | Environment loading |

---

## Exports

```python
from eval_workbench.shared.database.neon import (
    # Configuration
    NeonSettings,
    get_neon_settings,
    reset_neon_settings_cache,

    # Database Managers
    NeonConnection,
    AsyncNeonConnection,

    # Task Execution
    QueueExecutor,
)
```
