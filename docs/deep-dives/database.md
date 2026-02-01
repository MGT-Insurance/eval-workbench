# Database Integration (Neon/PostgreSQL)

The shared database module provides synchronous and asynchronous interfaces for
PostgreSQL/Neon with connection pooling, DataFrame I/O, and concurrent task
execution.

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
│  NeonConnection       │         │  AsyncNeonDatabaseManager │
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
from shared.database.neon import get_neon_settings, reset_neon_settings_cache

# Get cached settings
settings = get_neon_settings()

# Clear cache when environment changes
reset_neon_settings_cache()
```

---

## NeonConnection (Synchronous)

Synchronous database manager for scripts, data analysis, and standard web apps.

### Initialization

```python
from shared.database.neon import NeonConnection

# Using environment variable
db = NeonConnection()

# Using explicit connection string
db = NeonConnection(connection_string="postgresql://user:pass@host/db")
```

### Context Manager (Recommended)

```python
with NeonConnection() as db:
    users = db.fetch_all("SELECT * FROM users")
    # Pool automatically closed on exit
```

### Query Methods

#### fetch_all() - Retrieve All Rows

```python
def fetch_all(
    self,
    query: str,
    params: Optional[Union[tuple, dict]] = None
) -> List[Dict[str, Any]]
```

**Example:**
```python
# Positional parameters
results = db.fetch_all(
    "SELECT * FROM users WHERE age > %s",
    (18,)
)

# Named parameters
results = db.fetch_all(
    "SELECT * FROM users WHERE email = %(email)s",
    {"email": "user@example.com"}
)
```

#### fetch_one() - Retrieve Single Row

```python
def fetch_one(
    self,
    query: str,
    params: Optional[Union[tuple, dict]] = None
) -> Optional[Dict[str, Any]]
```

**Example:**
```python
user = db.fetch_one("SELECT * FROM users WHERE id = %s", (1,))
if user:
    print(user['name'])  # Access as dictionary
```

#### execute_commit() - Execute and Commit

```python
def execute_commit(
    self,
    query: str,
    params: Optional[Union[tuple, dict]] = None
) -> None
```

**Example:**
```python
db.execute_commit(
    "INSERT INTO users (name, email) VALUES (%s, %s)",
    ("Alice", "alice@example.com")
)

db.execute_commit(
    "UPDATE users SET status = %s WHERE id = %s",
    ("active", 42)
)
```

### DataFrame Utilities

#### fetch_dataframe() - Query to DataFrame

```python
def fetch_dataframe(
    self,
    query: str,
    params: Optional[Union[tuple, dict]] = None
) -> pd.DataFrame
```

**Example:**
```python
df = db.fetch_dataframe(
    "SELECT id, name, score FROM results WHERE score > %s",
    (75,)
)
print(df.describe())
```

#### upload_dataframe() - Insert DataFrame

```python
def upload_dataframe(
    self,
    df: pd.DataFrame,
    table_name: str
) -> None
```

**Behavior:**
- Converts NaN/None to NULL
- Batches inserts by `db_upload_chunk_size` (default: 1000 rows)
- Logs row count on success

**Example:**
```python
import pandas as pd

df = pd.DataFrame({
    'id': [1, 2, 3],
    'value': [100.5, 200.3, None],
    'name': ['Alice', 'Bob', 'Charlie']
})

db.upload_dataframe(df, 'measurements')
# Logs: "Successfully uploaded 3 rows to 'measurements'."
```

### Table Management

#### create_table() - Safe Table Creation

```python
def create_table(
    self,
    table_name: str,
    schema: Optional[str] = None,
    columns: Optional[List[Tuple[str, str]]] = None
) -> None
```

**Using columns list (recommended):**
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

**Using raw schema:**
```python
db.create_table(
    table_name='users',
    schema='id SERIAL PRIMARY KEY, name VARCHAR(255), email VARCHAR(255) UNIQUE'
)
```

**SQL Type Validation:**
- Only allows: letters, digits, underscore, space, parentheses, comma, period
- Prevents SQL injection attacks

### Health Check

```python
def check_health(self) -> bool:
    """Test database connectivity. Returns True if successful."""

if db.check_health():
    print("Database is healthy")
```

### Cleanup

```python
db.close()  # Close the connection pool
```

---

## AsyncNeonDatabaseManager (Asynchronous)

Async database manager for FastAPI, high-concurrency workloads, and AI agents.
Identical API but fully async.

### Initialization

```python
from shared.database.neon import AsyncNeonDatabaseManager

db = AsyncNeonDatabaseManager()  # Pool opens lazily
```

### Async Context Manager

```python
async with AsyncNeonDatabaseManager() as db:
    users = await db.fetch_all("SELECT * FROM users")
    # Pool automatically closed on exit
```

### Async Methods

All methods parallel the sync API but are async:

```python
async def fetch_all(self, query, params=None) -> List[Dict]: ...
async def fetch_one(self, query, params=None) -> Optional[Dict]: ...
async def execute_commit(self, query, params=None) -> None: ...
async def fetch_dataframe(self, query, params=None) -> pd.DataFrame: ...
async def upload_dataframe(self, df, table_name) -> None: ...
async def create_table(self, table_name, schema=None, columns=None) -> None: ...
async def check_health(self) -> bool: ...
async def close(self) -> None: ...
```

### Concurrent Queries

```python
async with AsyncNeonDatabaseManager() as db:
    # Execute multiple queries concurrently
    users, orders, products = await asyncio.gather(
        db.fetch_all("SELECT * FROM users"),
        db.fetch_all("SELECT * FROM orders"),
        db.fetch_all("SELECT * FROM products"),
    )
```

---

## QueueExecutor

Worker pool pattern using `asyncio.Queue` for concurrent task execution.

### Initialization

```python
from shared.database.neon import QueueExecutor

executor = QueueExecutor(
    num_workers=5,      # Number of worker coroutines
    maxsize=0,          # Queue size limit (0 = unlimited)
    executor=None,      # Optional thread pool for sync functions
)
```

### Context Manager Usage

```python
async def process_item(item_id: int):
    await asyncio.sleep(1)
    return f"Processed {item_id}"

async def main():
    async with QueueExecutor(num_workers=10) as executor:
        # Submit tasks
        futures = [
            executor.submit(process_item, i)
            for i in range(100)
        ]

        # Gather results
        results = await asyncio.gather(*futures)
        return results

results = asyncio.run(main())
```

### Methods

#### start() - Activate Workers

```python
async def start(self):
    """Start worker coroutines. Must call before submitting tasks."""
```

#### stop() - Graceful Shutdown

```python
async def stop(self, timeout: Optional[float] = None):
    """Stop worker pool gracefully with optional timeout."""
```

#### submit() - Queue Task

```python
def submit(
    self,
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any
) -> asyncio.Future:
    """
    Submit function to be executed by worker pool.
    Supports both sync and async functions.
    """
```

### Backpressure Control

```python
# With queue size limit - blocks when full
async with QueueExecutor(num_workers=5, maxsize=100) as executor:
    # If queue has 100 tasks and workers are slow,
    # submit() will await until queue has space
    future = executor.submit(slow_function)
```

---

## Error Handling

All database methods follow consistent error handling:

```python
try:
    result = db.fetch_all(query, params)
except psycopg.Error as e:
    # Database-specific errors
    logger.error(f"Database query failed: {e}")
    raise
```

**Error Types:**
- `psycopg.Error`: Base exception for all psycopg errors
- `psycopg.DatabaseError`: Database-specific errors
- `psycopg.IntegrityError`: Constraint violations
- `psycopg.OperationalError`: Connection/operational issues
- `ValueError`: Invalid configuration or parameters

---

## Code Examples

### Pattern 1: Simple Synchronous Queries

```python
from shared.database.neon import NeonConnection

db = NeonConnection()

user = db.fetch_one("SELECT * FROM users WHERE id = %s", (1,))
users = db.fetch_all("SELECT * FROM users WHERE active = true")

db.close()
```

### Pattern 2: DataFrame Workflow

```python
with NeonConnection() as db:
    # Query to DataFrame
    df = db.fetch_dataframe("SELECT * FROM sales WHERE year = %s", (2024,))

    # Transform
    df['adjusted'] = df['value'] * 1.1

    # Upload back
    db.upload_dataframe(df, 'sales_adjusted')
```

### Pattern 3: Async with FastAPI

```python
from fastapi import FastAPI
from shared.database.neon import AsyncNeonDatabaseManager

app = FastAPI()
db = AsyncNeonDatabaseManager()

@app.on_event("startup")
async def startup():
    await db.pool.open()

@app.on_event("shutdown")
async def shutdown():
    await db.close()

@app.get("/users")
async def get_users():
    return await db.fetch_all("SELECT * FROM users")
```

### Pattern 4: Batch Operations

```python
with NeonConnection() as db:
    # Create table
    db.create_table('results', columns=[
        ('id', 'SERIAL PRIMARY KEY'),
        ('value', 'FLOAT'),
        ('name', 'VARCHAR(255)'),
    ])

    # Upload large DataFrame in batches
    import pandas as pd
    large_df = pd.read_csv('data.csv')  # 1M rows
    db.upload_dataframe(large_df, 'results')
    # Internally batches as: [0:1000], [1000:2000], ...
```

### Pattern 5: Parallel Processing with QueueExecutor

```python
from shared.database.neon import QueueExecutor

async def fetch_and_process(user_id: int):
    # Could be CPU-intensive or I/O work
    return user_id * 2

async def main():
    async with QueueExecutor(num_workers=10) as executor:
        futures = [
            executor.submit(fetch_and_process, i)
            for i in range(1000)
        ]
        results = await asyncio.gather(*futures)
        return results

results = asyncio.run(main())
```

### Pattern 6: Integration with Langfuse Joiner

```python
from shared.database.neon import NeonConnection
from implementations.athena.langfuse.join import AthenaNeonLangfuseJoiner

db = NeonConnection()
joiner = AthenaNeonLangfuseJoiner(neon_db=db, trace_loader=loader)

# Fetch cases using the database manager
cases = joiner.fetch_cases(
    limit=100,
    where="status = 'completed'",
)

db.close()
```

---

## Connection Pool Management

### Synchronous Pool

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

### Asynchronous Pool

```python
self.pool = AsyncConnectionPool(
    conninfo=self.dsn,
    min_size=0,
    max_size=20,
    open=False,          # Lazy initialization
)
```

### Statement Timeout

Applied per-connection when not using startup options:

```python
# If db_statement_timeout_ms > 0 and not using startup options:
SET statement_timeout = 60000;  # 60 seconds
```

---

## Dependencies

```
psycopg[binary]>=3.0         # PostgreSQL adapter
psycopg-pool>=3.0            # Connection pooling
pandas>=1.0                  # DataFrame support
pydantic>=2.0                # Settings validation
pydantic-settings>=2.0       # Environment loading
```

---

## Exports

```python
from shared.database.neon import (
    # Configuration
    NeonSettings,
    get_neon_settings,
    reset_neon_settings_cache,

    # Database Managers
    NeonConnection,
    AsyncNeonDatabaseManager,

    # Task Execution
    QueueExecutor,
)
```
