# Configuration

<div style="background: linear-gradient(135deg, #1E3A5F 0%, #0F2440 100%); padding: 24px; border-radius: 12px; color: white; margin: 20px 0;">

<p style="margin: 0; font-size: 16px; line-height: 1.6;">
<strong>Two complementary systems:</strong> a YAML config module with dot notation and env interpolation, and a Pydantic settings module with .env cascade. Together they handle all configuration needs from YAML files to environment variables.
</p>
</div>

---

## Config Module

**Source:** `src/eval_workbench/shared/config.py`

The config module provides a global dictionary-based configuration store with dot-notation access, environment variable interpolation, and context manager support.

### Loading Configuration

```python
from eval_workbench.shared.config import load, load_config

# Load YAML into global config (permanent)
load("path/to/config.yaml")

# Load as context manager (temporary — reverts on exit)
with load("path/to/config.yaml"):
    # config is active here
    pass
# config reverted to previous state

# Load dict directly
load({"source": {"type": "slack", "limit": 50}})

# Load YAML without modifying global config
cfg = load_config("path/to/config.yaml")
cfg = load_config("path/to/config.yaml", overrides={"source.limit": 100})
```

### Reading Values

```python
from eval_workbench.shared.config import get

# Dot notation access
source_type = get("source.type")
limit = get("source.limit", default=50)

# Raise on missing key
value = get("required.key", error=True)  # raises KeyError if missing

# Read from a specific config dict (not global)
value = get("source.type", cfg=my_config)
```

### Setting Values

```python
from eval_workbench.shared.config import set

# Temporary override via context manager
with set({"source.limit": 100, "sampling.n": 5}):
    # overridden values active here
    pass
# reverted
```

### Environment Variable Interpolation

YAML values containing `${VAR}` or `${VAR:-default}` are automatically resolved:

```yaml
# In YAML
environment: "${ENVIRONMENT:-production}"
database:
  connection_string: "${DATABASE_URL}"
  pool_size: "${DB_POOL_SIZE:-20}"
```

```python
# After loading
get("environment")                   # "production" (if ENVIRONMENT unset)
get("database.connection_string")    # value of DATABASE_URL env var
get("database.pool_size")            # "20" (if DB_POOL_SIZE unset)
```

---

## Settings Module

**Source:** `src/eval_workbench/shared/settings.py`

The settings module provides Pydantic-based settings with automatic .env file resolution and implementation-aware configuration.

### RepoSettingsBase

Base class for all settings models. Pre-configured for UTF-8 encoding and `extra='ignore'`.

```python
from eval_workbench.shared.settings import RepoSettingsBase, build_settings_config

class MySettings(RepoSettingsBase):
    model_config = build_settings_config()

    api_key: str
    timeout: int = 30
```

### .env Cascade

Settings are loaded from .env files in a specific order:

```
1. Repo root .env           →  /path/to/eval-workbench/.env
2. Implementation .env      →  /path/to/eval-workbench/src/.../implementations/{name}/.env
```

Later files override earlier ones. The cascade is resolved automatically.

### Key Functions

```python
from eval_workbench.shared.settings import (
    find_repo_root,
    infer_implementation_name,
    resolve_env_files,
    build_settings_config,
)

# Find repository root (searches for .git directory)
root = find_repo_root()

# Infer implementation from current path
name = infer_implementation_name()  # e.g., "athena"

# Resolve .env file paths
env_files = resolve_env_files()
# ["/.../eval-workbench/.env", "/.../implementations/athena/.env"]

# Build Pydantic settings config
config = build_settings_config(
    from_path=None,              # Start search path (default: cwd)
    implementation_name=None,    # Override implementation name
    env_prefix=None,             # Environment variable prefix
)
```

### Usage Pattern

```python
from pydantic_settings import BaseSettings
from eval_workbench.shared.settings import build_settings_config

class NeonSettings(BaseSettings):
    model_config = build_settings_config(env_prefix="DB_")

    database_url: str | None = None
    pool_min_size: int = 0
    pool_max_size: int = 20
    statement_timeout_ms: int = 60000
```

With a `.env` file:

```
DB_DATABASE_URL=postgresql://user:pass@host/db
DB_POOL_MAX_SIZE=10
```

---

## When to Use Which

| Need | Module | Example |
|------|--------|---------|
| YAML monitoring configs | `config` | `load("monitoring_slack.yaml")` |
| Runtime config overrides | `config` | `with set({"source.limit": 5}): ...` |
| Dot-notation nested access | `config` | `get("source.type")` |
| Environment-based settings | `settings` | `NeonSettings()` |
| Validated, typed settings | `settings` | Pydantic model with defaults |
| .env file resolution | `settings` | `build_settings_config()` |
| Config with env interpolation | `config` | `${DATABASE_URL}` in YAML |
