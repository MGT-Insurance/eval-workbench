<p align="center">
  <h1 align="center">Eval Workbench</h1>
  <p align="center">
    <strong>AI Evaluation Framework for MGT Insurance Products</strong>
  </p>
</p>

<p align="center">
  <a href="https://github.com/mgt-insurance/eval-workbench/actions/workflows/ci.yml">
    <img src="https://github.com/mgt-insurance/eval-workbench/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+">
  </a>
  <a href="https://github.com/astral-sh/ruff">
    <img src="https://img.shields.io/badge/code%20style-ruff-000000.svg" alt="Ruff">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT">
  </a>
</p>

<p align="center">
  <a href="https://mgt-insurance.github.io/eval-workbench/"><strong>Documentation</strong></a> ·
  <a href="https://github.com/mgt-insurance/eval-workbench/issues">Report Bug</a> ·
  <a href="https://github.com/mgt-insurance/eval-workbench/issues">Request Feature</a>
</p>

---

## Overview

Eval Workbench provides evaluation implementations built on top of the [Axion](https://github.com/ax-foundry/axion) framework. It enables comprehensive AI evaluation with support for multiple data sources, automated monitoring, and seamless integration with observability tools.

### Key Features

| Feature | Description |
|---------|-------------|
| **Langfuse Integration** | Prompt management, trace collection, and webhook-driven cache invalidation |
| **Online Monitoring** | Scheduled evaluation with configurable sampling and deduplication |
| **Database Persistence** | Batch upload evaluation results to Neon/Postgres |
| **Extensible Architecture** | Add new implementations without modifying shared code |

---

## Requirements

- **Python 3.12+**
- PostgreSQL/Neon database (optional, for result persistence)
- Langfuse account (optional, for prompt management)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/mgt-insurance/eval-workbench.git
cd eval-workbench

# (Optional) Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

---

## Quick Start

### 1. Configure Environment

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

### 2. Run an Evaluation

```python
from eval_workbench.shared.langfuse.trace import TraceCollection
from eval_workbench.implementations.athena.langfuse.prompt_patterns import ChatPromptPatterns

# Load and process traces
traces = TraceCollection(data, prompt_patterns=ChatPromptPatterns)
```

### 3. Start Online Monitoring

```bash
# Run with Langfuse data source
python scripts/run_monitoring.py monitoring_langfuse.yaml

# Run with Slack data source
python scripts/run_monitoring.py monitoring_slack.yaml

# Run with options
python scripts/run_monitoring.py monitoring_langfuse.yaml --limit 10 --hours-back 24
```

---

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src

# Run specific test file
pytest tests/shared/metrics/slack/test_engagement.py -v
```

---

## Project Structure

```
eval-workbench/
├── src/
│   └── eval_workbench/            # Main package
│       ├── shared/                # Reusable utilities and base classes
│       │   ├── langfuse/          # Langfuse integration (prompts, traces, webhooks)
│       │   ├── metrics/           # Metric base classes and Slack analyzers
│       │   ├── monitoring/        # Online monitoring and scheduling
│       │   ├── database/          # Neon/Postgres connection management
│       │   └── slack/             # Slack API utilities
│       │
│       └── implementations/       # Product-specific implementations
│           └── athena/            # Athena evaluation metrics
│               ├── config/        # YAML configuration files
│               ├── extractors/    # Data extraction logic
│               ├── langfuse/      # Prompt patterns and joins
│               └── metrics/       # Athena-specific metrics
│
├── scripts/                       # CLI tools and runners
│   ├── run_monitoring.py          # Main monitoring script
│   └── create_evaluation_tables.py
│
├── tests/                         # Test suite
├── docs/                          # MkDocs documentation source
└── pyproject.toml                 # Project configuration
```

---

## Configuration

### Environment Variables

Environment variables are loaded in two layers (later overrides earlier):

1. **Repo root `.env`** → Global defaults
2. **`implementations/<name>/.env`** → Per-implementation overrides

<details>
<summary><strong>Langfuse Settings</strong></summary>

```env
LANGFUSE_PUBLIC_KEY=""
LANGFUSE_SECRET_KEY=""
LANGFUSE_HOST="https://us.cloud.langfuse.com"
LANGFUSE_DEFAULT_LABEL="production"
LANGFUSE_DEFAULT_CACHE_TTL_SECONDS=60
```

</details>

<details>
<summary><strong>Database Settings</strong></summary>

```env
DATABASE_URL=""
DB_POOL_MIN_SIZE=0
DB_POOL_MAX_SIZE=20
DB_CONNECT_TIMEOUT_SECONDS=10
DB_STATEMENT_TIMEOUT_MS=60000
DB_APPLICATION_NAME=""
DB_UPLOAD_CHUNK_SIZE=1000
```

</details>

<details>
<summary><strong>Slack Settings</strong></summary>

```env
SLACK_ATHENA_TOKEN=""
SLACK_AIMEE_TOKEN=""
SLACK_CANARY_TOKEN=""
SLACK_PROMETHEUS_TOKEN=""
SLACK_QUILL_TOKEN=""
```

</details>

<details>
<summary><strong>Webhook Settings</strong></summary>

```env
LANGFUSE_WEBHOOK_SECRET=""
LANGFUSE_WEBHOOK_NOTIFY_URL=""
LANGFUSE_SLACK_CHANNEL_ID=""
LANGFUSE_SLACK_REQUEST_TIMEOUT_SECONDS=10
LANGFUSE_SLACK_RETRY_MAX_ATTEMPTS=3
```

</details>

### YAML Configuration

Monitoring pipelines are configured via YAML files with environment variable interpolation:

```yaml
source:
  type: langfuse
  hours_back: ${HOURS_BACK:-24}
  limit: ${LIMIT:-100}

sampling:
  strategy: random
  n: 30

metrics:
  model: gpt-4o
```

Config files are located in `src/eval_workbench/implementations/<name>/config/`.

---

## CI/CD

### GitHub Actions Workflows

| Workflow | Trigger | Description |
|----------|---------|-------------|
| **CI** | Push/PR to main | Runs linting (ruff) and tests (pytest) |
| **Docs** | Push to main | Builds and deploys documentation to GitHub Pages |
| **Monitoring** | Scheduled (30min) | Runs automated evaluation (controlled by `MONITORING_ENABLED` variable) |

### Pre-commit Hooks

```bash
# Run all hooks manually
pre-commit run --all-files

# Hooks include:
# - ruff (linting + formatting)
# - trailing whitespace removal
# - YAML validation
# - Large file checks
# - nbstripout (Jupyter notebook output stripping)
```

---

## Monitoring Script Reference

```bash
python scripts/run_monitoring.py [config_file] [options]
```

| Option | Description |
|--------|-------------|
| `config_file` | YAML config file (default: `monitoring.yaml`) |
| `--limit N` | Override item limit |
| `--hours-back N` | Override hours to look back (Langfuse) |
| `--days-back N` | Override days to look back (Langfuse) |
| `--no-publish` | Skip publishing results |
| `--no-dedup` | Process all items (skip deduplication) |
| `--use-db-store` | Use database for deduplication instead of CSV |
| `-o, --output FILE` | Save results to CSV |
| `-v, --verbose` | Enable debug logging |

**Environment Variables:**

```bash
DEDUPLICATE=false        # Disable deduplication
SCORED_ITEMS_FILE=path   # Custom scored items CSV path
USE_DB_STORE=true        # Use database-backed deduplication
ENVIRONMENT=production   # Environment name for run naming
```

---

## Development

### Code Style

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting:

```bash
# Format code
ruff format .

# Check linting
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Adding a New Implementation

1. Create directory: `src/eval_workbench/implementations/<name>/`
2. Add `settings.py` extending `RepoSettingsBase`
3. Add `langfuse/prompt_patterns.py` for trace extraction
4. Add metrics in `metrics/` directory
5. Create config YAML in `config/`
6. Add `.env` for implementation-specific settings

### Notebook Development

For notebooks under `src/eval_workbench/implementations/<name>/notebooks/`:

```bash
# Install in editable mode (one-time setup)
pip install -e .

# Imports will work without sys.path hacks:
from eval_workbench.shared.langfuse.trace import TraceCollection
from eval_workbench.implementations.athena.settings import AthenaSettings
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/ -v`)
5. Run linting (`pre-commit run --all-files`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
