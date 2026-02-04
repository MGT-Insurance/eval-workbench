# Eval Workbench – MGT AI Evaluation

This repository contains specific implementations built on top of the
[Axion](https://github.com/ax-foundry/axion) or any other Evaluation Module.
This architecture separates the core evaluation framework from individual
evaluation implementations, enabling better tracking, ability to create custom
sharable tooling, easier sharing, and a clear separation of concerns.

---

### architecture
Get the full repo-level architecture overview, system diagram, and runtime flows:
[Architecture overview](deep-dives/architecture.md)

---

### environment variables
Environment variables are loaded with Pydantic Settings in two layers, with later
files overriding earlier ones:

- 1) Repo root `.env` provides global defaults.
- 2) `implementations/<name>/.env` provides per-implementation overrides.

Use `.env.example` files as the canonical templates for what each layer supports.
When you add a new setting, define it on a settings class that inherits
`shared.settings.RepoSettingsBase` and sets `model_config` via
`build_settings_config(from_path=Path(__file__))`, then document the value in the
relevant `.env.example`.

---

### langfuse settings
Prompt management uses these Langfuse settings:
```
LANGFUSE_PUBLIC_KEY=""
LANGFUSE_SECRET_KEY=""
LANGFUSE_HOST=""
LANGFUSE_DEFAULT_LABEL="production"
LANGFUSE_DEFAULT_CACHE_TTL_SECONDS=60
```

---

### database settings
The Neon/Postgres helper reads these variables (all optional unless your app
needs a DB connection):
```
DATABASE_URL=""
DB_POOL_MIN_SIZE=0
DB_POOL_MAX_SIZE=20
DB_CONNECT_TIMEOUT_SECONDS=10
DB_STATEMENT_TIMEOUT_MS=60000
DB_USE_STARTUP_STATEMENT_TIMEOUT=false
DB_APPLICATION_NAME=""
DB_UPLOAD_CHUNK_SIZE=1000
```

---

### notebook imports
For notebooks under any `src/implementations/<name>/notebooks/`, install this
repo in editable mode so `shared` and `implementations.<name>` imports work
without `sys.path` hacks:
```
pip install -e .
```

---

### prompt patterns
Prompt extraction uses an explicit strategy passed to `Trace` or
`TraceCollection`. For Athena:
```
from eval_workbench.implementations.athena.langfuse.prompt_patterns import (
    ChatPromptPatterns,
    WorkflowPromptPatterns,
)
from eval_workbench.shared.langfuse.trace import TraceCollection

chat_traces = TraceCollection(data, prompt_patterns=ChatPromptPatterns)
recommendation_traces = TraceCollection(data, prompt_patterns=WorkflowPromptPatterns)
```

---

### langfuse webhooks
Use a webhook to invalidate prompt cache on updates:
```
export LANGFUSE_PUBLIC_KEY="..."
export LANGFUSE_SECRET_KEY="..."
export LANGFUSE_WEBHOOK_SECRET="whsec_..."
export LANGFUSE_WEBHOOK_NOTIFY_URL="https://your-app/notify"
export LANGFUSE_SLACK_CHANNEL_ID="C0123456789"
export LANGFUSE_SLACK_REQUEST_TIMEOUT_SECONDS="10"
export LANGFUSE_SLACK_RETRY_MAX_ATTEMPTS="3"
export LANGFUSE_SLACK_RETRY_BACKOFF_SECONDS="0.5"
export LANGFUSE_SLACK_RETRY_MAX_BACKOFF_SECONDS="4.0"
export SLACK_ATHENA_TOKEN="xoxb-..."
export SLACK_AIMEE_TOKEN="xoxb-..."
export SLACK_CANARY_TOKEN="xoxb-..."
export SLACK_PROMETHEUS_TOKEN="xoxb-..."
export SLACK_QUILL_TOKEN="xoxb-..."

uvicorn shared.langfuse.webhook:app --host 0.0.0.0 --port 5001
```

Configure Langfuse to POST to `https://your-host/webhooks/langfuse` for
`prompt.created`, `prompt.updated`, and `prompt.deleted` events.

Test: update a prompt in Langfuse and confirm a message posts to the Slack
channel. Slack posting happens in a background task and includes retry/backoff
with rate-limit handling.

---

### pre-commit
Formatting is managed via pre-commit hooks.
```
# Run on all files
pre-commit run --all-files

# Install to run after every commit
pre-commit install
```
