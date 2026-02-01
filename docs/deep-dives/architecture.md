# Architecture

This repo is a **thin, implementation-focused layer** built on top of **Axion** (an external evaluation framework) plus a set of shared utilities for:

- **Online monitoring** (scheduled “production-ish” evaluation runs against live data sources)
- **Offline analysis** (loading traces/datasets locally, running metrics, exporting results)
- **Prompt management + cache invalidation** (Langfuse + webhook + Slack notifications)
- **Optional persistence** (Neon/Postgres for normalized results)

Key directories:

- `src/shared/`: cross-implementation primitives (monitoring pipeline, Langfuse helpers, Slack helpers, DB helpers, shared metrics)
- `src/implementations/athena/`: a concrete implementation (extractors, metrics, Langfuse prompt patterns, YAML configs)
- `scripts/`: runnable entrypoints (notably `scripts/run_monitoring.py`)
- `.github/workflows/`: CI + scheduled monitoring automation
- `data/`: local + CI-cached dedup state (`data/scored_items.csv`) and other artifacts

---

## Repo-wide system diagram (ASCII)

```
                         ┌──────────────────────────────────────────────┐
                         │                 GitHub Actions               │
                         │                                              │
                         │  CI: lint/test (push/PR)                      │
                         │  Scheduled Monitoring: cron (hourly)          │
                         └───────────────┬──────────────────────────────┘
                                         │
                                         │ runs
                                         ▼
                          ┌───────────────────────────────────────────┐
                          │          scripts/run_monitoring.py         │
                          │  - parse args/env                           │
                          │  - load YAML config                          │
                          │  - set up ScoredItemsStore (optional)        │
                          └────────────────┬───────────────────────────┘
                                           │
                                           │ constructs from YAML
                                           ▼
                          ┌───────────────────────────────────────────┐
                          │       shared.monitoring.OnlineMonitor      │
                          │  Responsibilities:                           │
                          │  - load config (YAML + overrides)            │
                          │  - fetch from source                          │
                          │  - deduplicate via ScoredItemsStore           │
                          │  - sample items (strategy)                    │
                          │  - run Axion evaluation_runner                │
                          │  - publish results (Langfuse / Neon)          │
                          └───────────┬───────────────┬─────────────────┘
                                      │               │
                          fetch items │               │ publish
                                      │               │
                                      ▼               ▼
                ┌─────────────────────────────┐   ┌──────────────────────────────┐
                │ shared.monitoring.sources   │   │        Publishing Sinks       │
                │                             │   │                              │
                │  DataSource (interface)     │   │  Langfuse (experiments/obs)   │
                │   ├─ LangfuseDataSource     │   │   - results.publish_as_...     │
                │   └─ SlackDataSource        │   │                              │
                └──────────┬───────────┬──────┘   │                              │
                           │           │          │  Neon/Postgres (optional)     │
                           │           │          │   - shared.database.neon      │
                           │           │          └──────────────────────────────┘
                 Langfuse  │           │ Slack
                  traces   │           │ messages/threads
                           │           │
                           ▼           ▼
          ┌────────────────────────┐   ┌─────────────────────────────────┐
          │  Langfuse API / Axion   │   │            Slack API            │
          │  - LangfuseTraceLoader  │   │  - shared.slack.* helpers       │
          └───────────┬────────────┘   └─────────────────────────────────┘
                      │
                      │ per-trace extraction
                      ▼
     ┌────────────────────────────────────────────────────────────────────┐
     │             Implementations (example: Athena)                       │
     │                                                                    │
     │  src/implementations/athena/                                        │
     │   - extractors/*: Trace -> Axion DatasetItem                         │
     │   - metrics/*: scoring logic (LLM + heuristics)                      │
     │   - langfuse/prompt_patterns.py: regex extraction patterns           │
     │   - config/*.yaml: monitor definitions (source, sampling, publish)  │
     │                                                                    │
     │  Metric registry (import-time registration)                          │
     └──────────────────────────┬─────────────────────────────────────────┘
                                │
                                │ calls into
                                ▼
                     ┌───────────────────────────────────┐
                     │            Axion (external)        │
                     │  - DatasetItem                     │
                     │  - evaluation_runner               │
                     │  - tracing integration             │
                     └───────────────────────────────────┘


  Prompt lifecycle / cache invalidation side-channel:

        Langfuse Prompt UI/API ──(prompt.updated webhook)──► FastAPI app
                                                     shared.langfuse.webhook:app
                                                           │
                                                           ├─ invalidate local prompt cache
                                                           ├─ optional notify URL callback
                                                           └─ Slack alert (async task)
```

---

## Core runtime flows

### 1) Scheduled online monitoring (production path)

**Trigger**: `.github/workflows/monitoring.yml` runs hourly (cron) or via manual dispatch.

**Flow**:

1. GitHub Actions runs `python scripts/run_monitoring.py <config>.yaml`.
2. `scripts/run_monitoring.py`:
   - loads `src/implementations/athena/config/<config>.yaml`
   - optionally enables dedup via `data/scored_items.csv` (cached across workflow runs)
3. `OnlineMonitor.from_yaml(...)` builds:
   - a `DataSource` (`LangfuseDataSource` or `SlackDataSource`)
   - a `SamplingStrategy` (e.g. random \(n=10\))
   - a metric configuration mapping to registered metrics
4. `OnlineMonitor.run_async(...)`:
   - fetches items from the source
   - filters out already-scored items (`ScoredItemsStore`)
   - samples the remainder (strategy)
   - runs Axion `evaluation_runner(...)`
   - records dedup IDs
5. If configured, publishing happens (via `publish=True` on `run_async(...)` / `run(...)`):
   - to **Langfuse** as an experiment and/or observability events
   - optionally to **Neon/Postgres** in normalized tables

Note: the scripts currently call a convenience method named `run_and_publish(...)`, but the underlying monitor implementation publishes through the `publish` flag on `run_async(...)`.

**Why this works well**:

- The pipeline is **idempotent** (dedup) and safe to run repeatedly on a schedule.
- The inputs are **pluggable** (source/extractor) and the scoring is **configurable** (metrics_config).

---

### 2) Local/dev monitoring (continuous runner)

**Trigger**: `shared.monitoring.scheduler.MonitoringScheduler` (APScheduler) for local environments.

**Flow**:

- You add one or more YAML configs, each with either:
  - `interval_minutes`, or
  - `cron` (crontab string)
- The scheduler calls `monitor.run_async(publish=True)` on schedule.

This mirrors the GitHub Actions cron path but runs in a long-lived process.

---

### 3) Offline trace analysis (notebooks / scripts)

The trace wrapper layer (`shared.langfuse.trace`) supports a “domain-friendly” view of Langfuse traces:

- `TraceCollection` wraps a list of trace payloads
- `Trace` groups observations into named “steps”
- `PromptPatternsBase` (and implementation-specific subclasses) provides regex-based extraction so you can do:
  - `trace.recommendation.variables.some_field` (via dot-access wrappers)

This is used to build `DatasetItem` objects for evaluation and/or exploratory analysis.

---

### 4) Prompt management and cache invalidation (Langfuse → app)

`shared.langfuse.prompt.LangfusePromptManager` caches prompt fetches (TTL) and supports explicit invalidation.

**Flow**:

1. A prompt changes in Langfuse (create/update/delete).
2. Langfuse POSTs to the FastAPI endpoint `shared.langfuse.webhook:app` at `/webhooks/langfuse`.
3. The webhook verifies a signature and then:
   - marks the prompt stale in `LangfusePromptManager`
   - optionally notifies an external listener URL
   - posts a Slack alert asynchronously

This keeps “prompt-as-code” consumers consistent with the latest prompt versions without restarting long-running processes.

---

## Architectural boundaries (how things fit)

### “Shared core” vs “Implementation”

- `src/shared/` is the **platform**: stable abstractions (monitoring pipeline, data source interface, settings, DB clients).
- `src/implementations/<name>/` is **product-specific glue**:
  - how to convert source data into evaluation inputs (extractors)
  - which metrics exist and how they’re parameterized
  - what a “monitor run” means for that product (YAML configs)

This separation makes it easy to add a new implementation without rewriting the monitoring machinery.

---

## Design patterns used (explicitly)

### Strategy pattern

- **Data acquisition**: `DataSource` with `LangfuseDataSource` and `SlackDataSource`
- **Sampling**: `SamplingStrategy` implementations selected from config (e.g., all/random/most_recent/oldest)
- **Prompt variable extraction**: `PromptPatternsBase` subclasses per implementation (e.g., workflow vs chat patterns)

**Why**: swapping behavior without changing the `OnlineMonitor` orchestration logic.

---

### Factory + configuration-as-code

- `OnlineMonitor.from_yaml(...)` acts as a **factory** that translates YAML into concrete objects:
  - loads dotted-path extractors/classes
  - chooses source type and sampling strategy
  - threads raw config through for publishing metadata (agent/channel/component)

**Why**: the runtime topology is controlled by config, not code edits.

---

### Plugin/registry pattern (metrics)

Implementation metrics are registered in a module-level registry (e.g. Athena recommendation metrics), and the monitor imports the registry to ensure it’s populated before evaluation.

**Why**: the evaluation runner can refer to metrics by name/config while the implementation decides what exists.

---

### Adapter / ports-and-adapters (lightweight hexagonal)

You can view the monitor as a “use-case orchestrator” with ports:

- **Input ports**: `DataSource.fetch_items()` returning Axion `DatasetItem`s
- **Output ports**: publish-to-Langfuse and push-to-DB

Concrete adapters:

- Langfuse API (trace loader, experiments, prompt API)
- Slack API (exporting messages, posting notifications)
- Neon/Postgres (connection pool + dataframe upload)

**Why**: keeps the orchestration stable even as integrations evolve.

---

### Idempotency + dedup (stateful edge)

- `ScoredItemsStore` is a simple **append-only CSV** store keyed by `(source_key, monitor_key, item_id)`.
- GitHub Actions caches it across runs, which makes hourly monitoring “pick up where it left off”.

**Why**: online monitoring should be repeatable and safe under retries.

---

### Settings layering (12-factor style)

Settings are loaded via Pydantic Settings in two layers (documented in `README.md`):

- repo root `.env` (global defaults)
- per-implementation `.env` overrides

**Why**: one codebase supports multiple deployments/implementations with minimal drift.

---

## “Where to look” cheat-sheet

- **Online monitoring orchestrator**: `src/shared/monitoring/monitor.py`
- **Data sources**: `src/shared/monitoring/sources.py`
- **Dedup store**: `src/shared/monitoring/scored_items.py`
- **Local scheduler**: `src/shared/monitoring/scheduler.py`
- **Athena monitor config**: `src/implementations/athena/config/monitoring.yaml`
- **Langfuse trace wrappers**: `src/shared/langfuse/trace.py`
- **Langfuse prompts + caching**: `src/shared/langfuse/prompt.py`
- **Langfuse webhook (invalidate + Slack notify)**: `src/shared/langfuse/webhook.py`
- **Neon/Postgres helper**: `src/shared/database/neon.py`
- **Scheduled run automation**: `.github/workflows/monitoring.yml`
