# Architecture

<div style="background: linear-gradient(135deg, #1E3A5F 0%, #0F2440 100%); padding: 24px; border-radius: 12px; color: white; margin: 20px 0;">

<p style="margin: 0; font-size: 16px; line-height: 1.6;">
<strong>A thin, implementation-focused layer</strong> built on top of <a href="https://github.com/ax-foundry/axion" style="color: #7BB8E0;">Axion</a> — providing online monitoring, offline analysis, prompt management with cache invalidation, and optional Neon/Postgres persistence.
</p>
</div>

## Key Directories

<table>
<tr>
<td width="50%" valign="top">

<h3><strong>Shared Core</strong></h3>
<strong><code>src/shared/</code></strong>

<p>Cross-implementation primitives — monitoring pipeline, Langfuse helpers, Slack helpers, DB helpers, and shared metrics.</p>

</td>
<td width="50%" valign="top">

<h3><strong>Implementations</strong></h3>
<strong><code>src/implementations/athena/</code></strong>

<p>Concrete implementation — extractors, metrics, Langfuse prompt patterns, and YAML monitoring configs.</p>

</td>
</tr>
<tr>
<td width="50%" valign="top">

<h3><strong>Scripts</strong></h3>
<strong><code>scripts/</code></strong>

<p>Runnable entrypoints, notably <code>scripts/run_monitoring.py</code> for triggering evaluation runs.</p>

</td>
<td width="50%" valign="top">

<h3><strong>Automation</strong></h3>
<strong><code>.github/workflows/</code></strong>

<p>CI + scheduled monitoring automation. Hourly cron jobs and manual dispatch for production runs.</p>

</td>
</tr>
</table>

---

## System Diagram

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
                 Langfuse  │           │ Slack    └──────────────────────────────┘
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
```

<div style="background: rgba(30, 58, 95, 0.1); border-left: 3px solid #1E3A5F; padding: 16px; margin: 20px 0;">

**Prompt lifecycle / cache invalidation side-channel:**

Langfuse Prompt UI/API sends a `prompt.updated` webhook to the FastAPI app (`shared.langfuse.webhook:app`), which invalidates the local prompt cache, optionally fires a notify URL callback, and posts a Slack alert asynchronously.
</div>

---

## Core Runtime Flows

### 1. Scheduled online monitoring

<div style="background: rgba(30, 58, 95, 0.1); border-left: 3px solid #1E3A5F; padding: 16px; margin: 20px 0;">

**Trigger:** `.github/workflows/monitoring.yml` runs hourly (cron) or via manual dispatch.
</div>

1. GitHub Actions runs `python scripts/run_monitoring.py <config>.yaml`
2. `scripts/run_monitoring.py` loads `src/implementations/athena/config/<config>.yaml` and optionally enables dedup via `data/scored_items.csv` (cached across workflow runs)
3. `OnlineMonitor.from_yaml(...)` builds a `DataSource`, a `SamplingStrategy`, and a metric configuration mapping to registered metrics
4. `OnlineMonitor.run_async(...)` fetches items, filters already-scored items, samples the remainder, and runs Axion `evaluation_runner(...)`
5. If configured, results are published to **Langfuse** (experiments/observability) and optionally to **Neon/Postgres** (normalized tables)

!!! tip "Why this works well"
    The pipeline is **idempotent** (dedup) and safe to run repeatedly on a schedule. Inputs are **pluggable** (source/extractor) and scoring is **configurable** (metrics_config).

---

### 2. Local/dev monitoring

<div style="background: rgba(30, 58, 95, 0.1); border-left: 3px solid #1E3A5F; padding: 16px; margin: 20px 0;">

**Trigger:** `shared.monitoring.scheduler.MonitoringScheduler` (APScheduler) for local environments.
</div>

Add one or more YAML configs with either `interval_minutes` or `cron` (crontab string). The scheduler calls `monitor.run_async(publish=True)` on schedule — mirroring the GitHub Actions cron path in a long-lived process.

---

### 3. Offline trace analysis

The trace wrapper layer (`shared.langfuse.trace`) supports a "domain-friendly" view of Langfuse traces:

- `TraceCollection` wraps a list of trace payloads
- `Trace` groups observations into named "steps"
- `PromptPatternsBase` provides regex-based extraction for dot-access: `trace.recommendation.variables.some_field`

This builds `DatasetItem` objects for evaluation and/or exploratory analysis.

---

### 4. Prompt management and cache invalidation

`shared.langfuse.prompt.LangfusePromptManager` caches prompt fetches (TTL) and supports explicit invalidation.

1. A prompt changes in Langfuse (create/update/delete)
2. Langfuse POSTs to `shared.langfuse.webhook:app` at `/webhooks/langfuse`
3. The webhook verifies a signature, marks the prompt stale, optionally notifies an external listener, and posts a Slack alert asynchronously

!!! info "Why this matters"
    Keeps "prompt-as-code" consumers consistent with the latest prompt versions without restarting long-running processes.

---

## Architectural Boundaries

<div style="background: linear-gradient(135deg, #1E3A5F 0%, #0F2440 100%); padding: 24px; border-radius: 12px; color: white; margin: 20px 0;">

<p style="margin: 0; font-size: 16px; line-height: 1.6;">
<strong>Shared core vs Implementation.</strong> <code style="color: #7BB8E0;">src/shared/</code> is the platform — stable abstractions (monitoring pipeline, data source interface, settings, DB clients). <code style="color: #7BB8E0;">src/implementations/&lt;name&gt;/</code> is product-specific glue — extractors, metrics, and YAML configs. This separation makes adding a new implementation trivial.
</p>
</div>

---

## Design Patterns

<div class="rule-grid">
  <div class="rule-card">
    <span class="rule-card__number">S</span>
    <p class="rule-card__title">Strategy</p>
    <p class="rule-card__desc"><strong>Data acquisition:</strong> <code>DataSource</code> with <code>LangfuseDataSource</code> and <code>SlackDataSource</code>. <strong>Sampling:</strong> strategy implementations from config. <strong>Extraction:</strong> <code>PromptPatternsBase</code> subclasses per implementation.</p>
  </div>
  <div class="rule-card">
    <span class="rule-card__number">F</span>
    <p class="rule-card__title">Factory + Config-as-Code</p>
    <p class="rule-card__desc"><code>OnlineMonitor.from_yaml(...)</code> translates YAML into concrete objects — resolving extractor registry keys (with dotted-path fallback), choosing sources, and selecting sampling strategies. Runtime topology is controlled by config, not code edits.</p>
  </div>
  <div class="rule-card">
    <span class="rule-card__number">R</span>
    <p class="rule-card__title">Plugin Registry</p>
    <p class="rule-card__desc">Implementation metrics registered in a module-level registry. The monitor imports the registry to populate it before evaluation. The runner refers to metrics by name while the implementation decides what exists.</p>
  </div>
  <div class="rule-card">
    <span class="rule-card__number">A</span>
    <p class="rule-card__title">Adapter / Ports-and-Adapters</p>
    <p class="rule-card__desc"><strong>Input ports:</strong> <code>DataSource.fetch_items()</code> returning an Axion <code>Dataset</code>. <strong>Output ports:</strong> publish-to-Langfuse, push-to-DB. Keeps orchestration stable as integrations evolve.</p>
  </div>
</div>

### Idempotency + Dedup

`ScoredItemsStore` is a simple **append-only CSV** store keyed by `(source_key, monitor_key, item_id)`. GitHub Actions caches it across runs, making hourly monitoring "pick up where it left off."

### Settings Layering (12-Factor)

Settings are loaded via Pydantic Settings in two layers:

- Repo root `.env` (global defaults)
- Per-implementation `.env` overrides

One codebase supports multiple deployments/implementations with minimal drift.

---

## Extractors

<div markdown style="background: rgba(30, 58, 95, 0.1); border-left: 3px solid #1E3A5F; padding: 16px; margin: 20px 0;">

**Extractors convert raw data into Axion `DatasetItem` objects.** Each data source (Langfuse traces, database rows, Slack messages) needs an extractor function to normalize the data into a standardized format for evaluation.

</div>

### Function Signature Pattern

All extractors follow the same pattern — a function that takes a source-specific input and returns a `DatasetItem`:

```python
from axion.dataset import DatasetItem

# Langfuse trace extractor
def extract_recommendation(trace: Trace) -> DatasetItem:
    return DatasetItem(
        id=trace.id,
        input={"query": ..., "context": ...},
        expected_output=...,
        actual_output=...,
        additional_input={...},
        metadata={...},
    )

# Database row extractor
def extract_recommendation_from_row(row: dict) -> DatasetItem:
    return DatasetItem(
        id=row["id"],
        input={"quote_locator": row["quote_locator"]},
        actual_output=row["recommendation"],
        metadata={...},
    )
```

### Athena Extractors

**Recommendation extractor** (`implementations/athena/extractors/recommendation.py`)

- `extract_recommendation(trace)` — Converts a Langfuse trace from the recommendation step into a DatasetItem. Extracts quote locator, underwriting flags, recommendations, citations, and latency.
- `extract_recommendation_from_row(row)` — Alternative extractor for database rows from `athena_cases`. Sorts recommendation entries by timestamp, extracts the most recent.

**Location extraction extractor** (`implementations/athena/extractors/location_extraction.py`)

- `extract_location_extraction(trace)` — Converts a Langfuse trace from the location-extraction step. Extracts quote data, product initiate, and generation output.

### Writing a New Extractor

1. Create a file in `implementations/<name>/extractors/`
2. Write a function matching the signature: `(source_data) -> DatasetItem`
3. Reference it in your monitoring YAML config:

```yaml
source:
  type: langfuse
  extractor: "eval_workbench.implementations.my_agent.extractors.my_extractor"
```

For Neon sources, use the dotted path to the function:

```yaml
source:
  type: neon
  extractor: "eval_workbench.implementations.my_agent.extractors.extract_from_row"
```

---

## Quick Reference

<table>
<tr>
<td width="50%" valign="top">
<h3><strong>Core Pipeline</strong></h3>
<strong>Monitoring &amp; Orchestration</strong>
<table>
<tr><th>File</th><th>Purpose</th></tr>
<tr><td><code>shared/monitoring/monitor.py</code></td><td>Online monitoring orchestrator</td></tr>
<tr><td><code>shared/monitoring/sources.py</code></td><td>Data sources</td></tr>
<tr><td><code>shared/monitoring/scored_items.py</code></td><td>Dedup store</td></tr>
<tr><td><code>shared/monitoring/scheduler.py</code></td><td>Local scheduler</td></tr>
</table>
</td>
<td width="50%" valign="top">
<h3><strong>Integrations</strong></h3>
<strong>External Systems &amp; Config</strong>
<table>
<tr><th>File</th><th>Purpose</th></tr>
<tr><td><code>shared/langfuse/trace.py</code></td><td>Trace wrappers</td></tr>
<tr><td><code>shared/langfuse/prompt.py</code></td><td>Prompts + caching</td></tr>
<tr><td><code>shared/langfuse/webhook.py</code></td><td>Cache invalidation + Slack</td></tr>
<tr><td><code>shared/database/neon.py</code></td><td>Neon/Postgres helper</td></tr>
</table>
</td>
</tr>
<tr>
<td colspan="2">
<table>
<tr><th>File</th><th>Purpose</th></tr>
<tr><td><code>implementations/athena/config/monitoring.yaml</code></td><td>Athena monitor config</td></tr>
<tr><td><code>.github/workflows/monitoring.yml</code></td><td>Scheduled run automation</td></tr>
</table>
</td>
</tr>
</table>
