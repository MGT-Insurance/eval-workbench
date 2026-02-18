# Scripts Reference

<div style="background: linear-gradient(135deg, #1E3A5F 0%, #0F2440 100%); padding: 24px; border-radius: 12px; color: white; margin: 20px 0;">

<p style="margin: 0; font-size: 16px; line-height: 1.6;">
<strong>Runnable entrypoints for monitoring, table creation, and KPI population.</strong> All scripts live in <code style="color: #7BB8E0;">scripts/</code> and can be run directly with Python.
</p>
</div>

<div class="rule-grid">

<div class="rule-card">
<span class="rule-card__number">1</span>
<p class="rule-card__title">monitoring_entrypoint.py</p>
<p class="rule-card__desc">Run a single monitoring pass from a YAML config. Designed for GitHub Actions cron.</p>
</div>

<div class="rule-card">
<span class="rule-card__number">2</span>
<p class="rule-card__title">monitoring_dry_run.py</p>
<p class="rule-card__desc">Run the same monitor flow locally with publishing disabled (no DB/Langfuse push).</p>
</div>

<div class="rule-card">
<span class="rule-card__number">3</span>
<p class="rule-card__title">create_evaluation_tables.py</p>
<p class="rule-card__desc">Create <code>evaluation_dataset</code>, <code>evaluation_results</code> tables, and a joined view.</p>
</div>

<div class="rule-card">
<span class="rule-card__number">4</span>
<p class="rule-card__title">create_athena_kpi_objects.py</p>
<p class="rule-card__desc">Create the <code>agent_kpi_logs</code> EAV table for KPI observations.</p>
</div>

<div class="rule-card">
<span class="rule-card__number">5</span>
<p class="rule-card__title">create_rule_extractions_table.py</p>
<p class="rule-card__desc">Create the <code>rule_extractions</code> table for memory pipeline output.</p>
</div>

<div class="rule-card">
<span class="rule-card__number">6</span>
<p class="rule-card__title">populate_athena_kpis.py</p>
<p class="rule-card__desc">Populate <code>agent_kpi_logs</code> from <code>athena_cases</code> and <code>evaluation_results</code>.</p>
</div>

</div>

---

## monitoring_entrypoint.py

Run a single `OnlineMonitor` pass from a YAML config file and exit. Designed for GitHub Actions cron runs (run-once per schedule).

```bash
python scripts/monitoring/monitoring_entrypoint.py [config_file]
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `config_file` | `monitoring_langfuse.yaml` | Config file path or name. Searches `implementations/athena/config/` if not found directly. `.yaml` extension optional. |

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `DEDUPLICATE` | `true` | Enable/disable deduplication |
| `ENVIRONMENT` | — | Production environment designation |
| `OPENAI_API_KEY` | — | OpenAI API key (for LLM metrics) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (for LLM metrics) |
| `DATABASE_URL` | — | PostgreSQL connection string |
| `LANGFUSE_*` | — | Langfuse credentials |

**Examples:**

```bash
# Default config
python scripts/monitoring/monitoring_entrypoint.py

# Specific config (with or without extension)
python scripts/monitoring/monitoring_entrypoint.py monitoring_slack
python scripts/monitoring/monitoring_entrypoint.py monitoring_slack.yaml

# Full path
python scripts/monitoring/monitoring_entrypoint.py src/eval_workbench/implementations/athena/config/monitoring_neon.yaml

# Disable deduplication
DEDUPLICATE=false python scripts/monitoring/monitoring_entrypoint.py monitoring_slack
```

---

## monitoring_dry_run.py

Run a local dry-run of `OnlineMonitor` from YAML config using the same setup path as the GitHub Action entrypoint, but with `publish=False` enforced.

```bash
python scripts/monitoring/monitoring_dry_run.py [config_file] [--summary-only]
```

Behavior:

- Fetches source items and runs evaluation normally.
- Respects deduplication (`DEDUPLICATE=true/false`).
- Does **not** push to DB.
- Does **not** publish to Langfuse experiment/observability.

Examples:

```bash
# Mimic Athena Langfuse monitor, no publishing side effects
python scripts/monitoring/monitoring_dry_run.py monitoring_langfuse

# Use a full path and print JSON summary only
python scripts/monitoring/monitoring_dry_run.py src/eval_workbench/implementations/athena/config/monitoring_langfuse.yaml --summary-only
```

---

## create_evaluation_tables.py

Creates the core evaluation persistence tables and a joined view.

```bash
python scripts/create_evaluation_tables.py
```

**Tables created:**

**`evaluation_dataset`** — Stores dataset items for evaluation.

| Column | Type | Description |
|--------|------|-------------|
| `dataset_id` | TEXT PK | Unique identifier |
| `query` | TEXT | Input query |
| `expected_output` | TEXT | Ground truth |
| `actual_output` | TEXT | Model output |
| `conversation` | JSONB | Full conversation |
| `additional_input` | JSONB | Extra context |
| `source_type` | TEXT | Data source type |
| `environment` | TEXT | Production/staging |
| `trace_id` | TEXT | Langfuse trace ID |
| `created_at` | TIMESTAMPTZ | Row creation time |

**`evaluation_results`** — Stores metric evaluation results.

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | TEXT | Evaluation run identifier |
| `dataset_id` | TEXT FK | Foreign key to evaluation_dataset |
| `metric_name` | TEXT | Name of metric |
| `metric_score` | DOUBLE | Numeric score |
| `passed` | BOOLEAN | Whether metric passed |
| `explanation` | TEXT | Score explanation |
| `signals` | JSONB | Detailed signal data |
| `model_name` | TEXT | LLM model used |
| `timestamp` | TIMESTAMPTZ | Evaluation time |

Primary key: `(run_id, dataset_id, metric_name)`

**`evaluation_view`** — Joined view combining both tables.

**Requires:** `DATABASE_URL` environment variable.

---

## create_athena_kpi_objects.py

Creates the `agent_kpi_logs` EAV (Entity-Attribute-Value) table for storing agent KPI observations.

```bash
python scripts/create_athena_kpi_objects.py
```

**Table: `agent_kpi_logs`**

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | UUID |
| `created_at` | TIMESTAMPTZ | Observation timestamp |
| `source_name` | TEXT | Source (e.g., `athena`) |
| `kpi_name` | TEXT | KPI name (e.g., `stp_rate`) |
| `kpi_category` | TEXT | Category (see below) |
| `dataset_id` | TEXT | Reference to specific case |
| `numeric_value` | DOUBLE | KPI numeric value |
| `text_value` | TEXT | KPI text value |
| `json_value` | JSONB | KPI JSON value |
| `source_component` | TEXT | Component (e.g., `underwriter`) |
| `source_step` | TEXT | Pipeline step |
| `environment` | TEXT | Defaults to `production` |
| `tags` | JSONB | Flexible tagging |
| `metadata` | JSONB | Additional metadata |

**KPI Categories:** `operational_efficiency`, `risk_accuracy`, `data_integrity`, `commercial_impact`

**Indexes:** Composite on `(source_name, kpi_name)`, `kpi_category`, `dataset_id`, `created_at`, and a time-series index on `(source_name, kpi_name, created_at DESC)`.

**Requires:** `DATABASE_URL` environment variable.

---

## create_rule_extractions_table.py

Creates the `rule_extractions` table for storing structured rules extracted by the memory pipeline.

```bash
python scripts/create_rule_extractions_table.py
```

**Table: `rule_extractions`**

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT PK | Unique identifier |
| `created_at` | TIMESTAMPTZ | Row creation time |
| `batch_id` | TEXT | Batch identifier |
| `agent_name` | TEXT | Agent (defaults to `athena`) |
| `raw_text` | TEXT | Original text content |
| `raw_text_hash` | TEXT | Hash for deduplication |
| `risk_factor` | TEXT | Risk factor name |
| `risk_category` | TEXT | Risk category |
| `rule_name` | TEXT | Rule name |
| `product_type` | TEXT | Product type |
| `action` | TEXT | Rule action |
| `outcome_description` | TEXT | Expected outcome |
| `mitigants` | JSONB | Mitigating factors |
| `threshold` | JSONB | Threshold definitions |
| `data_fields` | JSONB | Required data fields |
| `ingestion_status` | TEXT | pending / processed / error |

**Indexes:** On `ingestion_status`, `agent_name`, `batch_id`, `rule_name`, and composite `(risk_factor, product_type)`.

**Requires:** `DATABASE_URL` environment variable.

---

## populate_athena_kpis.py

Populates `agent_kpi_logs` by reading from `athena_cases` and `evaluation_results`, then inserting deduplicated KPI observations.

```bash
python scripts/populate_athena_kpis.py
```

```bash
# Scheduled-style backfill window
python scripts/populate_athena_kpis.py --since-days 2
```

**KPIs computed:**

| KPI | Category | Source | Logic |
|-----|----------|--------|-------|
| `stp_rate` | operational_efficiency | athena_cases | 1.0 if no UW flags, else 0.0 |
| `time_to_quote` | operational_efficiency | athena_cases | `executionTimeMs / 1000` |
| `referral_rate` | operational_efficiency | athena_cases | 1.0 if UW flags present |
| `bindable_quote_rate` | commercial_impact | athena_cases | 1.0 if no UW flags |
| `decision_variance` | risk_accuracy | evaluation_results | 0.0 if outcome matches, else 1.0 |
| `referral_accuracy` | risk_accuracy | evaluation_results | metric_score from Refer Reason |
| `faithfulness_score` | data_integrity | evaluation_results | metric_score from UW Faithfulness |
| `hallucination_count` | data_integrity | evaluation_results | Count from signals JSONB |

**Behavior:**

- Reads from source database, writes to target database
- `incremental` mode (default): inserts only new `(dataset_id, kpi_name)` rows
- `backfill` mode: overwrites existing `(dataset_id, kpi_name)` rows in the lookback window
- Safe to re-run (idempotent)
- Default lookback: 30 days

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--since-days` | `30` | Lookback window for source rows |
| `--source-database-url` | env fallback | Source DB URL (falls back to `SOURCE_DATABASE_URL`, then `DATABASE_URL`) |
| `--target-database-url` | env fallback | Target DB URL (falls back to `TARGET_DATABASE_URL`, then `DATABASE_URL`) |
| `--write-mode` | `incremental` | `incremental` or `backfill` |

**Requires:** `DATABASE_URL` environment variable (or separate source/target URLs).

### Scheduled automation (GitHub Actions)

Workflow file: `.github/workflows/populate_athena_kpis.yml`

- Cron: daily at 00:00 UTC
- Scheduled runs are gated by repo variable `KPI_POPULATION_ENABLED=true`
- Manual trigger available via **Run workflow**
- Manual trigger supports `write_mode` (`incremental` or `backfill`)
- Secrets:
  - `DATABASE_URL` for same source/target DB, or
  - `SOURCE_DATABASE_URL` + `TARGET_DATABASE_URL` for split DBs
