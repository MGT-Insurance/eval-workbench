# Evaluation Schema

<div style="background: linear-gradient(135deg, #1E3A5F 0%, #0F2440 100%); padding: 24px; border-radius: 12px; color: white; margin: 20px 0;">

<p style="margin: 0; font-size: 16px; line-height: 1.6;">
<strong>A 3-stage data model for evaluation persistence.</strong> Raw input data flows into <code style="color: #7BB8E0;">evaluation_dataset</code>, metric scores flow into <code style="color: #7BB8E0;">evaluation_results</code>, and the <code style="color: #7BB8E0;">evaluation_view</code> joins both for unified analysis.
</p>
</div>

## Data Model

```
┌─────────────────────┐     ┌─────────────────────┐
│  evaluation_dataset │────<│  evaluation_results │
│  (Input/Ground Truth)│ 1:N │  (Metric Scores)    │
└─────────────────────┘     └─────────────────────┘
            │                         │
            └────────┬────────────────┘
                     ▼
           ┌─────────────────────┐
           │   evaluation_view   │
           │   (Joined Analysis) │
           └─────────────────────┘
```

<div class="rule-grid">
  <div class="rule-card">
    <span class="rule-card__number">1</span>
    <p class="rule-card__title">Dataset</p>
    <p class="rule-card__desc">Each evaluation item (prompt, response, context) gets a unique <code>dataset_id</code>. This is the ground truth input.</p>
  </div>
  <div class="rule-card">
    <span class="rule-card__number">2</span>
    <p class="rule-card__title">Results</p>
    <p class="rule-card__desc">Each metric evaluation produces a row keyed by <code>(run_id, dataset_id, metric_name)</code>. A single dataset item can have many metric scores across runs.</p>
  </div>
  <div class="rule-card">
    <span class="rule-card__number">3</span>
    <p class="rule-card__title">View</p>
    <p class="rule-card__desc">Joins results back to dataset context — enabling queries like "show me all failed metrics with their original prompts."</p>
  </div>
</div>

---

## evaluation_dataset

<div style="background: rgba(30, 58, 95, 0.1); border-left: 3px solid #1E3A5F; padding: 16px; margin: 20px 0;">

Stores the input data for evaluation: queries, responses, context, and ground truth. **Primary Key:** `dataset_id`
</div>

### Core Fields

| Column | Type | Description |
|--------|------|-------------|
| `dataset_id` | TEXT | Unique identifier for the evaluation record |
| `query` | TEXT | The user's input/prompt |
| `expected_output` | TEXT | Reference/expected output (ground truth) |
| `actual_output` | TEXT | System's generated response |
| `created_at` | TIMESTAMPTZ | Record creation timestamp |

### Context & Metadata

| Column | Type | Description |
|--------|------|-------------|
| `additional_input` | JSONB | Extra context inputs (business logic, flags, data) |
| `acceptance_criteria` | JSONB | User-defined definitions of acceptable responses |
| `dataset_metadata` | JSONB | Additional metadata for the input data source |
| `user_tags` | JSONB | Custom tags applied to the record |
| `conversation` | JSONB | Multi-turn conversation structure |

### Source Tracking

| Column | Type | Description |
|--------|------|-------------|
| `source_type` | TEXT | Source channel (e.g., `'slack'`, `'online'`, `'langfuse'`) |
| `environment` | TEXT | Execution environment (e.g., `'production'`, `'staging'`) |
| `source_name` | TEXT | Name of the agent/system being evaluated |
| `source_component` | TEXT | Specific component of the agent |

### Tool & Retrieval

| Column | Type | Description |
|--------|------|-------------|
| `tools_called` | JSONB | Tools actually invoked by the agent |
| `expected_tools` | JSONB | Tools that should have been called |
| `retrieved_content` | JSONB | RAG context chunks retrieved |

### Evaluation Signals

| Column | Type | Description |
|--------|------|-------------|
| `judgment` | JSONB | Binary/categorical evaluation decision |
| `critique` | JSONB | Detailed reasoning for the judgment |
| `trace` | JSONB | Full execution trace information |
| `additional_output` | JSONB | Extra outputs (debug info, side effects) |

### Document & Reference

| Column | Type | Description |
|--------|------|-------------|
| `document_text` | TEXT | Full text of processed document |
| `actual_reference` | JSONB | Ranked list of retrieved documents |
| `expected_reference` | JSONB | Ground truth reference ranking |

### Observability

| Column | Type | Description |
|--------|------|-------------|
| `latency` | DOUBLE PRECISION | Response time in seconds |
| `trace_id` | TEXT | Link to distributed trace (Datadog, Langfuse, etc.) |
| `observation_id` | TEXT | ID of specific observation/span evaluated |
| `has_errors` | BOOLEAN | Whether tool errors occurred |

---

## evaluation_results

<div style="background: rgba(30, 58, 95, 0.1); border-left: 3px solid #1E3A5F; padding: 16px; margin: 20px 0;">

Stores metric scores and metadata from evaluation runs. **Primary Key:** `(run_id, dataset_id, metric_name)` (composite). **Foreign Key:** `dataset_id` &rarr; `evaluation_dataset.dataset_id`
</div>

### Score Fields

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | TEXT | Unique ID for the evaluation run |
| `dataset_id` | TEXT | Reference to the dataset item (FK) |
| `metric_name` | TEXT | Name of the metric evaluated |
| `metric_score` | DOUBLE PRECISION | Computed score value |
| `passed` | BOOLEAN | Whether score met threshold |
| `explanation` | TEXT | Justification for the score |
| `threshold` | DOUBLE PRECISION | Pass/fail cutoff value |

### Metric Metadata

| Column | Type | Description |
|--------|------|-------------|
| `metric_type` | TEXT | Type of node (`'metric'`, `'component'`) |
| `metric_category` | TEXT | Category (`'score'`, `'analysis'`, `'classification'`) |
| `metric_id` | TEXT | Unique identifier for this metric evaluation |
| `signals` | JSONB | Granular breakdown/intermediate signals |

### Hierarchy

| Column | Type | Description |
|--------|------|-------------|
| `parent` | TEXT | Parent component name (hierarchical evals) |
| `weight` | DOUBLE PRECISION | Weight relative to siblings |
| `source` | TEXT | Source of metric (evaluator class) |

### Run Context

| Column | Type | Description |
|--------|------|-------------|
| `evaluation_name` | TEXT | Name of experiment/test campaign |
| `eval_mode` | TEXT | Evaluation mode |
| `model_name` | TEXT | LLM used for evaluation |
| `llm_provider` | TEXT | Provider of the LLM |
| `timestamp` | TIMESTAMPTZ | When evaluation occurred |

### Cost & Versioning

| Column | Type | Description |
|--------|------|-------------|
| `cost_estimate` | DOUBLE PRECISION | Estimated cost of running this metric |
| `metric_metadata` | JSONB | Structured metadata (token usage, etc.) |
| `evaluation_metadata` | JSONB | Run-level metadata |
| `version` | TEXT | Version of metric logic |

---

## evaluation_view

<div style="background: linear-gradient(135deg, #1E3A5F 0%, #0F2440 100%); padding: 24px; border-radius: 12px; color: white; margin: 20px 0;">

<p style="margin: 0; font-size: 16px; line-height: 1.6;">
A database view joining <code style="color: #7BB8E0;">evaluation_results</code> with <code style="color: #7BB8E0;">evaluation_dataset</code> on <code style="color: #7BB8E0;">dataset_id</code> — providing a unified table for analysis.
</p>
</div>

```sql
CREATE VIEW evaluation_view AS
SELECT *
FROM evaluation_results r
LEFT JOIN evaluation_dataset d ON r.dataset_id = d.dataset_id;
```

Use `evaluation_view` when you need both metric scores and input context in the same query.

<table>
<tr>
<td width="50%" valign="top">

<h3><strong>From Dataset</strong></h3>
<strong>28 columns — Context</strong>

<p>Original input, expected output, source tracking, tool usage, retrieval context, and observability fields for each evaluated item.</p>

</td>
<td width="50%" valign="top">

<h3><strong>From Results</strong></h3>
<strong>23 columns — Metrics</strong>

<p>Scores, thresholds, explanations, hierarchy, run metadata, cost estimates, and versioning for each metric evaluation.</p>

</td>
</tr>
</table>

!!! note "Column Primary Key"
    Both tables have `dataset_id`. In the view, only one is kept (from `evaluation_results`). Use explicit column references if needed.
