# Evaluation Schema

This document describes the database schema used to store evaluation datasets and metric results. The schema follows a **3-stage data model**: raw input data flows into `evaluation_dataset`, metric scores flow into `evaluation_results`, and the `evaluation_view` joins both for unified analysis.

## Data Model Overview

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

**Stage 1: Dataset** — Each evaluation item (prompt, response, context) gets a unique `dataset_id`. This is the ground truth input.

**Stage 2: Results** — Each metric evaluation produces a row keyed by `(run_id, dataset_id, metric_name)`. A single dataset item can have many metric scores across multiple runs.

**Stage 3: View** — The view joins results back to their dataset context, enabling queries like "show me all failed metrics with their original prompts."

---

## evaluation_dataset

Stores the input data for evaluation: queries, responses, context, and ground truth.

**Primary Key:** `dataset_id`

### Field Reference

| Column | Type | Description |
|--------|------|-------------|
| `dataset_id` | TEXT | Unique identifier for the evaluation record |
| `query` | TEXT | The user's input/prompt |
| `expected_output` | TEXT | Reference/expected output (ground truth) |
| `actual_output` | TEXT | System's generated response |
| `additional_input` | JSONB | Extra context inputs (business logic, flags, data) |
| `acceptance_criteria` | JSONB | User-defined definitions of acceptable responses |
| `dataset_metadata` | JSONB | Additional metadata for the input data source |
| `user_tags` | JSONB | Custom tags applied to the record |
| `created_at` | TIMESTAMPTZ | Record creation timestamp |
| `conversation` | JSONB | Multi-turn conversation structure |
| `source_type` | TEXT | Source channel (e.g., `'slack'`, `'online'`, `'langfuse'`) |
| `environment` | TEXT | Execution environment (e.g., `'production'`, `'staging'`) |
| `source_name` | TEXT | Name of the agent/system being evaluated |
| `source_component` | TEXT | Specific component of the agent |
| `tools_called` | JSONB | Tools actually invoked by the agent |
| `expected_tools` | JSONB | Tools that should have been called |
| `retrieved_content` | JSONB | RAG context chunks retrieved |
| `judgment` | JSONB | Binary/categorical evaluation decision |
| `critique` | JSONB | Detailed reasoning for the judgment |
| `trace` | JSONB | Full execution trace information |
| `additional_output` | JSONB | Extra outputs (debug info, side effects) |
| `document_text` | TEXT | Full text of processed document |
| `actual_reference` | JSONB | Ranked list of retrieved documents |
| `expected_reference` | JSONB | Ground truth reference ranking |
| `latency` | DOUBLE PRECISION | Response time in seconds |
| `trace_id` | TEXT | Link to distributed trace (Datadog, Langfuse, etc.) |
| `observation_id` | TEXT | ID of specific observation/span evaluated |
| `has_errors` | BOOLEAN | Whether tool errors occurred |

---

## evaluation_results

Stores metric scores and metadata from evaluation runs.

**Primary Key:** `(run_id, dataset_id, metric_name)` (composite)

**Foreign Key:** `dataset_id` → `evaluation_dataset.dataset_id`

### Field Reference

| Column | Type | Description |
|--------|------|-------------|
| `run_id` | TEXT | Unique ID for the evaluation run |
| `dataset_id` | TEXT | Reference to the dataset item (FK) |
| `metric_name` | TEXT | Name of the metric evaluated |
| `metric_score` | DOUBLE PRECISION | Computed score value |
| `passed` | BOOLEAN | Whether score met threshold |
| `explanation` | TEXT | Justification for the score |
| `metric_type` | TEXT | Type of node (`'metric'`, `'component'`) |
| `metric_category` | TEXT | Category (`'score'`, `'analysis'`, `'classification'`) |
| `threshold` | DOUBLE PRECISION | Pass/fail cutoff value |
| `signals` | JSONB | Granular breakdown/intermediate signals |
| `metric_id` | TEXT | Unique identifier for this metric evaluation |
| `parent` | TEXT | Parent component name (hierarchical evals) |
| `weight` | DOUBLE PRECISION | Weight relative to siblings |
| `source` | TEXT | Source of metric (evaluator class) |
| `metric_metadata` | JSONB | Structured metadata (token usage, etc.) |
| `evaluation_name` | TEXT | Name of experiment/test campaign |
| `eval_mode` | TEXT | Evaluation mode |
| `cost_estimate` | DOUBLE PRECISION | Estimated cost of running this metric |
| `model_name` | TEXT | LLM used for evaluation |
| `llm_provider` | TEXT | Provider of the LLM |
| `timestamp` | TIMESTAMPTZ | When evaluation occurred |
| `evaluation_metadata` | JSONB | Run-level metadata |
| `version` | TEXT | Version of metric logic |

---

## evaluation_view

A database view that joins `evaluation_results` with `evaluation_dataset` on `dataset_id`, providing a unified table for analysis.

```sql
-- Conceptual definition
CREATE VIEW evaluation_view AS
SELECT *
FROM evaluation_results r
LEFT JOIN evaluation_dataset d ON r.dataset_id = d.dataset_id;
```

Use `evaluation_view` when you need both metric scores and input context in the same query.

### Columns from Dataset (Context)

All 28 columns from `evaluation_dataset` are available, providing the original input, expected output, and metadata for each evaluated item.

### Columns from Results (Metrics)

All 23 columns from `evaluation_results` are available, providing scores, thresholds, explanations, and run metadata.

!!! note "Column Primary Key"
    Both tables have `dataset_id`. In the view, only one is kept (from `evaluation_results`). Use explicit column references if needed.

---