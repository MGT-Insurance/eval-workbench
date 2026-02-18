# Graph Ingestion Pipeline

<div style="background: linear-gradient(135deg, #1E3A5F 0%, #0F2440 100%); padding: 24px; border-radius: 12px; color: white; margin: 20px 0;">

<p style="margin: 0; font-size: 16px; line-height: 1.6;">
<strong>Production-grade ingestion pipeline for populating the knowledge graph from historical data and live Slack conversations.</strong> Extracts underwriting rules from CSV retrospectives, human feedback, and monitored Slack threads — producing <em>proposals</em> that land in a review queue before touching the graph. Built for safety: every output is a pending proposal, never an automatic graph write.
</p>
</div>

## Overview

The pipeline has two independent ingestion paths plus a shared review-and-ingest workflow:

<div class="rule-grid">

<div class="rule-card">
<span class="rule-card__number">A</span>
<p class="rule-card__title">Batch CSV Ingestion</p>
<p class="rule-card__desc">Extract rules from 517 KB entries + 472 agent learnings CSVs. Triage, filter operational noise, run LLM extraction, save as proposals.</p>
</div>

<div class="rule-card">
<span class="rule-card__number">B</span>
<p class="rule-card__title">Ongoing Slack Hook</p>
<p class="rule-card__desc">After each monitoring run, extract rules from intervention/escalation threads and store ProductAnalyzer graph hints. Two paths, one hook.</p>
</div>

<div class="rule-card">
<span class="rule-card__number">R</span>
<p class="rule-card__title">Review & Ingest</p>
<p class="rule-card__desc">All proposals land as <code>pending_review</code>. Human review approves/rejects. Approved proposals are ingested into the graph via <code>run_from_db()</code>.</p>
</div>

</div>

---

## Key Design Principle

<div markdown style="background: rgba(30, 58, 95, 0.1); border-left: 3px solid #1E3A5F; padding: 16px; margin: 20px 0;">

**Proposals first, graph writes never automatic.** Every extraction — whether from a CSV retrospective or a live Slack thread — creates rows in `rule_extractions` with `review_status='pending_review'` and `ingestion_status='pending'`. Graph ingestion only happens after human review, via `run_from_db()` which filters on `review_status='approved'`.

</div>

This means:

- No operational noise, low-quality learnings, or hallucinated rules reach the graph
- Every graph mutation is traceable to a reviewed proposal with provenance
- Re-running extraction is safe (identity-based dedup, append-only with `superseded_by`)

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PART A: Batch CSV                                │
│                                                                         │
│  knowledge_base_entries.csv ──┐                                         │
│  agent_learnings.csv ─────────┤                                         │
│                               ▼                                         │
│                         ┌──────────┐    ┌───────────────┐               │
│                         │  Triage  │───▶│  extractable  │               │
│                         │  entry() │    │  _input()     │               │
│                         └──────────┘    └───────┬───────┘               │
│                                                 │                       │
│                           uw_rule ──────────────┤                       │
│                           data_quality ─────────┤                       │
│                           mixed ────────────────┤                       │
│                           operational ──▶ SKIP  │                       │
│                                                 ▼                       │
│                                        ┌────────────────┐               │
│                                        │ RuleExtractor  │               │
│                                        │  (gpt-4o LLM)  │               │
│                                        └───────┬────────┘               │
│                                                │                        │
│                                                ▼                        │
│                                   ┌────────────────────────┐            │
│                                   │ Post-extraction filter │            │
│                                   │ + Metadata override    │            │
│                                   │ + Confidence scoring   │            │
│                                   └───────────┬────────────┘            │
└───────────────────────────────────────────────┼─────────────────────────┘
                                                │
                                                ▼
                                  ┌──────────────────────────┐
                                  │   rule_extractions DB    │
                                  │                          │
                                  │  review_status =         │
                                  │    'pending_review'      │
                                  │  ingestion_status =      │
                                  │    'pending'             │
                                  │                          │
                                  └──────────┬───────────────┘
                                             │
┌────────────────────────────────────────────┼─────────────────────────────┐
│                     PART B: Slack Hook     │                             │
│                                            │                             │
│  monitor.run_async() → results             │                             │
│            │                               │                             │
│            ▼                               │                             │
│  ┌─────────────────┐  ┌───────────────┐    │                             │
│  │ Path 1: Graph   │  │ Path 2: Rule  │    │                             │
│  │ hints (no LLM)  │  │ extraction    │    │                             │
│  │                 │  │ (LLM + gates) │    │                             │
│  └────────┬────────┘  └───────┬───────┘    │                             │
│           │                   │            │                             │
│           └─────────┬─────────┘            │                             │
│                     │                      │                             │
│                     ▼                      │                             │
│              save_extractions() ──────────▶│                             │
└────────────────────────────────────────────┼─────────────────────────────┘
                                             │
                                             ▼
                                  ┌──────────────────────────┐
                                  │     Human Review         │
                                  │  approve / reject /      │
                                  │  defer                   │
                                  └──────────┬───────────────┘
                                             │
                                             ▼
                                  ┌──────────────────────────┐
                                  │   run_from_db()          │
                                  │   WHERE review_status =  │
                                  │     'approved'           │
                                  │   AND ingestion_status = │
                                  │     'pending'            │
                                  │   AND superseded_by IS   │
                                  │     NULL                 │
                                  └──────────┬───────────────┘
                                             │
                                             ▼
                                  ┌──────────────────────────┐
                                  │   Knowledge Graph        │
                                  │   (Zep / FalkorDB)       │
                                  └──────────────────────────┘
```

---

## Part A: Batch Ingestion

Extracts underwriting rules from two Neon DB tables (or CSV dumps as fallback):

| Source | DB Table / CSV | Rows | Description |
|--------|----------------|------|-------------|
| **Knowledge Base** | `knowledge_base_entries` | 517 | 472 retrospectives + 45 human feedback |
| **Agent Learnings** | `agent_learnings` | 472 | Structured retrospectives with rich metadata |

In production, the script reads directly from the Neon DB tables. CSV dumps can be used as a fallback via the `--from-csv` flag (useful for local development or when DB access is unavailable).

### Step 1: Preview

Before running any LLM calls, generate an ingestion index showing what would be processed:

```bash
# From DB (production default)
python scripts/ingest_knowledge_sources.py --preview

# From CSV dumps (fallback)
python scripts/ingest_knowledge_sources.py --preview --from-csv
```

Output shows triage counts per category, estimated LLM calls, and skipped entries (rejected, error model, non-athena agent).

### Step 2: Triage

Each entry is classified into one of five categories based on keyword scanning of the KB content and executive summary:

| Category | Action | Example Content |
|----------|--------|-----------------|
| `uw_rule` | Extract via LLM | "Crime limits standard is $1M for retail" |
| `data_quality` | Extract as `risk_category='data_quality'` | "Magic Dust discrepancy > 20%" |
| `operational` | **Skip** — no extraction | "OMI bulk upload timeout" |
| `mixed` | Extract UW content only (per-bullet filtering) | "Crime limits + OMI upload issue" |
| `triage_uncertain` | Include in preview for human review | Sparse learnings + partial alignment |

**Operational marker detection** uses a three-tier system configured in `ingestion_config.yaml`:

1. **Keywords** — `omi`, `socotra`, `bulk upload`, `deployment`, etc.
2. **Regex patterns** — `\bapi\s+(error|500|timeout|failure)\b`, etc.
3. **UW allowlist** — `acv`, `deductible`, `premium`, `wind exclusion`, etc.

The allowlist protects legitimate underwriting terms from being filtered. If text contains a UW term, only a strong regex pattern match will filter it — keyword alone is insufficient.

```python
# From ingest_knowledge_sources.py
def _has_operational_markers(text: str, config: dict) -> bool:
    # ...
    if has_uw_term:
        return has_pattern  # Strong match only
    return has_keyword or has_pattern
```

### Step 3: Build Extraction Input

Not all retrospective fields go to the LLM. Only extractable content is included:

**Included** (fed to extractor):

- KB `content` text
- `executiveSummary` (if it passes operational marker check)
- `keyLearnings` (per-bullet filtered)
- `futureRecommendations` (per-bullet filtered)

**Excluded** (stored as provenance only):

- `complexityFactors`, `dataGaps`, `missingFields`
- `timingAnalysis`, `stepBreakdown`, `bottlenecks`
- `workloadAnalysis`, `recommendationUpdates`

Per-bullet filtering means each `keyLearning` and `futureRecommendation` is checked individually. Operational bullets are excluded from extraction input but stored in provenance.

### Step 4: LLM Extraction

Each entry is one LLM call to `RuleExtractor` (gpt-4o). The extractor returns structured rule dicts with risk factors, thresholds, actions, mitigants, and more. See the [Knowledge Graph Memory](memory.md#stage-2-llm-extraction) guide for extraction details.

~490 entries = ~490 LLM calls, ~$5-8 estimated cost, ~60-90 min with rate limiting.

### Step 5: Post-Extraction Filtering & Metadata Override

After extraction, two things happen:

**Operational leakage check** — scans the extracted rule's `risk_factor`, `rule_name`, `outcome_description`, `historical_exceptions`, `source`, and `data_fields` for operational markers that survived input filtering. Leaked rules are tagged `source_category='operational_extracted'` (not discarded — available for auditing).

**Metadata override from CSV ground truth** — replaces LLM-inferred fields with known values:

- `source_type`: `sme` (human feedback) or `production` (retrospective)
- `decision_quality`: from `content_json.retrospective.decisionQuality.alignment`
- `product_type`: from `content_json.productType`

### Step 6: Confidence Scoring

Confidence is a composite of three factors:

| Factor | High | Medium | Low |
|--------|------|--------|-----|
| `source_quality` | approved SME feedback | production aligned | production partial, pending |
| `triage_category` | uw_rule | data_quality | mixed |
| `extraction_specificity` | has threshold + action + risk_factor | has action + risk_factor | vague/missing fields |

The factors are stored as `confidence_factors` JSONB alongside the confidence level, so reviewers can see *why* something is low/medium/high:

```json
{
  "source_quality": "approved_sme",
  "triage_category": "uw_rule",
  "extraction_specificity": "high",
  "has_threshold": true,
  "has_action": true,
  "has_risk_factor": true
}
```

### Step 7: Save Proposals

All extracted rules are saved to `rule_extractions` with full provenance:

```python
save_extractions(
    db, rules,
    batch_id=batch_id,
    agent_name='athena',
    raw_text=text,
    provenance={
        'kb_entry_id': kb_id,
        'learning_id': learning_id,
        'source_dataset': 'retrospective',
        'source_category': 'uw_rule',
        'proposal_kind': 'extracted_rule',
        'extractor_version': ext_version,
        'confidence_factors': factors,
        'review_status': 'pending_review',
    },
)
```

### CLI Reference

```bash
# Preview — reads from Neon DB, no LLM calls
python scripts/ingest_knowledge_sources.py --preview

# Dry run — extracts first 5 per category, inspect quality
python scripts/ingest_knowledge_sources.py --dry-run

# Full run — all entries from DB
python scripts/ingest_knowledge_sources.py --run

# Read from CSV dumps instead of DB
python scripts/ingest_knowledge_sources.py --run --from-csv

# Custom CSV paths (implies --from-csv)
python scripts/ingest_knowledge_sources.py --run \
  --kb-csv data/knowledge_base_entries.csv \
  --learnings-csv data/agent_learnings.csv

# Custom config
python scripts/ingest_knowledge_sources.py --run --config path/to/config.yaml
```

### Extraction Priority

Entries are processed in priority order based on `decisionQuality.alignment`:

1. **Approved human feedback** (highest value — SME-validated)
2. **Divergent** retrospectives (Athena and UW disagreed — reveals missing rules and mitigants)
3. **Partial** retrospectives (severity or solution was wrong)
4. **Aligned** retrospectives (validates existing knowledge)
5. **Pending** feedback

---

## Part B: Ongoing Slack Hook

After each monitoring run, the graph hook extracts proposals from monitored Slack threads. It hooks into the monitoring pipeline via `monitoring_entrypoint.py`, gated behind the `GRAPH_INGESTION=true` environment variable.

### How It Integrates

```python
# In monitoring_entrypoint.py — the hook runs after monitor.run_async()
results = await monitor.run_async(deduplicate=deduplicate, publish=True)

if graph_ingestion:
    from eval_workbench.shared.monitoring.graph_hook import post_process_for_graph

    with NeonConnection() as db:
        graph_stats = post_process_for_graph(results, db)
```

The hook leverages signals **already computed** by the monitoring pipeline's four sub-analyzers:

| Analyzer | Signals Used | Purpose |
|----------|-------------|---------|
| **ObjectiveAnalyzer** | `intervention_type`, `is_escalated`, `resolution_status` | Hard gate filtering |
| **SubjectiveAnalyzer** | `override_type`, `frustration_cause` | Metadata override |
| **FeedbackAttributionAnalyzer** | `failed_step`, `attribution_confidence` | Routing (rule vs data vs system) |
| **ProductAnalyzer** | `graph_hints[]`, `learnings[]`, `learning_categories[]` | Direct proposal input |

These signals are NOT re-extracted — they're queried from the already-computed evaluation results.

### Path 1: Graph Hints (No LLM)

`graph_hints` from the ProductAnalyzer are **already structured proposals**:

```python
class GraphHint:
    target_node_type: 'rule' | 'guardrail' | 'exception' | 'other'
    suggested_action: 'add' | 'edit' | 'remove' | 'review'
    rule_name_hint: Optional[str]
```

Each hint is stored verbatim as a `proposal_kind='graph_hint'` row with thread evidence from the ProductAnalyzer's `learnings` field. No LLM extraction needed.

### Path 2: Rule Extraction (LLM)

For threads that triggered intervention but may not have produced graph hints, rules are extracted via LLM.

**Hard gates** (ALL must pass):

| Gate | Condition | Rationale |
|------|-----------|-----------|
| 1 | Thread has an Athena recommendation | No recommendation = no rule to attribute |
| 2 | `has_intervention == True` OR `is_escalated == True` | Only threads with human corrections |
| 3 | `intervention_type` NOT in `{approval, support, clarification}` | These don't indicate rule problems |
| 4 | `failed_step` NOT in `{data_integrity_failure, system_tooling_failure, chat_interface}` | Not graph-actionable |

Threads that pass all gates are formatted with computed signals as context:

```
Source: Live Slack Conversation (Intervention Detected)
Channel: C12345
Thread: 1234567.890
Intervention Type: correction (from ObjectiveAnalyzer)
Override Type: full_override (from SubjectiveAnalyzer)
Frustration Cause: incorrect_recommendation (from SubjectiveAnalyzer)
Attribution: rule_engine_failure @ high (from FeedbackAttribution)

--- Thread Conversation ---
[Turn 0] athena: Based on the submission...
[Turn 1] underwriter: Actually, that's not right...

--- Intervention Context ---
Underwriter corrected the recommendation...
```

**Evidence snippets** are extracted deterministically from messages around the intervention point using correction keywords (`correct`, `actually`, `should be`, `override`, `decline`, `approve`).

**Post-extraction filters** apply the same operational marker check and additionally tag vague process rules (`risk_category='process'` + `action='verify'` + no threshold) as `source_category='process_from_slack'`.

### Enabling the Slack Hook

```bash
# Set env var before running monitoring
export GRAPH_INGESTION=true

# Run monitoring (the hook fires automatically)
python scripts/monitoring/monitoring_entrypoint.py monitoring_slack.yaml
```

---

## Provenance Model

Every proposal row carries full provenance tracking:

### Identity Keys

| Source | Identity Key | Purpose |
|--------|-------------|---------|
| KB feedback | `kb_entry_id` | 1:1 with knowledge base entry |
| Retrospectives | `learning_id` | 1:1 with agent_learnings row |
| Slack threads | `(slack_channel_id, slack_thread_ts)` | Per-thread tracking |

### Provenance Columns

| Column | Type | Description |
|--------|------|-------------|
| `kb_entry_id` | VARCHAR | Knowledge base entry ID |
| `learning_id` | VARCHAR | Agent learnings row ID |
| `source_dataset` | VARCHAR | `kb_feedback`, `retrospective`, `slack_thread`, `slack_graph_hint` |
| `source_category` | VARCHAR | `uw_rule`, `data_quality`, `operational`, `mixed`, etc. |
| `proposal_kind` | VARCHAR | `extracted_rule`, `graph_hint`, `mutation_proposal` |
| `approval_status` | VARCHAR | KB entry status (`approved`, `pending`, `rejected`) |
| `slack_channel_id` | VARCHAR | Slack channel ID |
| `slack_thread_ts` | VARCHAR | Slack thread timestamp |
| `langfuse_trace_id` | VARCHAR | Langfuse trace ID when available |
| `extractor_version` | VARCHAR | Hash of extractor prompt + model (first 12 chars of SHA-256) |
| `superseded_by` | VARCHAR | ID of newer extraction row (NULL = current) |
| `confidence_factors` | JSONB | Structured confidence breakdown |
| `evidence_snippet` | TEXT | Quoted spans justifying the extraction |

### Lifecycle Columns

| Column | Type | Description |
|--------|------|-------------|
| `review_status` | VARCHAR | `pending_review` → `approved` / `rejected` / `deferred` |
| `reviewed_by` | VARCHAR | Reviewer identifier |
| `reviewed_at` | TIMESTAMPTZ | Review timestamp |
| `ingestion_status` | VARCHAR | `pending` → `ingested` / `failed` |
| `ingested_at` | TIMESTAMPTZ | Graph ingestion timestamp |

### Two Separate Lifecycles

```
review_status:     pending_review → approved / rejected / deferred
                                      │
                                      ▼ (only approved rows)
ingestion_status:  pending ──────────────────→ ingested / failed
```

`review_status` tracks human approval. `ingestion_status` tracks graph writes. They are independent — a row can be `review_status='approved'` but `ingestion_status='pending'` (approved, waiting for ingestion batch).

---

## Deduplication

### Identity-Based Dedup (primary)

When the same source is reprocessed (e.g., re-running the CSV script with an updated extractor), the system uses identity keys to detect existing rows:

- **Same identity + same extractor version** → skip (idempotent re-run)
- **Same identity + different extractor version** → INSERT new row, mark old row `superseded_by = new_row_id`

Both rows are retained (append-only). Old rows are never deleted or updated.

```python
# Check for existing extractions with the same identity
existing = find_existing_by_identity(
    db,
    kb_entry_id=kb_id,
    learning_id=learning_id,
    extractor_version=ext_version,
)

# Same version? Skip.
same_version = [e for e in existing if e.get('extractor_version') == ext_version]
if same_version:
    continue  # Idempotent

# Different version? Supersede.
new_ids = save_extractions(db, rules, ...)
if existing and new_ids:
    supersede_rows(db, [e['id'] for e in existing], new_ids[0])
```

### Hash-Based Dedup (secondary safety net)

`raw_text_hash` (SHA-256 of input text) remains as an integrity check. Catches cases where identity keys match but input text was reformatted.

### Superseded Scoping

Supersession chains are per-identity, not global:

- KB feedback: within `kb_entry_id`
- Retrospectives: within `learning_id`
- Slack threads: within `(slack_channel_id, slack_thread_ts)`

---

## Enums

All status fields are enforced via Python string enums (DB stores VARCHAR):

```python
class ReviewStatus(str, Enum):
    PENDING_REVIEW = 'pending_review'
    APPROVED = 'approved'
    REJECTED = 'rejected'
    DEFERRED = 'deferred'

class ProposalKind(str, Enum):
    EXTRACTED_RULE = 'extracted_rule'
    GRAPH_HINT = 'graph_hint'
    MUTATION_PROPOSAL = 'mutation_proposal'

class IngestionStatus(str, Enum):
    PENDING = 'pending'
    INGESTED = 'ingested'
    FAILED = 'failed'

class SourceDataset(str, Enum):
    KB_FEEDBACK = 'kb_feedback'
    RETROSPECTIVE = 'retrospective'
    SLACK_THREAD = 'slack_thread'
    SLACK_GRAPH_HINT = 'slack_graph_hint'
    MANUAL_SEED = 'manual_seed'

class SourceCategory(str, Enum):
    UW_RULE = 'uw_rule'
    DATA_QUALITY = 'data_quality'
    OPERATIONAL = 'operational'
    MIXED = 'mixed'
    TRIAGE_UNCERTAIN = 'triage_uncertain'
    PROCESS_FROM_SLACK = 'process_from_slack'
    OPERATIONAL_EXTRACTED = 'operational_extracted'
```

---

## Review & Graph Ingestion

### Reviewing Proposals

Use `mark_reviewed()` to approve or reject proposals:

```python
from eval_workbench.shared.memory import mark_reviewed, ReviewStatus

# Approve a proposal
mark_reviewed(db, rule_id, review_status=ReviewStatus.APPROVED, reviewed_by='matt')

# Reject a proposal
mark_reviewed(db, rule_id, review_status=ReviewStatus.REJECTED, reviewed_by='matt')

# Defer for later
mark_reviewed(db, rule_id, review_status=ReviewStatus.DEFERRED, reviewed_by='matt')
```

### Ingesting Approved Proposals

After review, ingest approved proposals into the knowledge graph:

```python
from eval_workbench.implementations.athena.memory import AthenaRulePipeline

pipeline = AthenaRulePipeline(store=store, db=db)

# Ingest only approved, non-superseded proposals (default)
result = pipeline.run_from_db(require_review=True)

# Legacy mode: ingest all pending (ignores review_status)
result = pipeline.run_from_db(require_review=False)
```

The `run_from_db(require_review=True)` query:

```sql
SELECT * FROM rule_extractions
 WHERE review_status = 'approved'
   AND ingestion_status = 'pending'
   AND superseded_by IS NULL
   AND agent_name = %s
 ORDER BY created_at
```

### Pipeline persist_only Mode

Both Part A and Part B use the pipeline in `persist_only` mode — extraction + DB persistence happen, but graph ingestion is skipped:

```python
# persist_only=True: save to DB, skip graph writes
result = pipeline.run(
    raw_text,
    persist_only=True,      # No graph ingestion
    provenance=provenance,  # Provenance fields attached to all rules
)
```

---

## Schema Migration

Before first use, add provenance columns to the existing `rule_extractions` table:

```bash
python scripts/create_rule_extractions_table.py --migrate
```

Or create the table from scratch (includes all provenance columns):

```bash
python scripts/create_rule_extractions_table.py
```

The migration adds 16 columns using `ADD COLUMN IF NOT EXISTS` (safe to re-run) and creates 5 new indexes for review, identity keys, and source dataset filtering.

---

## Configuration

All filtering, triage, and extraction behavior is driven by `ingestion_config.yaml`:

```
src/eval_workbench/implementations/athena/config/ingestion_config.yaml
```

### Key Sections

**`operational_markers`** — Keywords, regex patterns, and UW allowlist for filtering operational content.

**`triage`** — Alignment values that trigger `triage_uncertain` classification.

**`extraction_priority`** — Order for processing retrospectives (`divergent` first).

**`slack_extraction_gates`** — Excluded intervention types and failure steps for Path 2.

**`evidence_snippet`** — Correction keywords and context window for extracting evidence from Slack threads.

---

## Environment Variables

| Variable | Required For | Default | Description |
|----------|-------------|---------|-------------|
| `DATABASE_URL` | A, B | — | Neon PostgreSQL connection string |
| `OPENAI_API_KEY` | A, B (Path 2) | — | For LLM extraction (gpt-4o) |
| `GRAPH_INGESTION` | B | `false` | Enable Slack graph hook |
| `DEDUPLICATE` | B | `true` | Enable monitoring dedup |

---

## File Layout

```
scripts/
  ingest_knowledge_sources.py         # Part A: batch CSV ingestion CLI
  create_rule_extractions_table.py    # Schema creation + migration
  monitoring/
    monitoring_entrypoint.py          # Modified: GRAPH_INGESTION hook

src/eval_workbench/
  shared/
    memory/
      enums.py                        # ReviewStatus, ProposalKind, SourceDataset, etc.
      persistence.py                  # save_extractions, find_existing_by_identity,
                                      # supersede_rows, mark_reviewed, etc.
    monitoring/
      graph_hook.py                   # Part B: post_process_for_graph()

  implementations/athena/
    config/
      ingestion_config.yaml           # Operational markers, gates, evidence config
    memory/
      pipeline.py                     # AthenaRulePipeline (persist_only + run_from_db)
      extractors.py                   # RuleExtractor (unchanged)
```

---

## Quick Start

### 1. Migrate the database

```bash
python scripts/create_rule_extractions_table.py --migrate
```

### 2. Preview the ingestion (reads from DB)

```bash
python scripts/ingest_knowledge_sources.py --preview

# Or from CSV dumps
python scripts/ingest_knowledge_sources.py --preview --from-csv
```

### 3. Run a dry run (5 entries per category)

```bash
python scripts/ingest_knowledge_sources.py --dry-run
```

### 4. Run full extraction

```bash
python scripts/ingest_knowledge_sources.py --run
```

### 5. Review proposals

```python
from eval_workbench.shared.memory import fetch_all_extractions, mark_reviewed, ReviewStatus
from eval_workbench.shared.database.neon import NeonConnection

with NeonConnection() as db:
    proposals = fetch_all_extractions(db, agent_name='athena', limit=100)

    for p in proposals:
        print(f"{p['id'][:8]}  {p['rule_name']:<40}  "
              f"conf={p['confidence']:<6}  src={p['source_dataset']}")

        # After reviewing:
        mark_reviewed(db, p['id'],
                      review_status=ReviewStatus.APPROVED,
                      reviewed_by='matt')
```

### 6. Ingest approved proposals into the graph

```python
from eval_workbench.shared.memory import ZepGraphStore
from eval_workbench.implementations.athena.memory import AthenaRulePipeline, ATHENA_ONTOLOGY

store = ZepGraphStore(agent_name='athena', ontology=ATHENA_ONTOLOGY)

with NeonConnection() as db:
    pipeline = AthenaRulePipeline(store=store, db=db)
    result = pipeline.run_from_db(require_review=True)
    print(f'Ingested: {result.items_ingested}')
```

### 7. Enable ongoing Slack hook

```bash
export GRAPH_INGESTION=true
python scripts/monitoring/monitoring_entrypoint.py monitoring_slack.yaml
```
