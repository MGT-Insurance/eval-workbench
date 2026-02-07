# Memory Dashboard Tab — Frontend Spec

## Overview

Dashboard tab for visualizing what the Athena agent has learned about underwriting rules. Two data sources:

- **Neon (Postgres)** — flat `rule_extractions` table with every extracted rule, raw source text, and ingestion status
- **Zep (Graph)** — knowledge graph with semantic relationships (RiskFactor -> Rule -> Outcome, Mitigant -> Rule)

No existing frontend or API layer. Backend is Python (FastAPI available as dependency). All data access functions exist and are documented below with exact return shapes.

---

## Page Layout

The Memory tab is a **single page** with a summary strip at the top, an optional conflict warning banner, and sub-tabs for the main content area.

```
┌─────────────────────────────────────────────────────────────┐
│  Memory                                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │
│  │ 12 Rules │ │ 11 Risk  │ │ 15 Mit-  │ │ 3 Hard   │       │
│  │          │ │ Factors  │ │ igants   │ │ Stops    │       │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │
│                                                             │
│  ┌─ small bar charts (inline, beside cards) ─────────────┐  │
│  │  rules by action: ██ decline ██ refer █ approve       │  │
│  │  rules by product: ██ ALL ██ Property █ BOP █ LRO     │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ⚠ 2 Conflicting Rules Detected (click to view)            │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ [Rules]  [Decision Quality]  [Hard Stops]  [Batches] │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │                                                      │   │
│  │  (sub-tab content area)                              │   │
│  │                                                      │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Component breakdown

| Component | Position | Always visible | Source |
|-----------|----------|---------------|--------|
| Summary cards | Top strip | Yes | `graph.summary()` + Neon status counts |
| Action/product bar charts | Top strip, beside cards | Yes | `graph.summary()` |
| Conflict warning banner | Below cards, above tabs | Only if conflicts exist | `graph.conflicting_rules()` across known risk factors |
| Sub-tabs | Main content area | Yes (one active) | See below |

### Sub-tabs

| Tab | Label | Content |
|-----|-------|---------|
| 1 | **Rules** | Filterable rules table with expandable row detail (includes inline decision path) |
| 2 | **Decision Quality** | Aligned vs divergent split view + soft thresholds |
| 3 | **Hard Stops** | Unmitigated decline rules list |
| 4 | **Batches** | Pipeline activity / batch history |

---

## Summary Cards (always visible)

**Source:** Zep — `graph.summary()` + Neon status counts

**Zep return shape:**
```json
{
  "total_nodes": 42,
  "total_edges": 58,
  "nodes_by_type": {
    "RiskFactor": 11,
    "Rule": 11,
    "Outcome": 7,
    "Mitigant": 15,
    "Source": 9
  },
  "edges_by_relation": {
    "TRIGGERS": 11,
    "RESULTS_IN": 11,
    "OVERRIDES": 18,
    "DERIVED_FROM": 9
  },
  "rules_by_action": {
    "decline": 5,
    "refer": 4,
    "approve_with_conditions": 1,
    "flag_for_review": 1
  },
  "rules_by_product": {
    "ALL": 6,
    "BOP": 2,
    "Property": 3,
    "LRO": 1
  }
}
```

**Neon status counts** (direct SQL):
```sql
SELECT ingestion_status, COUNT(*) as count
FROM rule_extractions WHERE agent_name = 'athena'
GROUP BY ingestion_status
```

**Cards to show:**
- **Rules** — `nodes_by_type.Rule` count
- **Risk Factors** — `nodes_by_type.RiskFactor` count
- **Mitigants** — `nodes_by_type.Mitigant` count
- **Hard Stops** — count from unmitigated declines (or pre-computed)

**Bar charts (small, inline):**
- Rules by action — horizontal bars, color-coded: red=decline, yellow=refer, blue=approve_with_conditions, gray=flag_for_review
- Rules by product — horizontal bars, neutral colors

---

## Conflict Warning Banner

**Source:** Zep — `graph.conflicting_rules(risk_factor)` run across known risk factors

**Behavior:**
- On page load, check for conflicts across all risk factors from `graph.list_all_rules()`
- If any conflicts found, show amber banner: "⚠ {N} Conflicting Rules Detected"
- Click banner → opens a modal or expandable section showing:

**Return shape:**
```json
[
  {
    "risk_factor": "Employee Count",
    "product_type": "BOP",
    "actions": ["decline", "refer"],
    "rules": [
      {"rule_name": "Rule A", "action": "decline", "properties": {}},
      {"rule_name": "Rule B", "action": "refer", "properties": {}}
    ]
  }
]
```

**Conflict detail layout:**
- Each conflict is a card showing: risk factor, product type, then the conflicting rules side-by-side with their actions highlighted in their badge colors

---

## Sub-tab 1: Rules

**Source:** Neon — `fetch_all_extractions(db, agent_name='athena')`

Dataset can also be read from CSV: `/Users/mattevanoff/Downloads/rule_extractions_v1.csv`

**Return shape** — `list[dict]`, each dict:
```json
{
  "id": "uuid-string",
  "created_at": "2026-02-06T12:00:00Z",
  "batch_id": "uuid-string",
  "agent_name": "athena",
  "raw_text": "Properties within 1 mile of coast require...",
  "raw_text_hash": "sha256-hex",
  "risk_factor": "Coastal Exposure",
  "risk_category": "location",
  "rule_name": "Coastal Wind Referral",
  "product_type": "Property",
  "action": "refer",
  "outcome_description": "Properties within 1 mile of coast require wind/hail review...",
  "mitigants": ["Hurricane shutters installed", "Miami-Dade rated roof"],
  "source": "Catastrophe Guidelines 2024",
  "source_type": "manual",
  "confidence": "high",
  "threshold": {"field": "distance_to_coast", "operator": "lt", "value": 1, "unit": "miles"},
  "threshold_type": "hard",
  "historical_exceptions": null,
  "decision_quality": null,
  "compound_trigger": null,
  "data_fields": null,
  "ingestion_status": "ingested",
  "ingestion_error": null,
  "ingested_at": "2026-02-06T12:00:05Z"
}
```

### Table columns

| Column | Source field | Display |
|--------|------------|---------|
| Rule Name | `rule_name` | Text, clickable to expand row |
| Risk Factor | `risk_factor` | Text |
| Action | `action` | Badge: red=`decline`, yellow=`refer`, blue=`approve_with_conditions`, gray=`flag_for_review` |
| Product | `product_type` | Text, `ALL` shown as a distinct style |
| Threshold | `threshold_type` | Badge: `hard` (solid) vs `soft` (outline) — soft means flexibility exists |
| Status | `ingestion_status` | Dot: green=`ingested`, gray=`pending`, red=`failed` |
| Confidence | `confidence` | Text or badge |
| Created | `created_at` | Relative time ("2 hours ago") |

### Filters (all optional, combinable)

- Action: multi-select (`decline`, `refer`, `approve_with_conditions`, `flag_for_review`)
- Product: multi-select (`ALL`, `BOP`, `Property`, `LRO`)
- Status: multi-select (`ingested`, `pending`, `failed`)
- Threshold type: `hard` / `soft`
- Risk category: text/select
- Batch: select dropdown
- Date range

### Expanded row detail (click a row to expand)

Two sections: **Rule Detail** and **Decision Path**.

**Rule Detail section** — fields not shown in the table:

- **Outcome Description** — full text
- **Mitigants** — bulleted list from the JSONB array
- **Threshold** — render the JSON object as human-readable: `distance_to_coast < 1 miles` (field + operator + value + unit)
- **Historical Exceptions** — italic text block, shows what UWs have actually done vs the manual
- **Source** — `source` + `source_type` badge (manual / compliance / production)
- **Decision Quality** — `aligned` vs `divergent` badge (if present)
- **Compound Trigger** — the boolean expression as code text (if present)
- **Data Fields** — code-formatted list of field paths (if present)
- **Raw Text** — collapsible/scrollable area showing the original source text
- **Ingestion Error** — red text block (only if `ingestion_status = 'failed'`)
- **IDs** — `id`, `batch_id` in small muted text

**Decision Path section** — inline graph trace for this rule:

**Source:** Zep — `graph.trace(rule.risk_factor, product_type=rule.product_type)`

Render as a horizontal flow:

```
[Coastal Exposure] --TRIGGERS--> [Coastal Wind Referral] --RESULTS_IN--> [Refer]
                                         ^
                            [Hurricane shutters installed] --OVERRIDES--
                            [Miami-Dade rated roof] --OVERRIDES--
```

- Nodes as rounded boxes, edges as labeled arrows
- Risk Factor node in blue, Rule node in orange, Outcome node in green, Mitigant nodes in purple
- Keep it compact — this is inside an expanded table row, not a full page

---

## Sub-tab 2: Decision Quality

**Source:** Neon — direct SQL

```sql
SELECT rule_name, risk_factor, action, product_type, threshold_type,
       historical_exceptions, decision_quality, outcome_description,
       threshold, mitigants
FROM rule_extractions
WHERE decision_quality IS NOT NULL
ORDER BY decision_quality, created_at DESC
```

### Layout — two-column split

**Left column: "Aligned"** (`decision_quality = 'aligned'`)
- Green section header
- Rules where the agent's learned behavior matches the manual
- Card list, each card shows:
  - Rule name
  - Action badge
  - Risk factor
  - Brief outcome description (truncated to 2 lines)

**Right column: "Divergent"** (`decision_quality = 'divergent'`)
- Orange/amber section header
- Rules where real UW decisions diverge from the manual — **these are the key insights**
- Card list, each card shows:
  - Rule name + action badge
  - `outcome_description` — what the manual says
  - `historical_exceptions` — **highlighted/emphasized** — what UWs actually did
  - `threshold` rendered as "manual says X, but UWs approved up to Y"
  - `mitigants` — what conditions enabled the divergent decisions

### Soft Thresholds section (below the split)

**Source:** Neon — direct SQL

```sql
SELECT rule_name, risk_factor, action, product_type,
       threshold, threshold_type, historical_exceptions,
       outcome_description, mitigants
FROM rule_extractions
WHERE threshold_type = 'soft'
ORDER BY created_at DESC
```

**Purpose:** Show rules where a hard number has learned flexibility. Rendered as a card list below the aligned/divergent split.

Each card:
- Rule name
- **Manual threshold:** `employee_count > 20 employees` (rendered from `threshold` JSON)
- **What actually happens:** `historical_exceptions` text
- **Mitigants that unlock flexibility:** bulleted `mitigants` list

---

## Sub-tab 3: Hard Stops

**Source:** Zep — `graph.unmitigated_declines()`

**Return shape** — `list[dict]`:
```json
[
  {
    "rule_name": "Cannabis Exclusion",
    "risk_factor": "Cannabis Operations",
    "product_type": "ALL"
  }
]
```

**Purpose:** Rules that are absolute hard stops — no mitigants, no flexibility, no overrides. The agent should never try to work around these.

**Layout:**
- Red-themed section header: "Hard Stops — No Overrides Available"
- Count badge: "{N} unmitigated decline rules"
- Card or list for each rule:
  - Rule name (bold)
  - Risk factor
  - Product type badge
  - Red "DECLINE" badge
- Optionally: link each rule to its expanded detail in the Rules tab

---

## Sub-tab 4: Batches

**Source:** Neon — direct SQL (aggregation query)

```sql
SELECT
    batch_id,
    COUNT(*) as rule_count,
    MIN(created_at) as started_at,
    SUM(CASE WHEN ingestion_status = 'ingested' THEN 1 ELSE 0 END) as ingested,
    SUM(CASE WHEN ingestion_status = 'pending' THEN 1 ELSE 0 END) as pending,
    SUM(CASE WHEN ingestion_status = 'failed' THEN 1 ELSE 0 END) as failed
FROM rule_extractions
WHERE agent_name = 'athena'
GROUP BY batch_id
ORDER BY MIN(created_at) DESC
```

**Purpose:** Pipeline run history — when the agent extracted rules, how many, success rate.

**Layout:**
- List of batches, most recent first
- Each batch row:
  - Batch ID (truncated UUID, e.g. `abc12345...`)
  - Timestamp (relative: "2 hours ago")
  - Rule count
  - Stacked progress bar: green (ingested) / gray (pending) / red (failed)
  - Percentage: "92% ingested"
- Click a batch → navigates to the **Rules** sub-tab filtered by that `batch_id`

---

## Backend Functions Reference

All functions are importable from Python. The frontend needs API endpoints wrapping these.

```python
# Neon (Postgres) — rule_extractions table
from eval_workbench.shared.database.neon import NeonConnection
from eval_workbench.shared.memory.persistence import (
    fetch_all_extractions,   # (db, agent_name='athena', limit=500) -> list[dict]
    fetch_pending,           # (db, agent_name='athena', batch_id=None) -> list[dict]
    save_extractions,        # (db, rules, batch_id, agent_name, raw_text) -> list[str]
    mark_ingested,           # (db, rule_id) -> None
    mark_failed,             # (db, rule_id, error) -> None
)

# Zep (Graph) — knowledge graph
from eval_workbench.implementations.athena.memory.analytics import AthenaGraphAnalytics

# graph = AthenaGraphAnalytics(store=store)
# graph.summary()                                          -> dict
# graph.query(input, product_type=None, limit=10)          -> list[dict]
# graph.trace(input, product_type=None, limit=25)          -> list[dict]
# graph.export(limit=500)                                  -> list[dict]
# graph.get_rule(rule_name, limit=25)                      -> dict | None
# graph.list_all_rules(limit=500)                          -> list[dict]
# graph.rules_by_action(action, limit=500)                 -> list[dict]
# graph.rules_by_product(product_type, limit=500)          -> list[dict]
# graph.uncovered_risks(limit=500)                         -> list[str]
# graph.mitigants_for_rule(rule_name, limit=25)            -> list[str]
# graph.rules_mitigated_by(mitigant, limit=25)             -> list[str]
# graph.unmitigated_declines(limit=500)                    -> list[dict]
# graph.conflicting_rules(risk_factor, limit=50)           -> list[dict]
# graph.overlapping_thresholds(field, limit=100)           -> list[dict]
# graph.evaluate(risk_factor, product_type, context, limit=25) -> list[dict]
```

### Connection setup
```python
# Neon
db = NeonConnection()  # reads DATABASE_URL from env
# use as context manager: with NeonConnection() as db: ...

# Zep
from eval_workbench.implementations.athena.memory import ATHENA_ONTOLOGY
from eval_workbench.shared.memory import ZepGraphStore

store = ZepGraphStore(agent_name='athena', ontology=ATHENA_ONTOLOGY)
graph = AthenaGraphAnalytics(store=store)
```

### Direct SQL for views not covered by existing functions
```python
# Pipeline activity (Batches sub-tab)
db.fetch_all("""
    SELECT batch_id, COUNT(*) as rule_count, MIN(created_at) as started_at,
           SUM(CASE WHEN ingestion_status = 'ingested' THEN 1 ELSE 0 END) as ingested,
           SUM(CASE WHEN ingestion_status = 'pending' THEN 1 ELSE 0 END) as pending,
           SUM(CASE WHEN ingestion_status = 'failed' THEN 1 ELSE 0 END) as failed
    FROM rule_extractions WHERE agent_name = %s
    GROUP BY batch_id ORDER BY MIN(created_at) DESC
""", ('athena',))

# Decision quality split (Decision Quality sub-tab)
db.fetch_all("""
    SELECT rule_name, risk_factor, action, product_type, threshold_type,
           historical_exceptions, decision_quality, outcome_description,
           threshold, mitigants
    FROM rule_extractions WHERE decision_quality IS NOT NULL
    ORDER BY decision_quality, created_at DESC
""")

# Soft thresholds (Decision Quality sub-tab, lower section)
db.fetch_all("""
    SELECT rule_name, risk_factor, action, product_type,
           threshold, threshold_type, historical_exceptions,
           outcome_description, mitigants
    FROM rule_extractions WHERE threshold_type = 'soft'
    ORDER BY created_at DESC
""")

# Status counts (summary cards)
db.fetch_all("""
    SELECT ingestion_status, COUNT(*) as count
    FROM rule_extractions WHERE agent_name = %s
    GROUP BY ingestion_status
""", ('athena',))
```

---

## Suggested API Endpoints

| Endpoint | Method | Backend call | Used by |
|----------|--------|-------------|---------|
| `/api/rules` | GET | `fetch_all_extractions(db)` | Rules sub-tab |
| `/api/rules?status=pending` | GET | `fetch_pending(db)` | Rules sub-tab filtered |
| `/api/rules?batch_id={id}` | GET | `fetch_all_extractions` + filter | Rules sub-tab filtered from Batches |
| `/api/rules/quality` | GET | Direct SQL (decision_quality) | Decision Quality sub-tab |
| `/api/rules/soft-thresholds` | GET | Direct SQL (threshold_type=soft) | Decision Quality sub-tab |
| `/api/batches` | GET | Direct SQL (GROUP BY batch_id) | Batches sub-tab |
| `/api/graph/summary` | GET | `graph.summary()` | Summary cards |
| `/api/graph/trace?q={risk}&product={type}` | GET | `graph.trace(q, product_type)` | Expanded row decision path |
| `/api/graph/rules` | GET | `graph.list_all_rules()` | Summary charts |
| `/api/graph/rule/{name}` | GET | `graph.get_rule(name)` | Expanded row detail |
| `/api/graph/unmitigated` | GET | `graph.unmitigated_declines()` | Hard Stops sub-tab |
| `/api/graph/conflicts?q={risk}` | GET | `graph.conflicting_rules(q)` | Conflict warning banner |
| `/api/status-counts` | GET | Direct SQL (GROUP BY status) | Summary cards |

---

## Environment Variables Required

```
MEMORY_DATABASE_URL=postgresql://...   # Neon Postgres connection
ZEP_API_KEY=...                 # Zep Cloud API key
```

---

## Build Priority

1. **Summary cards + bar charts** — top strip, always visible, quick win
2. **Rules sub-tab** — main content, filterable table with expandable rows
3. **Decision Quality sub-tab** — aligned vs divergent split + soft thresholds
4. **Hard Stops sub-tab** — small list, low effort
5. **Batches sub-tab** — pipeline history
6. **Expanded row decision path** — inline graph trace (most complex visual)
7. **Conflict warning banner** — check on page load, show if any exist
