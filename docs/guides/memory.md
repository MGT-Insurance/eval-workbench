# Knowledge Graph Memory

<div style="background: linear-gradient(135deg, #1E3A5F 0%, #0F2440 100%); padding: 24px; border-radius: 12px; color: white; margin: 20px 0;">

<p style="margin: 0; font-size: 16px; line-height: 1.6;">
<strong>Persistent knowledge graph memory for AI agents.</strong> Currently used by Athena (underwriting) to store risk factors, rules, decision logic, and their relationships — then retrieve them semantically at runtime. Built on <a href="https://www.getzep.com/" style="color: #7BB8E0;">Zep Cloud</a> Graphiti for graph storage and semantic search.
</p>
</div>

## How It Works (End-to-End)

The memory system has four stages. Raw knowledge goes in one end; structured, searchable
graph edges come out the other.

<div class="rule-grid">

<div class="rule-card">
<span class="rule-card__number">1</span>
<p class="rule-card__title">Raw Text</p>
<p class="rule-card__desc">Manuals, SME notes, training feedback, production learnings — unstructured text containing underwriting knowledge.</p>
</div>

<div class="rule-card">
<span class="rule-card__number">2</span>
<p class="rule-card__title">Extraction</p>
<p class="rule-card__desc"><code>RuleExtractor</code> (OpenAI) — LLM parses text into structured rule dicts with risk factors, thresholds, and actions.</p>
</div>

<div class="rule-card">
<span class="rule-card__number">3</span>
<p class="rule-card__title">Ingestion</p>
<p class="rule-card__desc"><code>ZepGraphStore</code> (Zep Cloud) — Edges stored in knowledge graph per agent with entity deduplication.</p>
</div>

<div class="rule-card">
<span class="rule-card__number">4</span>
<p class="rule-card__title">Retrieval</p>
<p class="rule-card__desc"><code>AthenaGraphAnalytics</code> — Semantic search returns relevant rules + graph paths for runtime decisions.</p>
</div>

</div>

---

### Stage 1: Raw Text Input

The input is unstructured text — anything that contains underwriting knowledge:

- **Underwriting manual excerpts** — formal rules and limits
- **Training session notes** — SME-taught heuristics and guidelines
- **SME interview transcripts** — tribal knowledge and known flexibility
- **Production decision logs** — the richest source: real outcomes, decision quality, learned patterns
- **Compliance bulletins** — hard regulatory constraints

Production case reviews are especially valuable because they reveal *multiple* rules
per text block, compound triggers, and cases where real decisions diverged from stated
rules (making a "hard" rule effectively soft):

```python
raw_texts = [
    # Manual excerpt — gives you the stated rule
    """Gas stations are automatically referred for senior underwriter review due to
    fuel storage liability. This applies to all LRO policies. However, if a fire
    suppression system is installed and the station is located more than 200ft from
    residential areas, the referral can be waived at the underwriter's discretion.
    Source: Underwriting Manual v3.2""",

    # Production case review — gives you compound triggers, decision quality, data fields
    """Athena correctly recommended decline for a lessor's risk warehouse with
    catastrophic recent loss ($600K representing 23.5% of building exposure),
    pricing 37% below minimum threshold, and prior carrier non-renewal. The
    combination of recent major loss + below-minimum pricing + prior carrier
    rejection is an automatic decline regardless of other positive factors.""",

    # Divergent production decision — reveals that a "hard" rule is actually soft
    """The property rate of 0.19% is below the 0.25% minimum threshold, which
    would normally trigger a referral. However, the underwriter approved because
    this was a BASIC cause of loss policy with wind/hail exclusion. The lower
    rate was appropriate for the limited coverage scope. Athena should learn
    that low property rates are often justified for BASIC CoL policies.""",

    # Data quality pattern — a non-traditional rule
    """The building exposure from Magic Dust was $2.3M vs customer input of $1.1M —
    a 109% discrepancy. The address format "404-600 Main St" indicates this may be
    a multi-location risk quoted as a single location. Flag for verification.""",
]
```

You can pass a single string, a list of strings, or a dict of strings. The pipeline
handles all three.

---

### Stage 2: LLM Extraction

The `RuleExtractor` sends raw text to an LLM (default: `gpt-4o`) with a system prompt
engineered to extract structured underwriting rules. The LLM returns JSON:

```python
from eval_workbench.implementations.athena.memory import RuleExtractor

extractor = RuleExtractor(model='gpt-4o')
rules = extractor.extract_batch([raw_text])
```

Each extracted rule is a dict. A single text block (especially production case reviews)
often yields **multiple** rules. The extractor distinguishes between hard deterministic
rules, soft guidelines, compound triggers, data quality patterns, and more.

**Hard rule** (no threshold flexibility):

```python
{
    "risk_factor": "Gas Station",
    "risk_category": "occupancy",
    "rule_name": "Gas Station Auto-Refer",
    "product_type": "LRO",
    "action": "refer",
    "outcome_description": "Gas stations must be referred to senior underwriter due to fuel storage liability.",
    "mitigants": ["Fire suppression system installed", "Located >200ft from residential"],
    "source": "Underwriting Manual v3.2",
    "source_type": "manual",
    "confidence": "high",
    "threshold": null,
    "threshold_type": null,
    "historical_exceptions": null,
    "decision_quality": null,
    "compound_trigger": null,
    "data_fields": []
}
```

**Compound trigger from a production case** (multiple conditions must combine):

```python
{
    "risk_factor": "Adverse Selection Trifecta",
    "risk_category": "eligibility",
    "rule_name": "Adverse Selection Auto-Decline",
    "product_type": "ALL",
    "action": "decline",
    "outcome_description": "Combination of recent major loss (>20% of building exposure) + below-minimum pricing + prior carrier non-renewal is an automatic decline regardless of other positive factors.",
    "mitigants": [],
    "source": "Production Decision Log - warehouse decline case",
    "source_type": "production",
    "confidence": "high",
    "threshold": null,
    "threshold_type": "hard",
    "historical_exceptions": null,
    "decision_quality": "aligned",
    "compound_trigger": "recent_major_loss AND below_minimum_pricing AND prior_carrier_rejection",
    "data_fields": ["auxData.rateData.input.total_loss_payment", "auxData.swallowProject.property_rate"]
}
```

**Soft threshold with divergent decision** (LLM learned the real rule is softer than the manual):

```python
{
    "risk_factor": "Property Rate Below Minimum",
    "risk_category": "pricing",
    "rule_name": "Low Property Rate Referral",
    "product_type": "Property",
    "action": "refer",
    "outcome_description": "Property rate below 0.25% triggers referral. However, rates below threshold are often legitimate for BASIC cause of loss policies with wind/hail exclusions.",
    "mitigants": ["BASIC cause of loss (limited coverage scope)", "Wind/hail exclusion in place"],
    "source": "Production Decision Log",
    "source_type": "production",
    "confidence": "high",
    "threshold": {"field": "property_rate", "operator": "lt", "value": 0.0025, "unit": "rate"},
    "threshold_type": "soft",
    "historical_exceptions": "Approved at 0.19% for newer construction with BASIC cause of loss and wind/hail exclusion",
    "decision_quality": "divergent",
    "compound_trigger": null,
    "data_fields": ["auxData.rateData.input.property_rate", "underwritingFlags.property_rate_low_Refer"]
}
```

**Data quality pattern** (non-traditional risk factor):

```python
{
    "risk_factor": "Building Exposure Discrepancy",
    "risk_category": "data_quality",
    "rule_name": "Magic Dust Exposure Mismatch Flag",
    "product_type": "ALL",
    "action": "flag_for_review",
    "outcome_description": "When Magic Dust building exposure differs from customer input by >20%, flag for verification. Address ranges (e.g. 404-600) may indicate multi-location risk.",
    "mitigants": ["Recent appraisal on file", "Physical inspection completed"],
    "source": "Production Decision Log",
    "source_type": "production",
    "confidence": "high",
    "threshold": {"field": "building_exposure_discrepancy_pct", "operator": "gt", "value": 20, "unit": "percent"},
    "threshold_type": "soft",
    "historical_exceptions": "Cases approved after visual verification confirmed Magic Dust data was incorrect",
    "decision_quality": "divergent",
    "compound_trigger": null,
    "data_fields": ["auxData.rateData.input.building_exposure", "magicDustByElement.building_area"]
}
```

The key extraction fields:

| Field | Purpose |
|-------|---------|
| `risk_category` | Category: `occupancy`, `construction`, `location`, `operations`, `eligibility`, `pricing`, `data_quality`, `geographic_appetite`, `catastrophe`, `classification`, `process` |
| `threshold` | Structured numeric boundary: `{field, operator, value, unit}` |
| `threshold_type` | `"hard"` (absolute, never exceeded) or `"soft"` (known flexibility) |
| `historical_exceptions` | Plain-text description of real-world deviations from the rule |
| `decision_quality` | From production cases: `"aligned"` (AI matched UW), `"divergent"` (disagreed), `"partial"` |
| `compound_trigger` | Multi-condition rule: `"condition_a AND condition_b AND condition_c"` |
| `data_fields` | System field paths referenced (e.g. `auxData.rateData.input.property_rate`) |
| `source_type` | Where the knowledge came from: `manual`, `training`, `sme`, `production`, `compliance` |
| `action` | Prescribed action: `refer`, `decline`, `approve_with_conditions`, `exclude`, `verify`, `flag_for_review` |

The extraction prompt teaches the LLM to:

- Extract **multiple rules** per text block (production case reviews often contain 3-5+)
- Look for **numeric thresholds** embedded in prose ("$75/sq ft", "0.25% rate", "$1.5M frame limit")
- Distinguish **hard vs soft** thresholds from language signals ("always" vs "generally")
- Capture **compound triggers** when multiple conditions must combine
- Recognize **data quality patterns** (exposure discrepancies, address ranges, zero-value fields)
- Recognize **geographic appetite** boundaries (Tier 1 counties, state exclusions)
- Recognize **pricing rules** (rate minimums, TIV/sqft thresholds)
- When a production case shows a **divergent decision**, extract both the original rule AND the learned exception

---

### Stage 3: Graph Ingestion

Each extracted rule is converted into a set of graph edges that follow the Athena ontology,
then ingested into Zep's knowledge graph.

The `_rule_to_ingest_payload()` method on the pipeline builds these edges. For hard rules:

```
Gas Station ──TRIGGERS──▶ Gas Station Auto-Refer ──RESULTS_IN──▶ Refer
                                    ▲
                                    │
            Fire suppression system ──OVERRIDES
            Located >200ft          ──OVERRIDES
                                    │
                                    ▼
            Gas Station Auto-Refer ──DERIVED_FROM──▶ Underwriting Manual v3.2
```

For production-sourced rules, all contextual metadata — thresholds, historical exceptions,
decision quality, compound triggers, and data field references — are embedded directly in
the edge properties and outcome description. This is intentional — Zep does semantic search
over fact text, so the nuance needs to be **in the facts**, not just in the graph topology.

**Soft threshold example:**

```
Employee Count ──TRIGGERS──▶ Swallow API Employee Limit ──RESULTS_IN──▶ Decline
                  │                       ▲
                  │ threshold: gt 20      │
                  │ threshold_type: soft  │
                  │ historical: "up to    Strong loss history ──OVERRIDES
                  │   23 approved"        Well-established biz ──OVERRIDES
                  │                       │
                  │                       ▼
                  │             Swallow API Employee Limit ──DERIVED_FROM──▶ Production Decisions
                  │
                  └─▶ Outcome includes: "Limit is 20, but UWs approved up to 23..."
```

**Compound trigger example:**

```
Adverse Selection Trifecta ──TRIGGERS──▶ Adverse Selection Auto-Decline ──RESULTS_IN──▶ Decline
                  │
                  │ threshold_type: hard
                  │ compound_trigger: "recent_major_loss AND below_minimum_pricing
                  │                    AND prior_carrier_rejection"
                  │ decision_quality: aligned
                  │ data_fields: [auxData.rateData.input.total_loss_payment, ...]
                  │
                  └─▶ Outcome includes: "Combination of major loss + low pricing +
                      prior rejection = automatic decline regardless of other positives"
```

When someone later searches "employee count 22", Zep's semantic search surfaces the fact
with the full context — including that 22 falls in the soft zone between the stated limit
(20) and the historical maximum (23). When someone searches "prior loss and coverage gap",
the compound trigger rule surfaces with the full multi-condition context.

In code:

```python
from eval_workbench.shared.memory import ZepGraphStore
from eval_workbench.implementations.athena.memory import AthenaRulePipeline, ATHENA_ONTOLOGY

store = ZepGraphStore(agent_name='athena', ontology=ATHENA_ONTOLOGY)
pipeline = AthenaRulePipeline(store=store)

# Full pipeline: raw text → LLM extraction → graph ingestion
result = pipeline.run(raw_text, batch_size=5)
print(f'Processed: {result.items_processed}, Ingested: {result.items_ingested}')
```

The `ZepGraphStore` serializes each edge set to JSON and calls `client.graph.add()`.
Each agent gets its own Zep user (e.g. `athena_global_rules`) for multi-tenancy isolation.

---

### Stage 4: Runtime Retrieval

At query time, `AthenaGraphAnalytics` provides three retrieval modes:

#### Quick search — `query()`

Returns matching edges filtered by product type. Use this for fast lookups
in the underwriting workflow.

```python
from eval_workbench.implementations.athena.memory import AthenaGraphAnalytics

analytics = AthenaGraphAnalytics(store=store)
results = analytics.query('Gas Station', product_type='LRO')

for r in results:
    print(f"{r['source']} --{r['relation']}--> {r['target']}")
```

There's also a domain alias:

```python
results = analytics.check_risk_appetite('Gas Station', product_type='LRO')
```

#### Decision path trace — `trace()`

Returns the full decision tree: RiskFactor → Rule → Outcome, with any applicable
mitigants. Use this for explainability and audit trails.

```python
paths = analytics.trace('Gas Station', product_type='LRO')

for path in paths:
    print(f"Risk:      {path['risk_factor']}")
    print(f"Rule:      {path['rule']}")
    print(f"Outcomes:  {path['outcomes']}")
    print(f"Mitigants: {path['mitigants']}")
```

#### Export — `export()`

Dumps the entire graph as structured dicts. Use this for audits, backups, or
feeding into other systems.

```python
all_rules = analytics.export()
analytics.export_to_json('rules_backup.json')
```

#### Visualization

Renders the graph as a networkx/matplotlib plot. Requires the `memory` optional
dependencies.

```python
analytics.visualize(title='Athena Underwriting Rules')
```

---

## Hard vs Soft Rules

<div markdown style="background: rgba(30, 58, 95, 0.1); border-left: 3px solid #1E3A5F; padding: 16px; margin: 20px 0;">

**Key concept:** Not all rules are equal. The memory system preserves the distinction between absolute limits and flexible guidelines, enabling nuanced AI recommendations.

</div>

Not all underwriting rules are binary. The system distinguishes between:

**Hard rules** — absolute limits with no flexibility. The rule is the rule.

- Cannabis operations → always decline
- Coastal distance < 1 mile → always refer (no exceptions without wind mitigation)
- These have `threshold_type: "hard"` (or `null` if there's no numeric boundary)

**Soft rules** — guidelines with known flexibility where UW discretion applies.

- Employee count > 20 → manual says decline, but UWs have approved up to 23
- Frame TIV > $1M → manual says refer, but seniors have approved $1.25M with sprinklers
- Vacant > 60 days → manual says decline, but approved with active renovation permits
- These have `threshold_type: "soft"` and a populated `historical_exceptions` field

**Why this matters for retrieval:** When Athena searches for "employee count 22",
Zep returns the fact with the full context: the stated limit (20), the threshold type
(soft), and the historical precedent (approved up to 23). Athena can then make a
nuanced recommendation instead of a blanket decline — e.g., "This exceeds the stated
limit of 20 but falls within the historical approval range of up to 23. Consider
approving if loss history is clean."

**How it gets into the graph:** The richness comes from two places:

1. **The extraction prompt** teaches the LLM to look for signals of flexibility in
   the source text ("generally", "typically", "has been approved", etc.) and to
   capture deviations in `historical_exceptions`.

2. **The pipeline** embeds threshold metadata and historical context directly into
   the edge properties and outcome description, so the full nuance is stored as
   searchable fact text in Zep.

**Beyond hard vs soft — other rule patterns the system captures:**

- **Compound triggers** — rules that only fire when multiple conditions combine
  (e.g. "recent major loss AND below-minimum pricing AND prior carrier rejection").
  The `compound_trigger` field captures the multi-condition logic.

- **Data quality rules** — non-traditional risk factors like exposure discrepancies,
  address range anomalies, or zero-value fields. Use `risk_category: "data_quality"`.

- **Geographic appetite** — hard boundaries based on state, county tier, or coastal
  distance. These are typically absolute ("California out of appetite"), with
  `risk_category: "geographic_appetite"`.

- **Pricing thresholds** — rate minimums, TIV/sqft floors, premium bounds. Often soft
  in practice because the stated minimum doesn't account for coverage scope variations.
  Use `risk_category: "pricing"`.

- **Process rules** — behavioral patterns like "refer for data correction, don't decline"
  when a blocking issue is correctable. Use `risk_category: "process"`.

**Source variety matters:** Soft rules emerge when you ingest from multiple source types.
A manual alone only gives you the hard limit. The flexibility appears when you also ingest:

- Production decision logs ("we approved 23 employees last quarter")
- SME interviews ("the 20 limit is a guideline, not a hard stop")
- Training notes ("for well-established businesses, we have room on employee count")

Set `source_type` accordingly so the graph captures the provenance.

---

## Ontology

<div markdown style="background: rgba(30, 58, 95, 0.1); border-left: 3px solid #1E3A5F; padding: 16px; margin: 20px 0;">

**Key concept:** The ontology defines the vocabulary of the knowledge graph — what types of nodes and edges exist and what they mean. Each implementation defines its own ontology via Pydantic frozen models that guide LLM extraction.

</div>

### Athena Ontology

**5 node types:**

| Node Type | Description |
|-----------|-------------|
| **RiskFactor** | An observable characteristic that affects underwriting (e.g. Gas Station, Employee Count) |
| **Rule** | An underwriting rule or guideline — can be hard (absolute) or soft (flexible). Has `threshold_type` property. |
| **Outcome** | The result of applying a rule (Refer, Decline, Approve with Conditions). Description includes threshold context. |
| **Mitigant** | A condition that can override or reduce a rule's impact |
| **Source** | Origin of the knowledge (manual, training, SME, production, compliance) |

**4 edge types (relations):**

| Relation | Source → Target | Key Properties |
|----------|----------------|----------------|
| **TRIGGERS** | RiskFactor → Rule | `threshold`, `threshold_type`, `historical_exceptions`, `confidence`, `decision_quality`, `compound_trigger`, `data_fields` |
| **RESULTS_IN** | Rule → Outcome | Outcome description enriched with threshold + historical context |
| **OVERRIDES** | Mitigant → Rule | A mitigant can override a rule |
| **DERIVED_FROM** | Rule → Source | Source `type`: manual, training, sme, production, compliance |

### Ontology Registry

Ontologies are registered at import time, following the same pattern as metric registration:

```python
from eval_workbench.shared.memory import ontology_registry

# Importing the athena memory module triggers registration
from eval_workbench.implementations.athena.memory import ATHENA_ONTOLOGY

ontology_registry.list_ontologies()  # ['athena_underwriting']
ontology_registry.get('athena_underwriting').to_prompt_context()  # Markdown for LLM prompts
```

### Creating a New Ontology

To add a knowledge graph for a new agent, define an `OntologyDefinition` and register it:

```python
from eval_workbench.shared.memory import (
    OntologyDefinition,
    NodeTypeDefinition,
    EdgeTypeDefinition,
    ontology_registry,
)

MY_ONTOLOGY = OntologyDefinition(
    name='my_agent_domain',
    version='1.0.0',
    description='Knowledge graph for ...',
    node_types=[
        NodeTypeDefinition(label='Concept', description='...', required_properties=['name']),
    ],
    edge_types=[
        EdgeTypeDefinition(relation='RELATES_TO', source_label='Concept', target_label='Concept', description='...'),
    ],
)

ontology_registry.register(MY_ONTOLOGY)
```

---

## Multi-Tenancy

Each agent gets its own isolated graph via a Zep user. The user ID is generated from
a template in settings:

```
user_id_template: "{agent_name}_global_rules"

athena  → athena_global_rules
abby  → abby_global_rules
```

This means agents cannot accidentally read or overwrite each other's knowledge. A future
MCP server can expose cross-agent search by querying multiple stores.

---

## Configuration

### YAML Config

```yaml
# implementations/athena/config/memory.yaml
memory:
  zep:
    api_key: "${ZEP_API_KEY}"
    admin_email: "athena@system.local"
    user_id_template: "{agent_name}_global_rules"
  extractor:
    model: "gpt-4o"
  pipeline:
    batch_size: 5
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `ZEP_API_KEY` | Zep Cloud API key | Yes |
| `ZEP_BASE_URL` | Zep Cloud base URL (self-hosted only) | No |

These can also be set in AthenaSettings:

```python
from eval_workbench.implementations.athena.settings import AthenaSettings

settings = AthenaSettings()
settings.zep_api_key   # from ZEP_API_KEY or .env
settings.zep_base_url  # from ZEP_BASE_URL or .env
```

### Installation

The memory module uses optional dependencies to keep the core package lightweight:

```bash
pip install eval-workbench[memory]
```

This installs `zep-cloud`, `networkx`, and `matplotlib`.

---

## Architecture

### File Layout

```
src/eval_workbench/
  shared/memory/                      # Shared abstractions
    __init__.py                       # Public API re-exports
    ontology.py                       # OntologyDefinition + OntologyRegistry
    store.py                          # BaseGraphStore ABC + data models
    pipeline.py                       # BasePipeline ABC
    analytics.py                      # BaseGraphAnalytics ABC
    settings.py                       # ZepSettings
    zep/
      store.py                        # ZepGraphStore (Zep Cloud backend)

  implementations/athena/memory/      # Athena-specific
    __init__.py                       # Ontology registration
    ontology.py                       # ATHENA_ONTOLOGY constant
    extractors.py                     # RuleExtractor (OpenAI)
    pipeline.py                       # AthenaRulePipeline
    analytics.py                      # AthenaGraphAnalytics
```

### Shared vs Implementation

The split follows the same pattern as metrics and monitoring:

- **`shared/memory/`** defines the **interfaces**: `BaseGraphStore`, `BasePipeline`,
  `BaseGraphAnalytics`, `OntologyDefinition`. These are stable abstractions that any
  agent can build on.

- **`implementations/athena/memory/`** provides the **Athena-specific glue**: the
  underwriting ontology, the rule extractor prompt, the pipeline that builds ontology-correct
  edges, and the analytics layer with domain methods like `check_risk_appetite()`.

To add memory for a new agent, you implement the same pattern — define an ontology,
write an extractor, subclass the pipeline and analytics ABCs.

### Data Flow Diagram

```
                    ┌───────────────────────────────────────┐
                    │         AthenaRulePipeline             │
                    │                                       │
  Raw text ────────▶│  1. RuleExtractor.extract_batch()     │
                    │     (OpenAI gpt-4o)                   │
                    │          │                             │
                    │          ▼                             │
                    │  2. _rule_to_ingest_payload()          │
                    │     (builds TRIGGERS, RESULTS_IN,      │
                    │      OVERRIDES, DERIVED_FROM edges)    │
                    │          │                             │
                    │          ▼                             │
                    │  3. ZepGraphStore.ingest()             │
                    │     (JSON → Zep Cloud API)            │
                    └───────────────────────────────────────┘
                                       │
                                       │  stored in Zep per-agent user
                                       ▼
                    ┌───────────────────────────────────────┐
                    │         AthenaGraphAnalytics           │
                    │                                       │
  "Gas Station" ───▶│  .query()          → matching edges   │
  "Gas Station" ───▶│  .trace()          → full paths       │
  "*"           ───▶│  .export()         → all knowledge    │
  "Gas Station" ───▶│  .check_risk_appetite() → alias       │
                    │                                       │
                    │  All go through ZepGraphStore.search() │
                    │  → semantic search over the graph      │
                    └───────────────────────────────────────┘
```

---

## MCP Server Readiness

The interfaces are designed so an MCP server can wrap them directly. All return types
are Pydantic models that serialize to JSON natively.

| MCP Tool | Maps To | Type |
|----------|---------|------|
| `memory_search` | `store.search(query)` | Query |
| `memory_check_risk` | `analytics.check_risk_appetite(...)` | Query |
| `memory_trace_path` | `analytics.trace(...)` | Query |
| `memory_export_rules` | `analytics.export()` | Query |
| `memory_list_ontologies` | `ontology_registry.list_ontologies()` | Query |
| `memory_get_ontology` | `ontology_registry.get(name).to_prompt_context()` | Query |
| `memory_ingest_rule` | `store.ingest(payload)` | Mutation |
| `memory_add_mitigant` | `store.add_mitigant(rule, mitigant)` | Mutation |
| `memory_run_pipeline` | `pipeline.run(raw_data)` | Mutation |
| `memory_clear_graph` | `store.clear()` | Mutation |

The ontology registry enables dynamic tool discovery per agent — an MCP server can
list available ontologies and expose agent-specific tools.

---

## Quick Start

### Full pipeline (raw text → graph → search)

```python
from eval_workbench.shared.memory import ZepGraphStore
from eval_workbench.implementations.athena.memory import (
    ATHENA_ONTOLOGY,
    AthenaRulePipeline,
    AthenaGraphAnalytics,
)

# 1. Build the store
store = ZepGraphStore(agent_name='athena', ontology=ATHENA_ONTOLOGY)

# 2. Ingest production learnings through the pipeline
pipeline = AthenaRulePipeline(store=store)
result = pipeline.run([
    """Athena correctly recommended decline for a lessor's risk warehouse with
    catastrophic recent loss ($600K representing 23.5% of building exposure),
    pricing 37% below minimum threshold, and prior carrier non-renewal. The
    combination of recent major loss + below-minimum pricing + prior carrier
    rejection is an automatic decline regardless of other positive factors.""",

    """The Swallow API employee count limit is 20 per the manual, but historical
    underwriters have approved up to 23 employees for well-established businesses
    with strong loss history. This is a soft threshold, not a hard stop.""",

    """California (CA) is currently out of appetite because the company is not yet
    approved to operate in that state. Geographic restrictions based on state
    approval should trigger early declination.""",
])
print(f'Processed: {result.items_processed}, Ingested: {result.items_ingested}')
```

**Sample output** (what the LLM extractor produces from the above):

```
Processed: 7, Ingested: 7
```

The extractor finds ~7 rules across the 3 texts. For example, the warehouse text
produces a compound trigger rule and a pricing rule. Here's what the extracted dicts
look like and the edges they generate:

```python
# Rule 1 (from the warehouse learning):
# {
#     "risk_factor": "Adverse Selection Trifecta",
#     "risk_category": "eligibility",
#     "rule_name": "Adverse Selection Auto-Decline",
#     "action": "decline",
#     "threshold_type": "hard",
#     "compound_trigger": "recent_major_loss AND below_minimum_pricing AND prior_carrier_rejection",
#     "decision_quality": "aligned",
#     "data_fields": ["auxData.rateData.input.total_loss_payment", ...]
# }
#
# → Adverse Selection Trifecta --TRIGGERS--> Adverse Selection Auto-Decline
# → Adverse Selection Auto-Decline --RESULTS_IN--> Decline
# → Adverse Selection Auto-Decline --DERIVED_FROM--> Production Decision Log

# Rule 2 (from the employee count learning):
# {
#     "risk_factor": "Employee Count",
#     "rule_name": "Swallow API Employee Limit",
#     "action": "decline",
#     "threshold": {"field": "employee_count", "operator": "gt", "value": 20},
#     "threshold_type": "soft",
#     "historical_exceptions": "UWs have approved up to 23 employees"
# }
#
# → Employee Count --TRIGGERS--> Swallow API Employee Limit
#     (threshold_type: soft, historical: "up to 23 approved")
# → Swallow API Employee Limit --RESULTS_IN--> Decline
#     (description includes "...but UWs have approved up to 23...")
# → Strong loss history --OVERRIDES--> Swallow API Employee Limit

# Rule 3 (from the California learning):
# {
#     "risk_factor": "California State",
#     "risk_category": "geographic_appetite",
#     "rule_name": "California Out of Appetite",
#     "action": "decline",
#     "threshold_type": "hard"
# }
```

Then at query time:

```python
# 3. Search
analytics = AthenaGraphAnalytics(store=store)

# Hard rule — clear answer
results = analytics.check_risk_appetite('California', product_type='BOP')
# → "California State --TRIGGERS--> California Out of Appetite"
#   threshold_type: hard, no exceptions

# Soft threshold — nuanced answer with historical context
results = analytics.check_risk_appetite('employee count 22', product_type='BOP')
# → "Employee Count --TRIGGERS--> Swallow API Employee Limit"
#   threshold_type: soft
#   historical: "UWs have approved up to 23 employees"
#   outcome: "Limit is 20 per manual, but UWs approved up to 23..."

# Compound trigger — surfaces the multi-condition rule
results = analytics.check_risk_appetite('prior loss and coverage gap', product_type='ALL')
# → "Adverse Selection Trifecta --TRIGGERS--> Adverse Selection Auto-Decline"
#   compound: "recent_major_loss AND below_minimum_pricing AND prior_carrier_rejection"
```

### From YAML config

```python
pipeline = AthenaRulePipeline.from_yaml(
    'src/eval_workbench/implementations/athena/config/memory.yaml'
)
result = pipeline.run(raw_texts)
```

### Pre-extracted rules (skip LLM)

If you already have structured rules (e.g. from a spreadsheet or manual curation),
you can skip the extractor and ingest directly:

```python
store = ZepGraphStore(agent_name='athena', ontology=ATHENA_ONTOLOGY)
pipeline = AthenaRulePipeline(store=store)

rule = {
    'risk_factor': 'Gas Station',
    'rule_name': 'Gas Station Auto-Refer',
    'product_type': 'LRO',
    'action': 'refer',
    'outcome_description': 'Referred due to fuel storage liability.',
    'mitigants': ['Fire suppression system installed'],
    'source': 'Underwriting Manual v3.2',
    'confidence': 'high',
}

payload = AthenaRulePipeline._rule_to_ingest_payload(rule)
store.ingest(payload)
```

### Adding a mitigant to an existing rule

```python
store.add_mitigant('Gas Station Auto-Refer', 'Annual fire inspection on file')
```

### Demo script

A runnable demo with sample rules is available at:

```bash
python private_scripts/run_memory_graph.py          # full demo
python private_scripts/run_memory_graph.py --clear   # clear + re-ingest
python private_scripts/run_memory_graph.py --export rules.json
```

Requires `ZEP_API_KEY` to be set.

---

## Agent Entry Points

This section describes how an AI agent (like Athena) actually *uses* the memory system
at runtime. There are two integration paths: direct Python API and MCP tools.

### Path 1: Direct Python API (current)

The agent's workflow code imports and calls the analytics layer directly. This is the
simplest path and works for agents running in the same Python process.

**At agent initialization:**

```python
from eval_workbench.shared.memory import ZepGraphStore
from eval_workbench.implementations.athena.memory import (
    ATHENA_ONTOLOGY,
    AthenaGraphAnalytics,
)

# Build once, reuse across the agent's lifetime
store = ZepGraphStore(agent_name='athena', ontology=ATHENA_ONTOLOGY)
analytics = AthenaGraphAnalytics(store=store)
```

**During underwriting evaluation** (the main entry point for the agent):

```python
def evaluate_submission(submission: dict) -> dict:
    """Where the agent calls into memory during its decision-making loop."""

    risk_factor = submission['occupancy_description']  # e.g. "Gas Station"
    product_type = submission['product_type']           # e.g. "LRO"

    # Primary entry point: check if this risk is in the knowledge graph
    rules = analytics.check_risk_appetite(risk_factor, product_type=product_type)

    for rule in rules:
        action = rule.get('action')          # "refer", "decline", etc.
        threshold_type = rule.get('threshold_type')  # "hard" or "soft"
        historical = rule.get('historical_exceptions')

        if action == 'decline' and threshold_type == 'hard':
            return {'decision': 'decline', 'reason': rule['outcome']}
        elif threshold_type == 'soft' and historical:
            # Nuanced decision — the agent can weigh flexibility
            ...

    # For explainability / audit trail
    paths = analytics.trace(risk_factor, product_type=product_type)
    return {'decision': '...', 'reasoning_chain': paths}
```

**For knowledge ingestion** (called by an admin workflow, CI/CD, or a learning pipeline —
not during real-time evaluation):

```python
from eval_workbench.implementations.athena.memory import AthenaRulePipeline

pipeline = AthenaRulePipeline(store=store)

# Ingest raw production learnings (LLM extraction + graph storage)
result = pipeline.run(raw_learnings_text, batch_size=5)
```

The key insight: **retrieval is on the hot path, ingestion is not.** The agent queries
the graph during every submission evaluation, but ingestion happens offline when new
learnings are collected.

### Path 2: MCP Tools (future)

When the memory system is wrapped in an MCP server, the agent's LLM calls tools
instead of Python functions. The tool signatures map directly to the Python API:

```
Agent LLM                          MCP Server                    Memory System
─────────                          ──────────                    ─────────────
"Check if Gas Station              memory_check_risk             analytics.check_risk_appetite(
 is in appetite for LRO"  ──────▶  {risk_factor: "Gas Station",    "Gas Station",
                                    product_type: "LRO"}          product_type="LRO"
                                          │                      )
                                          ▼                         │
                                   Returns JSON with               ▼
                                   matching rules, thresholds,   GraphSearchResult
                                   historical exceptions         → serialized to JSON
```

The MCP tool names and their Python API equivalents:

| Agent asks... | MCP Tool | Python Entry Point |
|---------------|----------|--------------------|
| "Is this risk in appetite?" | `memory_check_risk` | `analytics.check_risk_appetite(risk, product_type=...)` |
| "What rules apply here?" | `memory_search` | `store.search(query, limit=...)` |
| "Show the full decision path" | `memory_trace_path` | `analytics.trace(risk, product_type=...)` |
| "What does our ontology cover?" | `memory_get_ontology` | `ontology_registry.get(name).to_prompt_context()` |
| "Record this new learning" | `memory_run_pipeline` | `pipeline.run(raw_data)` |

In this path, the LLM decides *when* to query memory based on its system prompt. A
typical system prompt injection:

```python
# Inject the ontology into the agent's system prompt so it knows what to search for
ontology_context = ATHENA_ONTOLOGY.to_prompt_context()
system_prompt += f"""
You have access to an underwriting knowledge graph. Use the memory_check_risk tool
to look up any risk factor before making a recommendation. The graph contains:

{ontology_context}

Always check memory before recommending decline or referral.
"""
```

### Which path to choose

| Consideration | Direct Python API | MCP Tools |
|---------------|-------------------|-----------|
| Setup complexity | Lower — just import and call | Higher — requires MCP server |
| Agent autonomy | Agent code decides when to query | LLM decides when to query |
| Multi-language | Python only | Any language that speaks MCP |
| Latency | Lower — no HTTP hop | Slightly higher — MCP protocol overhead |
| Best for | Deterministic agent workflows | LLM-driven autonomous agents |

Most agents will start with the direct Python API and migrate to MCP tools when they
need the LLM to autonomously decide when to query memory.

---

## "Where to look" cheat-sheet

| What | Where |
|------|-------|
| Shared ABCs + data models | `src/eval_workbench/shared/memory/` |
| Zep backend implementation | `src/eval_workbench/shared/memory/zep/store.py` |
| Athena ontology definition | `src/eval_workbench/implementations/athena/memory/ontology.py` |
| LLM rule extractor | `src/eval_workbench/implementations/athena/memory/extractors.py` |
| Athena ingestion pipeline | `src/eval_workbench/implementations/athena/memory/pipeline.py` |
| Athena analytics / search | `src/eval_workbench/implementations/athena/memory/analytics.py` |
| YAML config | `src/eval_workbench/implementations/athena/config/memory.yaml` |
| Zep settings | `src/eval_workbench/shared/memory/settings.py` |
| Runnable demo | `private_scripts/run_memory_graph.py` |
