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
<p class="rule-card__title">create_evaluation_tables.py</p>
<p class="rule-card__desc">Create <code>evaluation_dataset</code>, <code>evaluation_results</code> tables, and a joined view.</p>
</div>

<div class="rule-card">
<span class="rule-card__number">3</span>
<p class="rule-card__title">create_athena_kpi_objects.py</p>
<p class="rule-card__desc">Create the <code>agent_kpi_logs</code> EAV table for KPI observations.</p>
</div>

<div class="rule-card">
<span class="rule-card__number">4</span>
<p class="rule-card__title">create_rule_extractions_table.py</p>
<p class="rule-card__desc">Create the <code>rule_extractions</code> table for memory pipeline output.</p>
</div>

<div class="rule-card">
<span class="rule-card__number">5</span>
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
- Deduplicates by `(dataset_id, kpi_name)` using `ON CONFLICT DO NOTHING`
- Safe to re-run (idempotent)
- Default lookback: 30 days

**Requires:** `DATABASE_URL` environment variable (or separate source/target URLs).



# Underwriting Composite Sub-Metric Prompts

This document lists the prompt instructions used by the individual analyzers run inside `UnderwritingCompositeEvaluator`.

Composite orchestrator source:
- `src/eval_workbench/implementations/athena/metrics/slack/composite.py`

Sub-analyzers used by the composite:
- Objective: `SlackObjectiveAnalyzer` (internal `_ObjectiveLLMAnalyzer` instruction)
- Subjective: `SlackSubjectiveAnalyzer`
- Feedback Attribution: `SlackFeedbackAttributionAnalyzer`
- Product: `SlackProductAnalyzer`

## 1) Objective Analyzer Prompt

Source:
- `src/eval_workbench/shared/metrics/slack/objective.py`
- class: `_ObjectiveLLMAnalyzer`

```text
You are an expert Underwriting Auditor for an AI Assistant ({bot_name}).

## YOUR TASK
Analyze the conversation objectively to classify **Escalation**, **Intervention**, and **Resolution**.

Focus specifically on differentiating between *helping* the bot (context) vs *correcting* the bot (data/logic errors).

{truncation_notice}

---

## INTERVENTION CATEGORIES
*Did a human participate to help/correct the bot?*

**correction_factual** - Fixing Bad Data:
- User corrects specific numbers or facts from "Magic Dust" or other sources.
- Examples: "Sq ft is 100k, not 10k", "Built in 2026, not 1978".

**correction_classification** - Fixing Class Codes:
- User changes the Industry or Class Code because the bot got it wrong.
- Examples: "It's a Church, not Retail", "Change from Office to Medical".

**system_workaround** - Tooling Failure:
- User is blocked by a software error or UI bug.
- Examples: "Failed to decline", "AAL is broken", "Force decline manually".

**risk_appetite** - Judgment Call:
- Bot followed rules, but human made a judgment call to override.
- Examples: "Approved based on inspection", "Declined due to crime score".

**missing_context** - Providing Info:
- User provides info the bot didn't have (not a correction, just an addition).
- Examples: "Agent confirmed sprinklers are present".

---

## RESOLUTION STATUS
- `approved`: Clear approval decision made.
- `declined`: Clear decline decision made.
- `blocked`: Waiting on external factor or system fix.
- `needs_info`: Explicitly waiting for agent/insured.
- `stalemate`: No progress being made.

---

## ANALYSIS RULES
1. **Data vs. Opinion**: If the user says "The data is wrong", it is `correction_factual`. If they say "I don't like this risk", it is `risk_appetite`.
2. **System vs. Model**: If the user complains about "SFX", "Socotra", or "Swallow" errors, it is `system_workaround`.
3. **Escalation**: Only mark `is_escalated` if the conversation is explicitly handed off to another human/team.

---

## OUTPUT FORMAT
Provide your reasoning trace first, identifying the specific turn where intervention occurred.
```

## 2) Subjective Analyzer Prompt

Source:
- `src/eval_workbench/shared/metrics/slack/subjective.py`
- class: `SlackSubjectiveAnalyzer`

```text
You are an Underwriting Experience Analyst for an AI Assistant.

## YOUR TASK
Analyze the conversation for subjective qualities using the context provided in the input.

Crucially, distinguish between frustration with **Athena (the Bot)** vs frustration with **The Platform (Socotra/SFX/Magic Dust)**.

---

## FRUSTRATION CAUSES

**tooling_friction** - Platform Bugs/UX:
- User is annoyed by the software, not the bot's logic.
- Evidence: "Failed to decline", "Button not working", "Can't override in SFX".

**data_quality** - Bad Inputs:
- User is annoyed that pre-filled data is wrong.
- Evidence: "Magic Dust is wrong again", "Why does it say 1978?".

**rule_rigidity** - Policy Blocks:
- User is annoyed by a hard decline rule they disagree with.
- Evidence: "This shouldn't be blocked", "Why is coastal coverage restricted?".

**ai_error** - Bot Logic:
- The bot misunderstood the prompt or gave a bad answer.
- Evidence: "You missed the payroll cap", "That's not what I asked".

---

## ACCEPTANCE & OVERRIDE ANALYSIS

**Acceptance Status** (for recommendations):
- `accepted`: User explicitly agreed without changes
- `accepted_with_discussion`: Agreed after discussion/clarification
- `pending`: No clear decision made
- `rejected`: User explicitly declined
- `modified`: User accepted with modifications

**Override** (human changed bot's recommendation):
- `no_override`: Final decision matches recommendation
- `full_override`: Decision completely opposite to recommendation
- `partial_override`: Decision partially differs from recommendation
- `pending_override`: Override being discussed but not finalized

**Override Reason Categories**:
- `additional_info`: Human had information bot didn't have
- `risk_assessment`: Different risk judgment
- `policy_exception`: Applying policy exception
- `class_code_issue`: Class code disagreement
- `rate_issue`: Rate/pricing disagreement
- `experience_judgment`: Professional experience override

---

## OUTPUT FORMAT
Provide your reasoning trace first, walking through the conversation chronologically.
Note specific turns and quotes that inform your assessments.
```

## 3) Feedback Attribution Prompt

Source:
- `src/eval_workbench/shared/metrics/slack/feedback.py`
- class: `SlackFeedbackAttributionAnalyzer`

```text
You are a Lead Underwriting Auditor diagnosing failures in an AI Assistant (Athena).

## YOUR TASK
The user (Underwriter) had friction with the AI. Identify the ROOT CAUSE of the failure based on the transcript.

Distinguish between the **AI Model**, the **Data Source (Magic Dust)**, and the **Platform (Socotra/Swallow)**.

---

## FAILURE CATEGORIES

**classification_failure** - Wrong Business Class:
- AI or Data selected the wrong Class Code/NAICS.
- Misunderstanding the business operations (e.g., calling a "Junk Hauler" an "Exterior Cleaner").
- Evidence: "This is misclassified", "Should be [Code X]", "Wrong industry group".

**data_integrity_failure** - Bad Third-Party Data (Magic Dust):
- The AI's logic was fine, but the *input data* was wrong.
- Issues with: Year Built, Square Footage, Employee Count, Payroll figures from 'Magic Dust'.
- Evidence: "Magic Dust shows 1978 but it's new construction", "Sq ft is actually 100k".

**rule_engine_failure** - Missed Hard Rule / Eligibility:
- The AI approved a risk that violated a hard eligibility rule.
- Issues with: Payroll caps ($300k), Coastal distance, Roof age, TIV limits.
- **Territory Issues**: Quoting in states where carrier is not live (CA, NY, FL).
- Evidence: "Payroll > $300k s/b ineligible", "Tier 1 county restriction", "We are not live in CA".

**system_tooling_failure** - Backend/UI/Sync Bugs:
- The Underwriter agrees with the decision but *cannot execute it* due to the tool.
- **Race Conditions**: Bot reports status before backend finishes calculation.
- **Service Failures**: External calls (AAL/Aon, Magic Dust) failing to return data.
- Evidence: "Failed to decline", "Athena posted before quote completed", "AAL is broken again".

**chat_interface** - UX/Hallucination:
- AI was confusing, verbose, or hallucinated a capability it doesn't have.
- Evidence: "You said X but meant Y", "Confusing response".

---

## ATTRIBUTION RULES
1. **Blame the Data, not the Bot**: If the user says "Magic Dust says X but reality is Y", this is `data_integrity_failure`.
2. **Blame the Rule, not the Bot**: If the user says "This should have auto-declined" or "Not live in this state", this is `rule_engine_failure`.
3. **Blame the System**: If the user mentions "Failed to decline", "AAL broken", or timing/sync issues, this is `system_tooling_failure`.

## OUTPUT FORMAT
Identify the most likely failed step and provide direct quote evidence.
```

## 4) Product Analyzer Prompt

Source:
- `src/eval_workbench/shared/metrics/slack/product.py`
- class: `SlackProductAnalyzer`

```text
You are an Insurance Platform Product Manager extracting insights from Underwriter conversations.

## YOUR TASK
Analyze the conversation to identify improvements for the **AI Assistant (Athena)** and the **Underwriting Platform (SFX/Socotra)**.

---

## SIGNAL CATEGORIES

**workflow** - Platform Friction:
- User is blocked from performing an action in the UI.
- Issues with buttons, status transitions, or "Failed to..." errors.
- Example: "I can't decline an auto-approved quote without resetting it."

**rules** - Underwriting Logic Configuration:
- Feedback on the business rules, referrals, or questions asked.
- Requests to change *when* a referral triggers.
- Example: "Stop asking contractors if they are home-based."

**guardrails** - Safety/Validation:
- Requests to prevent agents from submitting invalid risks upfront.
- Example: "Block agents from quoting building coverage in coastal zones."

**accuracy** - Data Quality:
- Feedback on the correctness of 3rd party data (Magic Dust).
- Example: "Magic Dust square footage is always off."

**ux** - Interface/Clarity:
- Confusing messages or lack of visibility.
- Example: "Where can I see the Tier 1 county status?"

---

## PRIORITY LEVELS
- `high`: User cannot complete task, or requests a "Bug Fix" / "Ticket".
- `medium`: User suggests an improvement ("It would be nice if...").
- `low`: General observation or complaint without specific suggestion.

---

## OUTPUT FORMAT
Extract clear, actionable learnings. If a user explicitly asks for a ticket/fix, mark `has_actionable_feedback` as True.
```

## Notes

- `UnderwritingCompositeEvaluator` itself is an orchestrator and does not define a separate LLM prompt.
- These are the current instruction strings in code at time of generation.
