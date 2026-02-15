# Slack Metrics

<div style="background: linear-gradient(135deg, #1E3A5F 0%, #0F2440 100%); padding: 24px; border-radius: 12px; color: white; margin: 20px 0;">

<p style="margin: 0; font-size: 16px; line-height: 1.6;">
<strong>5 Analyzers + 1 Composite Orchestrator</strong> for Slack conversation analytics. Split architecture separates heuristic pattern matching, objective factual analysis, subjective sentiment assessment, product insight extraction, and root-cause feedback attribution.
</p>
</div>

This module provides metrics for analyzing Slack conversations between users and
AI assistants (e.g., Athena). These metrics support KPI computation for
measuring AI assistant effectiveness in underwriting workflows.

---

## Analyzer Overview

<div class="rule-grid">

<div class="rule-card">
<a href="heuristic/" style="text-decoration: none; color: inherit;">
<span class="rule-card__number">H</span>
<p class="rule-card__title">Heuristic Analyzer</p>
<p class="rule-card__desc">Zero-cost pattern matching — interaction counts, engagement depth, recommendation detection. No LLM required.</p>
</a>
</div>

<div class="rule-card">
<a href="objective/" style="text-decoration: none; color: inherit;">
<span class="rule-card__number">O</span>
<p class="rule-card__title">Objective Analyzer</p>
<p class="rule-card__desc">LLM factual analysis (temp 0.0) — escalation, intervention, and resolution classification.</p>
</a>
</div>

<div class="rule-card">
<a href="subjective/" style="text-decoration: none; color: inherit;">
<span class="rule-card__number">S</span>
<p class="rule-card__title">Subjective Analyzer</p>
<p class="rule-card__desc">LLM sentiment & quality (temp 0.3) — sentiment, frustration, acceptance, override, satisfaction.</p>
</a>
</div>

<div class="rule-card">
<a href="product/" style="text-decoration: none; color: inherit;">
<span class="rule-card__number">P</span>
<p class="rule-card__title">Product Analyzer</p>
<p class="rule-card__desc">Product insight extraction — learnings, feature requests, priority classification.</p>
</a>
</div>

<div class="rule-card">
<a href="feedback/" style="text-decoration: none; color: inherit;">
<span class="rule-card__number">F</span>
<p class="rule-card__title">Feedback Attribution</p>
<p class="rule-card__desc">Root cause diagnosis — attributes failures to classification, data, rules, tooling, or interface.</p>
</a>
</div>

<div class="rule-card">
<a href="composite/" style="text-decoration: none; color: inherit;">
<span class="rule-card__number">C</span>
<p class="rule-card__title">Composite Orchestrator</p>
<p class="rule-card__desc">Orchestrates all analyzers in dependency order with conditional execution and context passing.</p>
</a>
</div>

</div>

---

## Architecture

| Analyzer | Class | Type | LLM Temp | Signals |
|----------|-------|------|----------|---------|
| [Heuristic](heuristic.md) | `SlackHeuristicAnalyzer` | Pattern matching | None | Interaction, Engagement, Recommendation |
| [Objective](objective.md) | `SlackObjectiveAnalyzer` | LLM classification | 0.0 | Escalation, Intervention, Resolution |
| [Subjective](subjective.md) | `SlackSubjectiveAnalyzer` | LLM assessment | 0.3 | Sentiment, Frustration, Acceptance, Override |
| [Product](product.md) | `SlackProductAnalyzer` | LLM extraction | 0.3 | Learnings, Feature Requests, Priority |
| [Feedback](feedback.md) | `SlackFeedbackAttributionAnalyzer` | LLM attribution | 0.2 | Failed Step, Remediation |
| [Composite](composite.md) | `UnderwritingCompositeEvaluator` | Orchestrator | Delegates | All of the above |

### Pipeline Flow

The composite orchestrator runs analyzers in strict dependency order:

```
Objective (always) → Subjective (if human messages) → Feedback (conditional) → Product (always*)
```

Feedback attribution only fires when friction is detected (`has_intervention` OR `is_escalated` OR `frustration_score > 0.5`), saving LLM costs on positive interactions.

---

## KPIs

| KPI | Source Analyzer | Formula |
|-----|-----------------|---------|
| `interaction_rate` | Heuristic | Interactive threads / Total eligible cases |
| `MAU` | Heuristic | Unique senders in 30 days |
| `engagement_rate` | Heuristic | Avg interactions per case |
| `stp_rate` | Objective | Threads with no intervention / Total |
| `escalation_rate` | Objective | Escalated cases / Total AI cases |
| `intervention_rate` | Objective | Threads with intervention / Total |
| `resolution_rate` | Objective | Resolved / Total |
| `stalemate_rate` | Objective | Stalemates / Total |
| `frustration_rate` | Subjective | Frustrated interactions / Total |
| `acceptance_rate` | Subjective | Accepted recommendations / Total |
| `override_rate` | Subjective | Overridden recommendations / Total |
| `override_satisfaction` | Subjective | Satisfactory overrides / Total overrides |

---

## Utility Functions

**File:** `utils.py`

Common utilities used across metrics:

| Function | Description |
|----------|-------------|
| `parse_slack_metadata()` | Extract thread_ts, channel_id, sender from additional_input |
| `extract_mentions()` | Extract @mentions from message text |
| `get_human_messages()` | Filter human messages from conversation |
| `get_ai_messages()` | Filter AI messages from conversation |
| `find_recommendation_turn()` | Find turn containing AI recommendation |
| `extract_recommendation_type()` | Extract approve/decline/review/hold |
| `extract_case_id()` | Extract case ID (MGT-BOP-XXXXXXX) |
| `extract_priority_score()` | Extract base/priority score |
| `count_questions()` | Count questions in text |
| `build_transcript()` | Build plain text transcript from conversation |
| `analyze_reactions()` | Analyze emoji reactions on messages |
| `detect_stalemate()` | Detect repeated bot messages |
| `calculate_time_to_resolution()` | Calculate time between first and last message |
