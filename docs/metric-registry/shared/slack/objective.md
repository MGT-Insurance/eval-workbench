# Objective Analyzer

<div style="background: linear-gradient(135deg, #1E3A5F 0%, #0F2440 100%); padding: 24px; border-radius: 12px; color: white; margin: 20px 0;">

<p style="margin: 0; font-size: 16px; line-height: 1.6;">
<strong>LLM-powered factual analysis at temperature 0.0.</strong> Classifies escalations, human interventions, and resolution status. Distinguishes data/logic errors from system issues. Falls back to heuristic on LLM failure.
</p>
</div>

<div class="at-a-glance">
<div class="at-a-glance__header">
<span class="at-a-glance__title">At a Glance</span>
<span class="at-a-glance__badge at-a-glance__badge--llm">LLM · temp 0.0</span>
</div>
<div class="at-a-glance__body">
<div class="at-a-glance__item">
<span class="at-a-glance__label">Class</span>
<span class="at-a-glance__value"><code>SlackObjectiveAnalyzer</code></span>
</div>
<div class="at-a-glance__item">
<span class="at-a-glance__label">Base</span>
<span class="at-a-glance__value"><code>BaseMetric</code></span>
</div>
<div class="at-a-glance__item">
<span class="at-a-glance__label">Source</span>
<span class="at-a-glance__value"><code>shared/metrics/slack/objective.py</code></span>
</div>
<div class="at-a-glance__item">
<span class="at-a-glance__label">Temperature</span>
<span class="at-a-glance__value">0.0 (deterministic)</span>
</div>
<div class="at-a-glance__item at-a-glance__item--wide">
<span class="at-a-glance__label">Signals</span>
<span class="at-a-glance__value"><code>EscalationSignals</code> · <code>InterventionSignals</code> · <code>ResolutionSignals</code></span>
</div>
</div>
</div>

---

## Constructor

```python
from eval_workbench.shared.metrics.slack.objective import SlackObjectiveAnalyzer

analyzer = SlackObjectiveAnalyzer(
    config=None,                # Optional[AnalyzerConfig]
    truncation_config=None,     # Optional[TruncationConfig] — conversation truncation
)
```

Temperature is forced to `0.0` for deterministic, reproducible classification.

---

## Signals

### EscalationSignals

| Signal | Type | Description |
|--------|------|-------------|
| `is_escalated` | bool | Whether conversation was escalated |
| `escalation_type` | str | Type: no_escalation, team_mention, explicit_handoff, error_escalation, complexity_escalation |
| `escalation_turn_index` | int | Turn where escalation occurred |
| `escalation_targets` | list | @mentioned users during escalation |
| `escalation_reason` | str | Reason for escalation |

### InterventionSignals

| Signal | Type | Description |
|--------|------|-------------|
| `has_intervention` | bool | Human intervened in the thread |
| `intervention_type` | str | Category of intervention (correction, missing context, approval, etc.) |
| `intervention_escalation` | str | Escalation class: hard / soft / authority / none |
| `is_stp` | bool | Straight-through processing (no intervention/escalation) |
| `intervention_summary` | str | Summary of what happened |
| `friction_point` | str | Concept causing friction (optional) |
| `issue_details` | str | Technical details (optional) |

### ResolutionSignals

| Signal | Type | Description |
|--------|------|-------------|
| `final_status` | str | approved / declined / blocked / needs_info / stalemate / pending |
| `is_resolved` | bool | Whether the thread reached a resolution |
| `resolution_type` | str | How it was resolved |
| `is_stalemate` | bool | Inactive beyond threshold |
| `time_to_resolution_seconds` | float | Time from first to last message (optional) |

---

## Internal Architecture

The analyzer uses a two-stage pattern:

1. **Outer class** (`SlackObjectiveAnalyzer`) handles orchestration, transcript building, and heuristic fallback
2. **Inner LLM class** (`_ObjectiveLLMAnalyzer`) handles the actual LLM call with structured input/output models

The LLM receives an `ObjectiveAnalysisInput` (transcript, recommendation context, mentions, human count) and returns an `ObjectiveAnalysisOutput` with escalation, intervention, and resolution fields.

---

## Sub-Metrics

<div class="sub-metrics">
<div class="sub-metric">
<span class="sub-metric__icon sub-metric__icon--classification">cls</span>
<span class="sub-metric__text">
<span class="sub-metric__name">escalation_type</span>
<span class="sub-metric__desc">Escalation type detected</span>
</span>
</div>
<div class="sub-metric">
<span class="sub-metric__icon sub-metric__icon--classification">cls</span>
<span class="sub-metric__text">
<span class="sub-metric__name">intervention_type</span>
<span class="sub-metric__desc">Intervention category</span>
</span>
</div>
<div class="sub-metric">
<span class="sub-metric__icon sub-metric__icon--classification">cls</span>
<span class="sub-metric__text">
<span class="sub-metric__name">resolution_status</span>
<span class="sub-metric__desc">Final thread status</span>
</span>
</div>
</div>

---

## Usage

```python
from eval_workbench.shared.metrics.slack.objective import SlackObjectiveAnalyzer

analyzer = SlackObjectiveAnalyzer()
result = await analyzer.execute(dataset_item)

signals = result.signals
print(f"Escalated: {signals.escalation.is_escalated}")
print(f"STP: {signals.intervention.is_stp}")
print(f"Status: {signals.resolution.final_status}")
```

---

## KPIs Supported

- `escalation_rate` — Escalated cases / Total AI cases
- `intervention_rate` — Threads with intervention / Total threads
- `stp_rate` — Threads with no intervention / Total threads
- `resolution_rate` — Resolved / Total
- `stalemate_rate` — Stalemates / Total
