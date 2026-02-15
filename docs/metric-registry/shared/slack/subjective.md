# Subjective Analyzer

<div style="background: linear-gradient(135deg, #1E3A5F 0%, #0F2440 100%); padding: 24px; border-radius: 12px; color: white; margin: 20px 0;">

<p style="margin: 0; font-size: 16px; line-height: 1.6;">
<strong>LLM-powered sentiment and quality assessment at temperature 0.3.</strong> Nuanced analysis of user sentiment, frustration, acceptance, override decisions, and satisfaction. Receives objective context to inform assessment.
</p>
</div>

<div class="at-a-glance">
<div class="at-a-glance__header">
<span class="at-a-glance__title">At a Glance</span>
<span class="at-a-glance__badge at-a-glance__badge--llm">LLM · temp 0.3</span>
</div>
<div class="at-a-glance__body">
<div class="at-a-glance__item">
<span class="at-a-glance__label">Class</span>
<span class="at-a-glance__value"><code>SlackSubjectiveAnalyzer</code></span>
</div>
<div class="at-a-glance__item">
<span class="at-a-glance__label">Base</span>
<span class="at-a-glance__value"><code>BaseMetric</code></span>
</div>
<div class="at-a-glance__item">
<span class="at-a-glance__label">Source</span>
<span class="at-a-glance__value"><code>shared/metrics/slack/subjective.py</code></span>
</div>
<div class="at-a-glance__item">
<span class="at-a-glance__label">Depends on</span>
<span class="at-a-glance__value">Objective Analyzer</span>
</div>
<div class="at-a-glance__item at-a-glance__item--wide">
<span class="at-a-glance__label">Signals</span>
<span class="at-a-glance__value"><code>SubjectiveAnalysisOutput</code> — sentiment, frustration, acceptance, override, satisfaction</span>
</div>
</div>
</div>

---

## Constructor

```python
from eval_workbench.shared.metrics.slack.subjective import SlackSubjectiveAnalyzer

analyzer = SlackSubjectiveAnalyzer(
    config=None,                # Optional[AnalyzerConfig]
    objective_context=None,     # Optional[Dict[str, Any]] — from objective analysis
)
```

Temperature is forced to `0.3` for nuanced but semi-consistent outputs.

---

## Signals

### Sentiment

| Signal | Type | Description |
|--------|------|-------------|
| `sentiment` | str | positive / neutral / frustrated / confused |
| `sentiment_trajectory` | str | improving / stable / worsening |
| `sentiment_indicators` | list | Detected sentiment signals |

### Frustration

| Signal | Type | Description |
|--------|------|-------------|
| `frustration_cause` | str | ai_error / data_quality / tooling_friction / rule_rigidity / slow_response / other |
| `frustration_indicators` | list | Specific frustration signals |
| `peak_frustration_turn` | int | Turn with highest frustration |

### Acceptance

| Signal | Type | Description |
|--------|------|-------------|
| `acceptance_status` | str | accepted / accepted_with_discussion / pending / rejected / modified |
| `is_accepted` | bool | Whether recommendation was accepted |
| `acceptance_turn_index` | int | Turn where decision was made |
| `decision_maker` | str | Who made the final decision |

### Override

| Signal | Type | Description |
|--------|------|-------------|
| `is_overridden` | bool | Whether recommendation was overridden |
| `override_type` | str | no_override / full_override / partial_override / pending_override |
| `final_decision` | str | What was actually decided |
| `override_reason` | str | Stated reason for override |
| `override_reason_category` | str | additional_info / risk_assessment / policy_exception / experience_judgment / etc. |

### Satisfaction

| Signal | Type | Description |
|--------|------|-------------|
| `satisfaction_score` | float | Quality score (0.0 - 1.0) |
| `has_clear_reason` | bool | Has stated reason for override |
| `has_supporting_evidence` | bool | Cites specific information |
| `is_actionable` | bool | Provides actionable feedback |
| `improvement_suggestions` | list | Suggestions for AI improvement |

---

## Objective Context

The subjective analyzer receives context from the objective analysis to inform its assessment. These fields are passed as input to the LLM:

- `objective_is_escalated` — Whether escalation was detected
- `objective_has_intervention` — Whether human intervention occurred
- `objective_intervention_type` — Type of intervention
- `objective_final_status` — Resolution status

This prevents the subjective analyzer from contradicting factual findings.

---

## Sub-Metrics

<div class="sub-metrics">
<div class="sub-metric">
<span class="sub-metric__icon sub-metric__icon--classification">cls</span>
<span class="sub-metric__text">
<span class="sub-metric__name">sentiment_category</span>
<span class="sub-metric__desc">Detected sentiment label</span>
</span>
</div>
<div class="sub-metric">
<span class="sub-metric__icon sub-metric__icon--classification">cls</span>
<span class="sub-metric__text">
<span class="sub-metric__name">frustration_cause</span>
<span class="sub-metric__desc">Root cause of frustration</span>
</span>
</div>
<div class="sub-metric">
<span class="sub-metric__icon sub-metric__icon--classification">cls</span>
<span class="sub-metric__text">
<span class="sub-metric__name">acceptance_status</span>
<span class="sub-metric__desc">Recommendation acceptance</span>
</span>
</div>
<div class="sub-metric">
<span class="sub-metric__icon sub-metric__icon--classification">cls</span>
<span class="sub-metric__text">
<span class="sub-metric__name">override_type</span>
<span class="sub-metric__desc">Override classification</span>
</span>
</div>
<div class="sub-metric">
<span class="sub-metric__icon sub-metric__icon--score">0–1</span>
<span class="sub-metric__text">
<span class="sub-metric__name">satisfaction_score</span>
<span class="sub-metric__desc">Override explanation quality</span>
</span>
</div>
</div>

---

## Usage

```python
from eval_workbench.shared.metrics.slack.subjective import SlackSubjectiveAnalyzer

# Standalone usage
analyzer = SlackSubjectiveAnalyzer()
result = await analyzer.execute(dataset_item)

# With objective context (typical in composite pipeline)
analyzer = SlackSubjectiveAnalyzer(
    objective_context={
        "is_escalated": False,
        "has_intervention": True,
        "intervention_type": "correction",
        "final_status": "approved",
    }
)
result = await analyzer.execute(dataset_item)

signals = result.signals
print(f"Sentiment: {signals.sentiment} ({signals.sentiment_trajectory})")
print(f"Override: {signals.override_type}")
```

---

## KPIs Supported

- `frustration_rate` — Frustrated interactions / Total interactions
- `acceptance_rate` — Recommendations accepted / Total recommendations
- `override_rate` — Recommendations overridden / Total recommendations
- `override_satisfaction` — Satisfactory overrides / Total overrides
- Sentiment distribution (positive / neutral / frustrated / confused)
