# Feedback Attribution Analyzer

<div style="background: linear-gradient(135deg, #1E3A5F 0%, #0F2440 100%); padding: 24px; border-radius: 12px; color: white; margin: 20px 0;">

<p style="margin: 0; font-size: 16px; line-height: 1.6;">
<strong>Root cause diagnosis for negative feedback scenarios.</strong> Attributes failures to specific pipeline stages — AI classification, third-party data, rule engine, platform tooling, or chat interface. Conditional execution: only runs when friction is detected.
</p>
</div>

<div class="at-a-glance">
<div class="at-a-glance__header">
<span class="at-a-glance__title">At a Glance</span>
<span class="at-a-glance__badge at-a-glance__badge--llm">LLM · temp 0.2</span>
<span class="at-a-glance__badge at-a-glance__badge--conditional">Conditional</span>
</div>
<div class="at-a-glance__body">
<div class="at-a-glance__item">
<span class="at-a-glance__label">Class</span>
<span class="at-a-glance__value"><code>SlackFeedbackAttributionAnalyzer</code></span>
</div>
<div class="at-a-glance__item">
<span class="at-a-glance__label">Base</span>
<span class="at-a-glance__value"><code>BaseMetric</code></span>
</div>
<div class="at-a-glance__item">
<span class="at-a-glance__label">Source</span>
<span class="at-a-glance__value"><code>shared/metrics/slack/feedback.py</code></span>
</div>
<div class="at-a-glance__item">
<span class="at-a-glance__label">Runs when</span>
<span class="at-a-glance__value">Negative feedback detected</span>
</div>
<div class="at-a-glance__item at-a-glance__item--wide">
<span class="at-a-glance__label">Signals</span>
<span class="at-a-glance__value"><code>FeedbackAttributionOutput</code> — failed_step, failure_evidence, remediation_hint</span>
</div>
</div>
</div>

---

## Constructor

```python
from eval_workbench.shared.metrics.slack.feedback import SlackFeedbackAttributionAnalyzer

analyzer = SlackFeedbackAttributionAnalyzer(
    config=None,                # Optional[AnalyzerConfig]
    analysis_context=None,      # Optional[Dict[str, Any]] — from other analyses
    sentiment_threshold=0.4,    # Sentiment cutoff for triggering analysis
)
```

Temperature is forced to `0.2` for focused, deterministic attribution.

---

## Signals

### FeedbackAttributionOutput

| Signal | Type | Description |
|--------|------|-------------|
| `has_negative_feedback` | bool | Whether negative feedback was detected |
| `failed_step` | str | Pipeline stage where failure occurred (see below) |
| `failure_evidence` | str | Evidence supporting the attribution |
| `confidence` | str | high / medium / low |
| `remediation_hint` | str | Suggested fix (optional) |
| `reasoning_trace` | str | LLM reasoning chain |

### Failed Step Categories

| Step | Description |
|------|-------------|
| `classification_failure` | AI classification or recommendation was wrong |
| `data_integrity_failure` | Third-party data (e.g., Magic Dust) was inaccurate |
| `rule_engine_failure` | Rule engine applied rules incorrectly |
| `system_tooling_failure` | Platform tooling (Socotra, SFX) issues |
| `chat_interface` | Slack formatting or UX problems |
| `unknown` | Cannot determine root cause |

---

## Conditional Execution

The feedback analyzer is designed to run **only when friction is detected**. In the composite pipeline, it fires when any of these conditions are true:

- `has_intervention` is `True`
- `is_escalated` is `True`
- `frustration_score > 0.5`

This prevents unnecessary LLM calls on positive interactions.

---

## Analysis Context

The analyzer receives context from upstream analyses:

- `sentiment_score` — Numeric sentiment (0-1)
- `frustration_score` — Frustration level
- `frustration_cause` — Root cause from subjective analysis
- `has_intervention` — Whether human intervened
- `intervention_type` — Type of intervention

---

## Sub-Metrics

<div class="sub-metrics">
<div class="sub-metric">
<span class="sub-metric__icon sub-metric__icon--classification">cls</span>
<span class="sub-metric__text">
<span class="sub-metric__name">failed_step</span>
<span class="sub-metric__desc">Pipeline stage attribution</span>
</span>
</div>
<div class="sub-metric">
<span class="sub-metric__icon sub-metric__icon--classification">cls</span>
<span class="sub-metric__text">
<span class="sub-metric__name">attribution_confidence</span>
<span class="sub-metric__desc">Confidence in attribution</span>
</span>
</div>
</div>

Only emitted when `has_negative_feedback` is `true`.

---

## Usage

```python
from eval_workbench.shared.metrics.slack.feedback import SlackFeedbackAttributionAnalyzer

analyzer = SlackFeedbackAttributionAnalyzer(
    analysis_context={
        "sentiment_score": 0.3,
        "frustration_score": 0.7,
        "frustration_cause": "ai_error",
        "has_intervention": True,
        "intervention_type": "correction",
    }
)
result = await analyzer.execute(dataset_item)

signals = result.signals
if signals.has_negative_feedback:
    print(f"Failed step: {signals.failed_step}")
    print(f"Evidence: {signals.failure_evidence}")
    print(f"Fix: {signals.remediation_hint}")
```

---

## KPIs Supported

- Failure attribution distribution across pipeline stages
- Root cause analysis for negative feedback trends
- Remediation tracking and prioritization
