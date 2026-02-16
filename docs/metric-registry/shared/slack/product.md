# Product Analyzer

<div style="background: linear-gradient(135deg, #1E3A5F 0%, #0F2440 100%); padding: 24px; border-radius: 12px; color: white; margin: 20px 0;">

<p style="margin: 0; font-size: 16px; line-height: 1.6;">
<strong>Actionable product insights extraction via LLM.</strong> Identifies UX issues, accuracy problems, workflow friction, feature requests, and rule configuration needs from Slack conversations.
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
<span class="at-a-glance__value"><code>SlackProductAnalyzer</code></span>
</div>
<div class="at-a-glance__item">
<span class="at-a-glance__label">Base</span>
<span class="at-a-glance__value"><code>BaseMetric</code></span>
</div>
<div class="at-a-glance__item">
<span class="at-a-glance__label">Source</span>
<span class="at-a-glance__value"><code>shared/metrics/slack/product.py</code></span>
</div>
<div class="at-a-glance__item">
<span class="at-a-glance__label">Temperature</span>
<span class="at-a-glance__value">0.3</span>
</div>
<div class="at-a-glance__item at-a-glance__item--wide">
<span class="at-a-glance__label">Signals</span>
<span class="at-a-glance__value"><code>ProductSignalsOutput</code> — learnings, feature requests, priority</span>
</div>
</div>
</div>

---

## Constructor

```python
from eval_workbench.shared.metrics.slack.product import SlackProductAnalyzer

analyzer = SlackProductAnalyzer(
    config=None,                # Optional[AnalyzerConfig]
    analysis_context=None,      # Optional[Dict[str, Any]] — from other analyses
)
```

---

## Signals

### ProductSignalsOutput

| Signal | Type | Description |
|--------|------|-------------|
| `learnings` | list[str] | Actionable insights extracted from conversation |
| `learning_categories` | list[str] | Categories: ux, accuracy, coverage, speed, workflow, rules, guardrails, other |
| `feature_requests` | list[str] | Explicit or implicit feature requests |
| `has_actionable_feedback` | bool | Whether actionable feedback was found |
| `priority_level` | str | high / medium / low / none |
| `suggested_action` | str | Recommended next step (optional) |
| `reasoning_trace` | str | LLM reasoning chain |

### Learning Categories

| Category | Description |
|----------|-------------|
| `ux` | User interface / experience issues |
| `accuracy` | Incorrect or imprecise outputs |
| `coverage` | Missing knowledge or capabilities |
| `speed` | Performance / response time concerns |
| `workflow` | Process or workflow friction |
| `rules` | Rule configuration needs |
| `guardrails` | Safety or boundary issues |
| `other` | Uncategorized feedback |

---

## Analysis Context

The product analyzer can receive context from other analyses to enrich its extraction:

- `has_intervention` — Whether human intervention occurred
- `intervention_type` — Type of intervention
- `is_frustrated` — Whether frustration was detected
- `sentiment` — User sentiment label

---

## Sub-Metrics

<div class="sub-metrics">
<div class="sub-metric">
<span class="sub-metric__icon sub-metric__icon--score">0–1</span>
<span class="sub-metric__text">
<span class="sub-metric__name">learnings</span>
<span class="sub-metric__desc">Number of learnings extracted</span>
</span>
</div>
<div class="sub-metric">
<span class="sub-metric__icon sub-metric__icon--score">0–1</span>
<span class="sub-metric__text">
<span class="sub-metric__name">feature_requests</span>
<span class="sub-metric__desc">Feature requests identified</span>
</span>
</div>
<div class="sub-metric">
<span class="sub-metric__icon sub-metric__icon--classification">cls</span>
<span class="sub-metric__text">
<span class="sub-metric__name">actionable_feedback</span>
<span class="sub-metric__desc">Actionable feedback exists</span>
</span>
</div>
<div class="sub-metric">
<span class="sub-metric__icon sub-metric__icon--classification">cls</span>
<span class="sub-metric__text">
<span class="sub-metric__name">priority_level</span>
<span class="sub-metric__desc">Priority classification</span>
</span>
</div>
</div>

---

## Usage

```python
from eval_workbench.shared.metrics.slack.product import SlackProductAnalyzer

analyzer = SlackProductAnalyzer()
result = await analyzer.execute(dataset_item)

signals = result.signals
print(f"Learnings: {signals.learnings}")
print(f"Priority: {signals.priority_level}")
print(f"Feature requests: {signals.feature_requests}")

# With context from other analyses
analyzer = SlackProductAnalyzer(
    analysis_context={
        "has_intervention": True,
        "intervention_type": "correction",
        "is_frustrated": False,
        "sentiment": "neutral",
    }
)
```

---

## KPIs Supported

- Product insight extraction for daily/weekly reports
- Feature request tracking and prioritization
- Friction point identification across conversations
