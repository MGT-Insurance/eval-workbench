# Recommendation Analyzer

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Extract AI recommendations from Slack conversations</strong><br>
<span class="badge" style="margin-top: 0.5rem;">Rule-Based</span>
<span class="badge" style="background: #667eea;">Analysis</span>
<span class="badge" style="background: #3b82f6;">Slack</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📊</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">—</code><br>
<small style="color: var(--md-text-muted);">Analysis only</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">⚡</span><br>
<strong>Default Threshold</strong><br>
<code style="font-size: 1.5rem; color: var(--md-primary);">—</code><br>
<small style="color: var(--md-text-muted);">Not applicable</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>conversation</code><br>
<small style="color: var(--md-text-muted);">Optional: additional_input</small>
</div>

</div>

!!! abstract "What It Measures"
    Recommendation Analyzer extracts **AI recommendations** from Slack conversations for KPI aggregation. It identifies recommendation type, turn index, confidence, and extracts case identifiers and priority scores.

    | Signal | Description |
    |--------|-------------|
    | `has_recommendation` | Whether AI made a recommendation |
    | `recommendation_type` | approve, decline, refer, etc. |
    | `recommendation_confidence` | AI's confidence level |
    | `case_id` | Extracted case identifier |

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.shared.metrics.slack.recommendation import RecommendationAnalyzer

    metric = RecommendationAnalyzer()

    item = DatasetItem(
        conversation=[
            {"role": "assistant", "content": "Case #12345: Recommend Approve with high confidence."},
        ]
    )

    result = await metric.execute(item)
    print(f"Has Recommendation: {result.signals.has_recommendation}")
    print(f"Type: {result.signals.recommendation_type}")
    print(f"Case ID: {result.signals.case_id}")
    ```

---

## Metric Diagnostics

<details markdown="1">
<summary><strong>📊 RecommendationResult Structure</strong></summary>

```python
RecommendationResult(
{
    "has_recommendation": true,
    "recommendation_type": "approve",
    "recommendation_turn_index": 0,
    "recommendation_confidence": 0.95,
    "case_id": "12345",
    "case_priority": "high"
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `has_recommendation` | `bool` | Recommendation found |
| `recommendation_type` | `str` | Type of recommendation |
| `recommendation_turn_index` | `int` | Turn with recommendation |
| `recommendation_confidence` | `float` | AI's confidence |
| `case_id` | `str` | Extracted case ID |
| `case_priority` | `str` | Case priority level |

</details>

---

## Quick Reference

!!! note "TL;DR"
    **Recommendation Analyzer** = What did the AI recommend?

    - **Use it when:** Extracting recommendation metadata
    - **Output type:** Analysis signals (no score)
    - **Key signals:** `has_recommendation`, `recommendation_type`, `case_id`

<div class="grid cards" markdown>

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Acceptance](./acceptance.md) · Override · Resolution

</div>
