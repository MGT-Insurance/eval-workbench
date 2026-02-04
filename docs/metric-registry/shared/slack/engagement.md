# Thread Engagement Analyzer

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Measure engagement depth within Slack threads</strong><br>
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
    Thread Engagement Analyzer measures the **depth and quality** of engagement within Slack threads. It counts back-and-forth exchanges, tracks response lengths, detects questions and @mentions, and identifies unique participants.

    | Signal | Description |
    |--------|-------------|
    | `interaction_depth` | Number of back-and-forth exchanges |
    | `question_count` | Questions asked in the thread |
    | `mention_count` | @mentions in the thread |
    | `unique_participants` | Distinct users involved |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Measuring conversation depth</li>
<li>Identifying highly interactive threads</li>
<li>Tracking participation patterns</li>
<li>Building engagement KPIs</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Need quality assessment</li>
<li>Single-message threads</li>
<li>Measuring sentiment</li>
</ul>
</div>

</div>

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.dataset import DatasetItem
    from shared.metrics.slack.engagement import ThreadEngagementAnalyzer

    metric = ThreadEngagementAnalyzer()

    item = DatasetItem(
        conversation=[
            {"role": "assistant", "content": "Here's my analysis."},
            {"role": "user", "content": "Can you clarify?"},
            {"role": "assistant", "content": "Sure, here's more detail."},
            {"role": "user", "content": "Thanks!"},
        ]
    )

    result = await metric.execute(item)
    print(f"Depth: {result.signals.interaction_depth}")
    print(f"Questions: {result.signals.question_count}")
    ```

---

## Metric Diagnostics

<details markdown="1">
<summary><strong>📊 ThreadEngagementResult Structure</strong></summary>

```python
ThreadEngagementResult(
{
    "interaction_depth": 4,
    "has_multiple_interactions": true,
    "avg_human_response_length": 45,
    "avg_ai_response_length": 120,
    "question_count": 1,
    "mention_count": 0,
    "unique_participants": 2
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `interaction_depth` | `int` | Back-and-forth exchange count |
| `has_multiple_interactions` | `bool` | More than one exchange |
| `avg_human_response_length` | `int` | Average human message length |
| `avg_ai_response_length` | `int` | Average AI message length |
| `question_count` | `int` | Questions detected |
| `mention_count` | `int` | @mentions detected |
| `unique_participants` | `int` | Distinct participants |

</details>

---

## Quick Reference

!!! note "TL;DR"
    **Thread Engagement Analyzer** = How deep and interactive is this conversation?

    - **Use it when:** Measuring conversation engagement patterns
    - **Output type:** Analysis signals (no score)
    - **Key signals:** `interaction_depth`, `question_count`

<div class="grid cards" markdown>

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Interaction](./interaction.md) · Sentiment · Resolution

</div>
