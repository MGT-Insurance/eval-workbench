# Slack Conversation Analyzer

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Comprehensive multi-metric analysis in a single pass</strong><br>
<span class="badge" style="margin-top: 0.5rem;">LLM-Powered</span>
<span class="badge" style="background: #667eea;">Composite</span>
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
<small style="color: var(--md-text-muted);">Varies by sub-metric</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>conversation</code><br>
<small style="color: var(--md-text-muted);">Optional: additional_input</small>
</div>

</div>

!!! abstract "What It Measures"
    The Slack Conversation Analyzer runs **multiple KPI metrics** in a single optimized pass. It analyzes interaction patterns, engagement depth, recommendations, escalations, sentiment, and more—returning a comprehensive signal structure for reporting.

    | Sub-Metric | Type | Description |
    |------------|------|-------------|
    | Interaction | Analysis | Message counts and patterns |
    | Engagement | Analysis | Depth and quality of exchanges |
    | Recommendation | Analysis | AI recommendation detection |
    | Escalation | Classification | Human escalation detection |
    | Frustration | Score | User frustration level |
    | Acceptance | Classification | Recommendation acceptance |
    | Override | Classification | Human override detection |
    | Intervention | Classification | Human intervention type |
    | Sentiment | Score | User sentiment analysis |
    | Resolution | Classification | Thread resolution status |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Running multiple Slack metrics</li>
<li>Building comprehensive dashboards</li>
<li>Batch processing conversations</li>
<li>Optimizing LLM call costs</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Only need one specific metric</li>
<li>Custom metric configuration needed</li>
<li>Non-Slack conversations</li>
</ul>
</div>

</div>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `metrics` | `list[str]` | All | Which metrics to run |
    | `frustration_threshold` | `float` | `0.6` | Frustration score threshold |
    | `satisfaction_threshold` | `float` | `0.7` | Satisfaction score threshold |
    | `sentiment_threshold` | `float` | `0.4` | Sentiment score threshold |

    **Available Metrics:**
    `interaction`, `engagement`, `recommendation`, `escalation`, `frustration`, `acceptance`, `override`, `satisfaction`, `intervention`, `sentiment`, `resolution`

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.dataset import DatasetItem
    from shared.metrics.slack.composite import SlackConversationAnalyzer

    metric = SlackConversationAnalyzer()

    item = DatasetItem(
        conversation=[
            {"role": "assistant", "content": "Recommend Approve."},
            {"role": "user", "content": "Thanks!"},
        ]
    )

    result = await metric.execute(item)
    # Access all sub-metric signals
    print(result.signals.interaction)
    print(result.signals.sentiment)
    print(result.signals.resolution)
    ```

=== ":material-cog-outline: Selective Metrics"

    ```python
    from axion.dataset import DatasetItem
    from shared.metrics.slack.composite import SlackConversationAnalyzer

    # Only run specific metrics
    metric = SlackConversationAnalyzer(
        metrics=["interaction", "sentiment", "resolution"],
        sentiment_threshold=0.4,
    )

    item = DatasetItem(conversation=[...])
    result = await metric.execute(item)
    ```

---

## Metric Diagnostics

<details markdown="1">
<summary><strong>📊 CompositeResult Structure</strong></summary>

```python
CompositeResult(
{
    "interaction": {
        "ai_message_count": 2,
        "human_message_count": 3,
        "is_interactive": true
    },
    "engagement": {
        "interaction_depth": 5,
        "question_count": 2
    },
    "recommendation": {
        "has_recommendation": true,
        "recommendation_type": "approve"
    },
    "sentiment": {
        "sentiment": "positive",
        "sentiment_score": 0.75
    },
    "resolution": {
        "is_resolved": true,
        "final_status": "approved"
    }
    // ... other metrics
}
)
```

</details>

---

## Quick Reference

!!! note "TL;DR"
    **Slack Conversation Analyzer** = All Slack KPI metrics in one efficient pass

    - **Use it when:** You need multiple Slack metrics together
    - **Score interpretation:** Varies by sub-metric
    - **Key benefit:** Single LLM call for multiple insights

<div class="grid cards" markdown>

- :material-link-variant: **All Sub-Metrics**

    [:octicons-arrow-right-24: Interaction](./interaction.md) · Engagement · Sentiment · Resolution · Override · Escalation

</div>
