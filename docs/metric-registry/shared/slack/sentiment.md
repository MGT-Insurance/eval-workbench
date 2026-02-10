# Sentiment Detector

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Detect user sentiment in Slack threads</strong><br>
<span class="badge" style="margin-top: 0.5rem;">LLM-Powered</span>
<span class="badge" style="background: #667eea;">Score</span>
<span class="badge" style="background: #3b82f6;">Slack</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Higher = more positive</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">⚡</span><br>
<strong>Default Threshold</strong><br>
<code style="font-size: 1.5rem; color: var(--md-primary);">0.4</code><br>
<small style="color: var(--md-text-muted);">Below = frustrated</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>conversation</code><br>
<small style="color: var(--md-text-muted);">Optional: additional_input</small>
</div>

</div>

!!! abstract "What It Measures"
    Sentiment Detector analyzes **user sentiment** in Slack threads, classifying it as positive, neutral, frustrated, or confused. It returns a calibrated sentiment score and identifies frustration indicators.

    | Score | Interpretation |
    |-------|----------------|
    | **0.7-1.0** | :material-check-all: Positive sentiment |
    | **0.4-0.7** | :material-alert: Neutral sentiment |
    | **0.0-0.4** | :material-close: Frustrated/negative |

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `frustration_threshold` | `float` | `0.4` | Score below which user is flagged frustrated |

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.shared.metrics.slack.sentiment import SentimentDetector

    metric = SentimentDetector(frustration_threshold=0.4)

    item = DatasetItem(
        conversation=[
            {"role": "assistant", "content": "Here's the recommendation."},
            {"role": "user", "content": "This still doesn't work??"},
        ]
    )

    result = await metric.execute(item)
    print(f"Sentiment: {result.signals.sentiment}")
    print(f"Score: {result.signals.sentiment_score}")
    print(f"Frustrated: {result.signals.is_frustrated}")
    ```

---

## Metric Diagnostics

<details markdown="1">
<summary><strong>📊 SentimentResult Structure</strong></summary>

```python
SentimentResult(
{
    "sentiment": "frustrated",
    "sentiment_score": 0.3,
    "is_frustrated": true,
    "is_positive": false,
    "is_confused": true,
    "frustration_indicators": ["question marks", "negative phrasing"],
    "peak_sentiment_turn": 1,
    "human_message_count": 1,
    "reasoning": "User shows frustration with repeated issues"
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `sentiment` | `str` | positive, neutral, frustrated, confused |
| `sentiment_score` | `float` | Calibrated score 0.0-1.0 |
| `is_frustrated` | `bool` | Score below threshold |
| `is_positive` | `bool` | Positive sentiment |
| `is_confused` | `bool` | User seems confused |
| `frustration_indicators` | `List[str]` | Detected indicators |
| `peak_sentiment_turn` | `int` | Most significant turn |

</details>

---

## Quick Reference

!!! note "TL;DR"
    **Sentiment Detector** = What's the user's overall sentiment?

    - **Use it when:** Monitoring user experience and satisfaction
    - **Score interpretation:** Higher = more positive
    - **Key signals:** `sentiment`, `sentiment_score`, `is_frustrated`

<div class="grid cards" markdown>

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Frustration](./frustration.md) · Escalation · Resolution

</div>
