# Frustration Detector

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Score user frustration levels in conversations</strong><br>
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
<small style="color: var(--md-text-muted);">Frustration level</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">⚡</span><br>
<strong>Default Threshold</strong><br>
<code style="font-size: 1.5rem; color: var(--md-primary);">0.6</code><br>
<small style="color: var(--md-text-muted);">Frustrated flag cutoff</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>conversation</code><br>
<small style="color: var(--md-text-muted);">Slack thread messages</small>
</div>

</div>

!!! abstract "What It Measures"
    Frustration Detector scores **user frustration levels** in Slack conversations. It analyzes human messages for frustration indicators, identifies the cause, and flags conversations where frustration exceeds the threshold.

    | Score | Interpretation |
    |-------|----------------|
    | **0.0-0.3** | :material-check-all: Calm, satisfied user |
    | **0.4-0.6** | :material-alert: Mild frustration |
    | **0.7-1.0** | :material-close: High frustration |

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.shared.metrics.slack.frustration import FrustrationDetector

    metric = FrustrationDetector()

    item = DatasetItem(
        conversation=[
            {"role": "assistant", "content": "Processing your request."},
            {"role": "user", "content": "This is the third time I've asked!"},
        ]
    )

    result = await metric.execute(item)
    print(f"Score: {result.signals.frustration_score}")
    print(f"Frustrated: {result.signals.is_frustrated}")
    print(f"Cause: {result.signals.frustration_cause}")
    ```

---

## Metric Diagnostics

<details markdown="1">
<summary><strong>📊 FrustrationResult Structure</strong></summary>

```python
FrustrationResult(
{
    "frustration_score": 0.75,
    "is_frustrated": true,
    "frustration_indicators": ["repetition", "exclamation"],
    "peak_frustration_turn": 1,
    "frustration_cause": "Repeated requests without resolution"
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `frustration_score` | `float` | Frustration level 0.0-1.0 |
| `is_frustrated` | `bool` | Score exceeds threshold |
| `frustration_indicators` | `List[str]` | Detected frustration signals |
| `peak_frustration_turn` | `int` | Highest frustration turn |
| `frustration_cause` | `str` | Identified cause |

</details>

---

## Quick Reference

!!! note "TL;DR"
    **Frustration Detector** = How frustrated is the user?

    - **Use it when:** Monitoring user frustration in conversations
    - **Score interpretation:** Higher = more frustrated
    - **Key signals:** `frustration_score`, `is_frustrated`

<div class="grid cards" markdown>

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Sentiment](./sentiment.md) · Escalation · Intervention

</div>
