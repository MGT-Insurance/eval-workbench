# Acceptance Detector

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Detect whether AI recommendations were accepted</strong><br>
<span class="badge" style="margin-top: 0.5rem;">LLM-Powered</span>
<span class="badge" style="background: #667eea;">Classification</span>
<span class="badge" style="background: #3b82f6;">Slack</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Acceptance score</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">⚡</span><br>
<strong>Default Threshold</strong><br>
<code style="font-size: 1.5rem; color: var(--md-primary);">0.5</code><br>
<small style="color: var(--md-text-muted);">Pass/fail cutoff</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>conversation</code><br>
<small style="color: var(--md-text-muted);">Slack thread messages</small>
</div>

</div>

!!! abstract "What It Measures"
    Acceptance Detector determines whether AI recommendations in Slack conversations were **accepted** by the human participants. It identifies the recommendation turn, analyzes post-recommendation messages, and classifies the acceptance status.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: Recommendation explicitly accepted |
    | **0.5** | :material-alert: Implicit or unclear acceptance |
    | **0.0** | :material-close: Recommendation rejected or overridden |

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.shared.metrics.slack.acceptance import AcceptanceDetector

    metric = AcceptanceDetector()

    item = DatasetItem(
        conversation=[
            {"role": "assistant", "content": "Recommend Approve based on clean history."},
            {"role": "user", "content": "Thanks, proceeding with approval."},
        ]
    )

    result = await metric.execute(item)
    print(f"Accepted: {result.signals.is_accepted}")
    print(f"Status: {result.signals.acceptance_status}")
    ```

---

## Metric Diagnostics

<details markdown="1">
<summary><strong>📊 AcceptanceResult Structure</strong></summary>

```python
AcceptanceResult(
{
    "acceptance_status": "accepted",
    "is_accepted": true,
    "acceptance_turn_index": 1,
    "decision_maker": "user",
    "turns_to_decision": 1,
    "has_recommendation": true
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `acceptance_status` | `str` | accepted, rejected, unclear |
| `is_accepted` | `bool` | Whether recommendation was accepted |
| `acceptance_turn_index` | `int` | Turn where acceptance occurred |
| `decision_maker` | `str` | Who made the acceptance decision |
| `turns_to_decision` | `int` | Messages between recommendation and decision |
| `has_recommendation` | `bool` | Whether a recommendation was found |

</details>

---

## Quick Reference

!!! note "TL;DR"
    **Acceptance Detector** = Did the user accept the AI's recommendation?

    - **Use it when:** Tracking whether AI recommendations are followed
    - **Score interpretation:** 1.0 = accepted, 0.0 = rejected
    - **Key output:** `is_accepted` boolean

<div class="grid cards" markdown>

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Override Detector](./override.md) · Recommendation Analyzer

</div>
