# Escalation Detector

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Detect when conversations escalate to human team members</strong><br>
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
<small style="color: var(--md-text-muted);">Classification score</small>
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
    Escalation Detector identifies when Slack conversations **escalate to human team members**. It analyzes @mentions, conversation flow, and message content to classify the type and reason for escalation.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: Clear escalation detected |
    | **0.5** | :material-alert: Possible escalation |
    | **0.0** | :material-close: No escalation |

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.shared.metrics.slack.escalation import EscalationDetector

    metric = EscalationDetector()

    item = DatasetItem(
        conversation=[
            {"role": "assistant", "content": "I'm unable to process this request."},
            {"role": "user", "content": "@team can someone help with this?"},
        ]
    )

    result = await metric.execute(item)
    print(f"Escalated: {result.signals.is_escalated}")
    print(f"Type: {result.signals.escalation_type}")
    print(f"Targets: {result.signals.escalation_targets}")
    ```

---

## Metric Diagnostics

<details markdown="1">
<summary><strong>📊 EscalationResult Structure</strong></summary>

```python
EscalationResult(
{
    "is_escalated": true,
    "escalation_type": "team_mention",
    "escalation_turn_index": 1,
    "escalation_targets": ["@team"],
    "escalation_reason": "User requested human assistance"
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `is_escalated` | `bool` | Whether escalation occurred |
| `escalation_type` | `str` | Type of escalation |
| `escalation_turn_index` | `int` | Turn where escalation occurred |
| `escalation_targets` | `List[str]` | Who was escalated to |
| `escalation_reason` | `str` | Reason for escalation |

</details>

---

## Quick Reference

!!! note "TL;DR"
    **Escalation Detector** = Did this conversation escalate to a human?

    - **Use it when:** Tracking when AI needs human help
    - **Score interpretation:** 1.0 = escalated, 0.0 = no escalation
    - **Key signals:** `is_escalated`, `escalation_type`

<div class="grid cards" markdown>

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Intervention](./intervention.md) · Frustration · Resolution

</div>
