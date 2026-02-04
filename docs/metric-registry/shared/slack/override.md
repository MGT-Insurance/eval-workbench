# Override Detector

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Detect when humans override AI recommendations</strong><br>
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
    Override Detector identifies when humans **override AI recommendations** in Slack conversations. It detects the recommendation, identifies the override, extracts the original vs. final decision, and categorizes the override reason.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: Clear override detected |
    | **0.5** | :material-alert: Possible override |
    | **0.0** | :material-close: No override |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Tracking override rates</li>
<li>Understanding AI vs. human decisions</li>
<li>Identifying improvement areas</li>
<li>Building override KPIs</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>No AI recommendation present</li>
<li>Need acceptance tracking (use Acceptance)</li>
<li>Measuring general intervention (use Intervention)</li>
</ul>
</div>

</div>

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.shared.metrics.slack.override import OverrideDetector

    metric = OverrideDetector()

    item = DatasetItem(
        conversation=[
            {"role": "assistant", "content": "Recommend Approve based on the data."},
            {"role": "user", "content": "We will decline anyway due to prior relationship."},
        ]
    )

    result = await metric.execute(item)
    print(f"Overridden: {result.signals.is_overridden}")
    print(f"Original: {result.signals.original_recommendation}")
    print(f"Final: {result.signals.final_decision}")
    print(f"Reason: {result.signals.override_reason}")
    ```

---

## Metric Diagnostics

<details markdown="1">
<summary><strong>📊 OverrideResult Structure</strong></summary>

```python
OverrideResult(
{
    "is_overridden": true,
    "override_type": "decision_change",
    "original_recommendation": "approve",
    "final_decision": "decline",
    "override_reason": "Prior relationship concerns",
    "override_reason_category": "business_judgment"
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `is_overridden` | `bool` | Whether override occurred |
| `override_type` | `str` | Type of override |
| `original_recommendation` | `str` | AI's recommendation |
| `final_decision` | `str` | Human's final decision |
| `override_reason` | `str` | Stated reason for override |
| `override_reason_category` | `str` | Categorized reason |

</details>

---

## Quick Reference

!!! note "TL;DR"
    **Override Detector** = Did the human override the AI's recommendation?

    - **Use it when:** Tracking when AI recommendations are changed
    - **Score interpretation:** 1.0 = overridden, 0.0 = no override
    - **Key signals:** `is_overridden`, `original_recommendation`, `final_decision`

<div class="grid cards" markdown>

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Acceptance](./acceptance.md) · Override Satisfaction · Intervention

</div>
