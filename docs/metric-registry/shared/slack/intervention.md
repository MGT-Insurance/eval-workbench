# Intervention Detector

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Detect and classify human intervention types</strong><br>
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
<small style="color: var(--md-text-muted);">Optional: additional_input</small>
</div>

</div>

!!! abstract "What It Measures"
    Intervention Detector identifies when a human **intervenes** in an AI-driven Slack thread and classifies the intervention type. It distinguishes between soft escalations (clarifications), hard escalations (corrections), authority escalations (approvals), and STP (straight-through processing with no human involvement).

    | Escalation Type | Description |
    |-----------------|-------------|
    | **none** | No intervention - STP workflow |
    | **soft** | Clarification or minor adjustment |
    | **hard** | Correction or significant change |
    | **authority** | Approval or authorization required |

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.shared.metrics.slack.intervention import InterventionDetector

    metric = InterventionDetector()

    item = DatasetItem(
        conversation=[
            {"role": "assistant", "content": "Recommend Approve."},
            {"role": "user", "content": "I need to override - decline due to roof age."},
        ]
    )

    result = await metric.execute(item)
    print(f"Has Intervention: {result.signals.has_intervention}")
    print(f"Type: {result.signals.intervention_type}")
    print(f"Escalation: {result.signals.escalation_type}")
    print(f"Is STP: {result.signals.is_stp}")
    ```

---

## Metric Diagnostics

<details markdown="1">
<summary><strong>📊 InterventionResult Structure</strong></summary>

```python
InterventionResult(
{
    "has_intervention": true,
    "intervention_type": "override",
    "escalation_type": "hard",
    "is_hard_escalation": true,
    "is_soft_escalation": false,
    "is_authority_escalation": false,
    "is_stp": false,
    "human_message_count": 1,
    "friction_point": "AI recommendation contradicted by user",
    "issue_details": "User disagrees with approve decision",
    "intervention_summary": "Override to decline",
    "reasoning": "User explicitly overrode AI recommendation"
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `has_intervention` | `bool` | Whether human intervened |
| `intervention_type` | `str` | Type of intervention |
| `escalation_type` | `str` | hard, soft, authority, or none |
| `is_hard_escalation` | `bool` | Correction/significant change |
| `is_soft_escalation` | `bool` | Clarification/minor adjustment |
| `is_authority_escalation` | `bool` | Approval required |
| `is_stp` | `bool` | Straight-through processing |
| `friction_point` | `str` | Where friction occurred |
| `issue_details` | `str` | Details of the issue |

</details>

---

## Quick Reference

!!! note "TL;DR"
    **Intervention Detector** = What type of human intervention occurred?

    - **Use it when:** Classifying intervention types and measuring STP
    - **Score interpretation:** 1.0 = intervention, 0.0 = STP
    - **Key signals:** `intervention_type`, `escalation_type`, `is_stp`

<div class="grid cards" markdown>

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Escalation](./escalation.md) · Override · Resolution

</div>
