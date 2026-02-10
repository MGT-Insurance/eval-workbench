# Override Satisfaction Analyzer

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Score the quality of override explanations</strong><br>
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
<small style="color: var(--md-text-muted);">Satisfaction score</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">⚡</span><br>
<strong>Default Threshold</strong><br>
<code style="font-size: 1.5rem; color: var(--md-primary);">0.7</code><br>
<small style="color: var(--md-text-muted);">Satisfactory cutoff</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>conversation</code><br>
<small style="color: var(--md-text-muted);">Slack thread with override</small>
</div>

</div>

!!! abstract "What It Measures"
    Override Satisfaction Analyzer scores the **quality of override explanations** in Slack conversations. It evaluates whether overrides have clear reasons, supporting evidence, and actionable guidance. Only runs when an override is detected.

    | Score | Interpretation |
    |-------|----------------|
    | **0.8-1.0** | :material-check-all: Well-documented override |
    | **0.5-0.7** | :material-alert: Adequate explanation |
    | **< 0.5** | :material-close: Poor or missing explanation |

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `satisfaction_threshold` | `float` | `0.7` | Minimum score for satisfactory |

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.shared.metrics.slack.override import OverrideSatisfactionAnalyzer

    metric = OverrideSatisfactionAnalyzer()

    item = DatasetItem(
        conversation=[
            {"role": "assistant", "content": "Recommend Approve."},
            {"role": "user", "content": "Override to Decline - roof is 25 years old per inspection report, exceeds our 20-year guideline."},
        ]
    )

    result = await metric.execute(item)
    print(f"Score: {result.signals.satisfaction_score}")
    print(f"Satisfactory: {result.signals.is_satisfactory}")
    print(f"Has Clear Reason: {result.signals.has_clear_reason}")
    ```

---

## Metric Diagnostics

<details markdown="1">
<summary><strong>📊 SatisfactionResult Structure</strong></summary>

```python
SatisfactionResult(
{
    "satisfaction_score": 0.85,
    "is_satisfactory": true,
    "has_clear_reason": true,
    "has_supporting_evidence": true,
    "is_actionable": true,
    "improvement_suggestions": []
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `satisfaction_score` | `float` | Quality score 0.0-1.0 |
| `is_satisfactory` | `bool` | Meets threshold |
| `has_clear_reason` | `bool` | Override reason is clear |
| `has_supporting_evidence` | `bool` | Evidence provided |
| `is_actionable` | `bool` | Contains actionable info |
| `improvement_suggestions` | `List[str]` | How to improve |

</details>

---

## Quick Reference

!!! note "TL;DR"
    **Override Satisfaction Analyzer** = Was the override well-documented?

    - **Use it when:** Evaluating override explanation quality
    - **Score interpretation:** Higher = better documentation
    - **Key signals:** `satisfaction_score`, `has_clear_reason`, `has_supporting_evidence`

<div class="grid cards" markdown>

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Override](./override.md) · Acceptance · Intervention

</div>
