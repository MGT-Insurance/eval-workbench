# Resolution Detector

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Determine the final outcome and resolution status</strong><br>
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
    Resolution Detector determines the **final outcome** of a Slack thread and whether the conversation is resolved. It classifies the status (approved, declined, blocked, needs_info, stalemate, pending) and detects stalemates based on inactivity.

    | Status | Description |
    |--------|-------------|
    | **approved** | Case approved and resolved |
    | **declined** | Case declined and resolved |
    | **blocked** | Blocked pending external action |
    | **needs_info** | Waiting for information |
    | **stalemate** | Inactive for extended period |
    | **pending** | Still in progress |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Tracking resolution rates</li>
<li>Identifying stale threads</li>
<li>Measuring time to resolution</li>
<li>Building outcome KPIs</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Only need recommendation (use Recommendation)</li>
<li>Thread just started</li>
<li>Measuring sentiment</li>
</ul>
</div>

</div>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `stalemate_hours` | `float` | `72.0` | Inactivity threshold for stalemate |

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.shared.metrics.slack.resolution import ResolutionDetector

    metric = ResolutionDetector(stalemate_hours=72.0)

    item = DatasetItem(
        conversation=[
            {"role": "assistant", "content": "Recommend Approve."},
            {"role": "user", "content": "Proceeding with approval. Binding now."},
        ]
    )

    result = await metric.execute(item)
    print(f"Status: {result.signals.final_status}")
    print(f"Resolved: {result.signals.is_resolved}")
    print(f"Stalemate: {result.signals.is_stalemate}")
    ```

---

## Metric Diagnostics

<details markdown="1">
<summary><strong>📊 ResolutionResult Structure</strong></summary>

```python
ResolutionResult(
{
    "final_status": "approved",
    "is_resolved": true,
    "resolution_type": "explicit",
    "is_stalemate": false,
    "time_to_resolution_seconds": 3600,
    "message_count": 2,
    "reasoning": "User explicitly confirmed approval and binding"
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `final_status` | `str` | Final outcome status |
| `is_resolved` | `bool` | Whether resolved |
| `resolution_type` | `str` | How it was resolved |
| `is_stalemate` | `bool` | Inactive too long |
| `time_to_resolution_seconds` | `int` | Time to resolve |
| `message_count` | `int` | Total messages |
| `reasoning` | `str` | Explanation |

</details>

---

## Quick Reference

!!! note "TL;DR"
    **Resolution Detector** = What's the final outcome and is it resolved?

    - **Use it when:** Tracking thread resolution and outcomes
    - **Score interpretation:** 1.0 = resolved, 0.0 = unresolved
    - **Key signals:** `final_status`, `is_resolved`, `is_stalemate`

<div class="grid cards" markdown>

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Recommendation](./recommendation.md) · Acceptance · Override

</div>
