# Slack Interaction Analyzer

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Extract interaction signals from Slack threads</strong><br>
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
    Slack Interaction Analyzer extracts **basic interaction signals** from Slack threads for KPI aggregation. It counts messages by role, determines interactivity, and extracts thread metadata.

    | Signal | Description |
    |--------|-------------|
    | `ai_message_count` | Number of AI messages |
    | `human_message_count` | Number of human messages |
    | `is_interactive` | Whether the thread has back-and-forth |
    | `is_ai_initiated` | Whether AI started the thread |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Building KPI dashboards</li>
<li>Counting interactions</li>
<li>Identifying interactive vs. one-way threads</li>
<li>Extracting thread metadata</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Need engagement depth (use Engagement)</li>
<li>Need quality analysis</li>
<li>Need sentiment analysis</li>
</ul>
</div>

</div>

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.shared.metrics.slack.interaction import SlackInteractionAnalyzer

    metric = SlackInteractionAnalyzer()

    item = DatasetItem(
        conversation=[
            {"role": "assistant", "content": "Hello, how can I help?"},
            {"role": "user", "content": "I need help with a quote."},
            {"role": "assistant", "content": "Sure, here's the analysis."},
        ]
    )

    result = await metric.execute(item)
    print(f"AI Messages: {result.signals.ai_message_count}")
    print(f"Human Messages: {result.signals.human_message_count}")
    print(f"Interactive: {result.signals.is_interactive}")
    ```

---

## Metric Diagnostics

<details markdown="1">
<summary><strong>📊 SlackInteractionResult Structure</strong></summary>

```python
SlackInteractionResult(
{
    "ai_message_count": 2,
    "human_message_count": 1,
    "total_turn_count": 3,
    "is_interactive": true,
    "is_ai_initiated": true,
    "has_human_response": true,
    "thread_id": "thread_123",
    "channel_id": "channel_456",
    "sender": "athena_bot"
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `ai_message_count` | `int` | AI message count |
| `human_message_count` | `int` | Human message count |
| `total_turn_count` | `int` | Total messages |
| `is_interactive` | `bool` | Has human participation |
| `is_ai_initiated` | `bool` | AI sent first message |
| `has_human_response` | `bool` | Human replied |
| `thread_id` | `str` | Thread identifier |
| `channel_id` | `str` | Channel identifier |
| `sender` | `str` | Initiating sender |

</details>

---

## Quick Reference

!!! note "TL;DR"
    **Slack Interaction Analyzer** = Basic interaction counts and metadata

    - **Use it when:** Need simple interaction metrics for dashboards
    - **Output type:** Analysis signals (no score)
    - **Key signals:** `ai_message_count`, `human_message_count`, `is_interactive`

<div class="grid cards" markdown>

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Engagement](./engagement.md) · Recommendation · Resolution

</div>
