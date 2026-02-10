# Slack Formatting Compliance

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Ensure output adheres to Slack mrkdwn formatting rules</strong><br>
<span class="badge" style="margin-top: 0.5rem;">Rule-Based</span>
<span class="badge" style="background: #667eea;">Compliance</span>
<span class="badge" style="background: #3b82f6;">Slack</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Compliance score</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">⚡</span><br>
<strong>Default Threshold</strong><br>
<code style="font-size: 1.5rem; color: var(--md-primary);">1.0</code><br>
<small style="color: var(--md-text-muted);">No violations allowed</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>actual_output</code><br>
<small style="color: var(--md-text-muted);">AI response text</small>
</div>

</div>

!!! abstract "What It Measures"
    Slack Formatting Compliance ensures AI output adheres to **Slack mrkdwn rules**. It detects violations like `**bold**` (should be `*bold*`), `# Headers` (not supported), and unwrapped numbers/currency that should use backticks.

    | Issue | Description |
    |-------|-------------|
    | **Double asterisk** | `**bold**` should be `*bold*` |
    | **Markdown headers** | `# Header` not supported |
    | **Unwrapped values** | Numbers/currency should use backticks |

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.shared.metrics.slack.slack_compliance import SlackFormattingCompliance

    metric = SlackFormattingCompliance()

    item = DatasetItem(
        actual_output="**Header** # Bad $500"
    )

    result = await metric.execute(item)
    print(f"Score: {result.score}")
    print(f"Issues: {result.signals.issues}")
    ```

=== ":material-check: Compliant Output"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.shared.metrics.slack.slack_compliance import SlackFormattingCompliance

    metric = SlackFormattingCompliance()

    item = DatasetItem(
        actual_output="*Header* with `$500` properly formatted"
    )

    result = await metric.execute(item)
    print(f"Score: {result.score}")  # 1.0
    ```

---

## Metric Diagnostics

<details markdown="1">
<summary><strong>📊 FormattingResult Structure</strong></summary>

```python
FormattingResult(
{
    "score": 0.7,
    "issues": [
        {
            "type": "double_asterisk",
            "context": "**Header**",
            "count": 1
        },
        {
            "type": "markdown_header",
            "context": "# Bad",
            "count": 1
        },
        {
            "type": "unwrapped_currency",
            "context": "$500",
            "count": 1
        }
    ]
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float` | Compliance score (deductions per issue) |
| `issues` | `List` | Detected formatting issues |

### Issue Types

| Type | Deduction | Description |
|------|-----------|-------------|
| `double_asterisk` | -0.1 | `**text**` instead of `*text*` |
| `markdown_header` | -0.1 | `# Header` used |
| `unwrapped_number` | -0.05 | Numbers not in backticks |
| `unwrapped_currency` | -0.05 | Currency not in backticks |

</details>

---

## Quick Reference

!!! note "TL;DR"
    **Slack Formatting Compliance** = Does the output follow Slack mrkdwn rules?

    - **Use it when:** Validating Slack message formatting
    - **Score interpretation:** 1.0 = compliant, lower = violations
    - **Key signals:** `score`, `issues`

<div class="grid cards" markdown>

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Interaction](./interaction.md) · Recommendation

</div>
