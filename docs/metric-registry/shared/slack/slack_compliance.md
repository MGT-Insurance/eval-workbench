# Slack Formatting Compliance

## At a Glance

- Score range: `0.0` to `1.0`
- Default threshold: `1.0`
- Required inputs: `actual_output`

## What It Measures

Ensures output adheres to strict Slack mrkdwn rules (no `#` headers, no
`**bold**`, and uses backticks for data values).

## How It Works

- Detects double-asterisk bold.
- Detects Markdown headers.
- Detects unwrapped numbers and currency values.
- Deducts points per issue type, clamped at 0.0.

## Configuration

- No custom parameters.

## Signals and Diagnostics

`FormattingResult` with:

- `score`
- `issues` (issue type, context, count)

## Example

```python
from axion.dataset import DatasetItem
from shared.metrics.slack.slack_compliance import SlackFormattingCompliance

metric = SlackFormattingCompliance()
item = DatasetItem(actual_output="**Header** # Bad $500")
result = await metric.execute(item)
```

## Quick Reference

- Metric key: `slack_compliance`
- Category: Slack
