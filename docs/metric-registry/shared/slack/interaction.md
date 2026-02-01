# Slack Interaction Analyzer

## At a Glance

- Score range: none (analysis-only)
- Default threshold: none
- Required inputs: `conversation`
- Optional inputs: `additional_input`

## What It Measures

Extracts interaction signals from Slack threads for KPI aggregation (message
counts, interaction flags, metadata).

## How It Works

- Counts AI and human messages.
- Determines if the thread is interactive.
- Extracts thread metadata (thread_id, channel_id, sender).

## Configuration

- No custom parameters.

## Signals and Diagnostics

`SlackInteractionResult` with:

- `ai_message_count`, `human_message_count`, `total_turn_count`
- `is_interactive`, `is_ai_initiated`, `has_human_response`
- `thread_id`, `channel_id`, `sender`

## Example

```python
from axion.dataset import DatasetItem
from shared.metrics.slack.interaction import SlackInteractionAnalyzer

metric = SlackInteractionAnalyzer()
item = DatasetItem(conversation=[{"role": "assistant", "content": "Hello"}])
result = await metric.execute(item)
```

## Quick Reference

- Metric key: `slack_interaction_analyzer`
- Category: Analysis
