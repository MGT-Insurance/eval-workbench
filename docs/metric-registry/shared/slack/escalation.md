# Escalation Detector

## At a Glance

- Score range: `0.0` to `1.0`
- Default threshold: `0.5`
- Required inputs: `conversation`

## What It Measures

Detects when Slack conversations escalate to human team members.

## How It Works

- Builds a transcript and detects @mentions.
- Uses LLM classification to identify escalation type.
- Outputs a classification score and escalation signals.

## Configuration

- No custom parameters.

## Signals and Diagnostics

`EscalationResult` with:

- `is_escalated`, `escalation_type`
- `escalation_turn_index`, `escalation_targets`, `escalation_reason`

## Example

```python
from axion.dataset import DatasetItem
from shared.metrics.slack.escalation import EscalationDetector

metric = EscalationDetector()
item = DatasetItem(conversation=[{"role": "user", "content": "@team please help"}])
result = await metric.execute(item)
```

## Quick Reference

- Metric key: `escalation_detector`
- Category: Classification
