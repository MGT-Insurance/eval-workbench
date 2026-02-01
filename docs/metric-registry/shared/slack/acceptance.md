# Acceptance Detector

## At a Glance

- Score range: `0.0` to `1.0`
- Default threshold: `0.5`
- Required inputs: `conversation`

## What It Measures

Determines whether AI recommendations in Slack conversations were accepted.

## How It Works

- Detects the recommendation turn and type.
- Uses LLM classification on post-recommendation messages.
- Maps acceptance status to a numeric score.

## Configuration

- No custom parameters.

## Signals and Diagnostics

`AcceptanceResult` with:

- `acceptance_status`, `is_accepted`
- `acceptance_turn_index`, `decision_maker`, `turns_to_decision`
- `has_recommendation`

## Example

```python
from axion.dataset import DatasetItem
from shared.metrics.slack.acceptance import AcceptanceDetector

metric = AcceptanceDetector()
item = DatasetItem(conversation=[{"role": "assistant", "content": "Recommend Approve"}])
result = await metric.execute(item)
```

## Quick Reference

- Metric key: `acceptance_detector`
- Category: Classification
