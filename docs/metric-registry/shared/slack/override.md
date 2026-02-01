# Override Detector

## At a Glance

- Score range: `0.0` to `1.0`
- Default threshold: `0.5`
- Required inputs: `conversation`

## What It Measures

Detects when humans override AI recommendations in Slack conversations.

## How It Works

- Detects recommendation turn and type.
- Uses LLM classification to detect overrides.
- Returns override status and reason category.

## Configuration

- No custom parameters.

## Signals and Diagnostics

`OverrideResult` with:

- `is_overridden`, `override_type`
- `original_recommendation`, `final_decision`
- `override_reason`, `override_reason_category`

## Example

```python
from axion.dataset import DatasetItem
from shared.metrics.slack.override import OverrideDetector

metric = OverrideDetector()
item = DatasetItem(conversation=[{"role": "user", "content": "We will decline anyway"}])
result = await metric.execute(item)
```

## Quick Reference

- Metric key: `override_detector`
- Category: Classification
