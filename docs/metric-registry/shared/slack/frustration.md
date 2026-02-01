# Frustration Detector

## At a Glance

- Score range: `0.0` to `1.0`
- Default threshold: `0.6`
- Required inputs: `conversation`

## What It Measures

Scores user frustration levels in Slack conversations.

## How It Works

- Builds a transcript and extracts human messages.
- Uses LLM to score frustration and identify causes.
- Flags `is_frustrated` when score exceeds threshold.

## Configuration

- No custom parameters.

## Signals and Diagnostics

`FrustrationResult` with:

- `frustration_score`, `is_frustrated`
- `frustration_indicators`, `peak_frustration_turn`, `frustration_cause`

## Example

```python
from axion.dataset import DatasetItem
from shared.metrics.slack.frustration import FrustrationDetector

metric = FrustrationDetector()
item = DatasetItem(conversation=[{"role": "user", "content": "This is wrong again"}])
result = await metric.execute(item)
```

## Quick Reference

- Metric key: `frustration_detector`
- Category: Score
