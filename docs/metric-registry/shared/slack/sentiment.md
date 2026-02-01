# Sentiment Detector

## At a Glance

- Score range: `0.0` to `1.0` (higher = more positive)
- Default threshold: `0.4` (below = frustrated)
- Required inputs: `conversation`
- Optional inputs: `additional_input`

## What It Measures

Detects user sentiment in Slack threads (positive / neutral / frustrated / confused) and returns a sentiment score.

## How It Works

- Builds a conversation transcript from the thread.
- If there are no human messages, returns a neutral baseline.
- Uses an LLM to classify sentiment and produce a calibrated `sentiment_score`.
- Falls back to heuristic text-pattern detection if the LLM fails.

## Configuration

- `frustration_threshold` (float, default `0.4`): Score below which the user is flagged as frustrated.

## Signals and Diagnostics

`SentimentResult` with:

- `sentiment`, `sentiment_score`
- `is_frustrated`, `is_positive`, `is_confused`
- `frustration_indicators`, `peak_sentiment_turn`
- `human_message_count`, `reasoning`

## Example

```python
from axion.dataset import DatasetItem
from shared.metrics.slack.sentiment import SentimentDetector

metric = SentimentDetector(frustration_threshold=0.4)
item = DatasetItem(conversation=[{"role": "user", "content": "This still doesn't work??"}])
result = await metric.execute(item)
```

## Quick Reference

- Metric key: `sentiment_detector`
- Category: Score
