# Override Satisfaction Analyzer

## At a Glance

- Score range: `0.0` to `1.0`
- Default threshold: `0.7`
- Required inputs: `conversation`

## What It Measures

Scores the quality of override explanations in Slack conversations.

## How It Works

- Runs only when an override is detected.
- Uses LLM analysis to score explanation quality.
- Returns satisfaction score and guidance signals.

## Configuration

- `satisfaction_threshold`: minimum score for satisfactory overrides

## Signals and Diagnostics

`SatisfactionResult` with:

- `satisfaction_score`, `is_satisfactory`
- `has_clear_reason`, `has_supporting_evidence`, `is_actionable`
- `improvement_suggestions`

## Example

```python
from axion.dataset import DatasetItem
from shared.metrics.slack.override import OverrideSatisfactionAnalyzer

metric = OverrideSatisfactionAnalyzer()
item = DatasetItem(conversation=[{"role": "user", "content": "Override due to roof age"}])
result = await metric.execute(item)
```

## Quick Reference

- Metric key: `override_satisfaction_analyzer`
- Category: Score
