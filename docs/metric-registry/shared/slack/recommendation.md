# Recommendation Analyzer

## At a Glance

- Score range: none (analysis-only)
- Default threshold: none
- Required inputs: `conversation`
- Optional inputs: `additional_input`

## What It Measures

Extracts AI recommendations from Slack conversations for KPI aggregation.

## How It Works

- Scans AI messages for recommendation patterns.
- Extracts recommendation type, turn index, and optional confidence.
- Extracts case identifiers and priority scores.

## Configuration

- No custom parameters.

## Signals and Diagnostics

`RecommendationResult` with:

- `has_recommendation`, `recommendation_type`
- `recommendation_turn_index`, `recommendation_confidence`
- `case_id`, `case_priority`

## Example

```python
from axion.dataset import DatasetItem
from shared.metrics.slack.recommendation import RecommendationAnalyzer

metric = RecommendationAnalyzer()
item = DatasetItem(conversation=[{"role": "assistant", "content": "Recommend Approve"}])
result = await metric.execute(item)
```

## Quick Reference

- Metric key: `recommendation_analyzer`
- Category: Analysis
