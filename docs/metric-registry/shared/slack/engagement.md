# Thread Engagement Analyzer

## At a Glance

- Score range: none (analysis-only)
- Default threshold: none
- Required inputs: `conversation`
- Optional inputs: `additional_input`

## What It Measures

Measures engagement depth within Slack threads (interaction depth, response
lengths, question counts, @mentions).

## How It Works

- Counts back-and-forth exchanges and message lengths.
- Detects questions and @mentions.
- Returns engagement signals without a numeric score.

## Configuration

- No custom parameters.

## Signals and Diagnostics

`ThreadEngagementResult` with:

- `interaction_depth`, `has_multiple_interactions`
- `avg_human_response_length`, `avg_ai_response_length`
- `question_count`, `mention_count`, `unique_participants`

## Example

```python
from axion.dataset import DatasetItem
from shared.metrics.slack.engagement import ThreadEngagementAnalyzer

metric = ThreadEngagementAnalyzer()
item = DatasetItem(conversation=[{"role": "user", "content": "Thanks"}])
result = await metric.execute(item)
```

## Quick Reference

- Metric key: `thread_engagement_analyzer`
- Category: Analysis
