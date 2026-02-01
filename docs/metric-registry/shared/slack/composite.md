# Slack Conversation Analyzer

## At a Glance

- Score range: none (analysis-only)
- Default threshold: none
- Required inputs: `conversation`
- Optional inputs: `additional_input`

## What It Measures

Comprehensive analysis of Slack conversations for all KPI signals in a single
pass.

## How It Works

- Builds transcript, mentions, and recommendation signals.
- Runs a single LLM call for classification/scoring.
- Returns nested signal structures for reporting.

## Configuration

- `metrics` (list[str] | None): Which metrics to run. If omitted, runs all available metrics.
- `frustration_threshold` (float, default `0.6`)
- `satisfaction_threshold` (float, default `0.7`)
- `sentiment_threshold` (float, default `0.4`)

## Signals and Diagnostics

Signals for:

- `interaction`, `engagement`, `recommendation`
- `escalation`, `frustration`, `acceptance`
- `override`, `satisfaction`
- `intervention`, `sentiment`, `resolution`

## Example

```python
from axion.dataset import DatasetItem
from shared.metrics.slack.composite import SlackConversationAnalyzer

metric = SlackConversationAnalyzer(
    metrics=["interaction", "engagement", "recommendation", "intervention", "sentiment", "resolution"],
    frustration_threshold=0.6,
    satisfaction_threshold=0.7,
    sentiment_threshold=0.4,
)
item = DatasetItem(conversation=[{"role": "user", "content": "Question"}])
result = await metric.execute(item)
```

## Quick Reference

- Metric key: `slack_conversation_analyzer`
- Category: Analysis
