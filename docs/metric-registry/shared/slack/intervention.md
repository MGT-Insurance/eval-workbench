# Intervention Detector

## At a Glance

- Score range: `0.0` to `1.0`
- Default threshold: `0.5`
- Required inputs: `conversation`
- Optional inputs: `additional_input`

## What It Measures

Detects when a human intervenes in an AI-driven Slack thread and classifies the intervention type (including escalation style and STP).

## How It Works

- Builds a conversation transcript and counts human messages.
- If there are no human messages, the thread is treated as STP (no intervention).
- Otherwise, uses an LLM to classify the intervention category and extract key details (friction point, issue details).
- Maps intervention category into escalation type (`hard` / `soft` / `authority` / `none`).
- Falls back to a simple heuristic if the LLM fails.

## Configuration

- No custom parameters.

## Signals and Diagnostics

`InterventionResult` with:

- `has_intervention`, `intervention_type`
- `escalation_type`, `is_hard_escalation`, `is_soft_escalation`, `is_authority_escalation`
- `is_stp`, `human_message_count`
- `friction_point`, `issue_details`, `intervention_summary`, `reasoning`

## Example

```python
from axion.dataset import DatasetItem
from shared.metrics.slack.intervention import InterventionDetector

metric = InterventionDetector()
item = DatasetItem(conversation=[{"role": "user", "content": "Need approval on this"}])
result = await metric.execute(item)
```

## Quick Reference

- Metric key: `intervention_detector`
- Category: Classification
