# Resolution Detector

## At a Glance

- Score range: `0.0` to `1.0`
- Default threshold: `0.5`
- Required inputs: `conversation`
- Optional inputs: `additional_input`

## What It Measures

Determines the final outcome of a Slack thread (approved / declined / blocked / needs_info / stalemate / pending) and whether the conversation is resolved.

## How It Works

- Builds a conversation transcript for outcome classification.
- Uses an LLM to label the final status and resolution type.
- Optionally uses `additional_input` to detect stalemates (inactivity) and compute time-to-resolution.
- Falls back to heuristic keyword matching if the LLM fails.

## Configuration

- `stalemate_hours` (float, default `72.0`): Inactivity window after which a thread is considered a stalemate (when not otherwise resolved).

## Signals and Diagnostics

`ResolutionResult` with:

- `final_status`, `is_resolved`, `resolution_type`
- `is_stalemate`, `time_to_resolution_seconds`, `message_count`
- `reasoning`

## Example

```python
from axion.dataset import DatasetItem
from shared.metrics.slack.resolution import ResolutionDetector

metric = ResolutionDetector(stalemate_hours=72.0)
item = DatasetItem(conversation=[{"role": "assistant", "content": "Approved"}])
result = await metric.execute(item, additional_input={"last_activity": 1735689600})
```

## Quick Reference

- Metric key: `resolution_detector`
- Category: Classification
