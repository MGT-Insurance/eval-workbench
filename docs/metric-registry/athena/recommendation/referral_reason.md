# Referral Reason Category

## At a Glance

- Score range: none (classification-only)
- Default threshold: none
- Required inputs: `actual_output`

## What It Measures

Classifies the primary referral or decline reason into a fixed category set
(e.g., Credit, Property Condition, Claims History).

## How It Works

- Detects negative outcomes.
- Uses LLM extraction to assign the primary category.
- Returns the category label as a signal.

## Configuration

- `recommendation_column_name`: additional output field to analyze
- `max_source_lines`: max source lines used for context

## Signals and Diagnostics

Signals include:

- `label`: selected `ReasonCategory` value

## Example

```python
from axion.dataset import DatasetItem
from implementations.athena.metrics.recommendation.referral_reason import ReferralReasonCategory

metric = ReferralReasonCategory()
item = DatasetItem(actual_output="Decline due to prior claims")
result = await metric.execute(item)
```

## Quick Reference

- Metric key: `referral_reason_category`
- Category: Classification
