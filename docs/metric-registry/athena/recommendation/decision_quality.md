# Decision Quality

## At a Glance

- Score range: `0.0` to `1.0`
- Default threshold: none (uses `score_range` only)
- Required inputs: `actual_output`, `expected_output`

## What It Measures

Evaluates whether the AI made the correct underwriting decision and whether its
reasoning aligns with the human notes.

## How It Works

- Extract the human decision and AI decision.
- Score the decision match.
- Optionally score reasoning coverage using extracted risk factors.
- Combine into an overall score with configurable weights.

## Configuration

- `outcome_weight`: weight for decision match
- `reasoning_weight`: weight for reasoning coverage
- `hard_fail_on_outcome_mismatch`: force score 0.0 if decision mismatch
- `recommendation_column_name`: additional output field to analyze

## Signals and Diagnostics

`DecisionQualityResult` with:

- `overall_score`
- `outcome_match`, `outcome_score`
- `human_decision_detected`, `ai_decision_detected`
- `reasoning_score`
- `missing_concepts`, `matched_concepts`

## Example

```python
from axion.dataset import DatasetItem
from implementations.athena.metrics.recommendation.decison_quality import DecisionQuality

metric = DecisionQuality()
item = DatasetItem(actual_output="Recommend Decline", expected_output="Decline due to roof age")
result = await metric.execute(item)
```

## Quick Reference

- Metric key: `decision_quality`
- Category: Athena
