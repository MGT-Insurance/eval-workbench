# Referral Reason Analysis

## At a Glance

- Score range: none (analysis-only)
- Default threshold: none
- Required inputs: `actual_output`

## What It Measures

Extracts and categorizes reasons for referral or decline outcomes, including
citations to supporting input data.

## How It Works

- Detects negative outcomes (referral/decline).
- Extracts reasons with LLM guidance.
- Maps reasons to categories and supporting citations.
- Returns structured signals (no numeric score).

## Configuration

- `recommendation_column_name`: additional output field to analyze
- `max_source_lines`: max source lines used for citation context

## Signals and Diagnostics

`ReasonAnalysisResult` with:

- `is_negative_outcome`, `outcome_label`
- `primary_reason`, `primary_category`
- `all_reasons`, `reason_count`
- `actionable_type`

## Example

```python
from axion.dataset import DatasetItem
from implementations.athena.metrics.recommendation.referral_reason import ReferralReasonAnalysis

metric = ReferralReasonAnalysis()
item = DatasetItem(actual_output="Refer due to roof age")
result = await metric.execute(item)
```

## Quick Reference

- Metric key: `referral_reason_analysis`
- Category: Analysis
