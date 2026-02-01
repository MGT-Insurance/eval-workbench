# Underwriting Completeness

## At a Glance

- Score range: `0.0` to `1.0`
- Default threshold: none
- Required inputs: `actual_output`

## What It Measures

Evaluates whether a recommendation contains a complete underwriting decision,
including decision, rationale, evidence, and next steps.

## How It Works

- Runs four criteria judges (Decision, Rationale, Evidence, NextStep).
- Applies configurable weights per criterion.
- Hard-gates the score to `0.0` if Decision is missing.

## Configuration

- `weights`: per-criterion weights (Decision/Rationale/Evidence/NextStep)

## Signals and Diagnostics

`UnderwritingCompletenessResult` with:

- `overall_score`
- `criteria`: per-criterion score, reasoning, evidence

## Example

```python
from axion.dataset import DatasetItem
from implementations.athena.metrics.recommendation.underwriting_completeness import UnderwritingCompleteness

metric = UnderwritingCompleteness()
item = DatasetItem(actual_output="Approve. Roof age 5 years. Revenue $1.2M. Next step: bind.")
result = await metric.execute(item)
```

## Quick Reference

- Metric key: `uw_completeness`
- Category: Athena
