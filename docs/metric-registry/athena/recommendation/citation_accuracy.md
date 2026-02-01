# Citation Accuracy

## At a Glance

- Score range: `0.0` to `1.0`
- Default threshold: `1.0`
- Required inputs: none (uses optional `actual_output`, `additional_output`, `additional_input`, `actual_reference`)

## What It Measures

Validates numeric citations like `[1]` against `actual_reference` and (optionally)
verifies referenced fields exist in the input data.

## How It Works

- Extract numeric citations from the output text.
- Match citations to `actual_reference` entries.
- If configured, verify referenced fields exist in `additional_input`.
- Compute score as valid citations / scorable citations.

## Configuration

- `validation_mode`: `ref_only` or `ref_plus_input`
- `output_key`: key in `additional_output` to analyze (fallback to `actual_output`)

## Signals and Diagnostics

`CitationAccuracyResult` with:

- `score`
- `total_citations`
- `scorable_citations`
- `valid_citations`
- `verdicts` (per-citation details)

## Example

```python
from axion.dataset import DatasetItem
from implementations.athena.metrics.recommendation.citation_accuracy import CitationAccuracy

metric = CitationAccuracy(validation_mode="ref_only")
item = DatasetItem(actual_output="Decision approved [1].", actual_reference=["[1] - quote.field"])
result = await metric.execute(item)
```

## Quick Reference

- Metric key: `citation_accuracy`
- Category: Athena
