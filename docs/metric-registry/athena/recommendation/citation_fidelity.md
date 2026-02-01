# Citation Fidelity

## At a Glance

- Score range: `0.0` to `1.0`
- Default threshold: `1.0`
- Required inputs: `actual_output`, `expected_output`

## What It Measures

Checks that bracketed citations (e.g., `[quote.field]`) point to real JSON keys
in `expected_output`. Optionally verifies the cited value appears in the text.

## How It Works

- Parse citations from `actual_output`.
- Resolve each citation path into `expected_output` JSON.
- Optionally verify the value appears near the citation text.
- Score = valid citations / total citations.

## Configuration

- `check_values`: verify cited values appear in text
- `window_chars`: how far back to search for a value
- `min_shared_tokens`, `fuzzy_threshold`, `numeric_tolerance`

## Signals and Diagnostics

`CitationFidelityResult` with:

- `score`
- `total_citations`
- `valid_citations`
- `verdicts` (per-citation status, reason)

## Example

```python
from axion.dataset import DatasetItem
from implementations.athena.metrics.recommendation.citation_fidelity import CitationFidelity

metric = CitationFidelity(check_values=True)
item = DatasetItem(
    actual_output="Premium is $1,200 [quote.premium].",
    expected_output={"quote": {"premium": 1200}},
)
result = await metric.execute(item)
```

## Quick Reference

- Metric key: `citation_fidelity`
- Category: Athena
