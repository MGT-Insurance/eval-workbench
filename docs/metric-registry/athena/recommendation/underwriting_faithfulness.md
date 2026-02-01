# Underwriting Faithfulness

## At a Glance

- Score range: `0.0` to `1.0`
- Default threshold: `0.9`
- Required inputs: none (uses `additional_input` if present)

## What It Measures

Checks whether factual claims in the recommendation are supported by the source
JSON input.

## How It Works

- Extracts atomic claims from the recommendation text.
- Finds relevant evidence lines in `additional_input`.
- Verifies each claim using LLM or heuristic mode.
- Scores as supported_claims / total_claims.

## Configuration

- `verification_mode`: `llm`, `heuristic`, or `heuristic_then_llm`
- `max_claims`: max claims to verify
- `max_concurrent`: concurrency limit for verification

## Signals and Diagnostics

`UnderwritingFaithfulnessResult` with:

- `overall_score`
- `total_claims`, `supported_claims`, `hallucinations`
- `claim_details`, `unverified_claims`

## Example

```python
from axion.dataset import DatasetItem
from implementations.athena.metrics.recommendation.underwriting_faithfulness import UnderwritingFaithfulness

metric = UnderwritingFaithfulness(verification_mode="heuristic")
item = DatasetItem(actual_output="Revenue is $1.2M.", additional_input={"revenue": 1200000})
result = await metric.execute(item)
```

## Quick Reference

- Metric key: `underwriting_faithfulness`
- Category: Athena
