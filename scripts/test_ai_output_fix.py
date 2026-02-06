"""
Quick smoke test for the ai_output -> actual_output rename.

Validates that DatasetItem construction no longer fails with 'Extra inputs are not permitted'
and that each metric's execute() can be called without validation errors.

Usage:
    python scripts/test_ai_output_fix.py
"""

import asyncio

from axion.dataset import DatasetItem


def test_dataset_item_rejects_ai_output():
    """Confirm that DatasetItem does NOT accept ai_output (the old field name)."""
    try:
        DatasetItem(**{"ai_output": "hello"})
        raise AssertionError("Expected ValidationError for ai_output, but none was raised")
    except Exception as e:
        assert "Extra inputs are not permitted" in str(e), f"Unexpected error: {e}"
    print("[PASS] DatasetItem correctly rejects ai_output")


def test_dataset_item_accepts_actual_output():
    """Confirm that DatasetItem accepts actual_output."""
    item = DatasetItem(**{"actual_output": "hello"})
    assert item.actual_output == "hello"
    print("[PASS] DatasetItem accepts actual_output")


async def test_underwriting_rules_execute():
    """Smoke-test UnderwritingRules.execute with a referral case (regex path only, no LLM)."""
    from eval_workbench.implementations.athena.metrics.recommendation.underwriting_rules import (
        UnderwritingRules,
    )

    metric = UnderwritingRules()
    item = {
        "actual_output": "Refer",
        "additional_output": {
            "brief_recommendation": "Referred due to convStoreTemp rule - this is a 7-Eleven franchise."
        },
    }
    result = await metric.execute(item)
    print(f"[PASS] UnderwritingRules.execute returned score={result.score}, explanation={result.explanation}")


async def test_underwriting_rules_approved():
    """Smoke-test UnderwritingRules.execute with an approved case (fast path)."""
    from eval_workbench.implementations.athena.metrics.recommendation.underwriting_rules import (
        UnderwritingRules,
    )

    metric = UnderwritingRules()
    item = {
        "actual_output": "Approved",
        "additional_output": {
            "brief_recommendation": "Standard BOP policy, all checks passed."
        },
    }
    result = await metric.execute(item)
    print(f"[PASS] UnderwritingRules.execute (approved) returned score={result.score}, explanation={result.explanation}")


async def test_decision_quality_input_model():
    """Validate CoverageCheckInput accepts actual_output."""
    from eval_workbench.implementations.athena.metrics.recommendation.decison_quality import (
        CoverageCheckInput,
        HumanRiskFactor,
    )

    inp = CoverageCheckInput(
        required_factors=[
            HumanRiskFactor(concept="Roof age exceeds threshold", impact="High"),
        ],
        actual_output="The roof is 28 years old, exceeding the 20-year limit.",
    )
    assert inp.actual_output is not None
    print("[PASS] CoverageCheckInput accepts actual_output")


async def test_refer_reason_input_model():
    """Validate ReasonExtractionInput accepts actual_output."""
    from eval_workbench.implementations.athena.metrics.recommendation.refer_reason import (
        ReasonExtractionInput,
    )

    inp = ReasonExtractionInput(
        actual_output="Decline. Roof age exceeds threshold.",
        source_data_summary="quote.property.roofAge: 28",
    )
    assert inp.actual_output is not None
    print("[PASS] ReasonExtractionInput accepts actual_output")


async def main():
    print("--- DatasetItem validation ---")
    test_dataset_item_rejects_ai_output()
    test_dataset_item_accepts_actual_output()

    print("\n--- Input model validation ---")
    await test_decision_quality_input_model()
    await test_refer_reason_input_model()

    print("\n--- UnderwritingRules execute (no LLM) ---")
    await test_underwriting_rules_execute()
    await test_underwriting_rules_approved()

    print("\nAll tests passed.")


if __name__ == "__main__":
    asyncio.run(main())
