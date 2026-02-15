from __future__ import annotations

import logging
from typing import Any

from axion._core.asyncio import run_async_function
from axion._handlers.llm.handler import LLMHandler
from axion.schema import RichBaseModel
from pydantic import Field

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """\
You are an expert underwriting knowledge extractor. Your job is to extract \
structured underwriting rules from raw text. The text may come from:

- Underwriting manuals (formal rules and guidelines)
- Training sessions or SME interviews
- Production decision logs and case reviews (these are the richest source — they \
contain real outcomes, decision quality assessments, and learned patterns)
- Compliance bulletins

A single piece of text often contains MULTIPLE rules. Extract them all.

For each rule found, return a JSON object with these fields:

CORE FIELDS:
- risk_factor: The risk being evaluated (e.g. "Gas Station", "Employee Count", \
"Building Exposure Discrepancy", "Multi-Location Risk", "Tier 1 Coastal County"). \
This can be a traditional risk factor OR a data quality/process pattern.
- risk_category: Category of risk. Use one of: occupancy, construction, location, \
operations, eligibility, pricing, data_quality, geographic_appetite, catastrophe, \
classification, process
- rule_name: A concise canonical name for the rule
- product_type: Insurance product type (LRO, BOP, GL, Property, Package, E&S, etc.) or "ALL"
- action: The prescribed action (refer, decline, approve_with_conditions, exclude, \
verify, flag_for_review)
- outcome_description: Human-readable description of the outcome. Include the FULL \
context: thresholds, limits, conditions, and any known flexibility. This should be a \
rich, self-contained sentence that someone can read and understand the rule completely.

CONTEXT FIELDS:
- mitigants: List of factors that can override or reduce the rule impact (can be empty)
- source: Where this knowledge comes from (if mentioned)
- source_type: One of "manual", "training", "sme", "production", "compliance", "unknown"
- confidence: Your confidence in the extraction (high, medium, low)

THRESHOLD FIELDS:
- threshold: An object describing the numeric boundary, if the rule involves one. \
Fields: "field" (what is measured, e.g. "employee_count", "property_rate", "tiv", \
"building_exposure", "tiv_per_sqft", "distance_to_coast", "vacancy_days", \
"building_age_years"), "operator" (one of "gt", "gte", "lt", "lte", "eq", "between"), \
"value" (the threshold number or [low, high] for "between"), "unit" (if applicable). \
Set to null if the rule has no numeric threshold.
- threshold_type: "hard" if the limit is absolute and never exceeded, "soft" if there \
is known flexibility or historical deviation, or null if not applicable.
- historical_exceptions: A string describing any known cases where the rule was bent \
or overridden in practice. Set to null if no exceptions are known.

PRODUCTION DECISION FIELDS (set all to null if not from a production case):
- decision_quality: If the text describes a production case, classify the outcome as \
"aligned" (AI recommendation matched UW decision), "divergent" (AI and UW disagreed), \
or "partial" (partially matched). Null if not from a production case.
- compound_trigger: If the rule only fires when MULTIPLE conditions are met together, \
describe the compound condition as a string (e.g. "recent_major_loss AND \
below_minimum_pricing AND prior_carrier_rejection"). Null if single-factor trigger.
- data_fields: List of system field paths referenced in the text (e.g. \
["auxData.rateData.input.property_rate", "underwritingFlags"]). Empty list if none.

Return a JSON object with a "rules" key containing an array of rule objects. \
If no rules are found, return {"rules": []}.

IMPORTANT GUIDELINES:

1. EXTRACT MULTIPLE RULES per text block. Production case reviews often contain 3-5+ \
distinct rules. Each "Key Learning" bullet and "Future Recommendation" bullet may \
contain a separate extractable rule.

2. Pay close attention to NUMERIC THRESHOLDS embedded in prose. Examples: \
"$75/sq ft minimum", "0.25% rate threshold", "$1.5M frame construction limit", \
"40+ year old systems". Always capture these in the threshold object.

3. Distinguish HARD vs SOFT thresholds:
   - Hard: "always decline", "automatic referral", "ineligible", "excluded"
   - Soft: "generally", "typically", "guideline", "has been approved", "at discretion", \
   case was approved despite exceeding limit, divergent decision quality

4. COMPOUND TRIGGERS: When a rule only fires when multiple conditions combine \
(e.g. "recent loss + low pricing + prior rejection = decline"), capture the \
compound_trigger field. The risk_factor should be the primary factor.

5. DATA QUALITY RULES: Patterns like "Magic Dust discrepancy > 20%", \
"address range indicates multi-location", "exposure_value_units = 0" are valid rules \
with risk_category "data_quality". Extract them.

6. GEOGRAPHIC APPETITE: Tier 1/Tier 2 county restrictions, state-level exclusions \
(e.g. "California out of appetite"), coastal distance limits — these are hard \
appetite boundaries. Use risk_category "geographic_appetite".

7. PRICING RULES: Rate minimums (e.g. "property_rate below 0.25%"), TIV/sqft \
minimums (e.g. "$75/sqft"), premium thresholds. Use risk_category "pricing".

8. When the text describes a DIVERGENT production decision (AI recommended X but UW \
did Y), extract BOTH the original rule AND the learned exception. The divergent \
outcome often reveals that a rule is softer than the manual states.

9. SKIP: General commentary, formatting preferences ("be more concise"), error \
reports without actionable rules, and vague statements.
"""


class RuleExtractionInput(RichBaseModel):
    text: str = Field(..., description='Raw text to extract underwriting rules from.')


class Threshold(RichBaseModel):
    field: str | None = None
    operator: str | None = None
    value: float | int | list[float] | list[int] | None = None
    unit: str | None = None


class Rule(RichBaseModel):
    risk_factor: str | None = None
    risk_category: str | None = None
    rule_name: str | None = None
    product_type: str | None = None
    action: str | None = None
    outcome_description: str | None = None
    mitigants: list[str] = Field(default_factory=list)
    source: str | None = None
    source_type: str | None = None
    confidence: str | None = None
    threshold: Threshold | None = None
    threshold_type: str | None = None
    historical_exceptions: str | None = None
    decision_quality: str | None = None
    compound_trigger: str | None = None
    data_fields: list[str] = Field(default_factory=list)


class RuleExtractionOutput(RichBaseModel):
    rules: list[Rule] = Field(default_factory=list)


class RuleExtractionHandler(LLMHandler[RuleExtractionInput, RuleExtractionOutput]):
    """Structured LLM handler for underwriting rule extraction."""

    instruction = EXTRACTION_SYSTEM_PROMPT
    input_model = RuleExtractionInput
    output_model = RuleExtractionOutput
    generation_fake_sample = False
    as_structured_llm = True
    fallback_to_parser = True

    def __init__(self, llm: Any) -> None:
        self.llm = llm
        super().__init__()


class RuleExtractor:
    """Extracts structured underwriting rules from raw text using an LLM."""

    def __init__(self, model: str = 'gpt-4o', provider: str | None = None) -> None:
        from axion.llm_registry import LLMRegistry

        self.model = model
        registry = LLMRegistry(provider=provider) if provider else LLMRegistry()
        self._llm = registry.get_llm(model_name=model)
        self._handler = RuleExtractionHandler(self._llm)

    def extract_batch(self, texts: list[str]) -> list[dict]:
        """Extract rules from a batch of text inputs.

        Parameters
        ----------
        texts:
            Raw text strings containing underwriting rules or learnings.

        Returns
        -------
        List of structured rule dicts.
        """
        combined = '\n\n---\n\n'.join(texts)
        # Avoid printing raw extraction text (can be large/sensitive).
        # Use debug logging with truncation for local troubleshooting.
        if logger.isEnabledFor(logging.DEBUG):
            snippet = combined[:500]
            logger.debug(
                'RuleExtractor combined input (len=%d): %r%s',
                len(combined),
                snippet,
                '...' if len(combined) > len(snippet) else '',
            )

        try:
            output = run_async_function(
                self._handler.execute, RuleExtractionInput(text=combined)
            )
            return [rule.model_dump() for rule in output.rules]
        except Exception as exc:
            logger.error('Rule extraction failed: %s', exc)

        return []
