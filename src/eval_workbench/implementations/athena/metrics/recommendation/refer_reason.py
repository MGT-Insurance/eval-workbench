from enum import Enum
from typing import Any, Dict, List, Optional

from axion._core.logging import get_logger
from axion._core.schema import RichBaseModel, StrictBaseModel
from axion._core.tracing import trace
from axion._core.types import MetricCategory
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

from eval_workbench.implementations.athena.utils import detect_outcome

logger = get_logger(__name__)


class ReasonCategory(str, Enum):
    """
    High-level categories for refer to underwriter/decline reasons.
    Maps to business-meaningful buckets for analytics.
    """

    COVERAGE_EXCESSIVE = 'Excessive Coverage Requested'
    COVERAGE_INADEQUATE = 'Inadequate / Suspicious Valuation'
    INSURABLE_INTEREST = 'Ownership / Insurable Interest Issue'
    PRICING_LOW = 'Pricing Anomaly - Too Low'
    PRICING_HIGH = 'Pricing Anomaly - Too High'
    DATA_CONFLICT = 'Data Conflict / Mismatch'
    DATA_INVALID = 'Implausible / Invalid Data'
    DATA_MISSING = 'Missing / Unverifiable Data'
    CLASSIFICATION_ERROR = 'Industry / Class Code Error'
    UNRELATED_OPERATIONS = 'Unrelated / Ancillary Operations'
    STARTUP_RISK = 'Startup / New Venture'
    MULTI_LOCATION = 'Multi-Location / Complex Ops'
    PROPERTY_CONDITION = 'Property Condition Concerns'
    PROPERTY_EXPOSURE = 'Construction / Exposure Threshold'
    LOCATION_RISK = 'Location / CAT Risk'
    OPERATIONAL_MISMATCH = 'Financial / Operational Inconsistency'
    COMPLIANCE = 'Sanctions / Watchlists'
    PROCEDURAL_HOLD = 'Procedural / Temporary Block'
    OTHER = 'Other / Unknown'


# Category metadata for analytics (priority + actionable type)
CATEGORY_METADATA: Dict[ReasonCategory, Dict[str, str]] = {
    ReasonCategory.COMPLIANCE: {'actionable': 'policy', 'priority': '0'},
    ReasonCategory.PRICING_HIGH: {'actionable': 'market', 'priority': '1'},
    ReasonCategory.PRICING_LOW: {'actionable': 'market', 'priority': '1'},
    ReasonCategory.COVERAGE_EXCESSIVE: {'actionable': 'market', 'priority': '2'},
    ReasonCategory.COVERAGE_INADEQUATE: {'actionable': 'market', 'priority': '2'},
    ReasonCategory.PROPERTY_CONDITION: {'actionable': 'market', 'priority': '3'},
    ReasonCategory.PROPERTY_EXPOSURE: {'actionable': 'market', 'priority': '3'},
    ReasonCategory.LOCATION_RISK: {'actionable': 'market', 'priority': '3'},
    ReasonCategory.OPERATIONAL_MISMATCH: {'actionable': 'market', 'priority': '3'},
    ReasonCategory.STARTUP_RISK: {'actionable': 'market', 'priority': '3'},
    ReasonCategory.MULTI_LOCATION: {'actionable': 'market', 'priority': '3'},
    ReasonCategory.CLASSIFICATION_ERROR: {'actionable': 'policy', 'priority': '4'},
    ReasonCategory.UNRELATED_OPERATIONS: {'actionable': 'policy', 'priority': '4'},
    ReasonCategory.INSURABLE_INTEREST: {'actionable': 'policy', 'priority': '4'},
    ReasonCategory.DATA_CONFLICT: {'actionable': 'system', 'priority': '5'},
    ReasonCategory.DATA_INVALID: {'actionable': 'system', 'priority': '5'},
    ReasonCategory.DATA_MISSING: {'actionable': 'system', 'priority': '5'},
    ReasonCategory.PROCEDURAL_HOLD: {'actionable': 'system', 'priority': '5'},
    ReasonCategory.OTHER: {'actionable': 'unknown', 'priority': '5'},
}


class ExtractedReason(RichBaseModel):
    """A single extracted reason from the recommendation."""

    reason_text: str = Field(
        ..., description='The specific reason extracted from the text.'
    )
    category: ReasonCategory = Field(
        ..., description='High-level category for the reason.'
    )
    reasoning: str = Field(
        ..., description='Explanation of why this was identified as a key reason.'
    )


class ReasonAnalysisResult(RichBaseModel):
    """The structured result for refer to underwriter reason analysis."""

    is_negative_outcome: bool = Field(
        ..., description='True if the outcome is a refer to underwriter or decline.'
    )
    outcome_label: str = Field(
        ...,
        description='Normalized outcome: Refer to Underwriter/Decline/Approved/Unknown',
    )
    primary_reason: Optional[ExtractedReason] = Field(
        None, description='The most significant reason for the negative outcome.'
    )
    primary_category: ReasonCategory = Field(
        ..., description='Category of the primary reason.'
    )
    all_reasons: List[ExtractedReason] = Field(
        default_factory=list, description='All extracted reasons.'
    )
    reason_count: int = Field(default=0, description='Number of reasons detected.')
    actionable_type: str = Field(
        default='unknown',
        description='Whether the reason is market-driven, system-driven, or policy-driven.',
    )
    model_config = {'extra': 'forbid'}


class ReasonExtractionInput(StrictBaseModel):
    """Input for the LLM reason extractor."""

    ai_output: str = Field(..., description="The AI's recommendation/decision text.")
    source_data_summary: str = Field(
        default='', description='Flattened summary of source data for context.'
    )


class ReasonExtractionOutput(StrictBaseModel):
    """Output from the LLM reason extractor."""

    reasons: List[ExtractedReason] = Field(
        default_factory=list,
        description='All identified reasons for the negative outcome.',
    )


class ReasonExtractor(BaseMetric[ReasonExtractionInput, ReasonExtractionOutput]):
    """LLM-based extractor for decline/refer to underwriter reasons."""

    instruction = """
You are an Underwriting Analyst. Extract the specific reasons why this quote was
DECLINED or REFERRED TO UNDERWRITER for human review.

For each reason you identify:
1. Extract the exact reason text from the recommendation
2. Categorize it into ONE of these categories:
   - COVERAGE_EXCESSIVE: Coverage limits materially exceed typical thresholds for this class or business size.
   - COVERAGE_INADEQUATE: Coverage amounts appear insufficient or implausible for the property or operations.
   - INSURABLE_INTEREST: Coverage requested for property the insured does not own or may not have insurable interest in.
   - PRICING_LOW: Calculated premium appears unusually low relative to exposure.
   - PRICING_HIGH: Calculated premium appears unusually high relative to exposure.
   - DATA_CONFLICT: Material discrepancies exist between customer-provided and third-party data.
   - DATA_INVALID: One or more inputs appear invalid or implausible.
   - DATA_MISSING: Required data is missing or cannot be verified.
   - CLASSIFICATION_ERROR: Business classification appears incorrect or overly generic.
   - UNRELATED_OPERATIONS: Business conducts operations outside its primary stated class.
   - STARTUP_RISK: Business has limited or no operating history.
   - MULTI_LOCATION: Risk involves multiple locations requiring manual review.
   - PROPERTY_CONDITION: Property condition presents elevated risk requiring review.
   - PROPERTY_EXPOSURE: Property characteristics exceed standard underwriting thresholds.
   - LOCATION_RISK: Location presents elevated catastrophe or environmental risk.
   - OPERATIONAL_MISMATCH: Operational metrics are internally inconsistent.
   - COMPLIANCE: Potential regulatory or sanctions-related concern.
   - PROCEDURAL_HOLD: Refer to underwriter is procedural rather than risk-based.
   - OTHER: Issue does not clearly fit a defined category.
3. Explain briefly why this is a key reason

Focus on the PRIMARY reasons driving the decision - not every minor detail.
Order reasons by importance (most important first).
"""
    input_model = ReasonExtractionInput
    output_model = ReasonExtractionOutput
    examples = [
        # Example 1: Property Condition
        (
            ReasonExtractionInput(
                ai_output=(
                    'I must refer this quote for underwriter review. The building has a '
                    'roof that is 28 years old, which exceeds our 20-year threshold. '
                    'Additionally, the electrical wiring is original knob-and-tube from '
                    'the 1950s construction.'
                ),
                source_data_summary=(
                    'quote.property.roofAge: 28\n'
                    'quote.property.wiringType: Knob and Tube\n'
                    'quote.property.yearBuilt: 1955'
                ),
            ),
            ReasonExtractionOutput(
                reasons=[
                    ExtractedReason(
                        reason_text='Roof is 28 years old, exceeding 20-year threshold',
                        category=ReasonCategory.PROPERTY_CONDITION,
                        reasoning='Explicit mention of roof age exceeding policy threshold.',
                    ),
                    ExtractedReason(
                        reason_text='Original knob-and-tube wiring from 1950s',
                        category=ReasonCategory.PROPERTY_CONDITION,
                        reasoning='Outdated wiring type is a known fire hazard concern.',
                    ),
                ]
            ),
        ),
        # Example 2: Data Missing / Unverifiable
        (
            ReasonExtractionInput(
                ai_output=(
                    'This application requires refer to underwriter. We cannot verify the reported '
                    'annual revenue of $2.5M against any third-party sources. The business '
                    'was established in 2024 and has no financial history available.'
                ),
                source_data_summary=(
                    'quote.business.annualRevenue: 2500000\n'
                    'quote.business.establishedYear: 2024\n'
                    'thirdParty.revenue: null'
                ),
            ),
            ReasonExtractionOutput(
                reasons=[
                    ExtractedReason(
                        reason_text='Cannot verify reported annual revenue of $2.5M',
                        category=ReasonCategory.DATA_MISSING,
                        reasoning='Third-party revenue data is unavailable for verification.',
                    ),
                    ExtractedReason(
                        reason_text='New business (2024) with no financial history',
                        category=ReasonCategory.STARTUP_RISK,
                        reasoning='Newly established businesses lack verifiable history.',
                    ),
                ]
            ),
        ),
        # Example 3: Classification Error
        (
            ReasonExtractionInput(
                ai_output=(
                    'Decline. The submission lists the business as "Retail Store (NOC)", '
                    'but the narrative and website indicate a church with onsite daycare. '
                    'The classification is too generic and does not match operations.'
                ),
                source_data_summary=(
                    'quote.business.naicsCode: 452990\n'
                    'quote.business.description: Church with daycare and events'
                ),
            ),
            ReasonExtractionOutput(
                reasons=[
                    ExtractedReason(
                        reason_text='Business classification is generic and mismatched to operations',
                        category=ReasonCategory.CLASSIFICATION_ERROR,
                        reasoning='Listed class does not align with described operations.',
                    ),
                ]
            ),
        ),
        # Example 4: Coverage Excessive + Data Conflict
        (
            ReasonExtractionInput(
                ai_output=(
                    'Recommend refer to underwriter. The requested BPP limit of $3.5M appears excessive for '
                    'a business reporting only $800k in annual sales. Additionally, the '
                    'customer reports the building was built in 2015, but enrichment shows 1972.'
                ),
                source_data_summary=(
                    'quote.coverage.bppLimit: 3500000\n'
                    'quote.business.annualSales: 800000\n'
                    'quote.property.yearBuilt: 2015\n'
                    'enrichment.property.yearBuilt: 1972'
                ),
            ),
            ReasonExtractionOutput(
                reasons=[
                    ExtractedReason(
                        reason_text='BPP limit of $3.5M is excessive for $800k annual sales',
                        category=ReasonCategory.COVERAGE_EXCESSIVE,
                        reasoning='Coverage limit is materially higher than typical for revenue.',
                    ),
                    ExtractedReason(
                        reason_text='Year built conflicts between customer and enrichment',
                        category=ReasonCategory.DATA_CONFLICT,
                        reasoning='Material discrepancy exists across data sources.',
                    ),
                ]
            ),
        ),
    ]


@metric(
    name='Refer Reason',
    key='refer_reason',
    description='Extracts and categorizes reasons for refer to underwriter/decline outcomes.',
    metric_category=MetricCategory.CLASSIFICATION,
    required_fields=['actual_output'],
    tags=['athena', 'classification'],
)
class ReferReason(BaseMetric):
    """
    Analyzes refer to underwriter/decline decisions to extract actionable reasons.

    Returns:
        - explanation: The primary category label (e.g., "Coverage Limits")
        - signals: Full structured analysis with all reasons and metadata
    """

    def __init__(
        self,
        recommendation_column_name: str = 'brief_recommendation',
        max_source_lines: int = 50,
        **kwargs,
    ):
        """
        Args:
            recommendation_column_name: Column name for the recommendation text.
            max_source_lines: Maximum lines of source data to include for context.
            **kwargs: Additional arguments passed to BaseMetric.
        """
        super().__init__(**kwargs)
        self.recommendation_column_name = recommendation_column_name
        self.max_source_lines = max_source_lines
        self.reason_extractor = ReasonExtractor(**kwargs)

    def _flatten_source_data(self, data: Any, prefix: str = '') -> List[str]:
        """Flatten nested dict/list to 'path: value' lines for LLM context."""
        if data is None:
            return []

        lines = []

        if isinstance(data, dict):
            for key, value in data.items():
                path = f'{prefix}.{key}' if prefix else key
                if isinstance(value, dict):
                    lines.extend(self._flatten_source_data(value, path))
                elif isinstance(value, list):
                    for i, item in enumerate(value[:5]):
                        if isinstance(item, dict):
                            lines.extend(
                                self._flatten_source_data(item, f'{path}[{i}]')
                            )
                        else:
                            lines.append(f'{path}[{i}]: {item}')
                else:
                    lines.append(f'{path}: {value}')
        elif isinstance(data, list):
            for i, item in enumerate(data[:5]):
                if isinstance(item, dict):
                    lines.extend(self._flatten_source_data(item, f'{prefix}[{i}]'))
                else:
                    lines.append(f'{prefix}[{i}]: {item}')

        return lines

    @trace(name='ReferReason', capture_args=True, capture_response=True)
    async def execute(
        self, dataset_item: DatasetItem, **kwargs
    ) -> MetricEvaluationResult:
        # Gather text sources
        actual_output = self.get_field(dataset_item, 'actual_output') or ''
        detailed_text = dataset_item.additional_output.get(
            self.recommendation_column_name, ''
        )
        full_text = f'{actual_output}\n{detailed_text}'.strip()

        # Detect outcome
        is_negative, outcome_label = detect_outcome(
            actual_output, variant='refer_to_underwriter_reason'
        )

        # If not refer to underwriter outcome, return early
        if 'refer' not in outcome_label.lower():
            result = ReasonAnalysisResult(
                is_negative_outcome=False,
                outcome_label=outcome_label,
                primary_reason=None,
                primary_category=ReasonCategory.OTHER,
                all_reasons=[],
                reason_count=0,
                actionable_type='n/a',
            )
            return MetricEvaluationResult(
                score=None,
                explanation=outcome_label,
                signals=result,
            )

        # Prepare source data summary for context
        source_data = dataset_item.additional_input or {}
        source_lines = self._flatten_source_data(source_data)
        source_summary = '\n'.join(source_lines[: self.max_source_lines])

        # LLM extraction
        llm_input = ReasonExtractionInput(
            ai_output=full_text,
            source_data_summary=source_summary,
        )
        llm_result = await self.reason_extractor.execute(llm_input)

        # Select primary reason (first from LLM, already ordered by importance)
        primary_reason = llm_result.reasons[0] if llm_result.reasons else None

        # Determine primary category and actionable type
        primary_category = (
            primary_reason.category if primary_reason else ReasonCategory.OTHER
        )
        metadata = CATEGORY_METADATA.get(primary_category, {})
        actionable_type = metadata.get('actionable', 'unknown')

        # Build result
        result = ReasonAnalysisResult(
            is_negative_outcome=True,
            outcome_label=outcome_label,
            primary_reason=primary_reason,
            primary_category=primary_category,
            all_reasons=list(llm_result.reasons),
            reason_count=len(llm_result.reasons),
            actionable_type=actionable_type,
        )

        self.compute_cost_estimate([self.reason_extractor])

        return MetricEvaluationResult(
            score=None,
            explanation=primary_category.value,
            signals=result,
        )

    def get_signals(self, result: ReasonAnalysisResult) -> List[SignalDescriptor]:
        """Define how results appear in the Dashboard/UI."""
        signals = [
            SignalDescriptor(
                name='Primary Category',
                group='Overview',
                extractor=lambda r: r.primary_category.value,
                headline_display=True,
            ),
            SignalDescriptor(
                name='Outcome',
                group='Overview',
                extractor=lambda r: r.outcome_label,
                headline_display=False,
            ),
            SignalDescriptor(
                name='Actionable Type',
                group='Overview',
                extractor=lambda r: r.actionable_type.title(),
                description='Is this a market condition, system issue, or policy constraint?',
                headline_display=False,
            ),
            SignalDescriptor(
                name='Reason Count',
                group='Overview',
                extractor=lambda r: r.reason_count,
                headline_display=False,
            ),
        ]

        if result.primary_reason:
            signals.extend(
                [
                    SignalDescriptor(
                        name='Primary Reason',
                        group='Primary Reason',
                        extractor=lambda r: r.primary_reason.reason_text
                        if r.primary_reason
                        else 'N/A',
                        headline_display=False,
                    ),
                    SignalDescriptor(
                        name='Reasoning',
                        group='Primary Reason',
                        extractor=lambda r: r.primary_reason.reasoning
                        if r.primary_reason
                        else 'N/A',
                        headline_display=False,
                    ),
                ]
            )

        for i, reason in enumerate(result.all_reasons):
            group_name = f'Reason {i + 1}: {reason.category.value}'
            signals.extend(
                [
                    SignalDescriptor(
                        name='Reason Text',
                        group=group_name,
                        extractor=lambda r, idx=i: r.all_reasons[idx].reason_text,
                        headline_display=False,
                    ),
                    SignalDescriptor(
                        name='Reasoning',
                        group=group_name,
                        extractor=lambda r, idx=i: r.all_reasons[idx].reasoning,
                        headline_display=False,
                    ),
                ]
            )

        return signals
