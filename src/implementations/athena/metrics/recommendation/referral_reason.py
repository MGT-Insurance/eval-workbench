from enum import Enum
from typing import Any, Dict, List, Optional

from axion._core.logging import get_logger
from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion._core.types import MetricCategory
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

from implementations.athena.utils import detect_outcome

logger = get_logger(__name__)


class ReasonCategory(str, Enum):
    """
    High-level categories for referral/decline reasons.
    Maps to business-meaningful buckets for analytics.
    """

    CREDIT = 'Credit'
    PROPERTY_CONDITION = 'Property Condition'
    BUSINESS_OPERATIONS = 'Business Operations'
    DATA_MISSING = 'Data Missing'
    DATA_INCONSISTENCY = 'Data Inconsistency'
    CLASSIFICATION_ISSUE = 'Classification Issue'
    SANCTIONS = 'Sanctions'
    COVERAGE_LIMITS = 'Coverage Limits'
    CLAIMS_HISTORY = 'Claims History'
    LOCATION_RISK = 'Location Risk'
    UNKNOWN = 'Unknown'


# Category metadata for analytics (priority + actionable type)
CATEGORY_METADATA: Dict[ReasonCategory, Dict[str, str]] = {
    ReasonCategory.SANCTIONS: {'actionable': 'policy', 'priority': '0'},
    ReasonCategory.CREDIT: {'actionable': 'market', 'priority': '1'},
    ReasonCategory.CLAIMS_HISTORY: {'actionable': 'market', 'priority': '2'},
    ReasonCategory.PROPERTY_CONDITION: {'actionable': 'market', 'priority': '3'},
    ReasonCategory.LOCATION_RISK: {'actionable': 'market', 'priority': '4'},
    ReasonCategory.BUSINESS_OPERATIONS: {'actionable': 'policy', 'priority': '5'},
    ReasonCategory.COVERAGE_LIMITS: {'actionable': 'policy', 'priority': '5'},
    ReasonCategory.DATA_MISSING: {'actionable': 'system', 'priority': '5'},
    ReasonCategory.DATA_INCONSISTENCY: {'actionable': 'system', 'priority': '5'},
    ReasonCategory.CLASSIFICATION_ISSUE: {'actionable': 'policy', 'priority': '5'},
    ReasonCategory.UNKNOWN: {'actionable': 'unknown', 'priority': '5'},
}


class CitedReason(RichBaseModel):
    """A single extracted reason with supporting citation."""

    reason_text: str = Field(
        ..., description='The specific reason extracted from the text.'
    )
    category: ReasonCategory = Field(
        ..., description='High-level category for the reason.'
    )
    citation_path: Optional[str] = Field(
        None,
        description="JSON path to supporting data (e.g., 'quote.property.roofAge').",
    )
    citation_value: Optional[str] = Field(
        None, description='The value from the source data supporting this reason.'
    )
    reasoning: str = Field(
        ..., description='Explanation of why this was identified as a key reason.'
    )


class ReasonAnalysisResult(RichBaseModel):
    """The final structured result for referral reason analysis."""

    is_negative_outcome: bool = Field(
        ..., description='True if the outcome is a referral or decline.'
    )
    outcome_label: str = Field(
        ..., description='Normalized outcome: Referral/Decline/Approved/Unknown'
    )
    primary_reason: Optional[CitedReason] = Field(
        None, description='The most significant reason for the negative outcome.'
    )
    primary_category: ReasonCategory = Field(
        ..., description='Category of the primary reason.'
    )
    all_reasons: List[CitedReason] = Field(
        default_factory=list, description='All extracted reasons with citations.'
    )
    reason_count: int = Field(default=0, description='Number of reasons detected.')
    actionable_type: str = Field(
        default='unknown',
        description='Whether the reason is market-driven, system-driven, or policy-driven.',
    )
    model_config = {'extra': 'forbid'}


class ReasonExtractionInput(RichBaseModel):
    """Input for the LLM reason extractor."""

    ai_output: str = Field(..., description="The AI's recommendation/decision text.")
    source_data_summary: str = Field(
        default='', description='Flattened summary of source data for citation lookup.'
    )


class ExtractedReason(RichBaseModel):
    """A single reason extracted by the LLM."""

    reason_text: str = Field(
        ..., description='The specific reason for the decline/referral.'
    )
    category: ReasonCategory = Field(..., description='Best matching category.')
    supporting_data_path: Optional[str] = Field(
        None, description='JSON path to supporting data if identifiable.'
    )
    reasoning: str = Field(..., description='Why this was identified as a key reason.')


class ReasonExtractionOutput(RichBaseModel):
    """Output from the LLM reason extractor."""

    reasons: List[ExtractedReason] = Field(
        default_factory=list,
        description='All identified reasons for the negative outcome.',
    )


class ReasonExtractor(BaseMetric[ReasonExtractionInput, ReasonExtractionOutput]):
    """LLM-based extractor for decline/referral reasons."""

    instruction = """
You are an Underwriting Analyst. Extract the specific reasons why this quote was
DECLINED or REFERRED for human review.

For each reason you identify:
1. Extract the exact reason text from the recommendation
2. Categorize it into ONE of these categories:
   - Credit: Credit score, financial stability, bankruptcy, payment history
   - Property Condition: Roof age, building condition, wiring, structural issues, inspection required
   - Business Operations: Business type, prohibited operations, high-risk class, 24-hour ops
   - Data Missing: Unverifiable data, missing information, discrepancies, cannot verify
   - Data Inconsistency: Conflicting or mismatched values across sources (inputs vs enrichments)
   - Classification Issue: Industry code or class mismatch, incorrect NAICS, appetite misclassification
   - Sanctions: OFAC, watchlist, denied parties, embargoed entities
   - Coverage Limits: TIV too high, outside appetite, limits exceeded
   - Claims History: Prior claims, loss history, claim frequency
   - Location Risk: Flood zone, high crime, wildfire, CAT exposure, protection class
   - Unknown: If the reason doesn't fit any category
3. If the source data summary is provided, identify the JSON path that supports this reason
   (e.g., "quote.property.roofAge" or "applicant.creditScore")
4. Explain briefly why this is a key reason

Focus on the PRIMARY reasons driving the decision - not every minor detail.
Order reasons by importance (most important first).
"""
    input_model = ReasonExtractionInput
    output_model = ReasonExtractionOutput
    examples = [
        # Example 1: Property Condition (Roof Age)
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
                        supporting_data_path='quote.property.roofAge',
                        reasoning='Explicit mention of roof age exceeding policy threshold.',
                    ),
                    ExtractedReason(
                        reason_text='Original knob-and-tube wiring from 1950s',
                        category=ReasonCategory.PROPERTY_CONDITION,
                        supporting_data_path='quote.property.wiringType',
                        reasoning='Outdated wiring type is a known fire hazard concern.',
                    ),
                ]
            ),
        ),
        # Example 2: Data Missing / Unverifiable
        (
            ReasonExtractionInput(
                ai_output=(
                    'This application requires referral. We cannot verify the reported '
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
                        supporting_data_path='thirdParty.revenue',
                        reasoning='Third-party revenue data is null/unavailable for verification.',
                    ),
                    ExtractedReason(
                        reason_text='New business (2024) with no financial history',
                        category=ReasonCategory.DATA_MISSING,
                        supporting_data_path='quote.business.establishedYear',
                        reasoning='Newly established businesses lack verifiable history.',
                    ),
                ]
            ),
        ),
        # Example 3: Business Operations (Prohibited Class)
        (
            ReasonExtractionInput(
                ai_output=(
                    'Decline. This business operates as a cannabis dispensary, which is '
                    'a prohibited class under our current underwriting guidelines. We do '
                    'not offer coverage for cannabis-related businesses in any state.'
                ),
                source_data_summary=(
                    'quote.business.naicsCode: 453998\n'
                    'quote.business.description: Cannabis Retail Dispensary'
                ),
            ),
            ReasonExtractionOutput(
                reasons=[
                    ExtractedReason(
                        reason_text='Cannabis dispensary is a prohibited business class',
                        category=ReasonCategory.BUSINESS_OPERATIONS,
                        supporting_data_path='quote.business.description',
                        reasoning='Cannabis businesses explicitly listed as prohibited class.',
                    ),
                ]
            ),
        ),
        # Example 4: Multiple Categories
        (
            ReasonExtractionInput(
                ai_output=(
                    'This quote is declined for multiple reasons: (1) The property is '
                    'located in a high-crime area with a protection class of 9, (2) the '
                    "applicant's credit score of 520 is below our minimum of 600, and (3) "
                    'we were unable to verify occupancy status.'
                ),
                source_data_summary=(
                    'quote.location.crimeScore: High\n'
                    'quote.location.protectionClass: 9\n'
                    'applicant.creditScore: 520\n'
                    'quote.property.occupancy: null'
                ),
            ),
            ReasonExtractionOutput(
                reasons=[
                    ExtractedReason(
                        reason_text='High-crime area with protection class 9',
                        category=ReasonCategory.LOCATION_RISK,
                        supporting_data_path='quote.location.protectionClass',
                        reasoning='Protection class 9 indicates poor fire protection coverage.',
                    ),
                    ExtractedReason(
                        reason_text='Credit score 520 below minimum threshold of 600',
                        category=ReasonCategory.CREDIT,
                        supporting_data_path='applicant.creditScore',
                        reasoning='Credit score explicitly mentioned as below policy minimum.',
                    ),
                    ExtractedReason(
                        reason_text='Unable to verify occupancy status',
                        category=ReasonCategory.DATA_MISSING,
                        supporting_data_path='quote.property.occupancy',
                        reasoning='Occupancy data is null/unverifiable.',
                    ),
                ]
            ),
        ),
    ]


@metric(
    name='ReferralReasonAnalysis',
    key='referral_reason_analysis',
    description='Extracts and categorizes reasons for referral/decline outcomes with citations.',
    metric_category=MetricCategory.ANALYSIS,
    required_fields=['actual_output'],
    tags=['analysis'],
)
class ReferralReasonAnalysis(BaseMetric):
    """
    Analyzes referral/decline decisions to extract actionable reasons.

    This is an ANALYSIS metric - it produces structured insights (categories, citations,
    actionable types) rather than a numeric score. The output is used for business
    intelligence and understanding why the AI is blocking business.

    Uses LLM extraction to identify reasons and map them to source data citations.
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
            max_source_lines: Maximum lines of source data to include for citation context.
            **kwargs: Additional arguments passed to BaseMetric.
        """
        super().__init__(**kwargs)
        self.recommendation_column_name = recommendation_column_name
        self.max_source_lines = max_source_lines

        # Initialize the LLM extractor
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
                    for i, item in enumerate(value[:5]):  # Limit array expansion
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

    def _resolve_citation_value(self, data: Any, path: Optional[str]) -> Optional[str]:
        """Resolve a JSON path to get the actual value from source data."""
        if not path or not data:
            return None

        try:
            keys = path.replace('[', '.').replace(']', '').split('.')
            current = data
            for key in keys:
                if not key:
                    continue
                if isinstance(current, dict):
                    current = current.get(key)
                elif isinstance(current, list) and key.isdigit():
                    current = current[int(key)]
                else:
                    return None
                if current is None:
                    return None
            return str(current) if current is not None else None
        except (KeyError, IndexError, TypeError):
            return None

    @trace(name='ReferralReasonAnalysis', capture_args=True, capture_response=True)
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
            actual_output, variant='referral_reason'
        )

        # If not a negative outcome, return early with analysis result
        if not is_negative:
            result = ReasonAnalysisResult(
                is_negative_outcome=False,
                outcome_label=outcome_label,
                primary_reason=None,
                primary_category=ReasonCategory.UNKNOWN,
                all_reasons=[],
                reason_count=0,
                actionable_type='n/a',
            )
            return MetricEvaluationResult(
                score=None,  # Analysis metrics don't produce scores
                explanation=(
                    f"Outcome '{outcome_label}' is not a negative outcome - "
                    'no reason analysis needed.'
                ),
                signals=result,
            )

        # Prepare source data summary for citation context
        source_data = dataset_item.additional_input or {}
        source_lines = self._flatten_source_data(source_data)
        source_summary = '\n'.join(source_lines[: self.max_source_lines])

        # LLM extraction
        llm_input = ReasonExtractionInput(
            ai_output=full_text,
            source_data_summary=source_summary,
        )
        llm_result = await self.reason_extractor.execute(llm_input)

        # Convert to CitedReason objects with resolved citation values
        cited_reasons: List[CitedReason] = []
        for extracted in llm_result.reasons:
            citation_value = self._resolve_citation_value(
                source_data, extracted.supporting_data_path
            )
            cited_reasons.append(
                CitedReason(
                    reason_text=extracted.reason_text,
                    category=extracted.category,
                    citation_path=extracted.supporting_data_path,
                    citation_value=citation_value,
                    reasoning=extracted.reasoning,
                )
            )

        # Select primary reason (first from LLM, already ordered by importance)
        primary_reason = cited_reasons[0] if cited_reasons else None

        # Determine primary category and actionable type
        primary_category = (
            primary_reason.category if primary_reason else ReasonCategory.UNKNOWN
        )
        metadata = CATEGORY_METADATA.get(primary_category, {})
        actionable_type = metadata.get('actionable', 'unknown')

        # Build result
        result = ReasonAnalysisResult(
            is_negative_outcome=True,
            outcome_label=outcome_label,
            primary_reason=primary_reason,
            primary_category=primary_category,
            all_reasons=cited_reasons,
            reason_count=len(cited_reasons),
            actionable_type=actionable_type,
        )

        self.compute_cost_estimate([self.reason_extractor])

        # Build explanation summary
        if cited_reasons:
            reason_summary = ', '.join(r.category.value for r in cited_reasons[:3])
            explanation = (
                f'Extracted {len(cited_reasons)} reason(s) for {outcome_label}. '
                f'Primary category: {primary_category.value} ({actionable_type}). '
                f'Categories: [{reason_summary}]'
            )
        else:
            explanation = f'No specific reasons extracted for {outcome_label} outcome.'

        return MetricEvaluationResult(
            score=None,  # Analysis metrics don't produce scores
            explanation=explanation,
            signals=result,
        )

    def get_signals(self, result: ReasonAnalysisResult) -> List[SignalDescriptor]:
        """Define how results appear in the Dashboard/UI."""
        signals = []

        # 1. Headline: Primary Category
        signals.append(
            SignalDescriptor(
                name='Primary Category',
                group='Overview',
                extractor=lambda r: r.primary_category.value,
                headline_display=True,
            )
        )

        # 2. Outcome Label
        signals.append(
            SignalDescriptor(
                name='Outcome',
                group='Overview',
                extractor=lambda r: r.outcome_label,
                headline_display=False,
            )
        )

        # 3. Actionable Type (market/system/policy)
        signals.append(
            SignalDescriptor(
                name='Actionable Type',
                group='Overview',
                extractor=lambda r: r.actionable_type.title(),
                description=(
                    'Is this a market condition, system issue, or policy constraint?'
                ),
                headline_display=False,
            )
        )

        # 4. Reason Count
        signals.append(
            SignalDescriptor(
                name='Reason Count',
                group='Overview',
                extractor=lambda r: r.reason_count,
                headline_display=False,
            )
        )

        # 5. Primary Reason Details
        if result.primary_reason:
            signals.append(
                SignalDescriptor(
                    name='Primary Reason',
                    group='Primary Reason',
                    extractor=lambda r: r.primary_reason.reason_text
                    if r.primary_reason
                    else 'N/A',
                    headline_display=False,
                )
            )
            signals.append(
                SignalDescriptor(
                    name='Citation Path',
                    group='Primary Reason',
                    extractor=lambda r: r.primary_reason.citation_path
                    if r.primary_reason and r.primary_reason.citation_path
                    else 'None',
                    headline_display=False,
                )
            )
            signals.append(
                SignalDescriptor(
                    name='Citation Value',
                    group='Primary Reason',
                    extractor=lambda r: r.primary_reason.citation_value
                    if r.primary_reason and r.primary_reason.citation_value
                    else 'N/A',
                    headline_display=False,
                )
            )
            signals.append(
                SignalDescriptor(
                    name='Reasoning',
                    group='Primary Reason',
                    extractor=lambda r: r.primary_reason.reasoning
                    if r.primary_reason
                    else 'N/A',
                    headline_display=False,
                )
            )

        # 6. All Reasons (detailed breakdown)
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
                        name='Citation',
                        group=group_name,
                        extractor=lambda r, idx=i: (
                            f'{r.all_reasons[idx].citation_path}: '
                            f'{r.all_reasons[idx].citation_value}'
                        )
                        if r.all_reasons[idx].citation_path
                        else 'None',
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


@metric(
    name='ReferralReasonCategory',
    key='referral_reason_category',
    description='Classifies the primary referral/decline reason category.',
    metric_category=MetricCategory.CLASSIFICATION,
    required_fields=['actual_output'],
    tags=['athena', 'classification'],
)
class ReferralReasonCategory(BaseMetric):
    """
    Classifies the primary reason category for referral/decline outcomes.

    This is a CLASSIFICATION metric - it outputs a single label from the
    fixed ReasonCategory set.
    """

    def __init__(
        self,
        recommendation_column_name: str = 'brief_recommendation',
        max_source_lines: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.recommendation_column_name = recommendation_column_name
        self.max_source_lines = max_source_lines

        # Initialize the LLM extractor
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
                    for i, item in enumerate(value[:5]):  # Limit array expansion
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

    @trace(name='ReferralReasonCategory', capture_args=True, capture_response=True)
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
            actual_output, variant='referral_reason'
        )

        # If not a negative outcome, return UNKNOWN (still a fixed label)
        if not is_negative:
            label = ReasonCategory.UNKNOWN
            return MetricEvaluationResult(
                score=None,
                explanation=(
                    f"Outcome '{outcome_label}' is not a negative outcome - "
                    'no reason classification needed.'
                ),
                signals={'label': label.value},
            )

        # Prepare source data summary for better categorization
        source_data = dataset_item.additional_input or {}
        source_lines = self._flatten_source_data(source_data)
        source_summary = '\n'.join(source_lines[: self.max_source_lines])

        # LLM extraction
        llm_input = ReasonExtractionInput(
            ai_output=full_text,
            source_data_summary=source_summary,
        )
        llm_result = await self.reason_extractor.execute(llm_input)

        # Primary category from first extracted reason (ordered by importance)
        if llm_result.reasons:
            primary_category = llm_result.reasons[0].category
            explanation = f'Primary category: {primary_category.value}'
        else:
            primary_category = ReasonCategory.UNKNOWN
            explanation = 'No specific reasons extracted; defaulted to Unknown.'

        return MetricEvaluationResult(
            score=None,
            explanation=explanation,
            signals={'label': primary_category.value},
        )

    def get_signals(self, result: Dict[str, str]) -> List[SignalDescriptor]:
        """Define how results appear in the Dashboard/UI."""
        return [
            SignalDescriptor(
                name='Primary Category',
                group='Overview',
                extractor=lambda r: r.get('label', ReasonCategory.UNKNOWN.value),
                headline_display=True,
            )
        ]
