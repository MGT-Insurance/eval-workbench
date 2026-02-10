import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Pattern, Tuple, cast

from axion._core.logging import get_logger
from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

from eval_workbench.implementations.athena.metrics.recommendation.underwriting_faithfulness import (
    flatten_json,
)
from eval_workbench.implementations.athena.utils import detect_outcome

logger = get_logger(__name__)


class TriggerName(str, Enum):
    """
    The 'Athena' Universe of Underwriting Triggers.
    Matches the Capability Mapping Doc.
    """

    # Hard Rules
    CONV_STORE_TEMP = 'convStoreTemp'
    BPP_TO_SALES = 'bppToSalesRatio'
    ORG_EST_YEAR = 'orgEstYear'
    NON_OWNED_BLDG = 'nonOwnedBuildingCoverage'
    BUSINESS_NOC = 'businessNOC'
    BPP_VALUE = 'bppValue'
    HOME_BASED = 'homeBasedBPP'
    CLAIMS_HISTORY = 'claimsHistory'
    NUM_EMPLOYEES = 'numberOfEmployees'

    # Fallback / Catch-all
    UNKNOWN = 'unknown_trigger'
    NONE = 'none'  # Clean pass (Auto-Approve)


@dataclass
class TriggerSpec:
    """Configuration for a single underwriting trigger rule."""

    name: TriggerName
    patterns: List[str]
    description: str  # For LLM prompt generation
    priority: int  # Lower = higher priority
    severity: Literal['hard', 'soft'] = 'hard'


TRIGGER_SPECS: List[TriggerSpec] = [
    TriggerSpec(
        name=TriggerName.CONV_STORE_TEMP,
        patterns=[
            r'convStoreTemp',
            r'Convenience Store.*Rule',
            r'rule.*9321',
            r'class.*CONVGAS',
            r'7[-\s]?eleven|circle\s?k|am\s?pm|wawa|sheetz',
            r'(convenience|liquor|package|corner)\s+store',
            r'c[-\s]?store',
            r'mini[-\s]?mart|minimart',
            r'bodega',
            r'(tobacco|liquor|alcohol|beer|wine|lottery).*sales?',
            r'gas\s?station|fuel\s?sales?',
            r'(convenience|liquor|package|mini[-\s]?mart|bodega|corner\s+store).{0,50}(24/7|24x7|open\s+24|24\s*hours?)',
        ],
        description='Convenience stores and liquor/package stores (often with tobacco/alcohol/lottery and sometimes 24/7 or fuel).',
        priority=1,
        severity='hard',
    ),
    TriggerSpec(
        name=TriggerName.CLAIMS_HISTORY,
        patterns=[
            r'claimsHistory',
            r'prior\s+claim',
            r'loss\s+history',
            r'previous\s+(claim|loss)',
            r'claim(s)?\s+(in|over)\s+the\s+(past|last)',
        ],
        description='Prior losses or claims history mentioned.',
        priority=2,
        severity='hard',
    ),
    TriggerSpec(
        name=TriggerName.ORG_EST_YEAR,
        patterns=[
            r'orgEstYear',
            r'established.*202[3-9]',
            r'incorporated.*202[3-9]',
            r'business.*<\s*3\s*years?',
            r'new\s+organization',
            r'(founded|started|opened).*202[3-9]',
        ],
        description='Business established within 3 years and requesting building coverage (new business + building risk).',
        priority=3,
        severity='hard',
    ),
    TriggerSpec(
        name=TriggerName.BPP_VALUE,
        patterns=[
            r'bppValue',
            r'contents.*>\s*\$?250[,.]?000',
            r'BPP.*exceeds?\s*\$?250',
            r'personal\s+property.*250',
        ],
        description='BPP/contents limit exceeds $250k (requires reasonableness/security review).',
        priority=4,
        severity='hard',
    ),
    TriggerSpec(
        name=TriggerName.BPP_TO_SALES,
        patterns=[
            r'bppToSalesRatio',
            r'contents.*sales.*ratio',
            r'<\s*10\s*%.*ratio',
            r'BPP.*to.*sales.*low',
            r'ratio.*contents.*revenue',
        ],
        description='Contents-to-sales ratio is below 10% (possible inadequate contents or unusual model).',
        priority=5,
        severity='soft',
    ),
    TriggerSpec(
        name=TriggerName.NON_OWNED_BLDG,
        patterns=[
            r'nonOwnedBuildingCoverage',
            r'tenant.*building\s+coverage',
            r'leased.*building\s+limit',
            r'renter.*requesting.*building',
            r'(triple[-\s]?net|nnn)\s+lease',
            r'net\s+lease',
            r'tenant.*(does\s+not\s+own|not\s+own).*building',
            r'building\s+coverage.*(lease|leased|tenant)',
        ],
        description="Building coverage requested but applicant doesn't own building (lease / NNN requirement).",
        priority=6,
        severity='soft',
    ),
    TriggerSpec(
        name=TriggerName.BUSINESS_NOC,
        patterns=[
            r'businessNOC',
            r'Not\s+Otherwise\s+Classified',
            r'classification\s+mismatch',
            r'NOC\s+class',
            r'unclear\s+business\s+type',
        ],
        description='Business selected as NOC / unclear operations; must classify and confirm appetite.',
        priority=7,
        severity='soft',
    ),
    TriggerSpec(
        name=TriggerName.HOME_BASED,
        patterns=[
            r'homeBasedBPP',
            r'residential.*location',
            r'home[-\s]?based\s+business',
            r'operates?\s+from\s+home',
        ],
        description='Home-based business seeking contents coverage (confirm commercial readiness and controls).',
        priority=8,
        severity='soft',
    ),
    TriggerSpec(
        name=TriggerName.NUM_EMPLOYEES,
        patterns=[
            r'numberOfEmployees',
            r'employee\s+count.*>\s*20',
            r'more\s+than\s+20\s+employees',
            r'exceeds?\s+employee\s+limit',
            r'headcount',
            r'staff(ing)?\s+of\s+\d{2,}',
        ],
        description='Employee count exceeds 20 (confirm small-business eligibility and staffing reasonableness).',
        priority=9,
        severity='soft',
    ),
]


class DetectionMethod(str, Enum):
    REGEX = 'regex'  # Fast, deterministic (100% confidence)
    LLM_FALLBACK = 'llm_fallback'  # Slow, probabilistic (<100% confidence)


class TriggerEvent(RichBaseModel):
    """A specific rule firing instance within a trace."""

    trigger_name: TriggerName = Field(..., description='The ID of the rule triggered.')
    detection_method: DetectionMethod = Field(..., description='How we found it.')
    context: Optional[str] = Field(
        None, description='Snippet where the rule was cited.'
    )
    confidence: float = Field(1.0, description='1.0 for Regex, variable for LLM.')


class TriggerReport(RichBaseModel):
    """The final structured signal object for the UI."""

    is_referral: bool = Field(..., description='Did the agent Refer/Decline?')
    active_triggers: List[TriggerEvent] = Field(default_factory=list)
    primary_referral_reason: TriggerName = Field(..., description='Top-level category.')
    summary_text: str = Field(..., description='Comma-separated triggers for tables.')
    # Enhanced fields
    outcome_label: str = Field(
        default='Unknown',
        description='Normalized outcome: Referral/Approved/Unknown',
    )
    trigger_count: int = Field(default=0, description='Number of triggers detected.')
    llm_fallback_used: bool = Field(
        default=False, description='True if LLM was invoked for ghost referral.'
    )
    min_confidence: float = Field(
        default=1.0, description='Minimum confidence across all detected triggers.'
    )
    has_hard_trigger: bool = Field(
        default=False, description='True if any hard-severity trigger was detected.'
    )
    unknown_reasoning: Optional[str] = Field(
        default=None,
        description='LLM explanation when outcome is referral but no trigger matched.',
    )
    model_config = {'extra': 'forbid'}


class GhostReferralInput(RichBaseModel):
    actual_output: str = Field(..., description='The full agent response.')


class GhostReferralOutput(RichBaseModel):
    likely_trigger: TriggerName = Field(
        ...
    )  # No description - $ref can't have keywords
    reasoning: str = Field(..., description='Why this rule applies.')


class GhostReferralClassifier(BaseMetric[GhostReferralInput, GhostReferralOutput]):
    instruction = """
    You are an Underwriting Classifier. The Agent has REFERRED or DECLINED a quote,
    but did not explicitly cite a rule code (e.g., 'convStoreTemp').

    Analyze the text and map it to the closest Athena Trigger.

    Triggers:
    - convStoreTemp: Convenience stores and liquor/package stores (often with tobacco/alcohol/lottery and sometimes 24/7 or fuel).
    - claimsHistory: Prior losses or claims mentioned.
    - orgEstYear: Business established within 3 years and requesting building coverage (new business + building risk).
    - bppValue: BPP/contents limit exceeds $250k (requires reasonableness/security review).
    - bppToSalesRatio: Contents-to-sales ratio is below 10% (possible inadequate contents or unusual model).
    - nonOwnedBuildingCoverage: Building coverage requested but applicant doesn't own building (lease / NNN requirement).
    - businessNOC: Business selected as NOC / unclear operations; must classify and confirm appetite.
    - homeBasedBPP: Home-based business seeking contents coverage (confirm commercial readiness and controls).
    - numberOfEmployees: Employee count exceeds 20 (confirm small-business eligibility and staffing reasonableness).

    If truly unknown, select 'unknown_trigger'.
    """
    input_model = GhostReferralInput
    output_model = GhostReferralOutput
    examples = [
        # Example 1: CONV_STORE_TEMP - 7-Eleven franchise with tobacco/alcohol
        (
            GhostReferralInput(
                actual_output='After reviewing the application, I must refer this quote. '
                'The business operates as a 7-Eleven convenience store with 24-hour operations '
                'and sells tobacco products, beer, and lottery tickets.'
            ),
            GhostReferralOutput(
                likely_trigger=TriggerName.CONV_STORE_TEMP,
                reasoning='7-Eleven franchise with tobacco, alcohol, lottery - classic convStoreTemp indicators.',
            ),
        ),
        # Example 2: ORG_EST_YEAR - New business requesting building coverage
        (
            GhostReferralInput(
                actual_output='This application requires referral. The business was just '
                'incorporated in 2024 and has less than one year of operating history. '
                'They are requesting building coverage and we cannot verify their track record.'
            ),
            GhostReferralOutput(
                likely_trigger=TriggerName.ORG_EST_YEAR,
                reasoning='Business incorporated in 2024 with less than 3 years history triggers orgEstYear.',
            ),
        ),
        # Example 3: BPP_TO_SALES - Low contents to revenue ratio
        (
            GhostReferralInput(
                actual_output='Referral needed. The applicant requests $15,000 in contents coverage '
                'but reports $500,000 in annual sales. This represents only a 3% ratio '
                'which seems inconsistent with typical retail operations.'
            ),
            GhostReferralOutput(
                likely_trigger=TriggerName.BPP_TO_SALES,
                reasoning='$15k contents vs $500k sales = 3% ratio, well below 10% threshold for bppToSalesRatio.',
            ),
        ),
        # Example 4: CLAIMS_HISTORY - Multiple prior claims
        (
            GhostReferralInput(
                actual_output='This quote must be referred for underwriter review. The applicant '
                'disclosed two water damage claims in the past three years and one theft '
                'claim from 2022. The loss history raises concerns about risk profile.'
            ),
            GhostReferralOutput(
                likely_trigger=TriggerName.CLAIMS_HISTORY,
                reasoning='Multiple prior claims (water damage, theft) mentioned - triggers claimsHistory.',
            ),
        ),
        # Example 5: UNKNOWN - Generic decline with no specific reason
        (
            GhostReferralInput(
                actual_output='After careful consideration, this application does not meet our '
                'current underwriting guidelines. We are unable to offer coverage at this time.'
            ),
            GhostReferralOutput(
                likely_trigger=TriggerName.UNKNOWN,
                reasoning='Generic decline language with no specific trigger indicators identifiable.',
            ),
        ),
        # Example 6: NON_OWNED_BLDG - Tenant requesting building coverage without ownership
        (
            GhostReferralInput(
                actual_output='Referral required. The applicant is a tenant and does not own the building, '
                'but the agent requested building coverage. Please provide the NNN (triple-net) lease.'
            ),
            GhostReferralOutput(
                likely_trigger=TriggerName.NON_OWNED_BLDG,
                reasoning='Tenant requesting building coverage without ownership; requires lease/NNN support.',
            ),
        ),
        # Example 7: NUM_EMPLOYEES - Employee count exceeds small business threshold
        (
            GhostReferralInput(
                actual_output='This needs referral. The insured reports 35 employees, which exceeds our small '
                'business eligibility threshold and requires confirmation of revenue and staffing reasonableness.'
            ),
            GhostReferralOutput(
                likely_trigger=TriggerName.NUM_EMPLOYEES,
                reasoning='Employee count exceeds 20, triggering numberOfEmployees eligibility review.',
            ),
        ),
        # Example 8: BUSINESS_NOC - Not Otherwise Classified / unclear operations
        (
            GhostReferralInput(
                actual_output='Referral needed. The business was submitted as NOC (Not Otherwise Classified) '
                'and the operations are unclear from the application. We need to determine the correct class.'
            ),
            GhostReferralOutput(
                likely_trigger=TriggerName.BUSINESS_NOC,
                reasoning='NOC/unclear operations require research and classification, matching businessNOC.',
            ),
        ),
    ]


class UnknownTriggerReasonInput(RichBaseModel):
    actual_output: str = Field(..., description='The full agent response.')


class UnknownTriggerReasonOutput(RichBaseModel):
    reasoning: str = Field(
        ..., description='Plain-English explanation for why this was referred.'
    )


class UnknownTriggerReasoner(
    BaseMetric[UnknownTriggerReasonInput, UnknownTriggerReasonOutput]
):
    instruction = """
    You are an Underwriting Analyst. The agent referred or declined a quote, but no
    known trigger matched. Read the text and provide a concise reason for why it
    appears to be referred. Do NOT map to a trigger code; instead summarize the
    likely issue in one sentence.
    """
    input_model = UnknownTriggerReasonInput
    output_model = UnknownTriggerReasonOutput


@metric(
    name='UW Rules',
    key='uw_rules',
    description='Tracks Athena referral triggers live. Uses Regex with LLM fallback.',
    score_range=(0, 1),  # 1.0 = Flagged/Referral, 0.0 = Clean
    required_fields=['actual_output'],
    tags=['heuristic', 'athena'],
)
class UnderwritingRules(BaseMetric):
    def __init__(
        self,
        trigger_specs: Optional[List[TriggerSpec]] = None,
        recommendation_column_name: str = 'brief_recommendation',
        use_unknown_reason_llm: bool = False,
        **kwargs,
    ):
        """
        Args:
            trigger_specs: Optional custom trigger specifications. Uses TRIGGER_SPECS if not provided.
            recommendation_column_name: The column name in the dataset item to use for the recommendation text.
            use_unknown_reason_llm: Whether to use the LLM to reason about why the outcome is referred.
            **kwargs: Additional arguments passed to BaseMetric.
        """
        super().__init__(**kwargs)
        self.recommendation_column_name = recommendation_column_name
        self.use_unknown_reason_llm = use_unknown_reason_llm
        # Use provided specs or fall back to default
        self._trigger_specs = trigger_specs or TRIGGER_SPECS

        # Build lookup dict by trigger name
        self._spec_by_name: Dict[TriggerName, TriggerSpec] = {
            spec.name: spec for spec in self._trigger_specs
        }

        # Pre-compile all regex patterns for speed
        self._compiled_patterns: Dict[TriggerName, List[Pattern]] = {
            spec.name: [re.compile(p, re.IGNORECASE) for p in spec.patterns]
            for spec in self._trigger_specs
        }

        # Initialize the fallback LLM classifier with dynamic instruction
        self.classifier = GhostReferralClassifier(**kwargs)
        # Keep the classifier trigger list in sync with TRIGGER_SPECS.
        self.classifier.instruction = self._build_llm_instruction()
        self.unknown_reasoner = UnknownTriggerReasoner(**kwargs)

    def _build_llm_instruction(self) -> str:
        """Generate LLM instruction dynamically from trigger specs."""
        trigger_lines = [
            f'- {spec.name.value}: {spec.description}'
            for spec in sorted(self._trigger_specs, key=lambda s: s.priority)
        ]
        triggers_text = '\n    '.join(trigger_lines)

        return f"""
    You are an Underwriting Classifier. The Agent has REFERRED or DECLINED a quote,
    but did not explicitly cite a rule code (e.g., 'convStoreTemp').

    Analyze the text and map it to the closest Athena Trigger.

    Triggers (ordered by priority):
    {triggers_text}

    If truly unknown, select 'unknown_trigger'.
    """

    def _flatten_additional_input(self, data: Optional[Dict]) -> Dict[str, str]:
        if not data:
            return {}

        flat_lines = flatten_json(data)
        flat_map: Dict[str, str] = {}
        for line in flat_lines:
            if ': ' in line:
                key, value = line.split(': ', 1)
                flat_map[key] = value
        return flat_map

    def _get_flat_value(
        self, flat_map: Dict[str, str], keys: List[str]
    ) -> Optional[str]:
        for key in keys:
            if key in flat_map:
                return flat_map[key]

        for key in keys:
            suffix = f'.{key}'
            for path, value in flat_map.items():
                if path.endswith(suffix) or path == key:
                    return value
        return None

    def _parse_number(self, value: Optional[str]) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).replace(',', '')
        match = re.search(r'-?\d+(?:\.\d+)?', text)
        if not match:
            return None
        return float(match.group(0))

    def _parse_bool(self, value: Optional[str]) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {'true', 'yes', '1'}:
            return True
        if text in {'false', 'no', '0'}:
            return False
        return None

    def _parse_building_coverage(self, value: Optional[str]) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if 'contents only' in text or text == 'contents':
            return False
        if 'building' in text:
            return True
        return None

    def _get_number(self, flat_map: Dict[str, str], keys: List[str]) -> Optional[float]:
        return self._parse_number(self._get_flat_value(flat_map, keys))

    def _get_bool(self, flat_map: Dict[str, str], keys: List[str]) -> Optional[bool]:
        return self._parse_bool(self._get_flat_value(flat_map, keys))

    def _select_primary_trigger(self, events: List[TriggerEvent]) -> TriggerName:
        """
        Select primary trigger using priority + confidence ranking.

        Priority is primary sort key (lower = higher priority).
        Confidence is secondary sort key (higher = better).
        """
        if not events:
            return TriggerName.NONE

        def sort_key(event: TriggerEvent) -> Tuple[int, float]:
            spec = self._spec_by_name.get(event.trigger_name)
            priority = spec.priority if spec else 999
            # Negate confidence so higher confidence comes first
            return (priority, -event.confidence)

        sorted_events = sorted(events, key=sort_key)
        return sorted_events[0].trigger_name

    def _deduplicate_events(self, events: List[TriggerEvent]) -> List[TriggerEvent]:
        """
        Merge events with the same trigger name, keeping highest confidence.
        """
        best_by_trigger: Dict[TriggerName, TriggerEvent] = {}

        for event in events:
            existing = best_by_trigger.get(event.trigger_name)
            if existing is None or event.confidence > existing.confidence:
                best_by_trigger[event.trigger_name] = event

        return list(best_by_trigger.values())

    @trace(name='UWRules', capture_args=True, capture_response=True)
    async def execute(
        self, item: DatasetItem | dict, callbacks: Any = None, **kwargs
    ) -> MetricEvaluationResult:
        # Axion's BaseMetric.execute supports DatasetItem or dict. Normalize to DatasetItem.
        dataset_item: DatasetItem
        if isinstance(item, DatasetItem):
            dataset_item = item
        else:
            dataset_item = DatasetItem(**cast(dict[str, Any], item))

        actual_output = self.get_field(dataset_item, 'actual_output') or ''
        full_text = dataset_item.additional_output.get(
            self.recommendation_column_name, ''
        )
        # Decline counts as a referral-class outcome for scoring/flags.
        is_referral, outcome_label = detect_outcome(
            actual_output, variant='underwriting_rules'
        )

        # Optional fast-path: if this wasn't a "Refer" decision, skip trigger detection.
        # This metric is primarily intended to explain and validate "Refer" outcomes.
        if outcome_label.upper() != 'REFER':
            primary_reason = TriggerName.NONE
            result_data = TriggerReport(
                is_referral=is_referral,
                active_triggers=[],
                primary_referral_reason=primary_reason,
                summary_text='',
                outcome_label=outcome_label,
                trigger_count=0,
                llm_fallback_used=False,
                min_confidence=1.0,
                has_hard_trigger=False,
                unknown_reasoning=None,
            )
            return MetricEvaluationResult(
                score=1.0, explanation=primary_reason, signals=result_data
            )

        detected_events: List[TriggerEvent] = []
        llm_fallback_used = False
        unknown_reasoning: Optional[str] = None

        # Structured Checks (based on additional_input)
        flat_map = self._flatten_additional_input(dataset_item.additional_input or {})

        bpp_limit = self._get_number(
            flat_map,
            [
                'context_data.auxData.rateData.output.input.bop_bpp_limit',
                'context_data.auxData.rateData.output.bop_bpp_limit',
                'bop_bpp_limit',
            ],
        )
        gross_sales = self._get_number(
            flat_map,
            [
                'context_data.auxData.rateData.output.input.bop_gross_sales',
                'context_data.auxData.rateData.output.bop_gross_sales',
                'bop_gross_sales',
            ],
        )
        num_employees = self._get_number(
            flat_map,
            [
                'context_data.auxData.rateData.output.input.bop_number_of_employees',
                'context_data.auxData.rateData.output.bop_number_of_employees',
                'bop_number_of_employees',
            ],
        )
        year_established = self._get_number(
            flat_map,
            [
                'context_data.auxData.rateData.output.input.bop_business_year_established',
                'context_data.auxData.rateData.output.bop_business_year_established',
                'bop_business_year_established',
            ],
        )
        claims_count = self._get_number(
            flat_map,
            [
                'context_data.auxData.rateData.output.input.bop_number_of_claims',
                'context_data.auxData.rateData.output.bop_number_of_claims',
                'bop_number_of_claims',
            ],
        )
        home_based = self._get_bool(
            flat_map,
            [
                'context_data.auxData.rateData.output.input.bop_home_based_business',
                'context_data.auxData.rateData.output.bop_home_based_business',
                'bop_home_based_business',
            ],
        )
        building_owned = self._get_bool(
            flat_map,
            [
                'context_data.auxData.rateData.output.input.bop_building_owned',
                'context_data.auxData.rateData.output.bop_building_owned',
                'bop_building_owned',
            ],
        )
        insure_building = self._get_flat_value(
            flat_map,
            [
                'context_data.auxData.rateData.output.input.bop_insure_buildings',
                'context_data.auxData.rateData.output.input.bop_insure_building',
                'context_data.auxData.rateData.output.bop_insure_buildings',
                'context_data.auxData.rateData.output.bop_insure_building',
                'bop_insure_buildings',
                'bop_insure_building',
            ],
        )

        building_coverage_requested = self._parse_building_coverage(insure_building)
        contents_only = (
            None
            if building_coverage_requested is None
            else not building_coverage_requested
        )

        if bpp_limit is not None and bpp_limit > 250000:
            detected_events.append(
                TriggerEvent(
                    trigger_name=TriggerName.BPP_VALUE,
                    detection_method=DetectionMethod.REGEX,
                    context=f'bop_bpp_limit={int(bpp_limit)}',
                    confidence=1.0,
                )
            )

        if bpp_limit is not None and gross_sales is not None and gross_sales > 0:
            ratio = bpp_limit / gross_sales
            if ratio < 0.10:
                detected_events.append(
                    TriggerEvent(
                        trigger_name=TriggerName.BPP_TO_SALES,
                        detection_method=DetectionMethod.REGEX,
                        context=f'bpp_to_sales_ratio={ratio:.2%}',
                        confidence=1.0,
                    )
                )

        if num_employees is not None and num_employees > 20:
            detected_events.append(
                TriggerEvent(
                    trigger_name=TriggerName.NUM_EMPLOYEES,
                    detection_method=DetectionMethod.REGEX,
                    context=f'bop_number_of_employees={int(num_employees)}',
                    confidence=1.0,
                )
            )

        if (
            year_established is not None
            and building_coverage_requested is True
            and (datetime.utcnow().year - int(year_established)) < 3
        ):
            detected_events.append(
                TriggerEvent(
                    trigger_name=TriggerName.ORG_EST_YEAR,
                    detection_method=DetectionMethod.REGEX,
                    context=f'bop_business_year_established={int(year_established)}',
                    confidence=1.0,
                )
            )

        if building_coverage_requested is True and building_owned is False:
            detected_events.append(
                TriggerEvent(
                    trigger_name=TriggerName.NON_OWNED_BLDG,
                    detection_method=DetectionMethod.REGEX,
                    context='building_coverage_requested=True; building_owned=False',
                    confidence=1.0,
                )
            )

        if home_based is True and contents_only is True:
            detected_events.append(
                TriggerEvent(
                    trigger_name=TriggerName.HOME_BASED,
                    detection_method=DetectionMethod.REGEX,
                    context='bop_home_based_business=True; contents_only=True',
                    confidence=1.0,
                )
            )

        if claims_count is not None and claims_count > 0:
            detected_events.append(
                TriggerEvent(
                    trigger_name=TriggerName.CLAIMS_HISTORY,
                    detection_method=DetectionMethod.REGEX,
                    context=f'bop_number_of_claims={int(claims_count)}',
                    confidence=1.0,
                )
            )

        # Regex Scan (Fast Path) using pre-compiled patterns
        for spec in self._trigger_specs:
            compiled_list = self._compiled_patterns.get(spec.name, [])
            for pattern in compiled_list:
                match = pattern.search(full_text)
                if match:
                    detected_events.append(
                        TriggerEvent(
                            trigger_name=spec.name,
                            detection_method=DetectionMethod.REGEX,
                            context=match.group(0),
                            confidence=1.0,
                        )
                    )
                    # Break inner loop (don't match same rule twice)
                    break

        # Deduplication Step
        detected_events = self._deduplicate_events(detected_events)

        # Ghost Referral Handling (LLM Fallback)
        if is_referral and not detected_events:
            llm_fallback_used = True
            llm_result = cast(
                Optional[GhostReferralOutput],
                await self.classifier.execute({'actual_output': full_text}),
            )

            if llm_result and llm_result.likely_trigger != TriggerName.UNKNOWN:
                detected_events.append(
                    TriggerEvent(
                        trigger_name=llm_result.likely_trigger,
                        detection_method=DetectionMethod.LLM_FALLBACK,
                        context=llm_result.reasoning,
                        confidence=0.8,
                    )
                )
            else:
                unknown_reasoning = llm_result.reasoning if llm_result else None

        # Compute Enhanced Signals
        trigger_count = len(detected_events)

        min_confidence = 1.0
        if detected_events:
            min_confidence = min(e.confidence for e in detected_events)

        # Check for hard triggers
        has_hard_trigger = any(
            self._spec_by_name.get(
                e.trigger_name,
                TriggerSpec(
                    name=e.trigger_name,
                    patterns=[],
                    description='',
                    priority=999,
                    severity='soft',
                ),
            ).severity
            == 'hard'
            for e in detected_events
        )

        # Priority-based Primary Selection
        primary_reason = self._select_primary_trigger(detected_events)
        if primary_reason == TriggerName.NONE and is_referral:
            primary_reason = TriggerName.UNKNOWN

        if self.use_unknown_reason_llm and primary_reason == TriggerName.UNKNOWN:
            reason_output = cast(
                Optional[UnknownTriggerReasonOutput],
                await self.unknown_reasoner.execute({'actual_output': full_text}),
            )
            if reason_output:
                unknown_reasoning = reason_output.reasoning

        summary_str = ', '.join([t.trigger_name.value for t in detected_events])

        # Score Definition:
        # when outcome aligns with triggers:
        # - referral/decline AND triggers present
        # - approval/unknown AND no triggers present
        # 0.0 for mismatches (approve with triggers, refer without triggers)
        score = 1.0 if (is_referral == bool(detected_events)) else 0.0
        if primary_reason == TriggerName.UNKNOWN:
            score = 0.0

        # Build Enriched TriggerReport
        result_data = TriggerReport(
            is_referral=is_referral,
            active_triggers=detected_events,
            primary_referral_reason=primary_reason,
            summary_text=summary_str,
            outcome_label=outcome_label,
            trigger_count=trigger_count,
            llm_fallback_used=llm_fallback_used,
            min_confidence=min_confidence,
            has_hard_trigger=has_hard_trigger,
            unknown_reasoning=unknown_reasoning,
        )

        # Compute cost estimate only if LLM fallback was used
        if llm_fallback_used:
            self.compute_cost_estimate([self.classifier])
        if self.use_unknown_reason_llm and primary_reason == TriggerName.UNKNOWN:
            self.compute_cost_estimate([self.unknown_reasoner])

        return MetricEvaluationResult(
            score=score, explanation=primary_reason, signals=result_data
        )

    def get_signals(self, result: TriggerReport) -> List[SignalDescriptor]:
        """
        Defines how the results appear in the Dashboard/UI.
        """
        signals = []

        # Headline: Primary Reason (e.g., "orgEstYear")
        signals.append(
            SignalDescriptor(
                name='Primary Trigger',
                group='Overview',
                extractor=lambda r: cast(
                    TriggerReport, r
                ).primary_referral_reason.value,
                headline_display=True,
            )
        )

        # Outcome Label (Referral/Approved/Unknown)
        signals.append(
            SignalDescriptor(
                name='Outcome Status',
                group='Overview',
                extractor=lambda r: cast(TriggerReport, r).outcome_label,
                headline_display=False,
            )
        )

        # Trigger Count
        signals.append(
            SignalDescriptor(
                name='Trigger Count',
                group='Overview',
                extractor=lambda r: cast(TriggerReport, r).trigger_count,
                headline_display=False,
            )
        )

        # LLM Fallback Used (Yes/No)
        signals.append(
            SignalDescriptor(
                name='LLM Fallback Used',
                group='Debug',
                extractor=lambda r: 'Yes'
                if cast(TriggerReport, r).llm_fallback_used
                else 'No',
                headline_display=False,
            )
        )

        signals.append(
            SignalDescriptor(
                name='Unknown Trigger Reason',
                group='Debug',
                extractor=lambda r: cast(TriggerReport, r).unknown_reasoning or 'N/A',
                headline_display=False,
            )
        )

        # Min Confidence (percentage)
        signals.append(
            SignalDescriptor(
                name='Min Confidence',
                group='Debug',
                extractor=lambda r: f'{cast(TriggerReport, r).min_confidence:.0%}',
                headline_display=False,
            )
        )

        # Has Hard Trigger (Yes/No)
        signals.append(
            SignalDescriptor(
                name='Has Hard Trigger',
                group='Overview',
                extractor=lambda r: 'Yes'
                if cast(TriggerReport, r).has_hard_trigger
                else 'No',
                headline_display=False,
            )
        )

        # Detection Method (for primary trigger)
        signals.append(
            SignalDescriptor(
                name='Detection Method',
                group='Debug',
                extractor=lambda r: next(
                    (
                        e.detection_method.value
                        for e in cast(TriggerReport, r).active_triggers
                        if e.trigger_name
                        == cast(TriggerReport, r).primary_referral_reason
                    ),
                    'None',
                ),
                headline_display=False,
            )
        )

        # Per-trigger details with severity labels
        for i, event in enumerate(result.active_triggers):
            spec = self._spec_by_name.get(event.trigger_name)
            severity = spec.severity if spec else 'unknown'

            signals.append(
                SignalDescriptor(
                    name=f'trigger_{i}',
                    group=f'Rule: {event.trigger_name.value}',
                    extractor=lambda r, idx=i: cast(TriggerReport, r)
                    .active_triggers[idx]
                    .context,
                    description=f'Detected via {event.detection_method.value} | Severity: {severity}',
                )
            )

        return signals
