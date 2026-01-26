from typing import List
from pydantic import Field

from axion._core.logging import get_logger
from axion._core.schema import RichBaseModel
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor

logger = get_logger(__name__)


class ReasoningGap(RichBaseModel):
    """A specific concept mentioned by the human but missed by the AI."""

    concept: str = Field(..., description="The specific risk factor or reason missed.")
    impact: str = Field(
        ..., description="Why this omission matters (e.g., 'Critical', 'Nuance')."
    )
    model_config = {"extra": "forbid"}


class ReasoningMatch(RichBaseModel):
    """A concept where both Human and AI aligned."""

    concept: str = Field(
        ..., description="The risk factor or reason identified by both."
    )
    model_config = {"extra": "forbid"}


class DecisionQualityResult(RichBaseModel):
    """The structured output for the Decision Quality metric."""

    # Outcome Component
    overall_score: float = Field(
        ..., description="Combined score of outcome and reasoning."
    )
    outcome_match: bool = Field(
        ..., description="True if the final decisions (Approve/Decline) match."
    )
    outcome_score: float = Field(..., description="1.0 for match, 0.0 for mismatch.")
    human_decision_detected: str = Field(
        ..., description="Normalized decision extracted from Human Notes."
    )
    ai_decision_detected: str = Field(
        ..., description="Normalized decision extracted from AI Output."
    )

    # Reasoning Component
    reasoning_score: float = Field(
        ..., description="Score based on coverage of human concepts."
    )
    missing_concepts: List[ReasoningGap] = Field(
        default_factory=list, description="List of reasons the AI missed."
    )
    matched_concepts: List[ReasoningMatch] = Field(
        default_factory=list, description="List of reasons both agreed on."
    )
    model_config = {"extra": "forbid"}


# --- 2. Reasoning Judge Component (LLM Model) ---


class ReasoningJudgeInput(RichBaseModel):
    human_notes: str = Field(
        ..., description="The underwriter's notes serving as the ground truth criteria."
    )
    ai_output: str = Field(..., description="The AI agent's full recommendation.")
    model_config = {"extra": "forbid"}


class ReasoningJudgeOutput(RichBaseModel):
    missing_concepts: List[ReasoningGap]
    matched_concepts: List[ReasoningMatch]
    reasoning_completeness_score: float = Field(
        ...,
        description="0.0 to 1.0 score representing how well the AI captured the human's reasoning.",
    )
    model_config = {"extra": "forbid"}


class ReasoningJudge(BaseMetric[ReasoningJudgeInput, ReasoningJudgeOutput]):
    instruction = """
    You are an Expert Underwriting Auditor. Compare the "AI Recommendation" against the "Human Underwriter Notes".

    Your Goal: Determine if the AI identified the SAME risk factors as the Human.

    Step 1: Extract every distinct risk concept or reason mentioned in the 'Human Underwriter Notes' (e.g., 'Firewall issues', 'Slip/Trip risk', 'High TIV').
    Step 2: Check if the 'AI Recommendation' explicitly addresses these concepts.
    Step 3: Categorize each concept as "Matched" (AI discussed it) or "Missing" (AI ignored it).

    CRITICAL:
    - Ignore extra information in the AI output. Focus ONLY on whether the Human's points were covered.
    - If the Human declined for "Roof Age", and the AI declined for "Credit Score" but did not mention Roof Age, mark "Roof Age" as MISSING.
    """
    input_model = ReasoningJudgeInput
    output_model = ReasoningJudgeOutput


@metric(
    name="DecisionQuality",
    key="decision_quality",
    description="Evaluates if the AI made the right decision for the right reasons.",
    required_fields=["actual_output", "expected_output"],
    score_range=(0, 1),
    tags=["Agent"],
)
class DecisionQuality(BaseMetric):
    def __init__(
        self,
        outcome_weight: float = 0.5,
        reasoning_weight: float = 0.5,
        hard_fail_on_outcome_mismatch: bool = True,
        **kwargs,
    ):
        """
        Args:
            outcome_weight: Weight given to the binary outcome match (default: 0.5).
            reasoning_weight: Weight given to the reasoning completeness score (default: 0.5).
            hard_fail_on_outcome_mismatch: If True, forces the total score to 0.0 if the
                                           outcome does not match, regardless of reasoning (default: True).
            **kwargs: Arguments passed to the underlying ReasoningJudge.
        """
        super().__init__(**kwargs)
        self.outcome_weight = outcome_weight
        self.reasoning_weight = reasoning_weight
        self.hard_fail_on_outcome_mismatch = hard_fail_on_outcome_mismatch

        self.reasoning_judge = ReasoningJudge(**kwargs)

    async def execute(
        self, dataset_item: DatasetItem, **kwargs
    ) -> MetricEvaluationResult:
        # Parse Decisions
        human_decision = dataset_item.expected_output or "UNKNOWN"
        ai_decision = dataset_item.actual_output or "UNKNOWN"

        # Simple string equality
        outcome_match = (
            human_decision.strip().lower() == ai_decision.strip().lower()
        ) and (human_decision != "UNKNOWN")
        outcome_score = 1.0 if outcome_match else 0.0

        # Reasoning Check (LLM Logic)
        human_text = (
            dataset_item.acceptance_criteria[0]
            if dataset_item.acceptance_criteria
            else human_decision
        )
        ai_text = dataset_item.additional_output.get(
            "brief_recommendation", dataset_item.actual_output
        )

        llm_input = ReasoningJudgeInput(human_notes=human_text, ai_output=ai_text)

        reasoning_result = await self.reasoning_judge.execute(llm_input)

        # Weighted Score
        final_score = (outcome_score * self.outcome_weight) + (
            reasoning_result.reasoning_completeness_score * self.reasoning_weight
        )

        # Apply Hard Fail Logic
        if self.hard_fail_on_outcome_mismatch and not outcome_match:
            final_score = 0.0

        result_data = DecisionQualityResult(
            overall_score=final_score,
            outcome_match=outcome_match,
            outcome_score=outcome_score,
            human_decision_detected=human_decision,
            ai_decision_detected=ai_decision,
            reasoning_score=reasoning_result.reasoning_completeness_score,
            missing_concepts=reasoning_result.missing_concepts,
            matched_concepts=reasoning_result.matched_concepts,
        )

        return MetricEvaluationResult(score=final_score, signals=result_data)

    def get_signals(self, result: DecisionQualityResult) -> List[SignalDescriptor]:
        """
        Define how the results are displayed in the UI.
        Uses extractors to pull data from the result object dynamically.
        """
        signals = []

        # Group 1: High Level Outcome
        signals.append(
            SignalDescriptor(
                name="Outcome Match",
                description="Did the AI and Human make the same final decision?",
                group="Decision Logic",
                extractor=lambda r: r.outcome_match,
                score_mapping={True: 1.0, False: 0.0},
                headline_display=True,
            )
        )

        signals.append(
            SignalDescriptor(
                name="Decisions",
                description="Comparison of detected decisions.",
                group="Decision Logic",
                extractor=lambda r: f"Human: {r.human_decision_detected} | AI: {r.ai_decision_detected}",
            )
        )

        # Group 2: Missing Concepts (The Critical Feedback)
        for i, gap in enumerate(result.missing_concepts):
            group_name = f"Missing: {gap.concept}"
            signals.extend(
                [
                    SignalDescriptor(
                        name=f"missed_concept_{i}",
                        group=group_name,
                        extractor=lambda r, idx=i: r.missing_concepts[idx].concept,
                        description="The concept missed by the AI.",
                    ),
                    SignalDescriptor(
                        name=f"impact_{i}",
                        group=group_name,
                        extractor=lambda r, idx=i: r.missing_concepts[idx].impact,
                        description="Why this omission matters.",
                        value=0.0,  # Visual indicator of failure
                        headline_display=False,
                    ),
                ]
            )

        # Group 3: Matched Concepts (The "Good" Stuff)
        for i, match in enumerate(result.matched_concepts):
            group_name = f"Matched: {match.concept}"
            signals.extend(
                [
                    SignalDescriptor(
                        name=f"matched_concept_{i}",
                        group=group_name,
                        extractor=lambda r, idx=i: r.matched_concepts[idx].concept,
                        description="AI correctly identified this factor.",
                        value=1.0,
                        headline_display=False,
                    )
                ]
            )

        return signals
