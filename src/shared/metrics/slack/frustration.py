from typing import List, Literal, Optional

from axion._core.schema import HumanMessage, RichBaseModel
from axion._core.tracing import trace
from axion._core.types import MetricCategory
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

from shared.metrics.slack.utils import build_transcript, get_human_messages

FrustrationCause = Literal[
    'none',
    'ai_error',
    'slow_response',
    'wrong_answer',
    'repeated_questions',
    'poor_understanding',
    'system_issue',
    'other',
]


class FrustrationInput(RichBaseModel):
    """Input model for frustration detection."""

    conversation_transcript: str = Field(
        description='Full conversation transcript between user and AI'
    )
    human_message_count: int = Field(
        description='Number of human messages in the conversation',
    )


class FrustrationOutput(RichBaseModel):
    """Output model for frustration detection."""

    frustration_score: float = Field(
        ge=0.0,
        le=1.0,
        description='Overall frustration score (0=calm, 1=very frustrated)',
    )
    frustration_indicators: List[str] = Field(
        default_factory=list,
        description='List of detected frustration signals',
    )
    peak_frustration_turn: Optional[int] = Field(
        default=None,
        description='Turn index with highest frustration (0-based)',
    )
    frustration_cause: FrustrationCause = Field(
        default='none',
        description='Primary cause of frustration',
    )
    reasoning: str = Field(
        default='',
        description='Explanation of the frustration assessment',
    )


class FrustrationResult(RichBaseModel):
    """Result model for frustration analysis."""

    frustration_score: float = Field(
        default=0.0,
        description='Overall frustration score (0-1)',
    )
    is_frustrated: bool = Field(
        default=False,
        description='Whether user appears frustrated (score >= threshold)',
    )
    frustration_indicators: List[str] = Field(
        default_factory=list,
        description='Detected frustration signals',
    )
    peak_frustration_turn: Optional[int] = Field(
        default=None,
        description='Turn with highest frustration',
    )
    frustration_cause: FrustrationCause = Field(
        default='none',
        description='Primary cause of frustration',
    )
    reasoning: str = Field(
        default='',
        description='Assessment explanation',
    )
    human_message_count: int = Field(
        default=0,
        description='Number of human messages analyzed',
    )


class FrustrationAnalyzer(BaseMetric[FrustrationInput, FrustrationOutput]):
    """Internal LLM-based analyzer for frustration detection."""

    instruction = """You are an expert at detecting user frustration in Slack conversations between users and an AI assistant.

**TASK**: Analyze the conversation to score the user's frustration level.

**FRUSTRATION INDICATORS**:
1. **Repeated rephrasing**: User restates the same question multiple times
2. **Punctuation emphasis**: Multiple question marks (???) or exclamation marks (!!!)
3. **Capitalization**: ALL CAPS words or phrases indicating emphasis/frustration
4. **Explicit complaints**: Direct expressions of frustration ("this is frustrating", "doesn't work")
5. **Sarcasm or negative tone**: Sarcastic remarks, dismissive language
6. **Escalation requests**: Asking for human help after AI interaction
7. **Giving up language**: "forget it", "never mind", abandoning the conversation

**FRUSTRATION CAUSES**:
- **ai_error**: AI made a clear mistake or error
- **slow_response**: Complaints about response time or delays
- **wrong_answer**: AI provided incorrect or unhelpful information
- **repeated_questions**: User had to repeat themselves multiple times
- **poor_understanding**: AI didn't understand what user was asking
- **system_issue**: Technical or system problems
- **other**: Other causes not listed above
- **none**: No frustration detected

**SCORING GUIDELINES**:
- 0.0-0.2: Calm, positive, or neutral interaction
- 0.2-0.4: Mild impatience or minor friction
- 0.4-0.6: Moderate frustration, some negative signals
- 0.6-0.8: Significant frustration, multiple negative signals
- 0.8-1.0: Severe frustration, user is clearly upset

**OUTPUT**:
- frustration_score: A score from 0.0 to 1.0
- frustration_indicators: List of specific signals detected (e.g., "repeated question at turn 3", "all caps usage")
- peak_frustration_turn: Turn number (0-indexed) where frustration peaked
- frustration_cause: Primary cause from the list above
- reasoning: Brief explanation (1-2 sentences) of your assessment"""

    input_model = FrustrationInput
    output_model = FrustrationOutput


@metric(
    name='Frustration Detector',
    key='frustration_detector',
    description='Scores user frustration level in Slack conversations.',
    required_fields=['conversation'],
    optional_fields=['additional_input'],
    metric_category=MetricCategory.SCORE,
    default_threshold=0.6,
    score_range=(0, 1),
    tags=['slack', 'multi_turn'],
)
class FrustrationDetector(BaseMetric):
    """
    LLM-based metric that scores user frustration in Slack conversations.

    Used for computing:
    - frustration_rate: Frustrated interactions / Total interactions

    Score interpretation:
    - 0.0-0.2: Calm interaction
    - 0.2-0.4: Mild impatience
    - 0.4-0.6: Moderate frustration
    - 0.6-0.8: Significant frustration
    - 0.8-1.0: Severe frustration

    Default threshold: 0.6 (indicates frustrated interaction)
    """

    def __init__(self, frustration_threshold: float = 0.6, **kwargs):
        """
        Initialize the frustration detector.

        Args:
            frustration_threshold: Score threshold for "frustrated" classification (default: 0.6)
        """
        super().__init__(**kwargs)
        self.frustration_threshold = frustration_threshold
        self.frustration_analyzer = FrustrationAnalyzer(**kwargs)

    @trace(name='FrustrationDetector', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """Detect frustration in Slack conversation."""

        if not item.conversation:
            return MetricEvaluationResult(
                score=0.0,
                explanation='No conversation provided.',
                signals=FrustrationResult(),
            )

        # Get human messages
        human_messages = get_human_messages(item.conversation)
        if not human_messages:
            return MetricEvaluationResult(
                score=0.0,
                explanation='No human messages to analyze.',
                signals=FrustrationResult(),
            )

        # Build transcript
        transcript = build_transcript(item.conversation)

        # Prepare input for LLM
        analysis_input = FrustrationInput(
            conversation_transcript=transcript,
            human_message_count=len(human_messages),
        )

        # Run LLM analysis
        try:
            llm_result = await self.frustration_analyzer.execute(analysis_input)

            result = FrustrationResult(
                frustration_score=llm_result.frustration_score,
                is_frustrated=llm_result.frustration_score
                >= self.frustration_threshold,
                frustration_indicators=llm_result.frustration_indicators,
                peak_frustration_turn=llm_result.peak_frustration_turn,
                frustration_cause=llm_result.frustration_cause,
                reasoning=llm_result.reasoning,
                human_message_count=len(human_messages),
            )

            explanation = (
                f'Frustration score: {result.frustration_score:.2f} '
                f'({"frustrated" if result.is_frustrated else "not frustrated"}). '
                f'Cause: {result.frustration_cause}. {result.reasoning}'
            )

        except Exception as e:
            # Fallback to heuristic if LLM fails
            result = self._heuristic_fallback(item, human_messages)
            explanation = f'Heuristic analysis (LLM failed: {e}): score={result.frustration_score:.2f}'

        return MetricEvaluationResult(
            score=result.frustration_score,
            explanation=explanation,
            signals=result,
        )

    def _heuristic_fallback(
        self, item: DatasetItem, human_messages: List[HumanMessage]
    ) -> FrustrationResult:
        """
        Fallback heuristic when LLM analysis fails.

        Uses simple pattern matching for frustration detection.
        """
        import re

        total_indicators = []
        frustration_scores = []
        peak_turn = None
        peak_score = 0.0

        for idx, msg in enumerate(item.conversation.messages):
            if not isinstance(msg, HumanMessage) or not msg.content:
                continue

            text = msg.content
            turn_score = 0.0
            turn_indicators = []

            # Multiple question marks
            if re.search(r'\?\?+', text):
                turn_score += 0.2
                turn_indicators.append('multiple question marks')

            # Multiple exclamation marks
            if re.search(r'!!+', text):
                turn_score += 0.2
                turn_indicators.append('multiple exclamation marks')

            # All caps words (3+ chars)
            caps_count = len(re.findall(r'\b[A-Z]{3,}\b', text))
            if caps_count > 0:
                turn_score += min(0.3, caps_count * 0.1)
                turn_indicators.append(f'{caps_count} ALL CAPS words')

            # Negative phrases
            negative_patterns = [
                (r'frustrat', 'frustration expressed'),
                (r'doesn\'?t?\s+work', 'complaint about not working'),
                (r'still\s+(?:not|wrong)', 'persistent issue'),
                (r'again\s+and\s+again', 'repetition complaint'),
                (r'forget\s+it|never\s*mind', 'giving up language'),
            ]
            for pattern, indicator in negative_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    turn_score += 0.2
                    turn_indicators.append(indicator)

            if turn_indicators:
                total_indicators.extend([f'Turn {idx}: {i}' for i in turn_indicators])
                frustration_scores.append(turn_score)

                if turn_score > peak_score:
                    peak_score = turn_score
                    peak_turn = idx

        # Calculate overall score
        overall_score = 0.0
        if frustration_scores:
            overall_score = min(1.0, sum(frustration_scores) / len(human_messages))

        return FrustrationResult(
            frustration_score=round(overall_score, 2),
            is_frustrated=overall_score >= self.frustration_threshold,
            frustration_indicators=total_indicators[:10],  # Limit to 10
            peak_frustration_turn=peak_turn,
            frustration_cause='other' if total_indicators else 'none',
            reasoning='Heuristic analysis based on text patterns',
            human_message_count=len(human_messages),
        )

    def get_signals(
        self, result: FrustrationResult
    ) -> List[SignalDescriptor[FrustrationResult]]:
        """Generate signal descriptors for frustration detection."""

        cause_scores = {
            'none': 0.0,
            'ai_error': 0.8,
            'slow_response': 0.5,
            'wrong_answer': 0.7,
            'repeated_questions': 0.6,
            'poor_understanding': 0.6,
            'system_issue': 0.5,
            'other': 0.5,
        }

        return [
            # Headline signals
            SignalDescriptor(
                name='frustration_score',
                extractor=lambda r: r.frustration_score,
                headline_display=True,
                description='Overall frustration score (0-1)',
            ),
            SignalDescriptor(
                name='is_frustrated',
                extractor=lambda r: r.is_frustrated,
                headline_display=True,
                description=f'Score >= {self.frustration_threshold}',
            ),
            SignalDescriptor(
                name='frustration_cause',
                extractor=lambda r: r.frustration_cause,
                score_mapping=cause_scores,
                description='Primary cause of frustration',
            ),
            # Detail signals
            SignalDescriptor(
                name='peak_frustration_turn',
                extractor=lambda r: r.peak_frustration_turn,
                description='Turn with highest frustration',
            ),
            SignalDescriptor(
                name='frustration_indicators',
                extractor=lambda r: '; '.join(r.frustration_indicators[:5])
                if r.frustration_indicators
                else None,
                description='Detected frustration signals',
            ),
            SignalDescriptor(
                name='reasoning',
                extractor=lambda r: r.reasoning,
                description='Assessment explanation',
            ),
            SignalDescriptor(
                name='human_message_count',
                extractor=lambda r: r.human_message_count,
                description='Human messages analyzed',
            ),
        ]
