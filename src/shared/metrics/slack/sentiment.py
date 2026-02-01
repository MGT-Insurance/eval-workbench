"""
Sentiment Detection Metric for Slack Conversations.

Analyzes user sentiment including positive, neutral, frustrated, and confused states.
"""

from typing import List, Literal, Optional

from axion._core.schema import HumanMessage, RichBaseModel
from axion._core.tracing import trace
from axion._core.types import MetricCategory
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

from shared.metrics.slack.utils import build_transcript, get_human_messages

SentimentType = Literal['positive', 'neutral', 'frustrated', 'confused']


class SentimentInput(RichBaseModel):
    """Input model for sentiment detection."""

    conversation_transcript: str = Field(
        description='Full conversation transcript between user and AI'
    )
    human_message_count: int = Field(
        description='Number of human messages in the conversation'
    )


class SentimentOutput(RichBaseModel):
    """Output model for sentiment detection."""

    sentiment: SentimentType = Field(default='neutral', description='User sentiment')
    sentiment_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description='Sentiment score (0=negative, 0.5=neutral, 1=positive)',
    )
    frustration_indicators: List[str] = Field(
        default_factory=list, description='Indicators of frustration'
    )
    peak_sentiment_turn: Optional[int] = Field(
        default=None, description='Turn with most extreme sentiment (0-indexed)'
    )
    reasoning: str = Field(default='', description='Brief analysis explanation')


class SentimentResult(RichBaseModel):
    """Result model for sentiment analysis."""

    sentiment: SentimentType = Field(default='neutral', description='User sentiment')
    sentiment_score: float = Field(
        default=0.5, description='Sentiment score (0-1, higher=more positive)'
    )
    is_frustrated: bool = Field(default=False, description='User showed frustration')
    is_positive: bool = Field(
        default=False, description='User showed positive sentiment'
    )
    is_confused: bool = Field(default=False, description='User showed confusion')
    frustration_indicators: List[str] = Field(
        default_factory=list, description='Indicators of frustration/negativity'
    )
    peak_sentiment_turn: Optional[int] = Field(default=None)
    reasoning: str = Field(default='')
    human_message_count: int = Field(default=0)


class SentimentAnalyzer(BaseMetric[SentimentInput, SentimentOutput]):
    """Internal LLM-based analyzer for sentiment detection."""

    instruction = """You are an expert at detecting user sentiment in Slack conversations between users and an AI assistant.

**TASK**: Analyze the conversation to determine user sentiment.

**SENTIMENT TYPES**:
- `positive`: Satisfied, appreciative, complimentary, happy with the interaction
- `neutral`: Matter-of-fact, professional, neither positive nor negative
- `frustrated`: Annoyed, upset, impatient, complaining, expressing dissatisfaction
- `confused`: Uncertain, asking for clarification, misunderstanding, lost

**SENTIMENT SCORE** (0.0 to 1.0):
- 0.0-0.2: Very negative/frustrated
- 0.2-0.4: Mildly negative
- 0.4-0.6: Neutral
- 0.6-0.8: Mildly positive
- 0.8-1.0: Very positive

**FRUSTRATION INDICATORS**:
- Multiple punctuation (???, !!!)
- ALL CAPS words
- Explicit complaints ("this is frustrating", "doesn't work")
- Sarcasm or dismissive language
- Giving up language ("forget it", "never mind")
- Repeated rephrasing of the same question

**POSITIVE INDICATORS**:
- Thank you messages
- Appreciation expressions
- Confirmation of helpful response
- Compliments

**OUTPUT**:
- sentiment: The overall user sentiment
- sentiment_score: 0.0-1.0 (higher = more positive)
- frustration_indicators: List of specific signals detected
- peak_sentiment_turn: Turn number (0-indexed) with most extreme sentiment
- reasoning: Brief explanation (1-2 sentences)"""

    input_model = SentimentInput
    output_model = SentimentOutput


@metric(
    name='Sentiment Detector',
    key='sentiment_detector',
    description='Detects user sentiment in Slack conversations (positive/neutral/frustrated/confused).',
    required_fields=['conversation'],
    optional_fields=['additional_input'],
    metric_category=MetricCategory.SCORE,
    default_threshold=0.4,
    score_range=(0, 1),
    tags=['slack', 'multi_turn', 'sentiment', 'feedback'],
)
class SentimentDetector(BaseMetric):
    """
    LLM-based metric that detects user sentiment in conversations.

    Used for computing:
    - frustration_rate: Frustrated interactions / Total interactions
    - sentiment distribution: positive/neutral/frustrated/confused

    Score interpretation (higher = more positive):
    - 0.0-0.2: Very frustrated/negative
    - 0.2-0.4: Mildly negative
    - 0.4-0.6: Neutral
    - 0.6-0.8: Mildly positive
    - 0.8-1.0: Very positive

    Default threshold: 0.4 (below = frustrated)
    """

    def __init__(self, frustration_threshold: float = 0.4, **kwargs):
        """
        Initialize the sentiment detector.

        Args:
            frustration_threshold: Score below which user is "frustrated" (default: 0.4)
        """
        super().__init__(**kwargs)
        self.frustration_threshold = frustration_threshold
        self.sentiment_analyzer = SentimentAnalyzer(**kwargs)

    @trace(name='SentimentDetector', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """Detect sentiment in conversation."""

        if not item.conversation:
            return MetricEvaluationResult(
                score=0.5,
                explanation='No conversation provided.',
                signals=SentimentResult(),
            )

        human_messages = get_human_messages(item.conversation)
        if not human_messages:
            return MetricEvaluationResult(
                score=0.5,
                explanation='No human messages to analyze.',
                signals=SentimentResult(),
            )

        transcript = build_transcript(item.conversation)

        try:
            analysis_input = SentimentInput(
                conversation_transcript=transcript,
                human_message_count=len(human_messages),
            )

            llm_result = await self.sentiment_analyzer.execute(analysis_input)

            result = SentimentResult(
                sentiment=llm_result.sentiment,
                sentiment_score=llm_result.sentiment_score,
                is_frustrated=llm_result.sentiment == 'frustrated'
                or llm_result.sentiment_score < self.frustration_threshold,
                is_positive=llm_result.sentiment == 'positive'
                or llm_result.sentiment_score >= 0.7,
                is_confused=llm_result.sentiment == 'confused',
                frustration_indicators=llm_result.frustration_indicators,
                peak_sentiment_turn=llm_result.peak_sentiment_turn,
                reasoning=llm_result.reasoning,
                human_message_count=len(human_messages),
            )

            explanation = (
                f'Sentiment: {result.sentiment} (score: {result.sentiment_score:.2f}). '
                f'{result.reasoning}'
            )

        except Exception as e:
            result = self._heuristic_fallback(item, human_messages)
            explanation = f'Heuristic analysis (LLM failed: {e}): {result.sentiment}'

        return MetricEvaluationResult(
            score=result.sentiment_score,
            explanation=explanation,
            signals=result,
        )

    def _heuristic_fallback(
        self, item: DatasetItem, human_messages: List[HumanMessage]
    ) -> SentimentResult:
        """Fallback heuristic when LLM fails."""
        import re

        total_indicators = []
        frustration_scores = []
        positive_signals = 0
        peak_turn = None
        peak_score = 0.0

        for idx, msg in enumerate(item.conversation.messages):
            if not isinstance(msg, HumanMessage) or not msg.content:
                continue

            text = msg.content
            turn_score = 0.0
            turn_indicators = []

            # Frustration patterns
            if re.search(r'\?\?+', text):
                turn_score += 0.2
                turn_indicators.append('multiple question marks')

            if re.search(r'!!+', text):
                turn_score += 0.2
                turn_indicators.append('multiple exclamation marks')

            caps_count = len(re.findall(r'\b[A-Z]{3,}\b', text))
            if caps_count > 0:
                turn_score += min(0.3, caps_count * 0.1)
                turn_indicators.append(f'{caps_count} ALL CAPS words')

            negative_patterns = [
                (r'frustrat', 'frustration expressed'),
                (r"doesn'?t?\s+work", 'complaint about not working'),
                (r'still\s+(?:not|wrong)', 'persistent issue'),
                (r'forget\s+it|never\s*mind', 'giving up language'),
            ]
            for pattern, indicator in negative_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    turn_score += 0.2
                    turn_indicators.append(indicator)

            # Positive patterns
            positive_patterns = [
                r'thank',
                r'great',
                r'perfect',
                r'appreciate',
                r'helpful',
            ]
            for pattern in positive_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    positive_signals += 1

            if turn_indicators:
                total_indicators.extend(turn_indicators)
                frustration_scores.append(turn_score)
                if turn_score > peak_score:
                    peak_score = turn_score
                    peak_turn = idx

        # Calculate sentiment score
        overall_frustration = 0.0
        if frustration_scores and human_messages:
            overall_frustration = min(
                1.0, sum(frustration_scores) / len(human_messages)
            )

        # Base score: 0.5 (neutral), adjust by frustration and positive signals
        sentiment_score = 0.5 - (overall_frustration * 0.5) + (positive_signals * 0.1)
        sentiment_score = max(0.0, min(1.0, sentiment_score))

        # Determine sentiment type
        if sentiment_score < self.frustration_threshold:
            sentiment: SentimentType = 'frustrated'
        elif sentiment_score >= 0.7:
            sentiment = 'positive'
        else:
            sentiment = 'neutral'

        return SentimentResult(
            sentiment=sentiment,
            sentiment_score=round(sentiment_score, 2),
            is_frustrated=sentiment == 'frustrated',
            is_positive=sentiment == 'positive',
            is_confused=False,
            frustration_indicators=total_indicators[:10],
            peak_sentiment_turn=peak_turn,
            reasoning='Heuristic analysis based on text patterns',
            human_message_count=len(human_messages),
        )

    def get_signals(
        self, result: SentimentResult
    ) -> List[SignalDescriptor[SentimentResult]]:
        """Generate signal descriptors."""

        sentiment_scores = {
            'positive': 1.0,
            'neutral': 0.5,
            'confused': 0.3,
            'frustrated': 0.1,
        }

        return [
            SignalDescriptor(
                name='sentiment',
                extractor=lambda r: r.sentiment,
                headline_display=True,
                score_mapping=sentiment_scores,
                description='User sentiment classification',
            ),
            SignalDescriptor(
                name='sentiment_score',
                extractor=lambda r: r.sentiment_score,
                headline_display=True,
                description='Sentiment score (0=negative, 1=positive)',
            ),
            SignalDescriptor(
                name='is_frustrated',
                extractor=lambda r: r.is_frustrated,
                headline_display=True,
                description=f'Score < {self.frustration_threshold}',
            ),
            SignalDescriptor(
                name='is_positive',
                extractor=lambda r: r.is_positive,
                description='Positive sentiment detected',
            ),
            SignalDescriptor(
                name='is_confused',
                extractor=lambda r: r.is_confused,
                description='Confusion detected',
            ),
            SignalDescriptor(
                name='frustration_indicators',
                extractor=lambda r: '; '.join(r.frustration_indicators[:5])
                if r.frustration_indicators
                else None,
                description='Detected negative signals',
            ),
            SignalDescriptor(
                name='peak_sentiment_turn',
                extractor=lambda r: r.peak_sentiment_turn,
                description='Turn with most extreme sentiment',
            ),
        ]
