# Individual metrics
from shared.metrics.slack.acceptance import AcceptanceDetector, AcceptanceResult

# Composite analyzers
from shared.metrics.slack.composite import (
    AcceptanceSignals,
    EngagementSignals,
    EscalationSignals,
    FrustrationSignals,
    InteractionSignals,
    OverrideSignals,
    RecommendationSignals,
    SatisfactionSignals,
    SlackAnalysisResult,
    SlackConversationAnalyzer,
)
from shared.metrics.slack.engagement import (
    ThreadEngagementAnalyzer,
    ThreadEngagementResult,
)
from shared.metrics.slack.escalation import EscalationDetector, EscalationResult
from shared.metrics.slack.frustration import FrustrationDetector, FrustrationResult
from shared.metrics.slack.interaction import (
    SlackInteractionAnalyzer,
    SlackInteractionResult,
)

# Feedback analysis metrics
from shared.metrics.slack.intervention import (
    EscalationCategory,
    InterventionCategory,
    InterventionDetector,
    InterventionResult,
    classify_escalation,
)
from shared.metrics.slack.override import OverrideDetector, OverrideResult
from shared.metrics.slack.recommendation import (
    RecommendationAnalyzer,
    RecommendationResult,
)
from shared.metrics.slack.resolution import (
    STALEMATE_THRESHOLD_SECONDS,
    ResolutionDetector,
    ResolutionResult,
    ResolutionStatus,
)
from shared.metrics.slack.sentiment import (
    SentimentDetector,
    SentimentResult,
    SentimentType,
)

# Utilities
from shared.metrics.slack.utils import (
    SlackMetadata,
    build_transcript,
    count_questions,
    detect_ai_sender,
    extract_case_id,
    extract_mentions,
    extract_priority_score,
    extract_recommendation_type,
    find_recommendation_turn,
    get_ai_messages,
    get_human_messages,
    has_recommendation_pattern,
    parse_slack_metadata,
)

__all__ = [
    # Individual metrics
    'SlackInteractionAnalyzer',
    'SlackInteractionResult',
    'ThreadEngagementAnalyzer',
    'ThreadEngagementResult',
    'RecommendationAnalyzer',
    'RecommendationResult',
    'EscalationDetector',
    'EscalationResult',
    'FrustrationDetector',
    'FrustrationResult',
    'AcceptanceDetector',
    'AcceptanceResult',
    'OverrideDetector',
    'OverrideResult',
    # Feedback analysis metrics
    'InterventionDetector',
    'InterventionResult',
    'InterventionCategory',
    'EscalationCategory',
    'classify_escalation',
    'SentimentDetector',
    'SentimentResult',
    'SentimentType',
    'ResolutionDetector',
    'ResolutionResult',
    'ResolutionStatus',
    'STALEMATE_THRESHOLD_SECONDS',
    # Composite analyzers
    'SlackConversationAnalyzer',
    'SlackAnalysisResult',
    'InteractionSignals',
    'EngagementSignals',
    'RecommendationSignals',
    'EscalationSignals',
    'FrustrationSignals',
    'AcceptanceSignals',
    'OverrideSignals',
    'SatisfactionSignals',
    # Utilities
    'build_transcript',
    'count_questions',
    'detect_ai_sender',
    'extract_case_id',
    'extract_mentions',
    'extract_priority_score',
    'extract_recommendation_type',
    'find_recommendation_turn',
    'get_ai_messages',
    'get_human_messages',
    'has_recommendation_pattern',
    'parse_slack_metadata',
    'SlackMetadata',
]
