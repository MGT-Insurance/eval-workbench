"""Slack conversation analysis metrics.

Standalone Metrics (can be run independently through evaluation_runner):
- SlackHeuristicAnalyzer: Zero-cost heuristic analysis (no LLM)
- SlackObjectiveAnalyzer: Factual classification (temp 0.0)
- SlackSubjectiveAnalyzer: Sentiment and quality analysis (temp 0.3)
- SlackProductAnalyzer: Product insights extraction
- SlackFeedbackAttributionAnalyzer: Failure attribution for negative feedback
"""

from eval_workbench.shared.metrics.slack.feedback import (
    FeedbackAttributionInput,
    FeedbackAttributionOutput,
    SlackFeedbackAttributionAnalyzer,
)
from eval_workbench.shared.metrics.slack.heuristic import (
    EngagementSignals,
    HeuristicAnalysisResult,
    InteractionSignals,
    RecommendationSignals,
    SlackHeuristicAnalyzer,
)
from eval_workbench.shared.metrics.slack.objective import (
    EscalationSignals,
    InterventionSignals,
    ObjectiveAnalysisInput,
    ObjectiveAnalysisOutput,
    ObjectiveAnalysisResult,
    ResolutionSignals,
    SlackObjectiveAnalyzer,
)
from eval_workbench.shared.metrics.slack.product import (
    ProductSignalsInput,
    ProductSignalsOutput,
    SlackProductAnalyzer,
)
from eval_workbench.shared.metrics.slack.subjective import (
    SlackSubjectiveAnalyzer,
    SubjectiveAnalysisInput,
    SubjectiveAnalysisOutput,
)
from eval_workbench.shared.metrics.slack.config import AnalyzerConfig, TruncationConfig
from eval_workbench.shared.metrics.slack.truncation import (
    HAS_TIKTOKEN,
    estimate_tokens,
    estimate_message_tokens,
    format_truncated_transcript,
    get_truncation_markers,
    truncate_conversation,
)
from eval_workbench.shared.metrics.slack.utils import (
    NEGATIVE_REACTIONS,
    POSITIVE_REACTIONS,
    ReactionSignals,
    SlackMetadata,
    StalemateSignals,
    analyze_reactions,
    build_transcript,
    calculate_time_to_resolution,
    count_questions,
    detect_stalemate,
    extract_case_id,
    extract_mentions,
    extract_priority_score,
    extract_recommendation_type,
    find_recommendation_turn,
    get_ai_messages,
    get_human_messages,
    parse_slack_metadata,
)

__all__ = [
    # === Standalone Metrics ===
    # Heuristic Analyzer (Zero Cost)
    'SlackHeuristicAnalyzer',
    'HeuristicAnalysisResult',
    'InteractionSignals',
    'EngagementSignals',
    'RecommendationSignals',
    # Objective Analyzer (LLM - Pass 1)
    'SlackObjectiveAnalyzer',
    'ObjectiveAnalysisResult',
    'ObjectiveAnalysisInput',
    'ObjectiveAnalysisOutput',
    'EscalationSignals',
    'InterventionSignals',
    'ResolutionSignals',
    # Subjective Analyzer (LLM - Pass 2)
    'SlackSubjectiveAnalyzer',
    'SubjectiveAnalysisInput',
    'SubjectiveAnalysisOutput',
    # Product Analyzer (Optional LLM)
    'SlackProductAnalyzer',
    'ProductSignalsInput',
    'ProductSignalsOutput',
    # Feedback Attribution Analyzer (Optional LLM)
    'SlackFeedbackAttributionAnalyzer',
    'FeedbackAttributionInput',
    'FeedbackAttributionOutput',
    # === Config ===
    'AnalyzerConfig',
    'TruncationConfig',
    # === Utilities ===
    'SlackMetadata',
    'ReactionSignals',
    'StalemateSignals',
    'parse_slack_metadata',
    'get_ai_messages',
    'get_human_messages',
    'build_transcript',
    'analyze_reactions',
    'detect_stalemate',
    'calculate_time_to_resolution',
    'count_questions',
    'extract_case_id',
    'extract_mentions',
    'extract_priority_score',
    'extract_recommendation_type',
    'find_recommendation_turn',
    'POSITIVE_REACTIONS',
    'NEGATIVE_REACTIONS',
    'estimate_tokens',
    'estimate_message_tokens',
    'truncate_conversation',
    'get_truncation_markers',
    'format_truncated_transcript',
    'HAS_TIKTOKEN',
]
