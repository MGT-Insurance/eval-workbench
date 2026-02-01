"""
Feedback Conversation Analysis Metrics (DEPRECATED).

This module has been moved to shared/metrics/slack/.

Please update your imports:

    # Old (deprecated)
    from implementations.athena.metrics.feedback import FeedbackConversationAnalyzer

    # New (recommended)
    from shared.metrics.slack import (
        InterventionDetector,
        SentimentDetector,
        ResolutionDetector,
    )

The new metrics follow the same pattern as other slack metrics and can be used
individually or combined via the existing SlackConversationAnalyzer.
"""

from __future__ import annotations

import warnings

from shared.metrics.slack.intervention import (
    EscalationCategory,
    InterventionCategory,
    InterventionDetector,
    InterventionResult,
    classify_escalation,
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

warnings.warn(
    'implementations.athena.metrics.feedback is deprecated. '
    'Use shared.metrics.slack instead: InterventionDetector, SentimentDetector, ResolutionDetector.',
    DeprecationWarning,
    stacklevel=2,
)

# Backwards compatibility aliases
FeedbackConversationAnalyzer = InterventionDetector  # Closest equivalent

__all__ = [
    # New names
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
    # Backwards compatibility
    'FeedbackConversationAnalyzer',
]
