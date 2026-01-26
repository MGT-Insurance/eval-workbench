"""
Feedback Analysis Module

Provides analytics for Athena AI underwriting conversations.

Components:
- schema: Data models for messages, contexts, metrics, and results
- handler: LLM handler for conversation analysis
- pipeline: ETL pipeline for processing conversations
- metric_computations: Per-conversation metric calculation
- aggregation_service: Period-based aggregation for dashboards
- jobs: Background aggregation jobs
"""

# Schema models
from implementations.athena.models.feedback_analysis.schema import (
    Message,
    ConversationContext,
    InterventionType,
    EscalationType,
    Sentiment,
    AuditResult,
    ConversationMetrics,
    AggregatedMetrics,
)

# Pipeline
from implementations.athena.models.feedback_analysis.pipeline import (
    run_pipeline,
    FeedbackAnalysisPipeline,
    PipelineResult,
)

# Handler
from implementations.athena.models.feedback_analysis.handler import (
    UnderwritingAuditHandler,
)

# Metric computation
from implementations.athena.models.feedback_analysis.metric_computations import (
    ConversationMetricCalculator,
    classify_escalation,
    parse_slack_timestamp,
    STALEMATE_THRESHOLD_SECONDS,
)

# Aggregation
from implementations.athena.models.feedback_analysis.aggregation_service import (
    MetricAggregationService,
)

# Jobs
from implementations.athena.models.feedback_analysis.jobs import (
    AggregationJobRunner,
    run_all_pending_aggregations,
)


__all__ = [
    # Schema
    'Message',
    'ConversationContext',
    'InterventionType',
    'EscalationType',
    'Sentiment',
    'AuditResult',
    'ConversationMetrics',
    'AggregatedMetrics',
    # Pipeline
    'run_pipeline',
    'FeedbackAnalysisPipeline',
    'PipelineResult',
    # Handler
    'UnderwritingAuditHandler',
    # Metric computation
    'ConversationMetricCalculator',
    'classify_escalation',
    'parse_slack_timestamp',
    'STALEMATE_THRESHOLD_SECONDS',
    # Aggregation
    'MetricAggregationService',
    # Jobs
    'AggregationJobRunner',
    'run_all_pending_aggregations',
]
