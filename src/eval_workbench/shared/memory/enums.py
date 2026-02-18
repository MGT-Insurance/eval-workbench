"""Enums for rule_extractions provenance and lifecycle tracking.

DB stores VARCHAR; Python code enforces valid values through these enums.
"""

from __future__ import annotations

from enum import Enum


class ReviewStatus(str, Enum):
    PENDING_REVIEW = 'pending_review'
    APPROVED = 'approved'
    REJECTED = 'rejected'
    DEFERRED = 'deferred'


class ProposalKind(str, Enum):
    EXTRACTED_RULE = 'extracted_rule'
    GRAPH_HINT = 'graph_hint'
    MUTATION_PROPOSAL = 'mutation_proposal'


class IngestionStatus(str, Enum):
    PENDING = 'pending'
    INGESTED = 'ingested'
    FAILED = 'failed'


class SourceDataset(str, Enum):
    KB_FEEDBACK = 'kb_feedback'
    RETROSPECTIVE = 'retrospective'
    SLACK_THREAD = 'slack_thread'
    SLACK_GRAPH_HINT = 'slack_graph_hint'
    MANUAL_SEED = 'manual_seed'


class SourceCategory(str, Enum):
    UW_RULE = 'uw_rule'
    DATA_QUALITY = 'data_quality'
    OPERATIONAL = 'operational'
    MIXED = 'mixed'
    TRIAGE_UNCERTAIN = 'triage_uncertain'
    PROCESS_FROM_SLACK = 'process_from_slack'
    OPERATIONAL_EXTRACTED = 'operational_extracted'
