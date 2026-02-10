from typing import Any, List, Optional

from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion._core.types import MetricCategory
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SubMetricResult
from pydantic import Field

from eval_workbench.shared.metrics.slack.config import AnalyzerConfig
from eval_workbench.shared.metrics.slack.feedback import (
    SlackFeedbackAttributionAnalyzer,
)
from eval_workbench.shared.metrics.slack.objective import SlackObjectiveAnalyzer
from eval_workbench.shared.metrics.slack.product import SlackProductAnalyzer
from eval_workbench.shared.metrics.slack.subjective import SlackSubjectiveAnalyzer


class UnderwritingCompositeResult(RichBaseModel):
    """Combined signals from all analyzers."""

    objective: Any = Field(default=None)
    subjective: Any = Field(default=None)
    feedback: Any = Field(default=None)
    product: Any = Field(default=None)


@metric(
    name='Underwriting Composite Evaluator',
    key='underwriting_composite_evaluator',
    required_fields=['conversation'],
    description='Orchestrates the full underwriting evaluation pipeline with dependency management.',
    metric_category=MetricCategory.ANALYSIS,
    tags=['slack', 'multi_turn'],
)
class UnderwritingCompositeEvaluator(BaseMetric):
    """
    The "General Manager" metric.

    It runs the sub-analyzers in a strict dependency order:
    1. Objective Analyzer (Get facts: Approved/Declined? Intervention?)
    2. Subjective Analyzer (Get sentiment, using Objective context)
    3. Feedback Analyzer (Get root cause, using Intervention + Sentiment context)
    4. Product Analyzer (Get feature requests)
    """

    is_multi_metric = True
    include_parent_score = False
    sub_metric_prefix = False

    def __init__(self, config: Optional[AnalyzerConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.config = config or AnalyzerConfig()

        # Initialize sub-analyzers
        self.objective = SlackObjectiveAnalyzer(config=self.config, **kwargs)
        self.subjective = SlackSubjectiveAnalyzer(config=self.config, **kwargs)
        self.feedback = SlackFeedbackAttributionAnalyzer(config=self.config, **kwargs)
        self.product = SlackProductAnalyzer(config=self.config, **kwargs)

    def _map_sentiment_to_score(self, sentiment: str) -> float:
        """Map categorical sentiment to the float expected by Feedback analyzer."""
        mapping = {'positive': 1.0, 'neutral': 0.5, 'confused': 0.3, 'frustrated': 0.0}
        return mapping.get(sentiment, 0.5)

    @trace(
        name='UnderwritingCompositeEvaluator', capture_args=True, capture_response=True
    )
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        """Execute the pipeline in sequence."""

        # OBJECTIVE
        # Did the human intervene? What was the outcome?
        obj_result = await self.objective.execute(item)
        obj_signals = obj_result.signals

        # Determine if we need deep analysis
        # If no intervention and resolved successfully, we might skip deep diagnostics
        needs_diagnosis = (
            obj_signals.intervention.has_intervention
            or obj_signals.escalation.is_escalated
        )

        # Prepare context for Subjective
        # Subjective needs to know if there was an escalation or intervention
        subj_context = {
            'is_escalated': obj_signals.escalation.is_escalated,
            'has_intervention': obj_signals.intervention.has_intervention,
            'intervention_type': obj_signals.intervention.intervention_type,
            'final_status': obj_signals.resolution.final_status,
        }

        # SUBJECTIVE
        # Always run subjective to catch "silent frustration" even if no intervention
        subj_result = await self.subjective.execute(
            item, objective_context=subj_context
        )
        subj_signals = subj_result.signals

        # Prepare context for Feedback
        # Feedback needs BOTH intervention facts AND sentiment score
        sentiment_score = self._map_sentiment_to_score(subj_signals.sentiment)

        # Calculate frustration score (1.0 is bad)
        frustration_score = 0.0
        if subj_signals.sentiment == 'frustrated':
            frustration_score = 1.0
        elif subj_signals.sentiment == 'confused':
            frustration_score = 0.5

        # Check if subjective analysis found frustration, which also triggers diagnosis
        if frustration_score > 0.5:
            needs_diagnosis = True

        feedback_context = {
            'has_intervention': obj_signals.intervention.has_intervention,
            'intervention_type': obj_signals.intervention.intervention_type,
            'sentiment_score': sentiment_score,
            'frustration_score': frustration_score,
            'frustration_cause': subj_signals.frustration_cause,
        }

        # FEEDBACK ATTRIBUTION
        # OPTIMIZATION: Only run expensive root cause analysis if there was a problem
        fb_result = None
        if needs_diagnosis:
            fb_result = await self.feedback.execute(
                item, analysis_context=feedback_context
            )

        # PRODUCT SIGNALS
        # Independent, can run with basic context. Good to run always for feature requests.
        prod_context = {
            'has_intervention': obj_signals.intervention.has_intervention,
            'intervention_type': obj_signals.intervention.intervention_type,
            'frustration_score': frustration_score,
            'sentiment': subj_signals.sentiment,
        }
        prod_result = await self.product.execute(item, analysis_context=prod_context)

        # Combine all signals into one master result
        composite_signals = UnderwritingCompositeResult(
            objective=obj_signals,
            subjective=subj_signals,
            feedback=fb_result.signals if fb_result else None,
            product=prod_result.signals,
        )

        return MetricEvaluationResult(
            score=None,
            explanation=f'Eval Complete. Status: {obj_signals.resolution.final_status} | Intervention: {obj_signals.intervention.intervention_type}',
            signals=composite_signals,
            # We store individual results in metadata so they can be accessed if needed
            metadata={
                'objective': obj_result,
                'subjective': subj_result,
                'feedback': fb_result,
                'product': prod_result,
            },
        )

    def get_sub_metrics(self, result: MetricEvaluationResult) -> List[SubMetricResult]:
        """
        Flatten the tree.
        Collects all sub-metrics from the children and presents them as one flat list.
        Each sub-metric's signals column is populated from its own metadata,
        which the child analyzers already fill with the targeted signal data.
        """
        all_metrics = []

        if not result.metadata:
            return []

        # Collect sub-metrics from each child analyzer
        # Objective Metrics
        if 'objective' in result.metadata:
            all_metrics.extend(
                self.objective.get_sub_metrics(result.metadata['objective'])
            )

        # Feedback Metrics (The most important technical ones)
        if 'feedback' in result.metadata and result.metadata['feedback']:
            all_metrics.extend(
                self.feedback.get_sub_metrics(result.metadata['feedback'])
            )

        # Product Metrics (Strategic)
        if 'product' in result.metadata:
            all_metrics.extend(self.product.get_sub_metrics(result.metadata['product']))

        # Subjective Metrics (Optional - can comment out if too noisy)
        if 'subjective' in result.metadata:
            all_metrics.extend(
                self.subjective.get_sub_metrics(result.metadata['subjective'])
            )

        # Populate signals from each sub-metric's own metadata
        # (child analyzers already put the targeted signal data there)
        for sub in all_metrics:
            sub.signals = {
                k: v for k, v in sub.metadata.items() if k != 'cost_estimate'
            }

        return all_metrics
