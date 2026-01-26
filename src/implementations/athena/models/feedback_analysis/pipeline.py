"""
Feedback Analysis Pipeline

ETL pipeline for processing Slack conversations through:
1. Data transformation (raw JSON -> typed contexts)
2. LLM analysis (conversation audit)
3. Metric computation (quantitative metrics)
4. Optional persistence (database storage)
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Set
import pandas as pd

from axion.llm_registry import LLMRegistry
from implementations.athena.models.feedback_analysis.schema import (
    ConversationContext,
    ConversationMetrics,
    AuditResult,
    Message,
)
from implementations.athena.models.feedback_analysis.handler import (
    UnderwritingAuditHandler,
)
from implementations.athena.models.feedback_analysis.metric_computations import (
    ConversationMetricCalculator,
    classify_escalation,
    parse_slack_timestamp,
)


@dataclass
class PipelineResult:
    """Result of running the feedback analysis pipeline."""

    contexts: List[ConversationContext] = field(default_factory=list)
    audit_results: List[AuditResult] = field(default_factory=list)
    metrics: List[ConversationMetrics] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    human_user_ids: Set[str] = field(default_factory=set)

    @property
    def success_count(self) -> int:
        return len(self.audit_results)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def total_count(self) -> int:
        return len(self.contexts)


class FeedbackAnalysisPipeline:
    """
    Pipeline for processing Slack conversations and computing analytics.

    Responsibilities:
    1. Transform raw JSON data to typed ConversationContext objects
    2. Run LLM analysis to produce AuditResult objects
    3. Compute per-conversation metrics
    4. Classify escalation types
    5. Optionally persist results to database
    """

    def __init__(self, llm=None, db_manager=None):
        """
        Initialize the pipeline.

        Args:
            llm: LLM instance for analysis. Defaults to registry LLM.
            db_manager: Optional database manager for persistence.
        """
        self.llm = llm or LLMRegistry().get_llm()
        self.audit_handler = UnderwritingAuditHandler(llm=self.llm)
        self.metric_calculator = ConversationMetricCalculator()
        self.db_manager = db_manager

    async def process(self, raw_data: List[Dict[str, Any]]) -> PipelineResult:
        """
        Process raw Slack data through the full pipeline.

        Args:
            raw_data: List of raw conversation JSON objects

        Returns:
            PipelineResult containing contexts, audit results, metrics, and errors
        """
        # 1. Transform to typed contexts (with timestamp parsing)
        contexts = self._transform_to_contexts(raw_data)
        print(f"Loaded {len(contexts)} conversations.")

        # 2. Run LLM analysis
        print("Running LLM analysis...")
        raw_results = await self._run_analysis(contexts)

        # 3. Process results: separate successes from errors
        audit_results = []
        errors = []
        successful_contexts = []

        for ctx, res in zip(contexts, raw_results):
            if isinstance(res, Exception):
                errors.append(
                    {
                        "thread_id": ctx.thread_id,
                        "error": str(res),
                        "error_type": type(res).__name__,
                    }
                )
                print(f"Error on {ctx.thread_id}: {res}")
            else:
                # 4. Classify escalation type
                res.escalation_type = classify_escalation(res.intervention_category)
                audit_results.append(res)
                successful_contexts.append(ctx)

        # 5. Compute per-conversation metrics
        print("Computing metrics...")
        metrics = []
        for ctx, res in zip(successful_contexts, audit_results):
            metric = self.metric_calculator.compute(ctx, res)
            metrics.append(metric)

        # 6. Collect human user IDs for MAU tracking
        human_user_ids = set()
        for ctx in successful_contexts:
            human_user_ids.update(ctx.human_participants)

        # 7. Persist if db configured
        if self.db_manager:
            await self._store_results(successful_contexts, audit_results, metrics)

        return PipelineResult(
            contexts=successful_contexts,
            audit_results=audit_results,
            metrics=metrics,
            errors=errors,
            human_user_ids=human_user_ids,
        )

    async def _run_analysis(self, contexts: List[ConversationContext]) -> List[Any]:
        """Run LLM analysis on all contexts concurrently."""
        tasks = [self.audit_handler.execute(input_data=ctx) for ctx in contexts]
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _store_results(
        self,
        contexts: List[ConversationContext],
        audit_results: List[AuditResult],
        metrics: List[ConversationMetrics],
    ) -> None:
        """Persist results to database. Override in subclass for actual implementation."""
        pass

    def _transform_to_contexts(
        self, raw_data: List[Dict[str, Any]]
    ) -> List[ConversationContext]:
        """
        Transform raw JSON data to typed ConversationContext objects.

        Handles:
        - Thread ID and URL extraction
        - Message transformation with timestamp parsing
        - Human participant collection
        - Thread timestamp computation
        """
        contexts = []

        for thread_root in raw_data:
            # 1. Get Thread ID, Channel ID, and URL from root object
            thread_id = str(thread_root.get("id", "unknown_id"))
            channel_id = thread_root.get("channel_id") or thread_root.get("channelId")
            slack_url = thread_root.get("messageUrl", "")

            # 2. Get Messages
            raw_msgs = thread_root.get("threadReplies", [])
            if not raw_msgs:
                # Fallback if no replies structure
                raw_msgs = [thread_root]

            # 3. Extract Metadata from first message
            first_msg_content = raw_msgs[0].get("content", "")
            case_id = self._extract_case_id(thread_root)
            title = self._extract_business_name(first_msg_content)

            # 4. Build Message Objects with timestamp parsing
            schema_messages = []
            human_participants = set()
            timestamps = []

            for i, m in enumerate(raw_msgs):
                ts_str = str(m.get("ts", ""))
                timestamp_utc = parse_slack_timestamp(ts_str)

                if timestamp_utc:
                    timestamps.append(timestamp_utc)

                is_bot = m.get("is_bot", False) or "bot_id" in m
                user_id = m.get("user_id") or m.get("user")

                # Track human participants
                if not is_bot and user_id:
                    human_participants.add(user_id)

                schema_messages.append(
                    Message(
                        ts=ts_str,
                        timestamp_utc=timestamp_utc,
                        sender=m.get("sender", "unknown"),
                        user_id=user_id,
                        is_bot=is_bot,
                        content=m.get("content", ""),
                        messageUrl=m.get("messageUrl"),
                        reply_count=m.get("reply_count", 0),
                        is_thread_reply=i > 0,  # First message is thread root
                    )
                )

            # 5. Compute thread timestamps
            thread_created_at = min(timestamps) if timestamps else None
            thread_last_activity_at = max(timestamps) if timestamps else None

            # 6. Create Context
            ctx = ConversationContext(
                thread_id=thread_id,
                channel_id=channel_id,
                case_id=case_id,
                slack_url=slack_url,
                title=title,
                messages=schema_messages,
                thread_created_at=thread_created_at,
                thread_last_activity_at=thread_last_activity_at,
                human_participants=list(human_participants),
            )
            contexts.append(ctx)

        return contexts

    @staticmethod
    def _extract_case_id(data: Dict[str, Any]) -> str:
        """Extract the ID directly from the data payload."""
        return str(data.get("id", "unknown"))

    @staticmethod
    def _extract_business_name(text: str) -> str:
        """Heuristic to get business name from first message."""
        if text and "New underwriting case:" in text:
            try:
                return text.split("New underwriting case:")[1].split("\n")[0].strip()
            except (IndexError, AttributeError):
                pass
        return "Unknown Business"


def format_conversation_log(messages: List[Message]) -> str:
    """Format conversation history into a readable string for the CSV."""
    formatted = []
    for m in messages:
        role = "BOT" if m.is_bot else "HUMAN"
        content_preview = m.content[:100] + "..." if len(m.content) > 100 else m.content
        formatted.append(f"[{role}] {m.sender}: {content_preview}")
    return "\n".join(formatted)


async def run_pipeline(raw_data_list: List[Dict]) -> pd.DataFrame:
    """
    Legacy function for backwards compatibility.

    Runs the pipeline and returns a DataFrame with dashboard data.
    """
    pipeline = FeedbackAnalysisPipeline()
    result = await pipeline.process(raw_data_list)

    # Build dashboard rows from results
    dashboard_rows = []
    for ctx, res, metric in zip(result.contexts, result.audit_results, result.metrics):
        row = {
            "Thread_ID": ctx.thread_id,
            "Channel_ID": ctx.channel_id,
            "Case_ID": ctx.case_id,
            "Business": ctx.title,
            "Slack_URL": ctx.slack_url,
            "Has_Intervention": res.has_human_intervention,
            "Intervention_Type": res.intervention_category.value,
            "Escalation_Type": res.escalation_type.value
            if res.escalation_type
            else None,
            "Friction_Point": res.friction_point,
            "Sentiment": res.human_sentiment.value,
            "Human_Summary": res.summary_of_human_input,
            "Final_Status": res.final_status,
            # Computed metrics
            "Total_Messages": metric.total_messages,
            "Human_Messages": metric.human_messages,
            "Bot_Messages": metric.bot_messages,
            "Duration_Seconds": metric.duration_seconds,
            "Exchange_Count": metric.exchange_count,
            "Ping_Pong_Ratio": metric.ping_pong_ratio,
            "Is_Resolved": metric.is_resolved,
            "Resolution_Type": metric.resolution_type,
            "Time_To_Resolution_Seconds": metric.time_to_resolution_seconds,
            "Is_Stalemate": metric.is_stalemate,
            "Has_Frustrated_Message": metric.has_frustrated_message,
            # Conversation preview
            "Full_Conversation": "\n".join(
                [
                    f"{'[BOT]' if m.is_bot else '[HUMAN]'} {m.sender}: {m.content}"
                    for m in ctx.messages
                ]
            ),
        }
        dashboard_rows.append(row)

    return pd.DataFrame(dashboard_rows)
