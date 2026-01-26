from typing import Tuple, List
from axion.handlers import LLMHandler

from implementations.athena.models.feedback_analysis.schema import (
    ConversationContext,
    AuditResult,
    Message,
)


class UnderwritingAuditHandler(LLMHandler[ConversationContext, AuditResult]):
    """
    Specialized handler to audit AI-Human underwriting conversations.
    """

    input_model = ConversationContext
    output_model = AuditResult
    as_structured_llm = False

    instruction = """
    You are a Senior Underwriting Auditor. Analyze the provided Slack conversation log.

    Context:
    - 'Athena' is the AI Underwriter.
    - Messages with 'is_bot=False' are Human Underwriters or Agents.

    Your Task:
    1. **Check for Humans**: Did a human post a message? If NO, set 'has_human_intervention' to False.
    2. **Categorize Intervention**: Why did the human speak? (Fixing data? Reporting a bug? Just approving?)
    3. **Extract Friction**: What specific underwriting rule or data point caused discussion?
    4. **Status**: What is the final state of the quote?
    """

    # Updated Examples with new Schema fields (thread_id, etc)
    examples: List[Tuple[ConversationContext, AuditResult]] = [
        (
            ConversationContext(
                thread_id="123456789",
                case_id="MGT-101",
                title="Joe's Roofers",
                slack_url="http://slack.com/archives/123",
                messages=[
                    Message(
                        ts="1",
                        sender="Athena",
                        is_bot=True,
                        content="Status: Blocked (RoofAge)",
                    ),
                    Message(
                        ts="2",
                        sender="U_HUMAN",
                        is_bot=False,
                        content="This is wrong. Override this.",
                    ),
                ],
            ),
            AuditResult(
                has_human_intervention=True,
                intervention_category="correction_factual",
                summary_of_human_input="Human overrode the roof age block.",
                friction_point="Roof Age Data",
                human_sentiment="neutral",
                final_status="Approved",
            ),
        )
    ]

    def __init__(self, llm, **kwargs):
        super().__init__(llm=llm, **kwargs)
