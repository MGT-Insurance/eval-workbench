from __future__ import annotations

import re
from typing import Dict

from shared.langfuse.trace import PromptPatternsBase, create_extraction_pattern


class WorkflowPromptPatterns(PromptPatternsBase):
    @staticmethod
    def _patterns_recommendation() -> Dict[str, str]:
        h_case = "CASE ASSESSMENT (from previous analysis)"
        h_context = "FULL CONTEXT DATA"
        h_flags = "UNDERWRITING FLAGS (Current Human Rules)"
        h_debug = "SWALLOW DEBUG DATA (Underwriting Rule Logic)"
        h_prev = "PREVIOUS RECOMMENDATIONS (Historical Context)"
        return {
            "caseAssessment": create_extraction_pattern(h_case, re.escape(h_context)),
            "contextData": create_extraction_pattern(h_context, re.escape(h_flags)),
            "underwritingFlags": create_extraction_pattern(h_flags, re.escape(h_debug)),
            "swallowDebugData": create_extraction_pattern(h_debug, f"{re.escape(h_prev)}|$"),
        }


class ChatPromptPatterns(PromptPatternsBase):
    @staticmethod
    def _patterns_chat() -> Dict[str, str]:
        # Chat headers include the colon inside the bold text.
        h_history = "**Conversation History:**"
        h_quote = "**Quote Context:**"
        h_message = "**Current User Message:**"
        h_context = "**Context:**"
        h_instructions = "**Instructions:**"

        return {
            "conversationHistory": rf"{re.escape(h_history)}\s*(.*?)\s*(?:{re.escape(h_quote)})",
            "quoteContext": rf"{re.escape(h_quote)}\s*(.*?)\s*(?:{re.escape(h_message)})",
            "currentUserMessage": rf"{re.escape(h_message)}\s*(.*?)\s*(?:{re.escape(h_context)})",
            "context": rf"{re.escape(h_context)}\s*(.*?)\s*(?:{re.escape(h_instructions)})",
            "instructions": rf"{re.escape(h_instructions)}\s*(.*?)\s*(?:$)",
        }


