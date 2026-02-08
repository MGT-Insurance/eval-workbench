from pydantic import Field

from axion._core.schema import RichBaseModel


class AnalyzerConfig(RichBaseModel):
    """
    Configurable domain context - decouples hardcoded 'Athena' references.

    This allows the analyzer to be used with different bot names and domain contexts
    without modifying the prompts directly.
    """

    bot_name: str = Field(
        default='Athena',
        description='Name of the AI assistant/bot in conversations',
    )
    domain_context: str = Field(
        default='insurance underwriting',
        description='Domain context for the conversations being analyzed',
    )


class TruncationConfig(RichBaseModel):
    """
    Head-Tail-Target truncation settings for long conversations.

    This strategy preserves:
    - First N turns (context establishment)
    - Last N turns (resolution)
    - Turns around a key event (e.g., recommendation)
    """

    max_tokens: int = Field(
        default=8000,
        description='Maximum tokens allowed in the transcript',
    )
    head_turns: int = Field(
        default=5,
        description='Number of turns to keep from the beginning of conversation',
    )
    tail_turns: int = Field(
        default=5,
        description='Number of turns to keep from the end of conversation',
    )
    target_context: int = Field(
        default=3,
        description='Number of turns to keep around a target event (e.g., recommendation)',
    )


__all__ = ['AnalyzerConfig', 'TruncationConfig']
