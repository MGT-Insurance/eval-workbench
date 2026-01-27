import re
from typing import List
from pydantic import Field
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from axion._core.schema import RichBaseModel
from axion._core.tracing import trace


class FormattingIssue(RichBaseModel):
    issue_type: str = Field(
        ...,
        description="The type of formatting error (e.g., 'Forbidden Double Asterisk').",
    )
    context: str = Field(..., description="Snippet of text where the issue occurred.")
    count: int = Field(..., description="Number of occurrences.")


class FormattingResult(RichBaseModel):
    score: float = Field(...)
    issues: List[FormattingIssue] = Field(...)


@metric(
    name="Slack Formatting Compliance",
    key="slack_compliance",
    description="Ensures output adheres to strict Slack mrkdwn rules (no # headers, no **bold**, uses backticks).",
    required_fields=["actual_output"],
    default_threshold=1.0,
    tags=["compliance", "programmatic"],
)
class SlackFormattingCompliance(BaseMetric):
    # Double asterisks are valid in MD but broken in Slack (which uses single *)
    DOUBLE_ASTERISK_PATTERN = re.compile(r"\*\*[^*]+\*\*")

    # Headers (# Header) break in Slack
    HEADER_PATTERN = re.compile(r"^#+\s", re.MULTILINE)

    # Heuristic: Money/Percent should often be in backticks.
    # Finds $500 or 50% NOT inside backticks.
    # Logic: Look for $ or % that is NOT preceded by `
    UNWRAPPED_DATA_PATTERN = re.compile(r"(?<!`)(\$[\d,.]+[KM]?|\d+%)(?!`)")

    @trace(name='SlackFormattingCompliance', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        text = item.actual_output
        issues = []

        # Check 1: Double Asterisks
        double_bolds = self.DOUBLE_ASTERISK_PATTERN.findall(text)
        if double_bolds:
            issues.append(
                FormattingIssue(
                    issue_type="Forbidden Double Asterisk (**)",
                    context=f"Found {len(double_bolds)} instances (e.g. '{double_bolds[0]}')",
                    count=len(double_bolds),
                )
            )

        # Check 2: Headers
        headers = self.HEADER_PATTERN.findall(text)
        if headers:
            issues.append(
                FormattingIssue(
                    issue_type="Forbidden Header (#)",
                    context=f"Found {len(headers)} instances.",
                    count=len(headers),
                )
            )

        # Check 3: Unwrapped Data (Soft Check - maybe penalize less or just warn)
        unwrapped = self.UNWRAPPED_DATA_PATTERN.findall(text)
        # Filter out common false positives if necessary
        if unwrapped:
            issues.append(
                FormattingIssue(
                    issue_type="Unwrapped Data Value",
                    context=f"Values like '{unwrapped[0]}' should likely be in backticks.",
                    count=len(unwrapped),
                )
            )

        # Scoring: Strict (1.0 = perfect). Deduct 0.25 per issue type found.
        # You can adjust weighting.
        score = max(0.0, 1.0 - (len(issues) * 0.25))

        explanation = (
            "Formatting: Perfect."
            if score == 1.0
            else f"Formatting Issues Found: {[i.issue_type for i in issues]}"
        )

        return MetricEvaluationResult(
            score=score,
            explanation=explanation,
            signals=FormattingResult(score=score, issues=issues),
        )

    def get_signals(self, result: FormattingResult) -> List[SignalDescriptor]:
        signals = []
        signals.append(
            SignalDescriptor(
                name="formatting_score",
                description="Score",
                extractor=lambda r: r.score,
                headline_display=True,
            )
        )
        for i, issue in enumerate(result.issues):
            signals.extend(
                [
                    SignalDescriptor(
                        name="issue_type",
                        group=f"❌ Issue {i + 1}",
                        description="Type",
                        extractor=lambda r, idx=i: r.issues[idx].issue_type,
                    ),
                    SignalDescriptor(
                        name="context",
                        group=f"❌ Issue {i + 1}",
                        description="Context",
                        extractor=lambda r, idx=i: r.issues[idx].context,
                    ),
                ]
            )
        return signals
