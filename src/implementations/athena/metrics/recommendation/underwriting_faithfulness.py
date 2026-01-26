import re
import asyncio
from typing import Any, Dict, List, Literal
from pydantic import Field

from axion._core.logging import get_logger
from axion._core.schema import RichBaseModel
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from axion._core.asyncio import SemaphoreExecutor

logger = get_logger(__name__)


STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
    'will', 'with', 'this', 'but', 'they', 'have', 'had', 'what', 'when',
    'where', 'who', 'which', 'why', 'how', 'been', 'since'
}

def flatten_json(y: Any, parent_key: str = '', sep: str = '.') -> List[str]:
    """Recursively flattens a nested dictionary/list into a list of 'path: value' strings."""
    out = []

    def _flatten(obj: Any, name: str):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(v, f"{name}{sep}{k}" if name else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _flatten(v, f"{name}[{i}]")
        else:
            out.append(f"{name}: {str(obj)}")

    _flatten(y, parent_key)
    return out

def find_relevant_context(claim: str, flattened_context: List[str], top_k: int = 5) -> List[str]:
    """Finds the most relevant data points for a claim using token overlap (ignoring stopwords)."""
    # Regex [^\W_]+ matches alphanumeric characters but excludes underscores,
    # effectively splitting snake_case into separate tokens (e.g., 'bop_protection_class' -> 'bop', 'protection', 'class')
    token_pattern = r'[^\W_]+'

    # Tokenize and remove stopwords
    claim_tokens = set(re.findall(token_pattern, claim.lower())) - STOPWORDS
    if not claim_tokens:
        # If claim is only stopwords, revert to full token set
        claim_tokens = set(re.findall(token_pattern, claim.lower()))

    scored_lines = []
    for line in flattened_context:
        line_tokens = set(re.findall(token_pattern, line.lower()))
        # Simple Jaccard-like intersection scoring
        if not line_tokens:
            continue

        # Check overlap against meaningful tokens
        score = len(claim_tokens.intersection(line_tokens))
        if score > 0:
            scored_lines.append((score, line))

    # Sort by score desc
    scored_lines.sort(key=lambda x: x[0], reverse=True)
    return [line for _, line in scored_lines[:top_k]]

# --- Components: LLM Models ---

class ClaimExtractionInput(RichBaseModel):
    text: str = Field(..., description="The underwriting recommendation text.")
    model_config = {'extra': 'forbid'}

class ClaimExtractionOutput(RichBaseModel):
    claims: List[str] = Field(..., description="List of atomic factual claims extracted from the text.")
    model_config = {'extra': 'forbid'}

class ClaimExtractor(BaseMetric[ClaimExtractionInput, ClaimExtractionOutput]):
    instruction = """
    Extract all verifiable factual claims from the text.
    Focus on: dollar amounts, dates, classifications, counts, and locations.
    Ignore subjective opinions.
    """
    input_model = ClaimExtractionInput
    output_model = ClaimExtractionOutput

class VerificationInput(RichBaseModel):
    claim: str = Field(..., description="The claim to verify.")
    evidence: str = Field(..., description="Candidate facts from the source data.")
    model_config = {'extra': 'forbid'}

class VerificationOutput(RichBaseModel):
    is_supported: bool = Field(..., description="True if the evidence supports the claim.")
    reason: str = Field(..., description="Explanation of the verdict.")
    model_config = {'extra': 'forbid'}

class FactVerifier(BaseMetric[VerificationInput, VerificationOutput]):
    instruction = """
    You are a strict data auditor. Determine if the 'Claim' is supported by the provided 'Evidence'.
    1. If Evidence explicitly matches the Claim (allowing for format diffs), set is_supported=True.
    2. If Evidence contradicts or is missing, set is_supported=False.
    """
    input_model = VerificationInput
    output_model = VerificationOutput


class HeuristicFactVerifier:
    """
    A traditional NLP verifier using token overlap/recall.
    Does not use an LLM, making it faster and free, but less semantically aware.
    Prioritizes numerical matches heavily.
    """
    def __init__(self, threshold: float = 0.75, **kwargs):
        self.threshold = threshold

    async def execute(self, input_data: VerificationInput) -> VerificationOutput:
        """Calculates weighted recall of the claim against the evidence."""
        # Use regex that splits on underscores to handle snake_case keys
        token_pattern = r'[^\W_]+'

        # 1. Clean and tokenize claim
        raw_claim_tokens = set(re.findall(token_pattern, input_data.claim.lower()))

        # Identify numbers: Digits OR Alphanumeric codes (e.g., "P2", "1st", "A1")
        # any(c.isdigit()) ensures "P2" is treated as a number, not text.
        numeric_tokens = {t for t in raw_claim_tokens if any(c.isdigit() for c in t)}

        # Identify text (removing numbers and stopwords)
        text_tokens = (raw_claim_tokens - numeric_tokens) - STOPWORDS

        # Fallback: if empty text tokens but no numbers, use raw (rare case)
        if not text_tokens and not numeric_tokens:
            text_tokens = raw_claim_tokens

        # 2. Tokenize the ENTIRE evidence block (Union approach)
        evidence_tokens = set(re.findall(token_pattern, input_data.evidence.lower()))

        # 3. Calculate Weighted Score
        # Strategy: If numbers exist, they account for 70% of the score. Text is 30%.
        # If no numbers, Text is 100%.

        score = 0.0
        reason_parts = []

        # Numeric Scoring
        if numeric_tokens:
            num_found = numeric_tokens.intersection(evidence_tokens)
            num_recall = len(num_found) / len(numeric_tokens)

            reason_parts.append(f"Numbers: {len(num_found)}/{len(numeric_tokens)} match")

            if text_tokens:
                text_found = text_tokens.intersection(evidence_tokens)
                text_recall = len(text_found) / len(text_tokens)
                reason_parts.append(f"Text: {len(text_found)}/{len(text_tokens)} match")

                # Weighted Average (Heuristic tuning: Numbers are more important)
                score = (0.7 * num_recall) + (0.3 * text_recall)
            else:
                score = num_recall
        else:
            # Text Only Scoring
            if text_tokens:
                text_found = text_tokens.intersection(evidence_tokens)
                score = len(text_found) / len(text_tokens)
                reason_parts.append(f"Text: {len(text_found)}/{len(text_tokens)} match")
            else:
                score = 0.0

        is_supported = score >= self.threshold
        status_text = "Supported" if is_supported else "Unsupported"

        # Detailed reasoning for debugging
        missing_numerics = numeric_tokens - evidence_tokens
        missing_text = text_tokens - evidence_tokens
        missing_info = []
        if missing_numerics:
            missing_info.append(f"Missing nums: {list(missing_numerics)}")
        if missing_text and not is_supported:
            # Only show missing text if it failed, to avoid noise
            missing_info.append(f"Missing words: {list(missing_text)[:3]}...")

        reason_str = f"Heuristic {status_text} (Score: {score:.0%}). " + "; ".join(reason_parts)
        if missing_info:
            reason_str += ". " + " ".join(missing_info)

        return VerificationOutput(is_supported=is_supported, reason=reason_str)



class UnderwritingFaithfulnessResult(RichBaseModel):
    overall_score: float
    total_claims: int
    supported_claims: int
    hallucinations: int
    claim_details: List[Dict[str, Any]]
    model_config = {'extra': 'forbid'}

@metric(
    name='Underwriting Faithfulness',
    key='underwriting_faithfulness',
    description='Checks if extracted claims in the recommendation exist in the source JSON.',
    required_fields=[],
    default_threshold=0.9,
    tags=['heuristic'],
)
class UnderwritingFaithfulness(BaseMetric):
    def __init__(self, verification_mode: Literal['llm', 'heuristic'] = 'llm', max_concurrent: int = 5, **kwargs):
        """
        Args:
            verification_mode: 'llm' for AI-based verification (slower, more accurate),
                               'heuristic' for word-overlap verification (faster, cheaper).
            max_concurrent: Maximum number of concurrent verification tasks.
            **kwargs: Additional arguments passed to BaseMetric/Components.
        """
        super().__init__(**kwargs)
        self.extractor = ClaimExtractor(**kwargs)
        self.semaphore_runner = SemaphoreExecutor(max_concurrent=max_concurrent)

        # Select the verifier based on the mode
        if verification_mode == 'heuristic':
            self.verifier = HeuristicFactVerifier(**kwargs)
        else:
            self.verifier = FactVerifier(**kwargs)

    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        # 1. Select Text Source
        text = ""
        if item.additional_output:
            text = item.additional_output.get('detailed_recommendation') or \
                   item.additional_output.get('brief_recommendation') or ""
        if not text:
            text = item.actual_output or ""

        # 2. Flatten Source of Truth
        flattened_facts_list = flatten_json(item.additional_input or {})

        if not flattened_facts_list:
             return MetricEvaluationResult(score=0.0, explanation="No source data found in additional_input.")

        # 3. Run Extraction
        extract_res = await self.extractor.execute(ClaimExtractionInput(text=text))
        claims = extract_res.claims

        if not claims:
            return MetricEvaluationResult(
                score=1.0,
                explanation="No verifiable claims found in text."
            )

        # 4. Run Verification (ASYNC / PARALLEL with THROTTLING)
        verification_tasks = []

        for claim in claims:
            # Filter evidence for this specific claim
            relevant_lines = find_relevant_context(claim, flattened_facts_list, top_k=15)
            evidence_text = "\n".join(relevant_lines)

            verify_input = VerificationInput(claim=claim, evidence=evidence_text)

            # Use the semaphore executor to wrap the verification call
            task = self.semaphore_runner.run(self.verifier.execute, verify_input)
            verification_tasks.append(task)

        # Run all verifications at once (controlled by semaphore)
        verification_results = await asyncio.gather(*verification_tasks)

        # 5. Compile Results
        results = []
        supported_count = 0

        for i, claim in enumerate(claims):
            verify_res = verification_results[i]

            is_pass = verify_res.is_supported
            reason = verify_res.reason

            if is_pass:
                supported_count += 1

            results.append({
                "claim": claim,
                "status": "✅ Supported" if is_pass else "❌ Hallucinated/Unsupported",
                "reason": reason
            })

        score = supported_count / len(claims) if claims else 1.0
        explanation = f"Faithfulness Score: {score:.2f}. Verified {supported_count}/{len(claims)} claims."

        return MetricEvaluationResult(
            score=score,
            explanation=explanation,
            signals=UnderwritingFaithfulnessResult(
                overall_score=score,
                total_claims=len(claims),
                supported_claims=supported_count,
                hallucinations=len(claims) - supported_count,
                claim_details=results
            )
        )

    def get_signals(self, result: UnderwritingFaithfulnessResult) -> List[SignalDescriptor]:
        signals = [
            SignalDescriptor(
                name="hallucination_count",
                description="Number of unsupported claims.",
                extractor=lambda r: r.hallucinations,
                headline_display=True
            )
        ]

        for i, det in enumerate(result.claim_details):
            group_name = f"Claim {i+1}"
            signals.extend([
                SignalDescriptor(
                    name="claim_text",
                    group=group_name,
                    description="Extracted Claim",
                    extractor=lambda r, idx=i: r.claim_details[idx]['claim']
                ),
                SignalDescriptor(
                    name="status",
                    group=group_name,
                    description="Verdict",
                    extractor=lambda r, idx=i: r.claim_details[idx]['status']
                ),
                SignalDescriptor(
                    name="reason",
                    group=group_name,
                    description="Judge Reasoning",
                    extractor=lambda r, idx=i: r.claim_details[idx]['reason']
                )
            ])

        return signals
