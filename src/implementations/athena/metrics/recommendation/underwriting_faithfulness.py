import asyncio
import re
from typing import Any, Dict, List, Literal

from axion._core.asyncio import SemaphoreExecutor
from axion._core.logging import get_logger
from axion._core.schema import RichBaseModel
from axion._core.tracing import trace
from axion.dataset import DatasetItem
from axion.metrics.base import BaseMetric, MetricEvaluationResult, metric
from axion.metrics.schema import SignalDescriptor
from pydantic import Field

from shared.metrics.flattening import flatten_json_lines as shared_flatten_json

logger = get_logger(__name__)


STOPWORDS = {
    'a',
    'an',
    'and',
    'are',
    'as',
    'at',
    'be',
    'by',
    'for',
    'from',
    'has',
    'he',
    'in',
    'is',
    'it',
    'its',
    'of',
    'on',
    'that',
    'the',
    'to',
    'was',
    'were',
    'will',
    'with',
    'this',
    'but',
    'they',
    'have',
    'had',
    'what',
    'when',
    'where',
    'who',
    'which',
    'why',
    'how',
    'been',
    'since',
}


def flatten_json(y: Any, parent_key: str = '', sep: str = '.') -> List[str]:
    """Recursively flattens a nested dictionary/list into a list of 'path: value' strings."""
    return shared_flatten_json(y, parent_key=parent_key, sep=sep)


def find_relevant_context(
    claim: str, flattened_context: List[str], top_k: int = 5
) -> List[str]:
    """Finds the most relevant data points for a claim using token overlap (ignoring stopwords)."""
    # Regex [^\W_]+ matches alphanumeric characters but excludes underscores,
    # effectively splitting snake_case into separate tokens (e.g., 'bop_protection_class' -> 'bop', 'protection', 'class')
    token_pattern = r'[^\W_]+'

    # Tokenize and remove stopwords
    claim_token_list = [
        t for t in re.findall(token_pattern, claim.lower()) if t not in STOPWORDS
    ]
    claim_tokens = set(claim_token_list)
    if not claim_tokens:
        # If claim is only stopwords, revert to full token set
        claim_token_list = re.findall(token_pattern, claim.lower())
        claim_tokens = set(claim_token_list)

    # Normalize numeric tokens to handle commas (e.g., "200,000" -> "200000")
    claim_numbers = {num.replace(',', '') for num in re.findall(r'\d[\d,]*', claim)}

    scored_lines = []
    preferred_lines = []
    anchor_tokens = [t for t in claim_token_list if len(t) >= 3 and not t.isdigit()]
    best_line_for_token: Dict[str, tuple[int, str]] = {}

    for line in flattened_context:
        line_tokens = set(re.findall(token_pattern, line.lower()))
        # Simple Jaccard-like intersection scoring
        if not line_tokens:
            continue

        # Check overlap against meaningful tokens
        score = len(claim_tokens.intersection(line_tokens))

        # Add numeric matches (normalized) as a stronger signal
        if claim_numbers:
            line_numbers = {
                num.replace(',', '') for num in re.findall(r'\d[\d,]*', line)
            }
            numeric_matches = len(claim_numbers.intersection(line_numbers))
            if numeric_matches:
                score += numeric_matches * 2

        if score > 0:
            scored_lines.append((score, line))
            for token in anchor_tokens:
                if token in line_tokens:
                    current = best_line_for_token.get(token)
                    if current is None or score > current[0]:
                        best_line_for_token[token] = (score, line)

    # Sort by score desc
    scored_lines.sort(key=lambda x: x[0], reverse=True)
    ranked_lines = [line for _, line in scored_lines]
    for token in anchor_tokens:
        if token in best_line_for_token:
            line = best_line_for_token[token][1]
            if line not in preferred_lines:
                preferred_lines.append(line)

    combined = preferred_lines + [
        line for line in ranked_lines if line not in preferred_lines
    ]
    if len(preferred_lines) >= top_k:
        return preferred_lines[:top_k]
    return combined[:top_k]


class ClaimExtractionInput(RichBaseModel):
    text: str = Field(..., description='The underwriting recommendation text.')
    model_config = {'extra': 'forbid'}


class ClaimExtractionOutput(RichBaseModel):
    claims: List[str] = Field(
        ..., description='List of atomic factual claims extracted from the text.'
    )
    model_config = {'extra': 'forbid'}


class ClaimExtractor(BaseMetric[ClaimExtractionInput, ClaimExtractionOutput]):
    instruction = """
    Extract all verifiable factual claims from the text.
    Focus on: dollar amounts, dates, classifications, counts, and locations.
    Ignore subjective opinions. Ignore values that are computed from the AI.
    """
    input_model = ClaimExtractionInput
    output_model = ClaimExtractionOutput


class VerificationInput(RichBaseModel):
    claim: str = Field(..., description='The claim to verify.')
    evidence: str = Field(..., description='Candidate facts from the source data.')
    model_config = {'extra': 'forbid'}


class VerificationOutput(RichBaseModel):
    is_supported: bool = Field(
        ..., description='True if the evidence supports the claim.'
    )
    reason: str = Field(..., description='Explanation of the verdict.')
    model_config = {'extra': 'forbid'}


class FactVerifier(BaseMetric[VerificationInput, VerificationOutput]):
    instruction = """
    You are a strict data auditor. Determine if the 'Claim' is supported by the provided 'Evidence'.
    1. If Evidence explicitly matches the Claim (allowing for format diffs), set is_supported=True.
    2. If Evidence contradicts or is missing, set is_supported=False.
    3. If the claim asserts a discrepancy between two values, mark supported as long as all
       numeric values mentioned in the claim appear somewhere in the Evidence, even if the
       Evidence doesn't explicitly tie them to the same element.

    MAGIC DUST NOTES:
    - Magic Dust (magicDustByElement, magicDustData, *_magicdust) is imputed data.
    - Quote fields are customer-provided and are the source of truth.
    - Discrepancies may be noted for risk context, but do not mark a claim unsupported
      solely because Magic Dust differs from customer input.
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

    @trace(name='HeuristicFactVerifier', capture_args=True, capture_response=True)
    async def execute(self, input_data: VerificationInput) -> VerificationOutput:
        """Calculates weighted recall of the claim against the evidence."""
        # Use regex that splits on underscores to handle snake_case keys
        token_pattern = r'[^\W_]+'
        number_pattern = r'\d[\d,]*'

        # Clean and tokenize claim
        raw_claim_tokens = set(re.findall(token_pattern, input_data.claim.lower()))

        # Identify numbers: handle comma-separated values (e.g., 5,167)
        numeric_tokens = {
            n.replace(',', '') for n in re.findall(number_pattern, input_data.claim)
        }
        comma_numbers = re.findall(r'\d{1,3}(?:,\d{3})+', input_data.claim)
        comma_parts = {part for num in comma_numbers for part in num.split(',')}
        # Also include alphanumeric codes (e.g., "P2", "1st", "A1")
        alnum_tokens = {
            t
            for t in raw_claim_tokens
            if any(c.isdigit() for c in t) and (not t.isdigit() or t not in comma_parts)
        }
        numeric_tokens = numeric_tokens.union(alnum_tokens)

        # Identify text (removing numbers and stopwords)
        text_tokens = (raw_claim_tokens - numeric_tokens) - STOPWORDS

        # Fallback: if empty text tokens but no numbers, use raw (rare case)
        if not text_tokens and not numeric_tokens:
            text_tokens = raw_claim_tokens

        # Tokenize the ENTIRE evidence block (Union approach)
        evidence_tokens = set(re.findall(token_pattern, input_data.evidence.lower()))
        evidence_numbers = {
            n.replace(',', '') for n in re.findall(number_pattern, input_data.evidence)
        }
        evidence_tokens = evidence_tokens.union(evidence_numbers)

        # Calculate Weighted Score
        # Strategy: If numbers exist, they account for 70% of the score. Text is 30%.
        # If no numbers, Text is 100%.

        score = 0.0
        reason_parts = []

        # Numeric Scoring
        if numeric_tokens:
            num_found = numeric_tokens.intersection(evidence_tokens)
            num_recall = len(num_found) / len(numeric_tokens)

            reason_parts.append(
                f'Numbers: {len(num_found)}/{len(numeric_tokens)} match'
            )

            if text_tokens:
                text_found = text_tokens.intersection(evidence_tokens)
                text_recall = len(text_found) / len(text_tokens)
                reason_parts.append(f'Text: {len(text_found)}/{len(text_tokens)} match')

                # Weighted Average (Heuristic tuning: Numbers are more important)
                score = (0.7 * num_recall) + (0.3 * text_recall)
            else:
                score = num_recall
        else:
            # Text Only Scoring
            if text_tokens:
                text_found = text_tokens.intersection(evidence_tokens)
                score = len(text_found) / len(text_tokens)
                reason_parts.append(f'Text: {len(text_found)}/{len(text_tokens)} match')
            else:
                score = 0.0

        is_supported = score >= self.threshold
        status_text = 'Supported' if is_supported else 'Unsupported'

        # Detailed reasoning for debugging
        missing_numerics = numeric_tokens - evidence_tokens
        missing_text = text_tokens - evidence_tokens
        missing_info = []
        if missing_numerics:
            missing_info.append(f'Missing nums: {list(missing_numerics)}')
        if missing_text and not is_supported:
            # Only show missing text if it failed, to avoid noise
            missing_info.append(f'Missing words: {list(missing_text)[:3]}...')

        reason_str = f'Heuristic {status_text} (Score: {score:.0%}). ' + '; '.join(
            reason_parts
        )
        if missing_info:
            reason_str += '. ' + ' '.join(missing_info)

        return VerificationOutput(is_supported=is_supported, reason=reason_str)


class UnderwritingFaithfulnessResult(RichBaseModel):
    overall_score: float
    total_claims: int
    supported_claims: int
    hallucinations: int
    claim_details: List[Dict[str, Any]]
    unverified_claims: List[str]
    model_config = {'extra': 'forbid'}


@metric(
    name='Underwriting Faithfulness',
    key='uw_faithfulness',
    description='Checks if extracted claims in the recommendation exist in the source JSON.',
    required_fields=[],
    default_threshold=0.9,
    tags=['athena'],
)
class UnderwritingFaithfulness(BaseMetric):
    def __init__(
        self,
        verification_mode: Literal['llm', 'heuristic', 'heuristic_then_llm'] = 'llm',
        max_claims: int = 10,
        max_concurrent: int = 5,
        **kwargs,
    ):
        """
        Args:
            verification_mode: 'llm' for AI-based verification (slower, more accurate),
                               'heuristic' for word-overlap verification (faster, cheaper),
                               'heuristic_then_llm' to fallback to LLM on failures.
            max_claims: Maximum number of claims to verify.
            max_concurrent: Maximum number of concurrent verification tasks.
            **kwargs: Additional arguments passed to BaseMetric/Components.
        """
        super().__init__(**kwargs)
        if verification_mode == 'heuristic':
            self.tags = ['athena', 'heuristic']
        self.extractor = ClaimExtractor(**kwargs)
        self.semaphore_runner = SemaphoreExecutor(max_concurrent=max_concurrent)
        self.max_claims = max_claims
        self.verification_mode = verification_mode
        # Select the verifier based on the mode
        if verification_mode == 'heuristic':
            self.verifier = HeuristicFactVerifier(**kwargs)
        elif verification_mode == 'heuristic_then_llm':
            self.heuristic_verifier = HeuristicFactVerifier(**kwargs)
            self.llm_verifier = FactVerifier(**kwargs)
        else:
            self.verifier = FactVerifier(**kwargs)

    @trace(name='UnderwritingFaithfulness', capture_args=True, capture_response=True)
    async def execute(self, item: DatasetItem, **kwargs) -> MetricEvaluationResult:
        # Select Text Source
        text = ''
        if item.additional_output:
            text = (
                item.additional_output.get('detailed_recommendation')
                or item.additional_output.get('brief_recommendation')
                or ''
            )
        if not text:
            text = item.actual_output or ''

        # Flatten Source of Truth
        flattened_facts_list = flatten_json(item.additional_input or {})

        if not flattened_facts_list:
            return MetricEvaluationResult(
                score=0.0, explanation='No source data found in additional_input.'
            )

        # Run Extraction
        extract_res = await self.extractor.execute(ClaimExtractionInput(text=text))
        claims = extract_res.claims

        if not claims:
            return MetricEvaluationResult(
                score=1.0, explanation='No verifiable claims found in text.'
            )

        # Run Verification (ASYNC / PARALLEL with THROTTLING)
        verification_tasks = []
        verify_inputs = []
        claims_to_verify = claims[: self.max_claims]
        for claim in claims_to_verify:
            # Filter evidence for this specific claim
            relevant_lines = find_relevant_context(
                claim, flattened_facts_list, top_k=15
            )
            evidence_text = '\n'.join(relevant_lines)

            verify_input = VerificationInput(claim=claim, evidence=evidence_text)
            verify_inputs.append(verify_input)

            if self.verification_mode != 'heuristic_then_llm':
                # Use the semaphore executor to wrap the verification call
                task = self.semaphore_runner.run(self.verifier.execute, verify_input)
                verification_tasks.append(task)

        failed_indices: list[int] = []
        if self.verification_mode == 'heuristic_then_llm':
            heuristic_tasks = [
                self.semaphore_runner.run(self.heuristic_verifier.execute, verify_input)
                for verify_input in verify_inputs
            ]
            heuristic_results = await asyncio.gather(*heuristic_tasks)
            failed_indices = [
                idx for idx, res in enumerate(heuristic_results) if not res.is_supported
            ]
            verification_results = list(heuristic_results)
            decision_sources = ['heuristic'] * len(verification_results)
            if failed_indices:
                llm_tasks = [
                    self.semaphore_runner.run(
                        self.llm_verifier.execute, verify_inputs[idx]
                    )
                    for idx in failed_indices
                ]
                llm_results = await asyncio.gather(*llm_tasks)
                for idx, llm_res in zip(failed_indices, llm_results):
                    verification_results[idx] = llm_res
                    decision_sources[idx] = 'llm'
        else:
            # Run all verifications at once (controlled by semaphore)
            verification_results = await asyncio.gather(*verification_tasks)
            decision_sources = [
                'heuristic' if self.verification_mode == 'heuristic' else 'llm'
            ] * len(verification_results)

        # Compile Results
        results = []
        supported_count = 0

        for i, claim in enumerate(claims_to_verify):
            verify_res = verification_results[i]

            is_pass = verify_res.is_supported
            reason = verify_res.reason

            if is_pass:
                supported_count += 1

            results.append(
                {
                    'claim': claim,
                    'status': '✅ Supported'
                    if is_pass
                    else '❌ Hallucinated/Unsupported',
                    'reason': reason,
                    'decision_source': decision_sources[i],
                }
            )

        score = supported_count / len(claims_to_verify) if claims_to_verify else 1.0
        explanation = f'Faithfulness Score: {score:.2f}. Verified {supported_count}/{len(claims_to_verify)} claims.'

        # Compute cost estimate based on verification mode
        used_sub_metrics: List[Any] = [self.extractor]
        if self.verification_mode == 'llm':
            used_sub_metrics.append(self.verifier)
        elif self.verification_mode == 'heuristic_then_llm':
            # Only add LLM verifier if it was actually used (had failures)
            if failed_indices:
                used_sub_metrics.append(self.llm_verifier)
        # Note: heuristic mode uses HeuristicFactVerifier which has no LLM cost

        self.compute_cost_estimate(used_sub_metrics)

        return MetricEvaluationResult(
            score=score,
            explanation=explanation,
            signals=UnderwritingFaithfulnessResult(
                overall_score=score,
                total_claims=len(claims_to_verify),
                supported_claims=supported_count,
                hallucinations=len(claims_to_verify) - supported_count,
                claim_details=results,
                unverified_claims=[
                    claim for claim in claims if claim not in claims_to_verify
                ],
            ),
        )

    def get_signals(
        self, result: UnderwritingFaithfulnessResult
    ) -> List[SignalDescriptor]:
        signals = [
            SignalDescriptor(
                name='hallucination_count',
                description='Number of unsupported claims.',
                extractor=lambda r: r.hallucinations,
                headline_display=True,
            )
        ]

        for i, det in enumerate(result.claim_details):
            group_name = f'Claim {i + 1}'
            signals.extend(
                [
                    SignalDescriptor(
                        name='claim_text',
                        group=group_name,
                        description='Extracted Claim',
                        extractor=lambda r, idx=i: r.claim_details[idx]['claim'],
                    ),
                    SignalDescriptor(
                        name='status',
                        group=group_name,
                        description='Verdict',
                        extractor=lambda r, idx=i: r.claim_details[idx]['status'],
                    ),
                    SignalDescriptor(
                        name='reason',
                        group=group_name,
                        description='Judge Reasoning',
                        extractor=lambda r, idx=i: r.claim_details[idx]['reason'],
                    ),
                    SignalDescriptor(
                        name='decision_source',
                        group=group_name,
                        description='Verifier used for the decision',
                        extractor=lambda r, idx=i: r.claim_details[idx][
                            'decision_source'
                        ],
                    ),
                ]
            )

        return signals
