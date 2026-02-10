# Recommendation Metrics

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Athena underwriting recommendation evaluation</strong><br>
<span class="badge" style="margin-top: 0.5rem;">7 Metrics</span>
<span class="badge" style="background: #667eea;">Underwriting</span>
</div>

This module provides metrics for evaluating Athena's underwriting recommendations.
These metrics cover citation verification, decision quality assessment, content
completeness, factual faithfulness, rule compliance, and referral reason analysis.

---

## Recommendation Metrics

<div class="grid-container">

<div class="grid-item">
<strong><a href="citation_accuracy/">Citation Accuracy</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Validate numeric citations against reference data</p>
<code>actual_output</code> <code>context</code>
</div>

<div class="grid-item">
<strong><a href="citation_fidelity/">Citation Fidelity</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Verify bracket-path citations resolve to valid JSON values</p>
<code>actual_output</code> <code>context</code>
</div>

<div class="grid-item">
<strong><a href="decision_quality/">Decision Quality</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Evaluate AI decision accuracy and reasoning alignment</p>
<code>actual_output</code> <code>expected_output</code>
</div>

<div class="grid-item">
<strong><a href="refer_reason/">Refer Reason</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Extract and categorize reasons for referral/decline outcomes</p>
<code>actual_output</code>
</div>

<div class="grid-item">
<strong><a href="underwriting_completeness/">Underwriting Completeness</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Ensure recommendations contain all required components</p>
<code>actual_output</code>
</div>

<div class="grid-item">
<strong><a href="underwriting_faithfulness/">Underwriting Faithfulness</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Verify factual claims are supported by source data</p>
<code>actual_output</code> <code>context</code>
</div>

<div class="grid-item">
<strong><a href="underwriting_rules/">Underwriting Rules</a></strong>
<p style="margin: 0.3rem 0; color: var(--md-text-secondary); font-size: 0.9rem;">Track referral triggers and validate outcome consistency</p>
<code>actual_output</code> <code>context</code>
</div>

</div>

## Overview

| Metric | Type | Score | Description |
|--------|------|-------|-------------|
| `CitationAccuracy` | Rule-Based | 0.0–1.0 | Validates numeric citations `[1]`, `[2]` match reference data |
| `CitationFidelity` | Rule-Based | 0.0–1.0 | Verifies bracket-path citations `[quote.field]` resolve to valid JSON values |
| `DecisionQuality` | LLM-Powered | 0.0–1.0 | Compares AI decision and reasoning against human decisions with hard-fail option |
| `ReferReason` | LLM-Powered | — | Extracts and categorizes reasons for referral/decline outcomes (analysis only) |
| `UnderwritingCompleteness` | LLM-Powered | 0.0–1.0 | Ensures recommendations include Decision, Rationale, Evidence, and Next Steps |
| `UnderwritingFaithfulness` | LLM-Powered | 0.0–1.0 | Detects hallucinations by verifying claims against source data (default threshold 0.9) |
| `UnderwritingRules` | Hybrid | 0.0–1.0 | Validates referral triggers and outcome consistency against underwriting guidelines |

## Metrics Detail

### CitationAccuracy (Rule-Based)

**File:** `citation_accuracy.py`

**Purpose:** Validate that numeric citations (e.g. `[1]`, `[2]`) in the AI recommendation
correctly reference entries in the provided context data.

**What it computes:**

- Extracts all numeric citation markers from the recommendation text
- Checks each citation index against the available reference data
- Produces a score based on the ratio of valid citations to total citations
- Default threshold of 1.0 requires all citations to be valid

**Score:** `valid_citations / total_citations` (1.0 if no citations found)

---

### CitationFidelity (Rule-Based)

**File:** `citation_fidelity.py`

**Purpose:** Verify that bracket-path citations (e.g. `[quote.field]`) in the recommendation
resolve to actual values in the source JSON context.

**What it computes:**

- Extracts bracket-path citation patterns from the recommendation text
- Attempts to resolve each path against the provided JSON context
- Produces a score based on the ratio of resolvable paths to total paths
- Default threshold of 1.0 requires all paths to resolve

**Score:** `resolvable_paths / total_paths` (1.0 if no paths found)

---

### DecisionQuality (LLM-Powered)

**File:** `decision_quality.py`

**Purpose:** Evaluate how well the AI's underwriting decision and reasoning align
with the human underwriter's decision.

**What it computes:**

- Compares the AI recommendation (approve/decline/refer) to the expected human decision
- Uses LLM to evaluate reasoning quality and alignment
- Produces a weighted quality score combining decision match and reasoning quality
- Supports a hard-fail option where decision mismatch results in a score of 0.0

**Score:** Weighted combination of decision accuracy and reasoning alignment (0.0–1.0)

---

### ReferReason (LLM-Powered)

**File:** `refer_reason.py`

**Purpose:** Extract and categorize the reasons behind referral or decline outcomes
for aggregate analysis.

**What it computes:**

- Uses LLM to analyze the recommendation text and identify stated reasons
- Categorizes reasons into predefined types (risk concerns, policy exceptions, missing information, etc.)
- Produces structured analysis output rather than a numeric score
- Designed for aggregate reporting and trend analysis across cases

**Score:** None (analysis-only metric)

---

### UnderwritingCompleteness (LLM-Powered)

**File:** `underwriting_completeness.py`

**Purpose:** Ensure that AI recommendations contain all required structural components
for a complete underwriting assessment.

**What it computes:**

- Uses LLM to check for four required components: **Decision**, **Rationale**, **Evidence**, and **Next Steps**
- Produces a weighted score based on which components are present and their quality
- Identifies missing or weak sections to guide recommendation improvements

**Required components:**

| Component | Description |
|-----------|-------------|
| Decision | Clear approve/decline/refer recommendation |
| Rationale | Reasoning behind the decision |
| Evidence | Supporting data and references |
| Next Steps | Actions to take or conditions to meet |

---

### UnderwritingFaithfulness (LLM-Powered)

**File:** `underwriting_faithfulness.py`

**Purpose:** Detect hallucinations by verifying that factual claims in the recommendation
are supported by the provided source data.

**What it computes:**

- Extracts factual claims from the recommendation text
- Uses LLM to verify each claim against the source context
- Classifies claims as supported, unsupported, or contradicted
- Produces a score based on the ratio of supported claims to total claims
- Default threshold of 0.9 sets a high bar for factual accuracy

**Score:** `supported_claims / total_claims` (0.0–1.0)

---

### UnderwritingRules (Hybrid)

**File:** `underwriting_rules.py`

**Purpose:** Validate that the AI recommendation correctly identifies referral triggers
and that the outcome is consistent with underwriting guidelines.

**What it computes:**

- Uses rule-based detection to identify referral triggers in the source data
- Compares detected triggers against the AI's stated outcome
- Validates that cases with referral triggers produce a "Refer" outcome
- Scoped to explicit Refer outcomes — approve/decline cases are not penalized
- Default threshold of 1.0 requires full consistency between triggers and outcome

**Score:** Consistency ratio between detected triggers and stated outcome (0.0–1.0)
