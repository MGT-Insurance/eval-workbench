# Decision Quality

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Evaluate AI decision accuracy and reasoning alignment</strong><br>
<span class="badge" style="margin-top: 0.5rem;">LLM-Powered</span>
<span class="badge" style="background: #667eea;">Quality</span>
<span class="badge" style="background: #6B7A3A;">Athena</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Weighted quality score</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">⚡</span><br>
<strong>Default Threshold</strong><br>
<code style="font-size: 1.5rem; color: var(--md-primary);">0.8</code><br>
<small style="color: var(--md-text-muted);">High Bar</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>actual_output</code> <code>expected_output</code><br>
<small style="color: var(--md-text-muted);">Human decision + AI recommendation</small>
</div>

</div>

!!! abstract "What It Measures"
    Decision Quality evaluates whether the AI made the **correct underwriting decision** (approve/decline/refer) and whether its **reasoning aligns** with the human underwriter's notes. It combines decision match scoring with reasoning coverage analysis.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: Correct decision + complete reasoning |
    | **0.7+** | :material-check: Correct decision, reasoning mostly aligned |
    | **0.5** | :material-alert: Decision match but reasoning gaps |
    | **0.0** | :material-close: Wrong decision (hard fail enabled) |

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric extracts decisions from both human and AI outputs, scores the match, and analyzes reasoning coverage.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[AI Recommendation]
            B[Human Decision/Notes]
        end

        subgraph EXTRACT["🔍 Step 1: Decision Extraction"]
            C[Detect Human Decision]
            D[Detect AI Decision]
            E["Approve / Decline / Refer"]
        end

        subgraph MATCH["⚖️ Step 2: Decision Match"]
            F[Compare Decisions]
            G["outcome_score"]
        end

        subgraph REASON["📝 Step 3: Reasoning Coverage"]
            H[Extract Risk Factors]
            I[Check Coverage in AI Output]
            J["reasoning_score"]
        end

        subgraph COMBINE["📊 Step 4: Final Score"]
            K[Apply Weights]
            L[Hard Fail Check]
            M["overall_score"]
        end

        A --> D
        B --> C
        C & D --> E
        E --> F
        F --> G
        B --> H
        A --> I
        H --> I
        I --> J
        G & J --> K
        K --> L
        L --> M

        style INPUT stroke:#8B9F4F,stroke-width:2px
        style EXTRACT stroke:#3b82f6,stroke-width:2px
        style MATCH stroke:#f59e0b,stroke-width:2px
        style REASON stroke:#8b5cf6,stroke-width:2px
        style COMBINE stroke:#10b981,stroke-width:2px
        style M fill:#8B9F4F,stroke:#6B7A3A,stroke-width:3px,color:#fff
    ```

=== ":material-scale-balance: Scoring System"

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">✅ DECISION MATCH</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #10b981;">1.0</div>
    <br><small>AI decision matches human decision exactly.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">❌ DECISION MISMATCH</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #ef4444;">0.0</div>
    <br><small>AI decision differs from human decision.</small>
    </div>

    </div>

    !!! tip "Score Formula"
        ```
        overall_score = (outcome_weight × outcome_score) + (reasoning_weight × reasoning_score)

        # If hard_fail_on_outcome_mismatch=True and decisions differ:
        overall_score = 0.0
        ```

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `outcome_weight` | `float` | `1.0` | Weight for decision match component |
    | `reasoning_weight` | `float` | `0.0` | Weight for reasoning coverage |
    | `hard_fail_on_outcome_mismatch` | `bool` | `True` | Force score to 0.0 if decisions differ |
    | `recommendation_column_name` | `str` | `brief_recommendation` | Additional output field to analyze |

    !!! info "Default Behavior"
        By default, `outcome_weight=1.0` and `reasoning_weight=0.0`, meaning the score is purely based on decision match (1.0 or 0.0). Set `reasoning_weight > 0` to include reasoning coverage analysis.

    !!! warning "Hard Fail Mode"
        When `hard_fail_on_outcome_mismatch=True` (default), any decision mismatch results in a score of 0.0, regardless of reasoning quality. Disable this for softer evaluation.

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.implementations.athena.metrics.recommendation.decision_quality import DecisionQuality

    metric = DecisionQuality()

    item = DatasetItem(
        actual_output="Recommend Decline due to building age exceeding 30 years.",
        expected_output="Decline - roof age and building condition are concerns.",
    )

    result = await metric.execute(item)
    print(result.pretty())
    # Overall: 0.85 (decision match + partial reasoning coverage)
    ```

=== ":material-cog-outline: Custom Weights"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.implementations.athena.metrics.recommendation.decision_quality import DecisionQuality

    # Prioritize reasoning over outcome
    metric = DecisionQuality(
        outcome_weight=0.4,
        reasoning_weight=0.6,
        hard_fail_on_outcome_mismatch=False,
    )

    item = DatasetItem(
        actual_output="Approve with conditions for roof repair.",
        expected_output="Approve - good risk profile, minor roof concerns noted.",
    )

    result = await metric.execute(item)
    print(f"Overall: {result.signals.overall_score}")
    print(f"Outcome Match: {result.signals.outcome_match}")
    print(f"Reasoning: {result.signals.reasoning_score}")
    ```

---

## Metric Diagnostics

Every evaluation is **fully interpretable**. Access detailed diagnostic results via `result.signals`.

```python
result = await metric.execute(item)
print(result.pretty())      # Human-readable summary
result.signals              # Full diagnostic breakdown
```

<details markdown="1">
<summary><strong>📊 DecisionQualityResult Structure</strong></summary>

```python
DecisionQualityResult(
{
    "overall_score": 1.0,
    "outcome_match": true,
    "outcome_score": 1.0,
    "human_decision_detected": "decline",
    "ai_decision_detected": "decline",
    "reasoning_score": 0.625,
    "matched_concepts": [
        {"concept": "building age"},
        {"concept": "condition"}
    ],
    "missing_concepts": [
        {"concept": "roof age", "impact": "High"}
    ]
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `overall_score` | `float` | Combined weighted score |
| `outcome_match` | `bool` | Whether decisions matched |
| `outcome_score` | `float` | Decision match score (0 or 1) |
| `human_decision_detected` | `str` | Extracted human decision |
| `ai_decision_detected` | `str` | Extracted AI decision |
| `reasoning_score` | `float \| None` | Impact-weighted reasoning coverage score (None when no risk factors extracted) |
| `matched_concepts` | `List[ReasoningMatch]` | Risk factors mentioned by AI (each has `concept`) |
| `missing_concepts` | `List[ReasoningGap]` | Risk factors AI missed (each has `concept` and `impact`) |

</details>

---

## Example Scenarios

=== "Pass (1.0)"

    !!! success "Decision Match (Default Weights)"

        **Human Decision:**
        > "Decline - prior claims history and high BPP value are concerns."

        **AI Recommendation:**
        > "Recommend Decline. The applicant has prior claims on record and the BPP coverage requested exceeds typical thresholds."

        **Analysis:**

        | Component | Score | Details |
        |-----------|-------|---------|
        | Decision Match | 1.0 | Both: Decline |
        | Reasoning Coverage | 1.0 | All factors mentioned |

        **Final Score:** `(1.0 × 1.0) + (0.0 × 1.0) = 1.0` :material-check-all:

        !!! tip "With custom weights `outcome_weight=0.6, reasoning_weight=0.4`"
            Score would be `(0.6 × 1.0) + (0.4 × 1.0) = 1.0`

=== "Partial (0.73)"

    !!! warning "Correct Decision, Missing Factors (Custom Weights)"

        With `outcome_weight=0.6, reasoning_weight=0.4`:

        **Human Decision:**
        > "Approve - good claims history, reasonable BPP, building in good condition."

        **AI Recommendation:**
        > "Recommend Approve based on clean claims history."

        **Analysis:**

        | Component | Score | Details |
        |-----------|-------|---------|
        | Decision Match | 1.0 | Both: Approve |
        | Reasoning Coverage | 0.33 | Only 1 of 3 factors mentioned |

        **Final Score:** `(0.6 × 1.0) + (0.4 × 0.33) = 0.73` :material-alert:

=== "Fail (0.0)"

    !!! failure "Wrong Decision (Hard Fail)"

        **Human Decision:**
        > "Decline - too many risk factors."

        **AI Recommendation:**
        > "Recommend Approve based on revenue metrics."

        **Analysis:**

        | Component | Score | Details |
        |-----------|-------|---------|
        | Decision Match | 0.0 | Human: Decline, AI: Approve |
        | Hard Fail | Triggered | Score forced to 0.0 |

        **Final Score:** `0.0` :material-close:

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">🎯</span>
<strong>Decision Accuracy</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Measures whether AI reaches the same conclusions as human experts.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">📝</span>
<strong>Reasoning Quality</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Ensures AI considers the same risk factors as human underwriters.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔄</span>
<strong>Calibration</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Helps identify where AI and human judgment diverge for retraining.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Decision Quality** = Does AI make the right decision for the right reasons?

    - **Use it when:** You have ground truth decisions and want to evaluate both outcome and reasoning
    - **Score interpretation:** Higher = better decision + reasoning alignment
    - **Key feature:** Configurable hard-fail on decision mismatch

<div class="grid cards" markdown>

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Underwriting Completeness](./underwriting_completeness.md) · Underwriting Faithfulness

</div>
