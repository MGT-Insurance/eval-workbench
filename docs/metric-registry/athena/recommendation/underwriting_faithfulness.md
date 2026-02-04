# Underwriting Faithfulness

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Verify factual claims are supported by source data</strong><br>
<span class="badge" style="margin-top: 0.5rem;">LLM-Powered</span>
<span class="badge" style="background: #667eea;">Faithfulness</span>
<span class="badge" style="background: #6B7A3A;">Athena</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Support ratio</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">⚡</span><br>
<strong>Default Threshold</strong><br>
<code style="font-size: 1.5rem; color: var(--md-primary);">0.9</code><br>
<small style="color: var(--md-text-muted);">High bar for factual accuracy</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>actual_output</code><br>
<small style="color: var(--md-text-muted);">Optional: additional_input (source data)</small>
</div>

</div>

!!! abstract "What It Measures"
    Underwriting Faithfulness checks whether **factual claims** in the AI recommendation are **supported by the source data**. It extracts atomic claims from the recommendation, finds relevant evidence in the input data, and verifies each claim. Unsupported claims are flagged as potential hallucinations.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: All claims verified against source data |
    | **0.9+** | :material-check: Nearly all claims supported |
    | **0.7** | :material-alert: Some claims not verifiable |
    | **< 0.7** | :material-close: Significant hallucination risk |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Detecting hallucinations</li>
<li>Verifying factual accuracy</li>
<li>Source data is available</li>
<li>High-stakes recommendations</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>No source data available</li>
<li>Evaluating opinions/judgments</li>
<li>Checking structural completeness</li>
<li>Output is purely generative</li>
</ul>
</div>

</div>

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric extracts claims from the recommendation and verifies each against the source data.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[AI Recommendation]
            B[Source Data JSON]
        end

        subgraph EXTRACT["🔍 Step 1: Claim Extraction"]
            C[Break into Atomic Claims]
            D["Claim List"]
        end

        subgraph EVIDENCE["📊 Step 2: Evidence Finding"]
            E[Search Source Data]
            F["Relevant Evidence Lines"]
        end

        subgraph VERIFY["⚖️ Step 3: Verification"]
            G[LLM or Heuristic Check]
            H["Supported / Unsupported"]
        end

        subgraph SCORE["📈 Step 4: Scoring"]
            I["Count Supported Claims"]
            J["Calculate Ratio"]
            K["overall_score"]
        end

        A --> C
        C --> D
        D --> E
        B --> E
        E --> F
        D & F --> G
        G --> H
        H --> I
        I --> J
        J --> K

        style INPUT stroke:#8B9F4F,stroke-width:2px
        style EXTRACT stroke:#3b82f6,stroke-width:2px
        style EVIDENCE stroke:#f59e0b,stroke-width:2px
        style VERIFY stroke:#8b5cf6,stroke-width:2px
        style SCORE stroke:#10b981,stroke-width:2px
        style K fill:#8B9F4F,stroke:#6B7A3A,stroke-width:3px,color:#fff
    ```

=== ":material-scale-balance: Verification Modes"

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #3b82f6; padding-left: 1rem;">
    <strong style="color: #3b82f6;">🤖 LLM Mode</strong>
    <br><small>Uses LLM to semantically verify claims against evidence. Most accurate but slower.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">⚡ Heuristic Mode</strong>
    <br><small>Uses pattern matching and fuzzy comparison. Faster but less nuanced.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #f59e0b; padding-left: 1rem;">
    <strong style="color: #f59e0b;">🔄 Heuristic-then-LLM</strong>
    <br><small>Tries heuristic first, falls back to LLM for uncertain cases.</small>
    </div>

    </div>

    !!! tip "Score Formula"
        ```
        overall_score = supported_claims / total_claims
        ```

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `verification_mode` | `str` | `llm` | `llm`, `heuristic`, or `heuristic_then_llm` |
    | `max_claims` | `int` | `50` | Maximum claims to verify |
    | `max_concurrent` | `int` | `10` | Concurrency limit for LLM verification |

    !!! info "Performance Tuning"
        For large recommendations, use `heuristic_then_llm` mode with appropriate `max_concurrent` to balance accuracy and speed.

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.implementations.athena.metrics.recommendation.underwriting_faithfulness import UnderwritingFaithfulness

    metric = UnderwritingFaithfulness(verification_mode="heuristic")

    item = DatasetItem(
        actual_output="The business has annual revenue of $1.2M and no prior claims.",
        additional_input={
            "financials": {"annual_revenue": 1200000},
            "claims": {"count": 0, "history": []}
        }
    )

    result = await metric.execute(item)
    print(result.pretty())
    # Score: 1.0 (both claims verified)
    ```

=== ":material-cog-outline: LLM Verification"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.implementations.athena.metrics.recommendation.underwriting_faithfulness import UnderwritingFaithfulness

    metric = UnderwritingFaithfulness(
        verification_mode="llm",
        max_claims=30,
        max_concurrent=5,
    )

    item = DatasetItem(
        actual_output="""
        Recommend approve based on:
        - Strong financials: $2.1M revenue
        - Clean 5-year claims history
        - Building constructed in 2019
        """,
        additional_input={
            "financials": {"revenue": 2100000},
            "claims": {"five_year_count": 0},
            "property": {"year_built": 2019}
        }
    )

    result = await metric.execute(item)
    print(f"Score: {result.signals.overall_score}")
    print(f"Supported: {result.signals.supported_claims}/{result.signals.total_claims}")
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
<summary><strong>📊 UnderwritingFaithfulnessResult Structure</strong></summary>

```python
UnderwritingFaithfulnessResult(
{
    "overall_score": 0.85,
    "total_claims": 4,
    "supported_claims": 3,
    "hallucinations": 1,
    "claim_details": [
        {
            "claim": "Revenue is $2.1M",
            "status": "supported",
            "evidence": "financials.revenue: 2100000",
            "confidence": 0.95
        },
        {
            "claim": "Building was constructed in 2019",
            "status": "supported",
            "evidence": "property.year_built: 2019",
            "confidence": 0.98
        },
        {
            "claim": "No claims in 5 years",
            "status": "supported",
            "evidence": "claims.five_year_count: 0",
            "confidence": 0.92
        },
        {
            "claim": "Premium is $1,200",
            "status": "unsupported",
            "evidence": null,
            "confidence": 0.1
        }
    ],
    "unverified_claims": ["Premium is $1,200"]
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `overall_score` | `float` | Proportion of supported claims |
| `total_claims` | `int` | Total atomic claims extracted |
| `supported_claims` | `int` | Claims verified against source |
| `hallucinations` | `int` | Claims not found in source |
| `claim_details` | `List` | Per-claim verification details |
| `unverified_claims` | `List[str]` | List of unsupported claims |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: All Claims Verified (Score: 1.0)</strong></summary>

!!! success "Fully Faithful Recommendation"

    **Recommendation:**
    > "Approve. Revenue is $1.5M. Building age is 8 years. Zero claims."

    **Source Data:**
    ```json
    {
        "financials": {"revenue": 1500000},
        "property": {"building_age": 8},
        "claims": {"count": 0}
    }
    ```

    **Analysis:**

    | Claim | Evidence | Status |
    |-------|----------|--------|
    | Revenue $1.5M | `financials.revenue: 1500000` | ✅ Supported |
    | Building age 8 years | `property.building_age: 8` | ✅ Supported |
    | Zero claims | `claims.count: 0` | ✅ Supported |

    **Final Score:** `3 / 3 = 1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Hallucination Detected (Score: 0.67)</strong></summary>

!!! warning "Unsupported Claim Found"

    **Recommendation:**
    > "Revenue is $1.5M. Building age is 8 years. **Premium is $1,200.**"

    **Source Data:**
    ```json
    {
        "financials": {"revenue": 1500000},
        "property": {"building_age": 8}
    }
    ```

    **Analysis:**

    | Claim | Evidence | Status |
    |-------|----------|--------|
    | Revenue $1.5M | `financials.revenue: 1500000` | ✅ Supported |
    | Building age 8 years | `property.building_age: 8` | ✅ Supported |
    | Premium $1,200 | Not found | ❌ Hallucination |

    **Final Score:** `2 / 3 = 0.67` :material-alert:

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Significant Hallucinations (Score: 0.25)</strong></summary>

!!! failure "Multiple Unsupported Claims"

    **Recommendation:**
    > "Revenue is $5M. Building is brand new. Located in a low-risk zone. Premium is competitive."

    **Source Data:**
    ```json
    {
        "financials": {"revenue": 1000000}
    }
    ```

    **Analysis:**

    | Claim | Evidence | Status |
    |-------|----------|--------|
    | Revenue $5M | Contradicts source ($1M) | ❌ False |
    | Building brand new | Not found | ❌ Hallucination |
    | Low-risk zone | Not found | ❌ Hallucination |
    | Competitive premium | Not found | ❌ Hallucination |

    **Final Score:** `0 / 4 = 0.0` :material-close:

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">🔍</span>
<strong>Hallucination Detection</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Catches AI claims that aren't grounded in actual data.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">✓</span>
<strong>Trust & Compliance</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Critical for regulated industries where false claims have consequences.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🛡️</span>
<strong>Risk Mitigation</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Prevents decisions based on fabricated information.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Underwriting Faithfulness** = Are the AI's factual claims actually in the source data?

    - **Use it when:** You have source data and need to verify factual accuracy
    - **Score interpretation:** Higher = more claims verified, fewer hallucinations
    - **Key feature:** Detects fabricated facts not in source data

<div class="grid cards" markdown>

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Citation Accuracy](./citation_accuracy.md) · Citation Fidelity · Underwriting Completeness

</div>
