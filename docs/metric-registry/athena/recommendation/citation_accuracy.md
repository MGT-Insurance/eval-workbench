# Citation Accuracy

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Validate numeric citations against reference data</strong><br>
<span class="badge" style="margin-top: 0.5rem;">Rule-Based</span>
<span class="badge" style="background: #667eea;">Verification</span>
<span class="badge" style="background: #6B7A3A;">Athena</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Validity ratio</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">⚡</span><br>
<strong>Default Threshold</strong><br>
<code style="font-size: 1.5rem; color: var(--md-primary);">1.0</code><br>
<small style="color: var(--md-text-muted);">All citations must be valid</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>actual_reference</code><br>
<small style="color: var(--md-text-muted);">Optional: actual_output, additional_output, additional_input</small>
</div>

</div>

!!! abstract "What It Measures"
    Citation Accuracy validates numeric citations like `[1]`, `[2]` in AI-generated text against the `actual_reference` data. It ensures every citation points to a real reference entry and optionally verifies that referenced fields exist in the input data.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: All citations reference valid entries |
    | **0.8+** | :material-check: Most citations valid, minor issues |
    | **0.5** | :material-alert: Half the citations are invalid |
    | **< 0.5** | :material-close: Significant citation errors |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>AI output contains numbered citations</li>
<li>You have reference data to validate against</li>
<li>Traceability of claims is important</li>
<li>Verifying underwriting recommendations</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Citations use bracket-path format (use Citation Fidelity)</li>
<li>No reference data available</li>
<li>Output doesn't contain citations</li>
<li>Free-form text without structured references</li>
</ul>
</div>

</div>

!!! tip "See Also: Citation Fidelity"
    **Citation Accuracy** validates numeric citations like `[1]` against reference lists.
    **[Citation Fidelity](./citation_fidelity.md)** validates bracket-path citations like `[quote.field]` against JSON data.

    Use Accuracy for numbered references; use Fidelity for JSON path citations.

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric extracts numeric citations from the output text and validates each against the reference data.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[AI Output Text]
            B[Reference Data]
            C[Input Data - Optional]
        end

        subgraph EXTRACT["🔍 Step 1: Citation Extraction"]
            D["Find [1], [2], etc."]
            E["Citation List"]
        end

        subgraph VALIDATE["⚖️ Step 2: Validation"]
            F[Match to Reference Entry]
            G[Optionally Check Input Fields]
            H["Valid / Invalid"]
        end

        subgraph SCORE["📊 Step 3: Scoring"]
            I["Count Valid Citations"]
            J["Calculate Ratio"]
            K["Final Score"]
        end

        A --> D
        D --> E
        E --> F
        B --> F
        F --> G
        C --> G
        G --> H
        H --> I
        I --> J
        J --> K

        style INPUT stroke:#8B9F4F,stroke-width:2px
        style EXTRACT stroke:#3b82f6,stroke-width:2px
        style VALIDATE stroke:#f59e0b,stroke-width:2px
        style SCORE stroke:#10b981,stroke-width:2px
        style K fill:#8B9F4F,stroke:#6B7A3A,stroke-width:3px,color:#fff
    ```

=== ":material-scale-balance: Scoring System"

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">✅ VALID</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #10b981;">1</div>
    <br><small>Citation matches a reference entry and field exists (if checked).</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">❌ INVALID</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #ef4444;">0</div>
    <br><small>Citation doesn't match any reference or field doesn't exist.</small>
    </div>

    </div>

    !!! tip "Score Formula"
        ```
        score = valid_citations / scorable_citations
        ```

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `validation_mode` | `str` | `ref_only` | `ref_only` or `ref_plus_input` |
    | `output_key` | `str` | `None` | Key in `additional_output` to analyze (fallback to `actual_output`) |

    !!! info "Validation Modes"
        - **ref_only**: Only check that citations match reference entries
        - **ref_plus_input**: Also verify referenced fields exist in `additional_input`

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.dataset import DatasetItem
    from implementations.athena.metrics.recommendation.citation_accuracy import CitationAccuracy

    metric = CitationAccuracy(validation_mode="ref_only")

    item = DatasetItem(
        actual_output="The policy was approved based on the roof age [1] and revenue [2].",
        actual_reference=["[1] - quote.roof_age", "[2] - quote.revenue"],
    )

    result = await metric.execute(item)
    print(result.pretty())
    # Score: 1.0 (2 of 2 citations valid)
    ```

=== ":material-cog-outline: With Input Validation"

    ```python
    from axion.dataset import DatasetItem
    from implementations.athena.metrics.recommendation.citation_accuracy import CitationAccuracy

    metric = CitationAccuracy(validation_mode="ref_plus_input")

    item = DatasetItem(
        actual_output="Premium is $1,200 [1].",
        actual_reference=["[1] - quote.premium"],
        additional_input={"quote": {"premium": 1200}},
    )

    result = await metric.execute(item)
    print(f"Score: {result.score}")
    print(f"Valid: {result.signals.valid_citations}/{result.signals.total_citations}")
    ```

---

## Metric Diagnostics

Every evaluation is **fully interpretable**. Access detailed diagnostic results via `result.signals` to understand exactly why a score was given.

```python
result = await metric.execute(item)
print(result.pretty())      # Human-readable summary
result.signals              # Full diagnostic breakdown
```

<details markdown="1">
<summary><strong>📊 CitationAccuracyResult Structure</strong></summary>

```python
CitationAccuracyResult(
{
    "score": 1.0,
    "total_citations": 2,
    "scorable_citations": 2,
    "valid_citations": 2,
    "verdicts": [
        {
            "citation": "[1]",
            "reference_match": "[1] - quote.roof_age",
            "status": "valid",
            "reason": "Reference entry found"
        },
        {
            "citation": "[2]",
            "reference_match": "[2] - quote.revenue",
            "status": "valid",
            "reason": "Reference entry found"
        }
    ]
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float` | Overall accuracy score |
| `total_citations` | `int` | Total citations found in output |
| `scorable_citations` | `int` | Citations that could be validated |
| `valid_citations` | `int` | Citations that passed validation |
| `verdicts` | `List` | Per-citation validation details |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: All Valid (Score: 1.0)</strong></summary>

!!! success "All Citations Match References"

    **Output:**
    > "The property qualifies for approval based on the building age [1] and claims history [2]."

    **Reference Data:**
    ```python
    ["[1] - property.building_age", "[2] - property.claims_count"]
    ```

    **Analysis:**

    | Citation | Reference Match | Status |
    |----------|-----------------|--------|
    | `[1]` | `[1] - property.building_age` | ✅ Valid |
    | `[2]` | `[2] - property.claims_count` | ✅ Valid |

    **Final Score:** `2 / 2 = 1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Missing Reference (Score: 0.5)</strong></summary>

!!! warning "Some Citations Invalid"

    **Output:**
    > "Coverage approved per [1]. Additional review needed per [3]."

    **Reference Data:**
    ```python
    ["[1] - quote.coverage", "[2] - quote.premium"]
    ```

    **Analysis:**

    | Citation | Reference Match | Status |
    |----------|-----------------|--------|
    | `[1]` | `[1] - quote.coverage` | ✅ Valid |
    | `[3]` | Not found | ❌ Invalid |

    **Final Score:** `1 / 2 = 0.5` :material-alert:

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">🔗</span>
<strong>Traceability</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Ensures every claim in AI output can be traced back to source data.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">✓</span>
<strong>Compliance</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Critical for regulatory requirements where decisions must be documented.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🛡️</span>
<strong>Trust</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Builds confidence that AI recommendations are grounded in real data.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Citation Accuracy** = Do numeric citations `[1]`, `[2]` point to valid reference entries?

    - **Use it when:** AI output has numbered citations and you have reference data
    - **Score interpretation:** Higher = more citations are valid
    - **Key difference:** Validates reference existence, not content accuracy

<div class="grid cards" markdown>

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Citation Fidelity](./citation_fidelity.md) · Underwriting Faithfulness

</div>
