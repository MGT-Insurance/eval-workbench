# Citation Fidelity

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Verify bracket-path citations resolve to valid JSON values</strong><br>
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
<small style="color: var(--md-text-muted);">All paths must resolve</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>actual_output</code> <code>expected_output</code><br>
<small style="color: var(--md-text-muted);">JSON data required</small>
</div>

</div>

!!! abstract "What It Measures"
    Citation Fidelity validates bracket-path citations like `[quote.premium]` or `[property.address]` against JSON data in `expected_output`. It ensures every cited path resolves to a real value and optionally verifies the cited value appears in the surrounding text.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: All citation paths resolve correctly |
    | **0.8+** | :material-check: Most paths valid, minor issues |
    | **0.5** | :material-alert: Half the citations are invalid |
    | **< 0.5** | :material-close: Many paths don't resolve |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Citations use JSON path format</li>
<li>You need to verify data accuracy</li>
<li>Checking that cited values match the text</li>
<li>Validating structured data references</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Citations use numeric format (use Citation Accuracy)</li>
<li>No JSON data available</li>
<li>Output doesn't contain bracket citations</li>
<li>Free-form references without paths</li>
</ul>
</div>

</div>

!!! tip "See Also: Citation Accuracy"
    **Citation Fidelity** validates path citations like `[quote.field]` against JSON.
    **[Citation Accuracy](./citation_accuracy.md)** validates numeric citations like `[1]` against reference lists.

    Use Fidelity for JSON paths; use Accuracy for numbered references.

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric parses bracket-path citations and resolves each against the JSON structure.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[AI Output Text]
            B[Expected Output JSON]
        end

        subgraph PARSE["🔍 Step 1: Parse Citations"]
            C["Find [path.to.field]"]
            D["Citation Path List"]
        end

        subgraph RESOLVE["⚖️ Step 2: Path Resolution"]
            E[Navigate JSON Structure]
            F[Optionally Check Value in Text]
            G["Resolved / Not Found"]
        end

        subgraph SCORE["📊 Step 3: Scoring"]
            H["Count Valid Paths"]
            I["Calculate Ratio"]
            J["Final Score"]
        end

        A --> C
        C --> D
        D --> E
        B --> E
        E --> F
        A --> F
        F --> G
        G --> H
        H --> I
        I --> J

        style INPUT stroke:#8B9F4F,stroke-width:2px
        style PARSE stroke:#3b82f6,stroke-width:2px
        style RESOLVE stroke:#f59e0b,stroke-width:2px
        style SCORE stroke:#10b981,stroke-width:2px
        style J fill:#8B9F4F,stroke:#6B7A3A,stroke-width:3px,color:#fff
    ```

=== ":material-scale-balance: Scoring System"

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">✅ VALID</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #10b981;">1</div>
    <br><small>Path resolves to a value; value appears in text (if check enabled).</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">❌ INVALID</strong>
    <div style="float: right; font-size: 1.5rem; font-weight: bold; color: #ef4444;">0</div>
    <br><small>Path doesn't exist or value not found in text.</small>
    </div>

    </div>

    !!! tip "Score Formula"
        ```
        score = valid_citations / total_citations
        ```

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `check_values` | `bool` | `False` | Verify cited values appear in text |
    | `window_chars` | `int` | `200` | Characters to search for value |
    | `min_shared_tokens` | `int` | `2` | Min tokens to match for text values |
    | `fuzzy_threshold` | `float` | `0.8` | Fuzzy match threshold for strings |
    | `numeric_tolerance` | `float` | `0.01` | Tolerance for numeric comparisons |

    !!! info "Value Checking"
        When `check_values=True`, the metric verifies that the JSON value actually appears in the text near the citation, preventing citations that point to valid paths but misrepresent the data.

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.implementations.athena.metrics.recommendation.citation_fidelity import CitationFidelity

    metric = CitationFidelity()

    item = DatasetItem(
        actual_output="The premium is $1,200 [quote.premium].",
        expected_output={"quote": {"premium": 1200}},
    )

    result = await metric.execute(item)
    print(result.pretty())
    # Score: 1.0 (path resolves correctly)
    ```

=== ":material-cog-outline: With Value Checking"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.implementations.athena.metrics.recommendation.citation_fidelity import CitationFidelity

    metric = CitationFidelity(check_values=True, window_chars=100)

    item = DatasetItem(
        actual_output="Premium is $1,200 [quote.premium]. Coverage limit $500,000 [quote.coverage].",
        expected_output={
            "quote": {
                "premium": 1200,
                "coverage": 500000
            }
        },
    )

    result = await metric.execute(item)
    print(f"Score: {result.score}")
    print(f"Valid: {result.signals.valid_citations}/{result.signals.total_citations}")
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
<summary><strong>📊 CitationFidelityResult Structure</strong></summary>

```python
CitationFidelityResult(
{
    "score": 1.0,
    "total_citations": 2,
    "valid_citations": 2,
    "verdicts": [
        {
            "citation": "[quote.premium]",
            "path": "quote.premium",
            "resolved_value": 1200,
            "status": "valid",
            "reason": "Path resolved successfully"
        },
        {
            "citation": "[quote.coverage]",
            "path": "quote.coverage",
            "resolved_value": 500000,
            "status": "valid",
            "reason": "Path resolved successfully"
        }
    ]
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float` | Overall fidelity score |
| `total_citations` | `int` | Total path citations found |
| `valid_citations` | `int` | Citations that resolved |
| `verdicts` | `List` | Per-citation resolution details |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: All Paths Resolve (Score: 1.0)</strong></summary>

!!! success "Valid JSON Path Citations"

    **Output:**
    > "Building age is 15 years [property.building_age]. Revenue: $2.5M [financials.revenue]."

    **Expected Output (JSON):**
    ```json
    {
        "property": {"building_age": 15},
        "financials": {"revenue": 2500000}
    }
    ```

    **Analysis:**

    | Citation | Path | Resolved Value | Status |
    |----------|------|----------------|--------|
    | `[property.building_age]` | `property.building_age` | `15` | ✅ Valid |
    | `[financials.revenue]` | `financials.revenue` | `2500000` | ✅ Valid |

    **Final Score:** `2 / 2 = 1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Invalid Path (Score: 0.5)</strong></summary>

!!! warning "Some Paths Don't Resolve"

    **Output:**
    > "Premium is $1,200 [quote.premium]. Deductible: $500 [quote.deductible]."

    **Expected Output (JSON):**
    ```json
    {
        "quote": {"premium": 1200}
    }
    ```

    **Analysis:**

    | Citation | Path | Resolved Value | Status |
    |----------|------|----------------|--------|
    | `[quote.premium]` | `quote.premium` | `1200` | ✅ Valid |
    | `[quote.deductible]` | `quote.deductible` | Not found | ❌ Invalid |

    **Final Score:** `1 / 2 = 0.5` :material-alert:

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">🎯</span>
<strong>Data Accuracy</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Ensures AI-cited values actually exist in the source data structure.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔍</span>
<strong>Value Verification</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Optional check that cited values match what's stated in the text.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">📊</span>
<strong>Structured Tracing</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Enables precise traceability through JSON path references.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Citation Fidelity** = Do bracket-path citations like `[quote.field]` resolve to valid JSON values?

    - **Use it when:** AI output uses JSON path citations and you have structured data
    - **Score interpretation:** Higher = more paths resolve correctly
    - **Key difference:** Validates path resolution; optionally checks value accuracy

<div class="grid cards" markdown>

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Citation Accuracy](./citation_accuracy.md) · Underwriting Faithfulness

</div>
