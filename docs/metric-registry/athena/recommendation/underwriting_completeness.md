# Underwriting Completeness

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Ensure recommendations contain all required components</strong><br>
<span class="badge" style="margin-top: 0.5rem;">LLM-Powered</span>
<span class="badge" style="background: #667eea;">Completeness</span>
<span class="badge" style="background: #6B7A3A;">Athena</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Weighted criteria score</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">⚡</span><br>
<strong>Default Threshold</strong><br>
<code style="font-size: 1.5rem; color: var(--md-primary);">—</code><br>
<small style="color: var(--md-text-muted);">No default threshold</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>actual_output</code><br>
<small style="color: var(--md-text-muted);">AI recommendation text</small>
</div>

</div>

!!! abstract "What It Measures"
    Underwriting Completeness evaluates whether a recommendation contains all required components of a complete underwriting decision: **Decision**, **Rationale**, **Evidence**, and **Next Steps**. Missing the decision component results in an automatic score of 0.0.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: All four components present and strong |
    | **0.7+** | :material-check: Decision clear, some components weaker |
    | **0.5** | :material-alert: Missing or weak supporting components |
    | **0.0** | :material-close: No clear decision (hard gate) |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Evaluating recommendation structure</li>
<li>Ensuring actionable outputs</li>
<li>Training agents on proper format</li>
<li>Quality assurance for underwriting</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Checking factual accuracy</li>
<li>Comparing to ground truth</li>
<li>Evaluating informal responses</li>
<li>Non-recommendation content</li>
</ul>
</div>

</div>

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric runs four specialized LLM judges to evaluate each component of the recommendation.

    ### Step-by-Step Process

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Input"]
            A[AI Recommendation]
        end

        subgraph JUDGES["⚖️ Four Criteria Judges"]
            B["🎯 Decision Judge"]
            C["📝 Rationale Judge"]
            D["📊 Evidence Judge"]
            E["➡️ Next Step Judge"]
        end

        subgraph SCORES["📊 Per-Criterion Scores"]
            F["Decision Score"]
            G["Rationale Score"]
            H["Evidence Score"]
            I["NextStep Score"]
        end

        subgraph COMBINE["🔄 Final Score"]
            J[Apply Weights]
            K[Check Decision Gate]
            L["overall_score"]
        end

        A --> B & C & D & E
        B --> F
        C --> G
        D --> H
        E --> I
        F & G & H & I --> J
        J --> K
        K --> L

        style INPUT stroke:#8B9F4F,stroke-width:2px
        style JUDGES stroke:#3b82f6,stroke-width:2px
        style SCORES stroke:#f59e0b,stroke-width:2px
        style COMBINE stroke:#10b981,stroke-width:2px
        style L fill:#8B9F4F,stroke:#6B7A3A,stroke-width:3px,color:#fff
    ```

=== ":material-scale-balance: Criteria Components"

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">🎯 Decision (Required)</strong>
    <br><small>Clear approve/decline/refer recommendation. <strong>Hard gate</strong>: missing = score 0.0</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #3b82f6; padding-left: 1rem;">
    <strong style="color: #3b82f6;">📝 Rationale</strong>
    <br><small>Explanation of why the decision was made.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #f59e0b; padding-left: 1rem;">
    <strong style="color: #f59e0b;">📊 Evidence</strong>
    <br><small>Supporting data points and facts cited.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">➡️ Next Steps</strong>
    <br><small>Clear guidance on what happens next.</small>
    </div>

    </div>

    !!! tip "Score Formula"
        ```
        overall_score = Σ (weight[i] × criterion_score[i])

        # If Decision score = 0:
        overall_score = 0.0  (hard gate)
        ```

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `weights` | `dict` | See below | Per-criterion weights |

    **Default Weights:**

    | Criterion | Default Weight |
    |-----------|----------------|
    | Decision | 0.4 |
    | Rationale | 0.25 |
    | Evidence | 0.2 |
    | NextStep | 0.15 |

    !!! warning "Hard Gate"
        If the Decision criterion scores 0.0 (no clear decision found), the overall score is forced to 0.0 regardless of other criteria.

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.dataset import DatasetItem
    from implementations.athena.metrics.recommendation.underwriting_completeness import UnderwritingCompleteness

    metric = UnderwritingCompleteness()

    item = DatasetItem(
        actual_output="""
        **Recommendation: Approve**

        The applicant presents a low-risk profile based on:
        - Building age: 5 years (excellent)
        - Clean claims history (0 claims)
        - Revenue: $1.2M annually

        Next steps: Proceed to bind coverage. No additional documentation required.
        """
    )

    result = await metric.execute(item)
    print(result.pretty())
    # Score: 0.95 (all components present and strong)
    ```

=== ":material-cog-outline: Custom Weights"

    ```python
    from axion.dataset import DatasetItem
    from implementations.athena.metrics.recommendation.underwriting_completeness import UnderwritingCompleteness

    # Prioritize evidence over next steps
    metric = UnderwritingCompleteness(
        weights={
            "Decision": 0.35,
            "Rationale": 0.25,
            "Evidence": 0.30,
            "NextStep": 0.10,
        }
    )

    item = DatasetItem(actual_output="Approve. Roof age 5 years. Revenue $1.2M. Next step: bind.")
    result = await metric.execute(item)
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
<summary><strong>📊 UnderwritingCompletenessResult Structure</strong></summary>

```python
UnderwritingCompletenessResult(
{
    "overall_score": 0.92,
    "criteria": {
        "Decision": {
            "score": 1.0,
            "reasoning": "Clear 'Approve' recommendation stated",
            "evidence": "Recommendation: Approve"
        },
        "Rationale": {
            "score": 0.9,
            "reasoning": "Strong explanation with multiple factors",
            "evidence": "low-risk profile, building age, claims history"
        },
        "Evidence": {
            "score": 0.85,
            "reasoning": "Specific data points cited",
            "evidence": "5 years, 0 claims, $1.2M"
        },
        "NextStep": {
            "score": 0.8,
            "reasoning": "Clear action specified",
            "evidence": "Proceed to bind coverage"
        }
    }
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `overall_score` | `float` | Weighted combination of criteria |
| `criteria` | `dict` | Per-criterion score, reasoning, evidence |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: Complete Recommendation (Score: 0.95)</strong></summary>

!!! success "All Components Present"

    **Recommendation:**
    > "**Approve** this application. The business has excellent financials with $2.1M annual revenue, no prior claims in 5 years, and the building is well-maintained (constructed 2019). Proceed to bind the policy immediately."

    **Analysis:**

    | Criterion | Score | Finding |
    |-----------|-------|---------|
    | Decision | 1.0 | Clear "Approve" |
    | Rationale | 0.95 | Multiple factors explained |
    | Evidence | 0.90 | Specific data cited |
    | NextStep | 0.85 | Clear action |

    **Final Score:** `0.95` :material-check-all:

</details>

<details markdown="1">
<summary><strong>⚠️ Scenario 2: Weak Components (Score: 0.65)</strong></summary>

!!! warning "Missing Elements"

    **Recommendation:**
    > "Approve. Good risk."

    **Analysis:**

    | Criterion | Score | Finding |
    |-----------|-------|---------|
    | Decision | 1.0 | Clear "Approve" |
    | Rationale | 0.4 | Vague "good risk" |
    | Evidence | 0.2 | No specific data |
    | NextStep | 0.0 | No next steps |

    **Final Score:** `0.65` :material-alert:

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: No Decision (Score: 0.0)</strong></summary>

!!! failure "Hard Gate Triggered"

    **Recommendation:**
    > "The building is 10 years old with $500k revenue. There have been 2 claims in the past 3 years."

    **Analysis:**

    | Criterion | Score | Finding |
    |-----------|-------|---------|
    | Decision | 0.0 | No decision stated |
    | Hard Gate | Triggered | Score forced to 0.0 |

    **Final Score:** `0.0` :material-close:

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">📋</span>
<strong>Actionable Output</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Ensures AI recommendations can be acted upon by underwriters.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">📝</span>
<strong>Documentation</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Complete recommendations create an audit trail for compliance.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🎓</span>
<strong>Training Signal</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Helps identify where AI outputs need structural improvement.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Underwriting Completeness** = Does the recommendation have all required parts?

    - **Use it when:** Evaluating the structure of AI recommendations
    - **Score interpretation:** Higher = more complete recommendation
    - **Key feature:** Hard gate on missing decision

<div class="grid cards" markdown>

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Decision Quality](./decision_quality.md) · Underwriting Faithfulness

</div>
