# Underwriting Rules

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Track referral triggers and validate outcome consistency</strong><br>
<span class="badge" style="margin-top: 0.5rem;">Hybrid</span>
<span class="badge" style="background: #667eea;">Rules</span>
<span class="badge" style="background: #6B7A3A;">Athena</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">🎯</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">0.0</code> ──────── <code style="font-size: 1.1rem;">1.0</code><br>
<small style="color: var(--md-text-muted);">Consistency score</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">⚡</span><br>
<strong>Default Threshold</strong><br>
<code style="font-size: 1.5rem; color: var(--md-primary);">1.0</code><br>
<small style="color: var(--md-text-muted);">Outcome must match triggers</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>actual_output</code> <code>additional_input</code><br>
<small style="color: var(--md-text-muted);">Recommendation + source data</small>
</div>

</div>

!!! abstract "What It Measures"
    Underwriting Rules tracks **referral triggers**, but it is now **scoped to explicit `Refer` outcomes**.

    - If the detected outcome is **not** `Refer` (e.g. Approved / Decline / Unknown), the metric **returns early** with `score=1.0` and does **not** run trigger detection.
    - If the detected outcome **is** `Refer`, the metric requires at least one trigger to be found; otherwise it is treated as `unknown_trigger`.

    | Score | Interpretation |
    |-------|----------------|
    | **1.0** | :material-check-all: Outcome is not `Refer` (skipped) **or** `Refer` with a trigger found |
    | **0.0** | :material-close: Outcome is `Refer` but **no** trigger could be detected (`unknown_trigger`) |

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Enforcing underwriting guidelines</li>
<li>Tracking referral reasons</li>
<li>Auditing decision consistency</li>
<li>Validating rule compliance</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>No structured input data</li>
<li>Rules don't apply</li>
<li>Evaluating reasoning quality</li>
<li>Non-underwriting decisions</li>
</ul>
</div>

</div>

---

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Detection Pipeline"

    The metric first detects the outcome label. Only **explicit `Refer`** outcomes proceed to trigger detection (structured checks → regex scan → LLM fallback).

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Inputs"]
            A[AI Recommendation]
            B[Source Data JSON]
        end

        subgraph OUTCOME["🎯 Step 1: Outcome Detection"]
            C[Detect AI Decision]
            D["Approve / Refer / Decline"]
        end

        subgraph GATE["🚦 Step 2: Scope Gate"]
            X{Outcome = Refer?}
            Y["Return early\nScore: 1.0"]
        end

        subgraph RULES["⚖️ Step 3: Trigger Detection"]
            E[Structured Checks]
            F[Regex Scan]
            G[LLM Fallback]
            H["Detected Triggers"]
        end

        subgraph VALIDATE["✓ Step 4: Refer Validation"]
            I{Triggers found?}
            K["Score: 1.0"]
            Z["Score: 0.0\nunknown_trigger"]
        end

        A --> C
        C --> D
        D --> X
        X -->|No| Y
        X -->|Yes| E
        B --> E
        A --> F
        E & F --> H
        H --> I
        I -->|Yes| K
        I -->|No triggers after structured/regex| G
        G --> H
        I -->|No triggers after fallback| Z

        style INPUT stroke:#8B9F4F,stroke-width:2px
        style OUTCOME stroke:#3b82f6,stroke-width:2px
        style GATE stroke:#64748b,stroke-width:2px
        style RULES stroke:#f59e0b,stroke-width:2px
        style VALIDATE stroke:#10b981,stroke-width:2px
        style K fill:#8B9F4F,stroke:#6B7A3A,stroke-width:3px,color:#fff
    ```

=== ":material-scale-balance: Scoring Logic"

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">✅ Score = 1.0</strong>
    <br><small>Outcome is not <code>Refer</code> (skipped), OR outcome is <code>Refer</code> and at least one trigger is detected.</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">❌ Score = 0.0</strong>
    <br><small>Outcome is <code>Refer</code> but no trigger could be detected (falls back to <code>unknown_trigger</code>).</small>
    </div>

    </div>

    !!! tip "Score Formula"
        ```
        if outcome_label != "Refer":
            score = 1.0
        else:
            score = 1.0 if bool(detected_triggers) else 0.0
        ```

</details>

---

## Referral Triggers

The metric detects the following referral triggers through structured data checks and regex patterns:

=== ":material-database: Structured Rules"

    | Trigger | Condition | Severity |
    |---------|-----------|----------|
    | **bppValue** | BPP limit > $250,000 | Hard |
    | **bppToSalesRatio** | BPP / sales < 10% | Soft |
    | **numberOfEmployees** | Employees > 20 | Soft |
    | **orgEstYear** | Business < 3 years + building coverage | Hard |
    | **nonOwnedBuildingCoverage** | Building coverage requested but not owned | Soft |
    | **homeBasedBPP** | Home-based + contents-only | Soft |
    | **claimsHistory** | Prior claims count > 0 | Hard |

=== ":material-regex: Regex Rules"

    **Hard Severity:**

    - `convStoreTemp` - Convenience/liquor/package store indicators (often tobacco/alcohol/lottery; sometimes 24/7 or fuel)
    - `claimsHistory` - Prior claims mentions
    - `orgEstYear` - New business + building coverage indicators
    - `bppValue` - Excessive BPP mentions

    **Soft Severity:**

    - `bppToSalesRatio` - Low ratio indicators
    - `nonOwnedBuildingCoverage` - Tenant building coverage / lease (including NNN/triple-net language)
    - `businessNOC` - Not Otherwise Classified
    - `homeBasedBPP` - Home-based business indicators
    - `numberOfEmployees` - High employee count (eligibility review)

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `recommendation_column_name` | `str` | `brief_recommendation` | Field in additional_output to analyze |

    !!! info "LLM Fallback"
        When a <strong>Refer</strong> outcome is detected but no triggers are found, the metric uses an LLM classifier to infer the closest trigger category.

        If the LLM cannot map to a known trigger, the metric records <code>unknown_trigger</code> and scores the case as <code>0.0</code>.

        The LLM classifier prompt is generated from the same `TRIGGER_SPECS` catalog used by regex detection so trigger descriptions stay in sync.

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.implementations.athena.metrics.recommendation.underwriting_rules import UnderwritingRules

    metric = UnderwritingRules()

    item = DatasetItem(
        actual_output="Recommend Refer due to high BPP coverage request.",
        additional_input={
            "context_data": {
                "auxData": {
                    "rateData": {
                        "output": {
                            "input": {
                                "bop_bpp_limit": 300000  # > $250k threshold
                            }
                        }
                    }
                }
            }
        }
    )

    result = await metric.execute(item)
    print(result.pretty())
    # Score: 1.0 (referral with trigger present)
    ```

=== ":material-cog-outline: Multiple Triggers"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.implementations.athena.metrics.recommendation.underwriting_rules import UnderwritingRules

    metric = UnderwritingRules()

    item = DatasetItem(
        actual_output="Refer - prior claims and new business.",
        additional_input={
            "bop_number_of_claims": 2,
            "bop_business_year_established": 2024,  # < 3 years
            "bop_insure_building": "building"
        }
    )

    result = await metric.execute(item)
    # Score: 1.0 (refer with multiple triggers)
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
<summary><strong>📊 UnderwritingRulesResult Structure</strong></summary>

```python
UnderwritingRulesResult(
{
    "score": 1.0,
    "is_referral": true,
    "outcome_label": "Referral",
    "primary_trigger": "bppValue",
    "detected_events": [
        {
            "trigger": "bppValue",
            "severity": "hard",
            "confidence": 0.95,
            "detection_method": "structured",
            "details": "BPP limit $300,000 exceeds $250,000 threshold"
        }
    ],
    "structured_values": {
        "bpp_limit": 300000,
        "gross_sales": 1500000
    }
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float` | 1.0 if consistent, 0.0 if not |
| `is_referral` | `bool` | Whether outcome is referral/decline |
| `outcome_label` | `str` | Normalized outcome label |
| `primary_trigger` | `str` | Most significant trigger |
| `detected_events` | `List` | All detected triggers with details |
| `structured_values` | `dict` | Extracted field values |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>✅ Scenario 1: Consistent Referral (Score: 1.0)</strong></summary>

!!! success "Referral Matches Triggers"

    **Recommendation:**
    > "Refer to underwriting - BPP coverage of $300,000 exceeds threshold."

    **Source Data:**
    ```json
    {"bop_bpp_limit": 300000}
    ```

    **Analysis:**

    | Component | Finding |
    |-----------|---------|
    | Outcome | Referral |
    | Trigger | bppValue (BPP > $250k) |
    | Match | ✅ Referral with trigger |

    **Final Score:** `1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>✅ Scenario 2: Non-Refer Outcome (Score: 1.0)</strong></summary>

!!! success "Skipped (Not in Scope)"

    **Recommendation:**
    > "Approve - all criteria within guidelines."

    **Source Data:**
    ```json
    {
        "bop_bpp_limit": 150000,
        "bop_number_of_claims": 0,
        "bop_number_of_employees": 10
    }
    ```

    **Analysis:**

    | Component | Finding |
    |-----------|---------|
    | Outcome | Approved (not `Refer`) |
    | Metric behavior | Returns early; does not run trigger detection |

    **Final Score:** `1.0` :material-check-all:

</details>

<details markdown="1">
<summary><strong>❌ Scenario 3: Refer with No Detected Trigger (Score: 0.0)</strong></summary>

!!! failure "Unknown Trigger"

    **Recommendation:**
    > "Refer to underwriting for review."

    **Source Data:**
    ```json
    {"bop_bpp_limit": 150000}
    ```

    **Analysis:**

    | Component | Finding |
    |-----------|---------|
    | Outcome | Refer |
    | Triggers | None detected |
    | Result | ❌ `unknown_trigger` |

    **Final Score:** `0.0` :material-close:

</details>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">📋</span>
<strong>Guideline Compliance</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Ensures AI follows established underwriting rules and thresholds.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔍</span>
<strong>Audit Trail</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Tracks exactly which triggers led to referral decisions.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">⚠️</span>
<strong>Risk Detection</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Catches <code>Refer</code> cases that can’t be mapped to a known trigger (<code>unknown_trigger</code>).</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Underwriting Rules** = For **`Refer`** outcomes, did we detect a valid referral trigger?

    - **Use it when:** Validating that AI follows underwriting guidelines
    - **Score interpretation:** 1.0 = not `Refer` (skipped) or `Refer` with trigger; 0.0 = `Refer` with no trigger (`unknown_trigger`)
    - **Key feature:** Multi-stage detection (structured + regex + LLM fallback)

<div class="grid cards" markdown>

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Decision Quality](./decision_quality.md) · Refer Reason

</div>
