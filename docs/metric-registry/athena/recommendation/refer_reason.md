# Refer Reason

<div style="border-left: 4px solid #8B9F4F; padding-left: 1rem; margin-bottom: 1.5rem;">
<strong style="font-size: 1.1rem;">Extract and categorize reasons for referral/decline outcomes</strong><br>
<span class="badge" style="margin-top: 0.5rem;">LLM-Powered</span>
<span class="badge" style="background: #667eea;">Analysis</span>
<span class="badge" style="background: #6B7A3A;">Athena</span>
</div>

## At a Glance

<div class="grid-container">

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📊</span><br>
<strong>Score Range</strong><br>
<code style="font-size: 1.1rem;">—</code><br>
<small style="color: var(--md-text-muted);">Analysis metric (no score)</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">⚡</span><br>
<strong>Default Threshold</strong><br>
<code style="font-size: 1.5rem; color: var(--md-primary);">—</code><br>
<small style="color: var(--md-text-muted);">Not applicable</small>
</div>

<div class="grid-item" style="text-align: center;">
<span style="font-size: 2rem;">📋</span><br>
<strong>Required Inputs</strong><br>
<code>actual_output</code><br>
<small style="color: var(--md-text-muted);">AI recommendation text</small>
</div>

</div>

!!! abstract "What It Measures"
    Refer Reason is an **analysis metric** that extracts and categorizes the reasons behind referral or decline outcomes. It identifies the primary reason category, all contributing reasons, and classifies the actionable type (market, system, or policy).

    | Output | Description |
    |--------|-------------|
    | `primary_category` | Main reason category (e.g., Claims History, BPP Value) |
    | `all_reasons` | Complete list of extracted reasons |
    | `actionable_type` | Classification: market, system, or policy |

<details markdown="1">
<summary><strong style="font-size: 1.1rem;">How It Works</strong></summary>

=== ":material-cog: Computation"

    The metric detects negative outcomes, then uses LLM extraction to identify and categorize all reasons.

    ```mermaid
    flowchart TD
        subgraph INPUT["📥 Input"]
            A[AI Recommendation]
        end

        subgraph DETECT["🔍 Step 1: Outcome Detection"]
            B[Check for Referral/Decline]
            C{Negative Outcome?}
        end

        subgraph EXTRACT["📝 Step 2: Reason Extraction"]
            D[LLM Analysis]
            E["Extract All Reasons"]
        end

        subgraph CATEGORIZE["🏷️ Step 3: Categorization"]
            F[Assign Categories]
            G[Determine Primary]
            H[Classify Actionable Type]
        end

        subgraph OUTPUT["📊 Output"]
            I["Structured Analysis"]
        end

        A --> B
        B --> C
        C -->|Yes| D
        C -->|No| I
        D --> E
        E --> F
        F --> G
        G --> H
        H --> I

        style INPUT stroke:#8B9F4F,stroke-width:2px
        style DETECT stroke:#3b82f6,stroke-width:2px
        style EXTRACT stroke:#f59e0b,stroke-width:2px
        style CATEGORIZE stroke:#8b5cf6,stroke-width:2px
        style OUTPUT stroke:#10b981,stroke-width:2px
        style I fill:#8B9F4F,stroke:#6B7A3A,stroke-width:3px,color:#fff
    ```

=== ":material-tag: Reason Categories"

    The metric uses 18 granular `ReasonCategory` values grouped by actionable type:

    **Coverage & Pricing**

    | Category | Description | Actionable |
    |----------|-------------|------------|
    | `Excessive Coverage Requested` | Coverage limits materially exceed typical thresholds | market |
    | `Inadequate / Suspicious Valuation` | Coverage amounts appear insufficient or implausible | market |
    | `Pricing Anomaly - Too Low` | Calculated premium appears unusually low | market |
    | `Pricing Anomaly - Too High` | Calculated premium appears unusually high | market |

    **Property & Location**

    | Category | Description | Actionable |
    |----------|-------------|------------|
    | `Property Condition Concerns` | Elevated risk from property condition | market |
    | `Construction / Exposure Threshold` | Property characteristics exceed thresholds | market |
    | `Location / CAT Risk` | Elevated catastrophe or environmental risk | market |

    **Data & Classification**

    | Category | Description | Actionable |
    |----------|-------------|------------|
    | `Data Conflict / Mismatch` | Discrepancies between customer and third-party data | system |
    | `Implausible / Invalid Data` | One or more inputs appear invalid | system |
    | `Missing / Unverifiable Data` | Required data cannot be verified | system |
    | `Industry / Class Code Error` | Business classification appears incorrect | policy |
    | `Unrelated / Ancillary Operations` | Operations outside primary stated class | policy |

    **Business & Operational**

    | Category | Description | Actionable |
    |----------|-------------|------------|
    | `Startup / New Venture` | Limited or no operating history | market |
    | `Multi-Location / Complex Ops` | Multiple locations requiring manual review | market |
    | `Financial / Operational Inconsistency` | Operational metrics are internally inconsistent | market |
    | `Ownership / Insurable Interest Issue` | Coverage for property without insurable interest | policy |

    **Compliance & Procedural**

    | Category | Description | Actionable |
    |----------|-------------|------------|
    | `Sanctions / Watchlists` | Potential regulatory or sanctions concern | policy |
    | `Procedural / Temporary Block` | Procedural rather than risk-based referral | system |
    | `Other / Unknown` | Does not fit a defined category | unknown |

</details>

---

## Configuration

=== ":material-tune: Parameters"

    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `recommendation_column_name` | `str` | `brief_recommendation` | Field in additional_output to analyze |
    | `max_source_lines` | `int` | `50` | Max source lines for context |

---

## Code Examples

=== ":material-play: Basic Usage"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.implementations.athena.metrics.recommendation.refer_reason import ReferReason

    metric = ReferReason()

    item = DatasetItem(
        actual_output="Refer to underwriting. Roof is 28 years old, exceeding 20-year threshold."
    )

    result = await metric.execute(item)
    print(result.explanation)
    # "Property Condition Concerns"

    print(result.signals.primary_category)
    # ReasonCategory.PROPERTY_CONDITION

    print(result.signals.all_reasons)
    # [ExtractedReason(reason_text="...", category=ReasonCategory.PROPERTY_CONDITION, reasoning="..."), ...]
    ```

=== ":material-cog-outline: Full Analysis"

    ```python
    from axion.dataset import DatasetItem
    from eval_workbench.implementations.athena.metrics.recommendation.refer_reason import ReferReason

    metric = ReferReason()

    item = DatasetItem(
        actual_output="""
        Refer to underwriting team.

        Reasons:
        - Business established in 2023 (less than 3 years)
        - BPP limit requested: $300,000 (exceeds threshold)
        - Home-based business requesting contents coverage
        """
    )

    result = await metric.execute(item)

    print(f"Outcome: {result.signals.outcome_label}")
    print(f"Primary Reason: {result.signals.primary_category}")
    print(f"Reason Count: {result.signals.reason_count}")
    print(f"Actionable Type: {result.signals.actionable_type}")

    for reason in result.signals.all_reasons:
        print(f"  - {reason['category']}: {reason['reasoning']}")
    ```

---

## Metric Diagnostics

Access detailed analysis results via `result.signals`.

```python
result = await metric.execute(item)
print(result.explanation)   # Primary category as explanation
result.signals              # Full analysis breakdown
```

<details markdown="1">
<summary><strong>📊 ReasonAnalysisResult Structure</strong></summary>

```python
ReasonAnalysisResult(
{
    "is_negative_outcome": true,
    "outcome_label": "Refer to Underwriter",
    "primary_reason": {
        "reason_text": "Roof is 28 years old, exceeding 20-year threshold",
        "category": "Property Condition Concerns",
        "reasoning": "Explicit mention of roof age exceeding policy threshold."
    },
    "primary_category": "Property Condition Concerns",
    "all_reasons": [
        {
            "reason_text": "Roof is 28 years old, exceeding 20-year threshold",
            "category": "Property Condition Concerns",
            "reasoning": "Explicit mention of roof age exceeding policy threshold."
        },
        {
            "reason_text": "Original knob-and-tube wiring from 1950s",
            "category": "Property Condition Concerns",
            "reasoning": "Outdated wiring type is a known fire hazard concern."
        }
    ],
    "reason_count": 2,
    "actionable_type": "market"
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `is_negative_outcome` | `bool` | Whether outcome is referral/decline |
| `outcome_label` | `str` | Refer to Underwriter, Decline, Approved, or Unknown |
| `primary_reason` | `ExtractedReason` | Most significant reason (`reason_text`, `category`, `reasoning`) |
| `primary_category` | `ReasonCategory` | Category enum of the primary reason |
| `all_reasons` | `List[ExtractedReason]` | All extracted reasons |
| `reason_count` | `int` | Number of reasons detected |
| `actionable_type` | `str` | market, system, policy, or unknown |

</details>

---

## Example Scenarios

=== "Single Reason"

    !!! info "Property Condition"

        **Recommendation:**
        > "Refer to underwriting - roof is 28 years old, exceeding 20-year threshold."

        **Analysis:**

        | Field | Value |
        |-------|-------|
        | Outcome | Refer to Underwriter |
        | Primary Category | Property Condition Concerns |
        | Reason Count | 1 |
        | Actionable Type | market |

        **Explanation:** `"Property Condition Concerns"`

=== "Multiple Reasons"

    !!! info "Multiple Factors"

        **Recommendation:**
        > "Refer - new business (2024), BPP limit of $3.5M appears excessive for $800k sales, and year built conflicts between customer data and enrichment."

        **Analysis:**

        | Field | Value |
        |-------|-------|
        | Outcome | Refer to Underwriter |
        | Primary Category | Excessive Coverage Requested |
        | Reason Count | 3 |
        | All Reasons | Excessive Coverage Requested, Data Conflict / Mismatch, Startup / New Venture |
        | Actionable Type | market |

        **Explanation:** `"Excessive Coverage Requested"`

=== "Approval (N/A)"

    !!! info "Not Applicable"

        **Recommendation:**
        > "Approve - all criteria within guidelines."

        **Analysis:**

        | Field | Value |
        |-------|-------|
        | Outcome | Approved |
        | Primary Category | None |
        | Reason Count | 0 |

        **Note:** Analysis only runs for negative outcomes.

---

## Actionable Types

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">🏪</span>
<strong>Market</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">External market conditions or factors outside control.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">⚙️</span>
<strong>System</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Configurable thresholds or rules that could be adjusted.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">📜</span>
<strong>Policy</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Fixed policy requirements or guidelines.</p>
</div>

</div>

---

## Why It Matters

<div class="grid-container">

<div class="grid-item">
<span style="font-size: 1.5rem;">📈</span>
<strong>Analytics</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Enables aggregation and trending of referral reasons.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🔍</span>
<strong>Root Cause</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Identifies patterns in why applications are declined.</p>
</div>

<div class="grid-item">
<span style="font-size: 1.5rem;">🎯</span>
<strong>Process Improvement</strong>
<p style="margin-top: 0.5rem; color: var(--md-text-secondary);">Helps identify where guidelines could be refined.</p>
</div>

</div>

---

## Quick Reference

!!! note "TL;DR"
    **Refer Reason** = Why did the AI refer or decline this application?

    - **Use it when:** You need to categorize and analyze referral/decline reasons
    - **Output type:** Analysis (no score)
    - **Key feature:** Structured extraction with category classification

<div class="grid cards" markdown>

- :material-link-variant: **Related Metrics**

    [:octicons-arrow-right-24: Underwriting Rules](./underwriting_rules.md) · Decision Quality

</div>
