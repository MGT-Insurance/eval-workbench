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

<div class="grid-container">

<div class="grid-item" style="border-left: 4px solid #10b981;">
<strong style="color: #10b981;">✅ Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Understanding referral patterns</li>
<li>Building analytics dashboards</li>
<li>Categorizing decline reasons</li>
<li>Training data analysis</li>
</ul>
</div>

<div class="grid-item" style="border-left: 4px solid #ef4444;">
<strong style="color: #ef4444;">❌ Don't Use When</strong>
<ul style="margin: 0.5rem 0 0 0; padding-left: 1.2rem;">
<li>Evaluating quality (use Decision Quality)</li>
<li>Checking rule compliance (use Underwriting Rules)</li>
<li>Approval outcomes</li>
<li>Scoring is required</li>
</ul>
</div>

</div>

---

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

    <div class="grid-container">

    <div class="grid-item" style="border-left: 4px solid #ef4444; padding-left: 1rem;">
    <strong style="color: #ef4444;">Claims History</strong>
    <br><small>Prior claims or loss history issues</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #f59e0b; padding-left: 1rem;">
    <strong style="color: #f59e0b;">BPP Value</strong>
    <br><small>Business personal property limits</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #3b82f6; padding-left: 1rem;">
    <strong style="color: #3b82f6;">New Business</strong>
    <br><small>Organization age concerns</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #8b5cf6; padding-left: 1rem;">
    <strong style="color: #8b5cf6;">Building Coverage</strong>
    <br><small>Non-owned building or coverage issues</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #10b981; padding-left: 1rem;">
    <strong style="color: #10b981;">Employee Count</strong>
    <br><small>High employee count concerns</small>
    </div>

    <div class="grid-item" style="border-left: 4px solid #6b7280; padding-left: 1rem;">
    <strong style="color: #6b7280;">Other</strong>
    <br><small>Classification, location, or other factors</small>
    </div>

    </div>

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
        actual_output="Decline due to prior claims history and high BPP coverage request."
    )

    result = await metric.execute(item)
    print(result.explanation)
    # "Claims History"

    print(result.signals.primary_category)
    # ReasonCategory.CLAIMS_HISTORY

    print(result.signals.all_reasons)
    # [{"category": "Claims History", ...}, {"category": "BPP Value", ...}]
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
<summary><strong>📊 ReferReasonResult Structure</strong></summary>

```python
ReferReasonResult(
{
    "is_negative_outcome": true,
    "outcome_label": "Referral",
    "primary_reason": {
        "category": "Claims History",
        "reasoning": "Prior claims mentioned as primary concern",
        "confidence": 0.92
    },
    "primary_category": "Claims History",
    "all_reasons": [
        {
            "category": "Claims History",
            "reasoning": "Prior claims mentioned",
            "confidence": 0.92
        },
        {
            "category": "BPP Value",
            "reasoning": "High BPP coverage request",
            "confidence": 0.85
        }
    ],
    "reason_count": 2,
    "actionable_type": "policy"
}
)
```

### Signal Fields

| Field | Type | Description |
|-------|------|-------------|
| `is_negative_outcome` | `bool` | Whether outcome is referral/decline |
| `outcome_label` | `str` | Referral, Decline, Approved, or Unknown |
| `primary_reason` | `dict` | Most significant reason details |
| `primary_category` | `str` | Category of primary reason |
| `all_reasons` | `List` | All extracted reasons |
| `reason_count` | `int` | Number of reasons detected |
| `actionable_type` | `str` | market, system, or policy |

</details>

---

## Example Scenarios

<details markdown="1">
<summary><strong>📊 Scenario 1: Single Clear Reason</strong></summary>

!!! info "Claims History"

    **Recommendation:**
    > "Decline - applicant has 3 claims in the past 2 years."

    **Analysis:**

    | Field | Value |
    |-------|-------|
    | Outcome | Decline |
    | Primary Category | Claims History |
    | Reason Count | 1 |
    | Actionable Type | policy |

    **Explanation:** `"Claims History"`

</details>

<details markdown="1">
<summary><strong>📊 Scenario 2: Multiple Reasons</strong></summary>

!!! info "Multiple Factors"

    **Recommendation:**
    > "Refer - new business (2024), high BPP ($350k), and 25 employees."

    **Analysis:**

    | Field | Value |
    |-------|-------|
    | Outcome | Referral |
    | Primary Category | New Business |
    | Reason Count | 3 |
    | All Reasons | New Business, BPP Value, Employee Count |
    | Actionable Type | system |

    **Explanation:** `"New Business"`

</details>

<details markdown="1">
<summary><strong>📊 Scenario 3: Approval (No Analysis)</strong></summary>

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

</details>

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
