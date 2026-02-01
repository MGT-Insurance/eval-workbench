## UnderwritingRules Metric

This document describes the `UnderwritingRules` metric in
`src/implementations/athena/metrics/recommendation/underwriting_rules.py`,
including detection logic and exact formulas used for each rule.

### Purpose

Track underwriting referral triggers in agent output and source data, and flag
whether the agent outcome (approve vs refer/decline) is consistent with those
triggers.

### Inputs

- `DatasetItem.actual_output` via `get_field("actual_output")`
- `DatasetItem.additional_output[recommendation_column_name]`
  - Default column name: `brief_recommendation`
- `DatasetItem.additional_input` (structured data, flattened)

### Outcome Detection

Outcome is derived from `detect_outcome(actual_output, variant="underwriting_rules")`:

- `is_referral`: `True` when the outcome is referral/decline
- `outcome_label`: normalized label for reporting

### Detection Pipeline

1. Structured checks on `additional_input` (deterministic rules)
2. Regex scan on `full_text` (recommendation text)
3. Deduplication per trigger (highest confidence kept)
4. LLM fallback if referral/decline and no triggers detected

### Structured Checks (Exact Formulas)

Structured fields are extracted from `additional_input` by flattening JSON and
looking up these keys (first match wins, suffix matches allowed):

- `bpp_limit` keys:
  - `context_data.auxData.rateData.output.input.bop_bpp_limit`
  - `context_data.auxData.rateData.output.bop_bpp_limit`
  - `bop_bpp_limit`
- `gross_sales` keys:
  - `context_data.auxData.rateData.output.input.bop_gross_sales`
  - `context_data.auxData.rateData.output.bop_gross_sales`
  - `bop_gross_sales`
- `num_employees` keys:
  - `context_data.auxData.rateData.output.input.bop_number_of_employees`
  - `context_data.auxData.rateData.output.bop_number_of_employees`
  - `bop_number_of_employees`
- `year_established` keys:
  - `context_data.auxData.rateData.output.input.bop_business_year_established`
  - `context_data.auxData.rateData.output.bop_business_year_established`
  - `bop_business_year_established`
- `claims_count` keys:
  - `context_data.auxData.rateData.output.input.bop_number_of_claims`
  - `context_data.auxData.rateData.output.bop_number_of_claims`
  - `bop_number_of_claims`
- `home_based` keys:
  - `context_data.auxData.rateData.output.input.bop_home_based_business`
  - `context_data.auxData.rateData.output.bop_home_based_business`
  - `bop_home_based_business`
- `building_owned` keys:
  - `context_data.auxData.rateData.output.input.bop_building_owned`
  - `context_data.auxData.rateData.output.bop_building_owned`
  - `bop_building_owned`
- `insure_building` keys:
  - `context_data.auxData.rateData.output.input.bop_insure_buildings`
  - `context_data.auxData.rateData.output.input.bop_insure_building`
  - `context_data.auxData.rateData.output.bop_insure_buildings`
  - `context_data.auxData.rateData.output.bop_insure_building`
  - `bop_insure_buildings`
  - `bop_insure_building`

Field parsing logic:

- `parse_number`: first numeric value in the string (commas ignored)
- `parse_bool`: `true/yes/1` -> `True`, `false/no/0` -> `False`
- `parse_building_coverage(insure_building)`:
  - returns `False` if text contains `contents only` or equals `contents`
  - returns `True` if text contains `building`
  - otherwise returns `None`

Derived flags:

- `building_coverage_requested = parse_building_coverage(insure_building)`
- `contents_only = None if building_coverage_requested is None else (not building_coverage_requested)`

Structured rule formulas (exact):

- `bppValue`:
  - Trigger when `bpp_limit is not None and bpp_limit > 250000`
- `bppToSalesRatio`:
  - Trigger when `bpp_limit is not None and gross_sales is not None and gross_sales > 0`
  - Ratio formula: `bpp_limit / gross_sales`
  - Trigger when `ratio < 0.10`
- `numberOfEmployees`:
  - Trigger when `num_employees is not None and num_employees > 20`
- `orgEstYear`:
  - Trigger when:
    - `year_established is not None`
    - `building_coverage_requested is True`
    - `(current_utc_year - int(year_established)) < 3`
- `nonOwnedBuildingCoverage`:
  - Trigger when `building_coverage_requested is True and building_owned is False`
- `homeBasedBPP`:
  - Trigger when `home_based is True and contents_only is True`
- `claimsHistory`:
  - Trigger when `claims_count is not None and claims_count > 0`

### Regex Rules (Text Scan)

Each trigger also has regex patterns used against the recommendation text.
If any pattern matches, the trigger fires with `DetectionMethod.REGEX`.

Hard severity:

- `convStoreTemp`:
  - `convStoreTemp`
  - `Convenience Store.*Rule`
  - `rule.*9321`
  - `class.*CONVGAS`
  - `7[-\s]?eleven|circle\s?k|am\s?pm|wawa|sheetz`
  - `(tobacco|liquor|alcohol|beer|wine|lottery).*sales?`
  - `gas\s?station|fuel\s?sales?`
- `claimsHistory`:
  - `claimsHistory`
  - `prior\s+claim`
  - `loss\s+history`
  - `previous\s+(claim|loss)`
  - `claim(s)?\s+(in|over)\s+the\s+(past|last)`
- `orgEstYear`:
  - `orgEstYear`
  - `established.*202[3-9]`
  - `incorporated.*202[3-9]`
  - `business.*<\s*3\s*years?`
  - `new\s+organization`
  - `(founded|started|opened).*202[3-9]`
- `bppValue`:
  - `bppValue`
  - `contents.*>\s*\$?250[,.]?000`
  - `BPP.*exceeds?\s*\$?250`
  - `personal\s+property.*250`

Soft severity:

- `bppToSalesRatio`:
  - `bppToSalesRatio`
  - `contents.*sales.*ratio`
  - `<\s*10\s*%.*ratio`
  - `BPP.*to.*sales.*low`
  - `ratio.*contents.*revenue`
- `nonOwnedBuildingCoverage`:
  - `nonOwnedBuildingCoverage`
  - `tenant.*building\s+coverage`
  - `leased.*building\s+limit`
  - `renter.*requesting.*building`
- `businessNOC`:
  - `businessNOC`
  - `Not\s+Otherwise\s+Classified`
  - `classification\s+mismatch`
  - `NOC\s+class`
  - `unclear\s+business\s+type`
- `homeBasedBPP`:
  - `homeBasedBPP`
  - `residential.*location`
  - `home[-\s]?based\s+business`
  - `operates?\s+from\s+home`
- `numberOfEmployees`:
  - `numberOfEmployees`
  - `employee\s+count.*>\s*20`
  - `more\s+than\s+20\s+employees`
  - `exceeds?\s+employee\s+limit`

### LLM Fallback (Ghost Referral)

If `is_referral` is `True` and no triggers were detected, the metric calls
`GhostReferralClassifier` to infer the closest trigger. If the classifier
returns `unknown_trigger`, no event is added.

LLM fallback confidence is set to `0.8` and detection method is `llm_fallback`.

### Primary Trigger Selection

Primary trigger is selected by:

1. Lowest `priority` (1 = highest)
2. Highest `confidence` (secondary tie-break)

If `is_referral` is `True` and the primary trigger is `none`, it is forced to
`unknown_trigger`.

### Score Formula

```
score = 1.0 if (is_referral == bool(detected_events)) else 0.0
```

Meaning:

- Referral/decline with at least one trigger -> score 1.0
- Approval/unknown with no triggers -> score 1.0
- Any mismatch -> score 0.0
