---
hide:
  - toc
---

# Eval Workbench

<div style="background: linear-gradient(135deg, #1E3A5F 0%, #0F2440 100%); padding: 24px; border-radius: 12px; color: white; margin: 20px 0;">

<p style="margin: 0; font-size: 16px; line-height: 1.6;">
<strong>Evaluation tooling and metrics for MGT agent workflows.</strong> Built on top of <a href="https://github.com/ax-foundry/axion" style="color: #7BB8E0;">Axion</a>. Eval Workbench separates the core evaluation framework from individual implementations — enabling better tracking, custom sharable tooling, and a clear separation of concerns.
</p>
</div>

<div class="seatbelt-hero">
  <div class="seatbelt-hero__text">
    <span class="seatbelt-hero__label">Philosophy</span>
    <p class="seatbelt-hero__quote">
      Measure what matters.<br>
      <em>Automate the rest.</em>
    </p>
    <p class="seatbelt-hero__sub">
      Eval Workbench provides a structured approach to agent evaluation — from Slack conversation analysis to underwriting recommendation scoring. Define metrics once, run them everywhere.
    </p>
  </div>
  <div class="seatbelt-hero__visual">
    <img src="assets/mountains-light.svg" alt="Eval Workbench" loading="lazy">
  </div>
</div>

<div class="rule-grid">
  <div class="rule-card">
    <span class="rule-card__number">1</span>
    <p class="rule-card__title">Shared metric registry</p>
    <p class="rule-card__desc">Metrics organized by scope — shared Slack metrics, Athena recommendation metrics — all versioned, documented, and reusable.</p>
  </div>
  <div class="rule-card">
    <span class="rule-card__number">2</span>
    <p class="rule-card__title">YAML-driven monitoring</p>
    <p class="rule-card__desc">Production monitoring and alerting configured through simple YAML files. No code changes needed to adjust thresholds or add new checks.</p>
  </div>
  <div class="rule-card">
    <span class="rule-card__number">3</span>
    <p class="rule-card__title">Langfuse integration</p>
    <p class="rule-card__desc">Prompt management, trace collection, and webhook-driven cache invalidation with Slack notifications — all built in.</p>
  </div>
  <div class="rule-card">
    <span class="rule-card__number">4</span>
    <p class="rule-card__title">Separation of concerns</p>
    <p class="rule-card__desc">Core framework stays clean. Implementation-specific logic lives in its own space. Extend without coupling.</p>
  </div>
</div>

[Get started](getting-started/index.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/MGT-Insurance/eval-workbench){ .md-button }

---

## **Component Arsenal**

<table>
<tr>
<td width="50%" valign="top">

<h3><strong>Metric Registry</strong></h3>
<strong>Shared &amp; Scoped Evaluation Metrics</strong>

<p>Evaluation metrics organized by scope — shared Slack metrics, Athena recommendation metrics, and more. Versioned, documented, and reusable across implementations.</p>

</td>
<td width="50%" valign="top">

<h3><strong>Langfuse Integration</strong></h3>
<strong>Prompt Management &amp; Tracing</strong>

<p>Prompt management, trace collection, and webhook-driven cache invalidation with Slack notifications. Full observability for every evaluation run.</p>

</td>
</tr>
<tr>
<td width="50%" valign="top">

<h3><strong>Monitoring &amp; Alerting</strong></h3>
<strong>YAML-Driven Production Pipelines</strong>

<p>Production monitoring and alerting with YAML-driven configuration. Define thresholds, schedules, and notification channels without touching code.</p>

</td>
<td width="50%" valign="top">

<h3><strong>Slack Analytics</strong></h3>
<strong>Conversation Analysis &amp; KPI Tracking</strong>

<p>Multi-agent evaluation across Slack-based workflows — sentiment, escalation, resolution, compliance, and more. Every conversation scored automatically.</p>

</td>
</tr>
<tr>
<td width="50%" valign="top">

<h3><strong>Knowledge Graph Memory</strong></h3>
<strong>Persistent Decision Context</strong>

<p>Decision memory system for persisting and recalling evaluation context across sessions. Build institutional knowledge from every evaluation cycle.</p>

</td>
<td width="50%" valign="top">

<h3><strong>Architecture &amp; Deep Dives</strong></h3>
<strong>System Design, Schema &amp; Runtime Flows</strong>

<p>Comprehensive documentation of system internals — database schema, async pipelines, and the full evaluation lifecycle from ingestion to scoring.</p>

</td>
</tr>
</table>

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Evaluation framework | [Axion](https://github.com/ax-foundry/axion) |
| Prompt management | [Langfuse](https://langfuse.com) |
| Configuration | [Pydantic](https://docs.pydantic.dev) Settings |
| Database | Neon / PostgreSQL |
| Task orchestration | Python async |
| Documentation | MkDocs Material |

---

## Quick Start

Install the repo in editable mode so all imports resolve:

```bash
pip install -e .
```

Set up pre-commit hooks:

```bash
pre-commit install
pre-commit run --all-files
```

Then head to [Getting Started](getting-started/index.md) for environment setup and configuration details.
