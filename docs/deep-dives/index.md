---
hide:
  - toc
---

# Deep Dives

<div style="background: linear-gradient(135deg, #1E3A5F 0%, #0F2440 100%); padding: 24px; border-radius: 12px; color: white; margin: 20px 0;">

<p style="margin: 0; font-size: 16px; line-height: 1.6;">
<strong>System internals, database schema, and runtime flows.</strong> These guides go beyond the surface — covering how the evaluation pipeline is wired together, how data flows from ingestion to scoring, and how persistence is structured.
</p>
</div>

<div class="rule-grid">
  <div class="rule-card">
    <span class="rule-card__number">1</span>
    <p class="rule-card__title">Architecture</p>
    <p class="rule-card__desc">System diagram, runtime flows, design patterns, and the shared-core vs implementation boundary. The full picture of how Eval Workbench orchestrates evaluation pipelines.</p>
    <p><a href="architecture/" class="md-button md-button--primary" style="margin-top: 8px;">Explore</a></p>
  </div>
  <div class="rule-card">
    <span class="rule-card__number">2</span>
    <p class="rule-card__title">Database</p>
    <p class="rule-card__desc">Neon/PostgreSQL integration — synchronous and async connection managers, DataFrame I/O, connection pooling, and the QueueExecutor for concurrent task execution.</p>
    <p><a href="database/" class="md-button md-button--primary" style="margin-top: 8px;">Explore</a></p>
  </div>
  <div class="rule-card">
    <span class="rule-card__number">3</span>
    <p class="rule-card__title">Schema</p>
    <p class="rule-card__desc">The 3-stage evaluation data model — from raw input in <code>evaluation_dataset</code>, to metric scores in <code>evaluation_results</code>, to the unified <code>evaluation_view</code>.</p>
    <p><a href="schema/" class="md-button md-button--primary" style="margin-top: 8px;">Explore</a></p>
  </div>
</div>
