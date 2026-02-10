---
hide:
  - toc
---

# Eval Workbench

**Evaluation tooling and metrics for MGT agent workflows.**

Built on top of [Axion](https://github.com/ax-foundry/axion) and other evaluation modules, Eval Workbench separates the core evaluation framework from individual implementations — enabling better tracking, custom sharable tooling, and a clear separation of concerns.

[Get started](getting-started/index.md){ .md-button .md-button--primary }
[View on GitHub](https://github.com/MGT-Insurance/eval-workbench){ .md-button }

---

## Features

<div class="feature-grid" markdown>

<div class="feature-card" markdown>

:material-chart-tree: **Metric Registry**

Evaluation metrics organized by scope — shared Slack metrics, Athena recommendation metrics, and more.

</div>

<div class="feature-card" markdown>

:material-connection: **Langfuse Integration**

Prompt management, trace collection, and webhook-driven cache invalidation with Slack notifications.

</div>

<div class="feature-card" markdown>

:material-monitor-dashboard: **Monitoring**

Production monitoring and alerting with YAML-driven configuration for evaluation pipelines.

</div>

<div class="feature-card" markdown>

:material-slack: **Slack Analytics**

Conversation analysis, KPI tracking, and multi-agent evaluation across Slack-based workflows.

</div>

<div class="feature-card" markdown>

:material-brain: **Knowledge Graph Memory**

Decision memory system for persisting and recalling evaluation context across sessions.

</div>

<div class="feature-card" markdown>

:material-layers-outline: **Architecture**

Deep dives into system design, database schema, and runtime flows.

</div>

</div>

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
