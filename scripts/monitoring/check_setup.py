#!/usr/bin/env python3
"""
Check monitoring setup (config + env vars) without running monitoring.

This script is meant for quick validation locally or in CI:
- Resolves and loads the YAML config (with ${ENV_VAR} interpolation)
- Checks for unresolved ${...} placeholders
- Infers required env vars from source/publishing/metrics_config
- Verifies the config can be constructed into an OnlineMonitor

Usage:
  python scripts/monitoring/check_setup.py monitoring_langfuse.yaml
  python scripts/monitoring/check_setup.py monitoring_slack.yaml -v
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Iterable

from eval_workbench.shared import config as cfg_loader
from eval_workbench.shared.monitoring import OnlineMonitor

logger = logging.getLogger(__name__)


def _resolve_config_path(config_file: str) -> Path:
    candidate = Path(config_file)
    if candidate.exists():
        return candidate

    config_name = config_file
    if not config_name.endswith((".yaml", ".yml")):
        config_name = f"{config_name}.yaml"

    repo_default = (
        Path("src") / "eval_workbench" / "implementations" / "athena" / "config" / config_name
    )
    if repo_default.exists():
        return repo_default

    raise FileNotFoundError(
        f"Config file not found: {config_file!r}. Tried '{repo_default}'."
    )


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, dict):
        for v in value.values():
            yield from _walk_strings(v)
        return
    if isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
        return


def _mask_env(name: str) -> str:
    if name not in os.environ:
        return "MISSING"
    raw = os.environ.get(name) or ""
    if not raw:
        return "SET(empty)"
    if len(raw) <= 6:
        return "SET(****)"
    return f"SET({raw[:2]}…{raw[-2:]})"


def _providers_from_metrics(metrics_cfg: dict[str, Any]) -> set[str]:
    providers: set[str] = set()
    for metric_cfg in metrics_cfg.values():
        if isinstance(metric_cfg, dict):
            provider = metric_cfg.get("llm_provider")
            if isinstance(provider, str) and provider.strip():
                providers.add(provider.strip().lower())
    return providers


def _has_any_truthy(cfg: dict[str, Any], dotted_key: str) -> bool:
    value = cfg_loader.get(dotted_key, cfg=cfg)
    return bool(value)


def _get_optional_str(cfg: dict[str, Any], dotted_key: str) -> str | None:
    value = cfg_loader.get(dotted_key, cfg=cfg)
    return value if isinstance(value, str) and value.strip() else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check monitoring config + env setup")
    parser.add_argument(
        "config_file",
        nargs="?",
        default="monitoring_langfuse.yaml",
        help="Config file path or name (resolved under src/eval_workbench/implementations/athena/config/)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        config_path = _resolve_config_path(args.config_file)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 2

    cfg = cfg_loader.load_config(config_path)

    logger.info("Config file: %s", config_path)
    logger.info("name=%r version=%r", cfg_loader.get("name", cfg=cfg), cfg_loader.get("version", cfg=cfg))

    source_type = cfg_loader.get("source.type", cfg=cfg) or "langfuse"
    logger.info("source.type=%r source.name=%r", source_type, cfg_loader.get("source.name", cfg=cfg))
    logger.info(
        "publishing: push_to_db=%s push_to_langfuse=%s experiment.enabled=%s",
        bool(cfg_loader.get("publishing.push_to_db", cfg=cfg)),
        bool(cfg_loader.get("publishing.push_to_langfuse", cfg=cfg)),
        bool(cfg_loader.get("publishing.experiment.enabled", default=False, cfg=cfg)),
    )

    # Report unresolved ${...} placeholders (env interpolation keeps original token if missing).
    unresolved = sorted({s for s in _walk_strings(cfg) if "${" in s})
    if unresolved:
        logger.warning("Found unresolved ${...} placeholders (env vars likely missing):")
        for s in unresolved[:50]:
            logger.warning("  %s", s)
        if len(unresolved) > 50:
            logger.warning("  ... and %d more", len(unresolved) - 50)

    required_envs: list[str] = []

    if str(source_type).lower() == "langfuse":
        required_envs += ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY"]

    if str(source_type).lower() == "slack":
        required_envs += ["SLACK_BOT_TOKEN"]

    if str(source_type).lower() == "neon":
        # NeonDataSource can accept connection_string in YAML, otherwise falls back to DATABASE_URL.
        if not _get_optional_str(cfg, "source.connection_string"):
            required_envs += ["DATABASE_URL"]

    # DB dedup store needs DB as well (unless connection string explicitly provided).
    if (cfg_loader.get("scored_store.type", cfg=cfg) in ("db", "database")) and not _get_optional_str(
        cfg, "scored_store.connection_string"
    ):
        required_envs += ["DATABASE_URL"]

    # Publishing to DB requires DB as well (unless connection string explicitly provided).
    if _has_any_truthy(cfg, "publishing.push_to_db") and not _get_optional_str(
        cfg, "publishing.database.connection_string"
    ):
        required_envs += ["DATABASE_URL"]

    metrics_cfg = cfg_loader.get("metrics_config", default={}, cfg=cfg) or {}
    providers = _providers_from_metrics(metrics_cfg if isinstance(metrics_cfg, dict) else {})
    if "openai" in providers:
        required_envs += ["OPENAI_API_KEY"]
    if "anthropic" in providers:
        required_envs += ["ANTHROPIC_API_KEY"]

    required_envs = sorted(set(required_envs))
    if required_envs:
        logger.info("Env var check (masked):")
        missing = []
        for name in required_envs:
            status = _mask_env(name)
            logger.info("  %s=%s", name, status)
            if status == "MISSING":
                missing.append(name)
    else:
        missing = []
        logger.info("No required env vars inferred from config")

    # Validate config can be constructed (no network calls here).
    try:
        _ = OnlineMonitor.from_yaml(config_path)
        logger.info("OnlineMonitor.from_yaml: OK")
    except Exception as e:
        logger.error("OnlineMonitor.from_yaml failed: %s", e, exc_info=args.verbose)
        return 1

    if missing or unresolved:
        logger.warning("Setup check completed with issues")
        return 1

    logger.info("Setup check: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

