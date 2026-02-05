#!/usr/bin/env python3
"""
Run a single OnlineMonitor pass from a YAML config and exit.

Designed for GitHub Actions cron runs (run-once per schedule).

Usage:
  python scripts/monitoring/monitoring_entrypoint.py monitoring_langfuse.yaml

Environment variables:
  - DEDUPLICATE: 'true'/'false' to enable deduplication (default: true)
  - ENVIRONMENT and any secrets (OPENAI_API_KEY, ANTHROPIC_API_KEY, LANGFUSE_*, DATABASE_URL, ...)
    are read implicitly by the YAML config interpolation and underlying clients.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from pathlib import Path

from eval_workbench.shared.monitoring import OnlineMonitor

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {'0', 'false', 'no', 'off', ''}


def _resolve_config_path(config_file: str) -> Path:
    candidate = Path(config_file)
    if candidate.exists():
        return candidate

    # Allow passing without extension.
    config_name = config_file
    if not config_name.endswith(('.yaml', '.yml')):
        config_name = f'{config_name}.yaml'

    repo_default = (
        Path('src')
        / 'eval_workbench'
        / 'implementations'
        / 'athena'
        / 'config'
        / config_name
    )
    if repo_default.exists():
        return repo_default

    raise FileNotFoundError(
        f"Config file not found: {config_file!r}. Tried '{repo_default}'."
    )


async def _run(config_path: Path) -> int:
    deduplicate = _env_bool('DEDUPLICATE', default=True)

    logger.info('Loading config: %s', config_path)
    logger.info('Deduplication: %s', 'enabled' if deduplicate else 'disabled')

    monitor = OnlineMonitor.from_yaml(config_path)
    results = await monitor.run_async(deduplicate=deduplicate, publish=True)

    if results is None:
        logger.info('No items to process')
        return 0

    try:
        df = results.to_dataframe()
        logger.info('Evaluated %s items', len(df))
    except Exception:
        logger.info('Monitoring run completed')

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description='Run monitoring once from YAML config')
    parser.add_argument(
        'config_file',
        nargs='?',
        default='monitoring_langfuse.yaml',
        help='Config file path or name (resolved under src/eval_workbench/implementations/athena/config/)',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )

    try:
        config_path = _resolve_config_path(args.config_file)
    except FileNotFoundError as e:
        logger.error('%s', e)
        return 2

    return asyncio.run(_run(config_path))


if __name__ == '__main__':
    raise SystemExit(main())

