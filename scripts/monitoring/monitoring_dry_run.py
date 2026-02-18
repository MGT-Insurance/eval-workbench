#!/usr/bin/env python3
"""
Run a local dry run of OnlineMonitor from YAML config.

This mirrors `monitoring_entrypoint.py` config loading and monitor execution,
but forces `publish=False` so no DB upload and no Langfuse publishing happens.
"""

from __future__ import annotations

import argparse
import asyncio
import json
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


def _summarize_results(results: object) -> dict:
    summary: dict[str, object] = {}
    try:
        df = results.to_dataframe()  # type: ignore[attr-defined]
        summary['rows'] = len(df)
        summary['columns'] = list(df.columns)
        if 'metric_name' in df.columns:
            summary['metric_names'] = sorted(
                [str(v) for v in df['metric_name'].dropna().unique().tolist()]
            )
        if 'dataset_id' in df.columns:
            summary['dataset_ids'] = [str(v) for v in df['dataset_id'].head(10).tolist()]
    except Exception:
        summary['rows'] = 'unknown'
        summary['columns'] = []
    return summary


async def _run(config_path: Path, *, summary_only: bool) -> int:
    deduplicate = _env_bool('DEDUPLICATE', default=True)

    logger.info('Loading config: %s', config_path)
    logger.info('Deduplication: %s', 'enabled' if deduplicate else 'disabled')
    logger.info('Mode: dry-run (publish disabled)')

    monitor = OnlineMonitor.from_yaml(config_path)
    # Core dry-run guarantee: no DB upload and no Langfuse push/publish.
    results = await monitor.run_async(deduplicate=deduplicate, publish=False)

    if results is None:
        logger.info('No items processed')
        print(json.dumps({'status': 'ok', 'processed': 0, 'publish': False}, indent=2))
        return 0

    summary = _summarize_results(results)
    payload = {
        'status': 'ok',
        'publish': False,
        'summary': summary,
    }
    print(json.dumps(payload, indent=2))

    if not summary_only:
        try:
            print('\nSample results:')
            print(results.to_dataframe().head(5).to_string(index=False))
        except Exception:
            pass

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Run monitoring once in local dry-run mode (no publishing).'
    )
    parser.add_argument(
        'config_file',
        nargs='?',
        default='monitoring_langfuse.yaml',
        help='Config file path or name (resolved under src/eval_workbench/implementations/athena/config/)',
    )
    parser.add_argument(
        '--summary-only',
        action='store_true',
        help='Print JSON summary only (skip sample rows).',
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

    return asyncio.run(_run(config_path, summary_only=args.summary_only))


if __name__ == '__main__':
    raise SystemExit(main())

