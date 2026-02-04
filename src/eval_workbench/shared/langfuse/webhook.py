"""
IMPORTANT: THIS IS CLAUDE GENERATED CODE FOR TESTING PURPOSES.
DONT USE IT FOR PRODUCTION. NEEDS REFACTORING.
"""

import asyncio
import hashlib
import hmac
import json
import logging
from typing import Any, Dict, cast
from urllib import request as urllib_request

from fastapi import FastAPI, Header, HTTPException, Request, status

from eval_workbench.shared.langfuse.prompt import (
    LangfusePromptManager,
    get_langfuse_settings,
)
from eval_workbench.shared.slack.service import (
    PostMessageOptions,
    SlackBlockBuilder,
    SlackHttpClient,
    SlackService,
)

logger = logging.getLogger(__name__)

app = FastAPI()
settings = get_langfuse_settings()
prompt_manager = LangfusePromptManager()
slack_client = SlackHttpClient(
    timeout_seconds=settings.langfuse_slack_request_timeout_seconds,
    max_attempts=settings.langfuse_slack_retry_max_attempts,
    backoff_seconds=settings.langfuse_slack_retry_backoff_seconds,
    max_backoff_seconds=settings.langfuse_slack_retry_max_backoff_seconds,
)
slack_service = SlackService(client=slack_client)


def _parse_signature_header(signature_header: str) -> tuple[str | None, str]:
    if 'v1=' in signature_header:
        parts: Dict[str, str] = {}
        for part in signature_header.split(','):
            part = part.strip()
            if '=' in part:
                key, value = part.split('=', 1)
                parts[key.strip()] = value.strip()
        return parts.get('t'), parts.get('v1', '').strip()
    return None, signature_header.strip()


def verify_signature(payload: bytes, signature_header: str | None, secret: str) -> None:
    if not signature_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail='Missing signature'
        )

    timestamp, provided_signature = _parse_signature_header(signature_header)
    if not provided_signature:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail='Invalid signature'
        )

    candidates = [
        hmac.new(
            key=secret.encode('utf-8'),
            msg=payload,
            digestmod=hashlib.sha256,
        ).hexdigest()
    ]
    if timestamp:
        signed_payload = f'{timestamp}.'.encode('utf-8') + payload
        candidates.append(
            hmac.new(
                key=secret.encode('utf-8'),
                msg=signed_payload,
                digestmod=hashlib.sha256,
            ).hexdigest()
        )

    if not any(
        hmac.compare_digest(candidate, provided_signature) for candidate in candidates
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail='Invalid signature'
        )


def _extract_prompt_name(payload: Dict[str, Any]) -> str | None:
    if 'name' in payload:
        return payload.get('name')
    if 'prompt' in payload and isinstance(payload['prompt'], dict):
        return payload['prompt'].get('name')
    return None


def _notify_external_listener(prompt_name: str, payload: Dict[str, Any]) -> None:
    notify_url = settings.langfuse_webhook_notify_url
    if not notify_url:
        return
    body = json.dumps(
        {
            'event': 'prompt.changed',
            'promptName': prompt_name,
            'payload': payload,
        }
    ).encode('utf-8')
    req = urllib_request.Request(
        notify_url,
        data=body,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urllib_request.urlopen(req, timeout=5) as response:
            logger.info('Prompt change notify status: %s', response.status)
    except Exception as exc:
        logger.warning('Prompt change notify failed: %s', exc)


async def _post_slack_alert(event_type: str, payload: Dict[str, Any]) -> None:
    channel_id = settings.langfuse_slack_channel_id
    if not channel_id:
        return
    prompt_name = _extract_prompt_name(payload)
    if not prompt_name:
        return
    prompt_version = payload.get('promptVersion')
    prompt_id = payload.get('promptId')
    prompt_url = payload.get('promptUrl') or payload.get('url')

    message_options = cast(
        PostMessageOptions,
        SlackBlockBuilder.format_prompt_change_alert(
            prompt_name=prompt_name,
            event_type=event_type,
            prompt_version=prompt_version,
            prompt_id=prompt_id,
            prompt_url=prompt_url,
        ),
    )
    result = await slack_service.post_message(
        channel_id,
        message_options,
        agent_id='athena',
    )
    if not result.get('success'):
        logger.warning('Slack alert failed: %s', result.get('error'))


async def _post_slack_alert_safe(event_type: str, payload: Dict[str, Any]) -> None:
    try:
        await _post_slack_alert(event_type, payload)
    except Exception as exc:
        logger.exception('Slack alert task failed: %s', exc)


@app.post('/webhooks/langfuse')
async def langfuse_webhook(
    request: Request,
    x_langfuse_signature: str | None = Header(
        default=None, alias='X-Langfuse-Signature'
    ),
) -> Dict[str, str]:
    secret = settings.langfuse_webhook_secret
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail='LANGFUSE_WEBHOOK_SECRET is not configured',
        )

    payload_bytes = await request.body()
    verify_signature(payload_bytes, x_langfuse_signature, secret)

    try:
        data = json.loads(payload_bytes)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail='Invalid JSON'
        ) from exc

    event_type = data.get('event')
    payload = data.get('data') or {}

    if event_type in {'prompt.created', 'prompt.updated', 'prompt.deleted'}:
        prompt_name = _extract_prompt_name(payload)
        if prompt_name:
            logger.info("Webhook received: %s for prompt '%s'", event_type, prompt_name)
            prompt_manager.mark_prompt_as_stale(prompt_name)
            prompt_manager.notify_prompt_change(prompt_name, payload)
            _notify_external_listener(prompt_name, payload)
            asyncio.create_task(_post_slack_alert_safe(event_type, payload))
            return {'status': 'success', 'message': f'Invalidated {prompt_name}'}

    return {'status': 'ignored', 'message': 'Event type not handled'}
