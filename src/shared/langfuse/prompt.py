import logging
import re
import time
from functools import lru_cache
from typing import (
    Optional,
    List,
    Dict,
    Any,
    Union,
    Sequence,
    Iterable,
    Literal,
    Callable,
    Set,
)
from langfuse import Langfuse
from langfuse.model import PromptClient
from pydantic import Field

from shared.settings import RepoSettingsBase, build_settings_config

logger = logging.getLogger(__name__)


class LangfuseSettings(RepoSettingsBase):
    model_config = build_settings_config(from_path=__file__)

    langfuse_public_key: str | None = Field(default=None)
    langfuse_secret_key: str | None = Field(default=None)
    langfuse_host: str | None = Field(default=None)
    langfuse_default_label: str = Field(default="production")
    langfuse_default_cache_ttl_seconds: int = Field(default=60)
    langfuse_webhook_secret: str | None = Field(default=None)
    langfuse_webhook_notify_url: str | None = Field(default=None)
    langfuse_slack_channel_id: str | None = Field(default=None)
    langfuse_slack_request_timeout_seconds: float = Field(default=10)
    langfuse_slack_retry_max_attempts: int = Field(default=3)
    langfuse_slack_retry_backoff_seconds: float = Field(default=0.5)
    langfuse_slack_retry_max_backoff_seconds: float = Field(default=4.0)


@lru_cache(maxsize=1)
def get_langfuse_settings() -> LangfuseSettings:
    return LangfuseSettings()


class LangfusePromptManager:
    """
    A wrapper class to manage Langfuse prompts.

    Attributes:
        client (Langfuse): The authenticated Langfuse client.
    """

    def __init__(
        self,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        host: Optional[str] = None,
    ):
        """
        Initialize the Langfuse client.
        Auto-detects credentials from environment variables (LANGFUSE_PUBLIC_KEY, etc.)
        if arguments are not provided.
        """
        settings = get_langfuse_settings()
        public_key = public_key or settings.langfuse_public_key
        secret_key = secret_key or settings.langfuse_secret_key
        host = host or settings.langfuse_host

        if not public_key or not secret_key:
            raise ValueError(
                "Langfuse credentials missing. Set LANGFUSE_PUBLIC_KEY and "
                "LANGFUSE_SECRET_KEY or pass them explicitly."
            )

        self.client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        self.default_label = settings.langfuse_default_label
        self.default_cache_ttl_seconds = settings.langfuse_default_cache_ttl_seconds
        self._stale_prompts: Set[str] = set()
        self._prompt_change_listeners: List[
            Callable[[str, Dict[str, Any] | None], None]
        ] = []

    def mark_prompt_as_stale(self, prompt_name: str) -> None:
        """Flags a prompt to be re-fetched immediately on next use."""
        logger.info("Prompt invalidation requested for: %s", prompt_name)
        self._stale_prompts.add(prompt_name)

    def on_prompt_change(
        self,
        listener: Callable[[str, Dict[str, Any] | None], None],
    ) -> None:
        """Register a callback for prompt change events."""
        self._prompt_change_listeners.append(listener)

    def notify_prompt_change(
        self,
        prompt_name: str,
        payload: Dict[str, Any] | None = None,
    ) -> None:
        """Notify registered listeners about a prompt change."""
        for listener in self._prompt_change_listeners:
            try:
                listener(prompt_name, payload)
            except Exception as exc:
                logger.exception("Prompt change listener failed: %s", exc)

    def create_or_update_prompt(
        self,
        name: str,
        prompt_content: Union[str, List[Dict[str, str]]],
        prompt_type: str = "text",
        labels: Optional[Sequence[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> PromptClient:
        """
        Creates a new prompt or updates an existing one by creating a new version.

        Args:
            name: Unique name of the prompt.
            prompt_content: String for 'text' type, or list of messages (dicts) for 'chat' type.
            type: "text" or "chat".
            labels: List of labels (e.g., ["production", "staging"]).
            config: JSON config (e.g., model parameters like temperature).

        Returns:
            PromptClient: The created prompt object.
        """
        # Note: In Langfuse, 'creating' a prompt with an existing name
        # automatically adds it as a new version.
        return self.client.create_prompt(
            name=name,
            prompt=prompt_content,
            type=prompt_type,
            config=config or {},
            labels=list(labels) if labels else [],
        )

    def get_prompt(
        self,
        name: str,
        version: Optional[int] = None,
        label: Optional[str] = None,
        cache_ttl_seconds: Optional[int] = None,
        retry_count: int = 0,
        retry_backoff_seconds: float = 0.2,
    ) -> PromptClient:
        """
        Fetches a prompt. Defaults to the 'production' label if version/label are omitted.

        Args:
            name: Name of the prompt.
            version: Specific version number (int).
            label: Specific label (e.g., "staging").
            cache_ttl_seconds: How long to cache the prompt locally (default 60s).
        """
        # Langfuse logic: If version is set, use it. If label is set, use it.
        # If neither, it defaults to label="production".
        resolved_label = (
            label if label is not None else (None if version else self.default_label)
        )
        resolved_cache_ttl = (
            self.default_cache_ttl_seconds
            if cache_ttl_seconds is None
            else cache_ttl_seconds
        )
        if name in self._stale_prompts:
            logger.info("Fetching fresh version for stale prompt: %s", name)
            resolved_cache_ttl = 0
            self._stale_prompts.discard(name)
        last_error: Optional[Exception] = None
        for attempt in range(retry_count + 1):
            try:
                return self.client.get_prompt(
                    name=name,
                    version=version,
                    label=resolved_label,
                    cache_ttl_seconds=resolved_cache_ttl,
                )
            except Exception as exc:
                last_error = exc
                if attempt >= retry_count:
                    break
                time.sleep(retry_backoff_seconds * (2**attempt))
        raise last_error

    def _extract_template_strings(self, prompt_obj: PromptClient) -> List[str]:
        raw_prompt = getattr(prompt_obj, "prompt", None)
        if raw_prompt is None:
            return []

        strings: List[str] = []

        def _walk(value: Any) -> None:
            if isinstance(value, str):
                strings.append(value)
                return
            if isinstance(value, dict):
                for v in value.values():
                    _walk(v)
                return
            if isinstance(value, list):
                for v in value:
                    _walk(v)

        _walk(raw_prompt)
        return strings

    def _find_placeholders(self, template_strings: Iterable[str]) -> List[str]:
        placeholder_re = re.compile(r"\{\{\s*([a-zA-Z_][\w\.]*)\s*\}\}")
        found: List[str] = []
        for text in template_strings:
            found.extend(placeholder_re.findall(text))
        return sorted(set(found))

    def _find_unrendered_placeholders(
        self, compiled: Union[str, List[Dict[str, Any]]]
    ) -> List[str]:
        placeholder_re = re.compile(r"\{\{\s*([a-zA-Z_][\w\.]*)\s*\}\}")
        found: List[str] = []
        if isinstance(compiled, str):
            found.extend(placeholder_re.findall(compiled))
            return sorted(set(found))

        for message in compiled:
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    found.extend(placeholder_re.findall(content))
        return sorted(set(found))

    def get_compiled_prompt(
        self,
        name: str,
        variables: Dict[str, Any],
        *,
        fallback: Optional[Union[str, List[Dict[str, Any]]]] = None,
        strict: bool = True,
        required_variables: Optional[Sequence[str]] = None,
        strict_mode: Literal["template", "required", "none"] = "template",
        fallback_on_compile_error: bool = True,
        render_with: Optional[
            Callable[[Union[str, List[Dict[str, Any]]], Dict[str, Any]], Any]
        ] = None,
        **kwargs,
    ) -> Union[str, List[Dict]]:
        """
        Fetches the production prompt and compiles it with variables immediately.

        Args:
            name: Name of the prompt.
            variables: Dict of variables to replace in the template (e.g. {"name": "Alice"}).
            fallback: Returned when the Langfuse API call fails.
            strict: If True, verify all expected variables are present.
            required_variables: Optional explicit list of required variables.
            strict_mode: Choose required-variable enforcement strategy.
            fallback_on_compile_error: Use fallback if compile raises.
            render_with: Optional external renderer (Jinja/Liquid/etc). Receives raw prompt and variables.
            **kwargs: Arguments passed to get_prompt (version, label).

        Returns:
            The raw compiled string (text) or list of messages (chat).
        """
        try:
            prompt_obj = self.get_prompt(name, **kwargs)
        except Exception as exc:
            if fallback is not None:
                logger.warning("Langfuse get_prompt failed for '%s': %s", name, exc)
                return fallback
            raise

        if strict and strict_mode != "none":
            if strict_mode == "required" and required_variables is not None:
                expected = set(required_variables)
            else:
                expected = set(
                    self._find_placeholders(self._extract_template_strings(prompt_obj))
                )
            missing = expected.difference(variables.keys())
            if missing:
                raise ValueError(f"Missing prompt variables: {sorted(missing)}")

        # .compile() replaces {{ key }} with values from 'variables'
        try:
            if render_with is not None:
                compiled = render_with(prompt_obj.prompt, variables)
            else:
                compiled = prompt_obj.compile(**variables)
        except Exception as exc:
            if fallback is not None and fallback_on_compile_error:
                logger.warning("Langfuse compile failed for '%s': %s", name, exc)
                return fallback
            raise

        if strict and strict_mode == "template":
            leftover = self._find_unrendered_placeholders(compiled)
            if leftover:
                raise ValueError(f"Unrendered prompt variables: {leftover}")

        return compiled

    def get_template(
        self,
        name: str,
        *,
        version: Optional[int] = None,
        label: Optional[str] = None,
        cache_ttl_seconds: Optional[int] = None,
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Fetch raw template from Langfuse (for external templating).
        """
        prompt_obj = self.get_prompt(
            name=name,
            version=version,
            label=label,
            cache_ttl_seconds=cache_ttl_seconds,
        )
        return prompt_obj.prompt

    def promote_version(
        self,
        name: str,
        version: int,
        labels: Sequence[str],
    ) -> None:
        """
        Promotes a specific version by assigning it a label (e.g., 'production').
        This effectively rolls forward or rolls back.
        """
        self.client.update_prompt(
            name=name,
            version=version,
            new_labels=list(labels),
        )
        logger.info("Promoted '%s' (v%s) to labels=%s.", name, version, labels)

    def get_prompt_config(self, name: str, **kwargs) -> Dict[str, Any]:
        """
        Helper to retrieve just the configuration (model, temp, etc.) of a prompt.
        """
        prompt = self.get_prompt(name, **kwargs)
        return prompt.config
