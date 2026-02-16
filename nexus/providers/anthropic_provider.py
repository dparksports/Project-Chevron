"""
Anthropic Provider — Claude 3.5, etc.
=======================================
Uses the anthropic SDK.
Install: pip install anthropic
"""

from __future__ import annotations

from nexus.providers.base import BaseProvider, ProviderConfig, ProviderResponse


class AnthropicProvider(BaseProvider):
    """Anthropic AI provider (Claude models)."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if not config.model:
            config.model = self.DEFAULT_MODEL
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                kwargs = {"api_key": self.config.api_key}
                if self.config.base_url:
                    kwargs["base_url"] = self.config.base_url
                self._client = anthropic.Anthropic(**kwargs)
            except ImportError:
                raise ImportError(
                    "Anthropic provider requires 'anthropic'. "
                    "Install with: pip install anthropic"
                )
        return self._client

    def generate(self, prompt: str, system_instruction: str = "") -> ProviderResponse:
        try:
            client = self._get_client()

            kwargs = {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_instruction:
                kwargs["system"] = system_instruction
            if self.config.temperature is not None:
                kwargs["temperature"] = self.config.temperature

            response = client.messages.create(**kwargs)

            content = ""
            if response.content:
                content = "".join(
                    block.text for block in response.content
                    if hasattr(block, "text")
                )

            return ProviderResponse(
                content=content,
                model=self.config.model,
                provider="anthropic",
                tokens_used=(response.usage.input_tokens + response.usage.output_tokens
                             if response.usage else 0),
                prompt_tokens=(response.usage.input_tokens if response.usage else 0),
                completion_tokens=(response.usage.output_tokens if response.usage else 0),
                finish_reason=response.stop_reason or "stop",
                raw_response=response,
            )
        except Exception as e:
            return ProviderResponse(
                content="",
                model=self.config.model,
                provider="anthropic",
                error=str(e),
            )

    def is_available(self) -> bool:
        return bool(self.config.api_key)
