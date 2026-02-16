"""
OpenAI Provider — GPT-4o, o1, etc.
====================================
Uses the openai SDK.
Install: pip install openai
"""

from __future__ import annotations

from nexus.providers.base import BaseProvider, ProviderConfig, ProviderResponse


class OpenAIProvider(BaseProvider):
    """OpenAI AI provider (GPT-4o, o1, etc.)."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if not config.model:
            config.model = self.DEFAULT_MODEL
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai
                kwargs = {"api_key": self.config.api_key}
                if self.config.base_url:
                    kwargs["base_url"] = self.config.base_url
                self._client = openai.OpenAI(**kwargs)
            except ImportError:
                raise ImportError(
                    "OpenAI provider requires 'openai'. "
                    "Install with: pip install openai"
                )
        return self._client

    def generate(self, prompt: str, system_instruction: str = "") -> ProviderResponse:
        try:
            client = self._get_client()

            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            choice = response.choices[0] if response.choices else None
            usage = response.usage

            return ProviderResponse(
                content=choice.message.content if choice else "",
                model=self.config.model,
                provider="openai",
                tokens_used=(usage.total_tokens if usage else 0),
                prompt_tokens=(usage.prompt_tokens if usage else 0),
                completion_tokens=(usage.completion_tokens if usage else 0),
                finish_reason=(choice.finish_reason if choice else "error"),
                raw_response=response,
            )
        except Exception as e:
            return ProviderResponse(
                content="",
                model=self.config.model,
                provider="openai",
                error=str(e),
            )

    def is_available(self) -> bool:
        return bool(self.config.api_key)
