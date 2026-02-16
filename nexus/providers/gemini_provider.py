"""
Gemini Provider — Google AI / Vertex AI
========================================
Uses the google-genai SDK.
Install: pip install google-genai
"""

from __future__ import annotations

from nexus.providers.base import BaseProvider, ProviderConfig, ProviderResponse


class GeminiProvider(BaseProvider):
    """Google Gemini AI provider."""

    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if not config.model:
            config.model = self.DEFAULT_MODEL
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.config.api_key)
            except ImportError:
                raise ImportError(
                    "Gemini provider requires 'google-genai'. "
                    "Install with: pip install google-genai"
                )
        return self._client

    def generate(self, prompt: str, system_instruction: str = "") -> ProviderResponse:
        try:
            from google.genai import types as genai_types

            client = self._get_client()
            gen_config = genai_types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )
            if system_instruction:
                gen_config.system_instruction = system_instruction

            response = client.models.generate_content(
                model=self.config.model,
                contents=prompt,
                config=gen_config,
            )

            return ProviderResponse(
                content=response.text or "",
                model=self.config.model,
                provider="gemini",
                raw_response=response,
                finish_reason="stop",
            )
        except Exception as e:
            return ProviderResponse(
                content="",
                model=self.config.model,
                provider="gemini",
                error=str(e),
            )

    def is_available(self) -> bool:
        return bool(self.config.api_key)
