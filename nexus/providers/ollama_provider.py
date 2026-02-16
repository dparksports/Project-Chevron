"""
Ollama Provider — Local LLMs
==============================
Uses HTTP requests to Ollama's REST API.
Install: https://ollama.com (no pip dependency needed)
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error

from nexus.providers.base import BaseProvider, ProviderConfig, ProviderResponse


class OllamaProvider(BaseProvider):
    """Ollama local LLM provider.

    Talks directly to Ollama's REST API — no external Python
    dependency required. Just install Ollama and pull a model.
    """

    DEFAULT_MODEL = "llama3.1"
    DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        if not config.model:
            config.model = self.DEFAULT_MODEL
        if not config.base_url:
            config.base_url = self.DEFAULT_BASE_URL

    def generate(self, prompt: str, system_instruction: str = "") -> ProviderResponse:
        try:
            url = f"{self.config.base_url}/api/generate"

            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            }
            if system_instruction:
                payload["system"] = system_instruction

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))

            return ProviderResponse(
                content=result.get("response", ""),
                model=self.config.model,
                provider="ollama",
                tokens_used=result.get("eval_count", 0) + result.get("prompt_eval_count", 0),
                prompt_tokens=result.get("prompt_eval_count", 0),
                completion_tokens=result.get("eval_count", 0),
                finish_reason="stop" if result.get("done") else "length",
                raw_response=result,
            )
        except urllib.error.URLError as e:
            return ProviderResponse(
                content="",
                model=self.config.model,
                provider="ollama",
                error=f"Cannot reach Ollama at {self.config.base_url}: {e}",
            )
        except Exception as e:
            return ProviderResponse(
                content="",
                model=self.config.model,
                provider="ollama",
                error=str(e),
            )

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            url = f"{self.config.base_url}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                result = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name", "") for m in result.get("models", [])]
            return any(self.config.model in m for m in models)
        except Exception:
            return False
