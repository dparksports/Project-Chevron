"""
AI Provider Base — Abstract Interface
=======================================
Provider-agnostic interface for AI code generation.
All providers (Gemini, OpenAI, Anthropic, Ollama) implement this.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class ProviderConfig:
    """Configuration for an AI provider.

    Designed to work with any provider — only populate the fields
    that apply to your chosen backend.
    """

    provider_name: str          # "gemini", "openai", "anthropic", "ollama"
    model: str = ""             # Model name (e.g., "gemini-2.0-flash", "gpt-4o")
    api_key: str = ""           # API key (not needed for Ollama)
    base_url: str = ""          # Custom endpoint (for Ollama, proxies, etc.)
    temperature: float = 0.1   # Low temp for deterministic code generation
    max_tokens: int = 4096      # Maximum response tokens
    extra: dict[str, Any] = field(default_factory=dict)  # Provider-specific options


@dataclass
class ProviderResponse:
    """Standardized response from any AI provider."""

    content: str                # The generated text/code
    model: str = ""             # Which model was actually used
    provider: str = ""          # Which provider backend
    tokens_used: int = 0        # Total tokens consumed
    prompt_tokens: int = 0      # Input tokens
    completion_tokens: int = 0  # Output tokens
    finish_reason: str = ""     # "stop", "length", "error", etc.
    raw_response: Any = None    # The raw provider response object
    error: Optional[str] = None # Error message if failed

    @property
    def success(self) -> bool:
        return self.error is None and len(self.content) > 0


class BaseProvider(ABC):
    """Abstract base class for AI providers.

    All providers must implement:
        - generate(): Send a prompt with system instruction, get text back
        - is_available(): Check if the provider is configured and reachable
    """

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, system_instruction: str = "") -> ProviderResponse:
        """Generate a response from the AI model.

        Args:
            prompt: The user prompt (e.g., "Implement the TodoStore module").
            system_instruction: The system prompt (from SCP Bridge).

        Returns:
            ProviderResponse with the generated content.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is configured and reachable."""
        ...

    @property
    def name(self) -> str:
        return self.config.provider_name

    @property
    def model(self) -> str:
        return self.config.model
