"""
AI Provider Abstraction Layer
==============================
Provider-agnostic interface for AI code generation.
Supports Gemini, OpenAI, Anthropic, and Ollama.
"""

from nexus.providers.base import BaseProvider, ProviderConfig, ProviderResponse
from nexus.providers.registry import get_provider, list_providers, register_provider

__all__ = [
    "BaseProvider", "ProviderConfig", "ProviderResponse",
    "get_provider", "list_providers", "register_provider",
]
