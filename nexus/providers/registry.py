"""
Provider Registry — Discover, Register, and Instantiate AI Providers
=====================================================================
Central registry that maps provider names to their implementation classes.
Supports lazy loading so importing nexus doesn't require all AI SDKs.
"""

from __future__ import annotations

from typing import Type, Optional

from nexus.providers.base import BaseProvider, ProviderConfig

# ─────────────────────────────────────────────────────────────
#  Registry
# ─────────────────────────────────────────────────────────────

_REGISTRY: dict[str, Type[BaseProvider]] = {}


def register_provider(name: str, provider_class: Type[BaseProvider]):
    """Register a provider class under a name."""
    _REGISTRY[name.lower()] = provider_class


def get_provider(config: ProviderConfig) -> BaseProvider:
    """Instantiate a provider from config.

    If the provider isn't registered yet, tries to import it lazily.

    Args:
        config: ProviderConfig with provider_name set.

    Returns:
        An instantiated BaseProvider subclass.

    Raises:
        ValueError: If the provider is not supported.
    """
    name = config.provider_name.lower()

    # Lazy-load built-in providers on first use
    if name not in _REGISTRY:
        _try_lazy_import(name)

    if name not in _REGISTRY:
        available = list(_REGISTRY.keys()) or ["(none registered)"]
        raise ValueError(
            f"Unknown provider '{name}'. Available: {available}. "
            f"Install the provider's SDK or register a custom provider."
        )

    return _REGISTRY[name](config)


def list_providers() -> list[str]:
    """List all registered provider names."""
    # Attempt to discover all built-in providers
    for name in ["gemini", "openai", "anthropic", "ollama"]:
        if name not in _REGISTRY:
            _try_lazy_import(name)
    return sorted(_REGISTRY.keys())


def _try_lazy_import(name: str):
    """Try to import and register a built-in provider."""
    try:
        if name == "gemini":
            from nexus.providers.gemini_provider import GeminiProvider
            register_provider("gemini", GeminiProvider)
        elif name == "openai":
            from nexus.providers.openai_provider import OpenAIProvider
            register_provider("openai", OpenAIProvider)
        elif name == "anthropic":
            from nexus.providers.anthropic_provider import AnthropicProvider
            register_provider("anthropic", AnthropicProvider)
        elif name == "ollama":
            from nexus.providers.ollama_provider import OllamaProvider
            register_provider("ollama", OllamaProvider)
    except ImportError:
        pass  # SDK not installed — provider won't be available
