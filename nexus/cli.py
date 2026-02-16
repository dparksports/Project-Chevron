"""
Nexus CLI — Command-Line Interface for the SCP-Powered AI IDE
===============================================================
Entry point for developers to interact with Nexus.

Usage:
    # Show architecture overview
    python -m nexus.cli overview --spec examples/turboscribe.chevron

    # Generate code for a single module
    python -m nexus.cli generate Transcriber --provider gemini --key YOUR_KEY

    # Generate all modules with verification
    python -m nexus.cli generate --all --provider gemini --key YOUR_KEY

    # Show context report (what the AI would see)
    python -m nexus.cli context Transcriber

    # Show session health metrics
    python -m nexus.cli health

    # List available AI providers
    python -m nexus.cli providers
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Add parent dir for chevron imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nexus.context_kernel import EntropyScorer, ContextPruner, SCPRetriever
from nexus.session_protocol import SessionState
from nexus.conductor import Conductor, Planner
from nexus.providers.base import ProviderConfig
from nexus.providers.registry import get_provider, list_providers


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def load_architecture(spec_path: str = None):
    """Load architecture spec from a Chevron file or turboscribe example."""
    # Try loading from turboscribe example (most common case)
    try:
        from examples.turboscribe_example import TURBOSCRIBE_SPEC
        return TURBOSCRIBE_SPEC
    except ImportError:
        pass

    # Try loading from scp_bridge template
    try:
        from scp_bridge import SCPBridge
        bridge = SCPBridge.from_template("todo_app")
        return bridge.spec
    except (ImportError, Exception):
        pass

    return None


def create_provider(args) -> object:
    """Create an AI provider from CLI arguments."""
    provider_name = getattr(args, "provider", "gemini")
    api_key = getattr(args, "key", "") or os.environ.get(
        {
            "gemini": "GOOGLE_API_KEY",
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }.get(provider_name, "API_KEY"),
        "",
    )

    config = ProviderConfig(
        provider_name=provider_name,
        model=getattr(args, "model", ""),
        api_key=api_key,
        base_url=getattr(args, "base_url", ""),
    )

    return get_provider(config)


# ─────────────────────────────────────────────────────────────
#  Commands
# ─────────────────────────────────────────────────────────────

def cmd_overview(args):
    """Show architecture overview."""
    arch = load_architecture(getattr(args, "spec", None))
    if not arch:
        print("✘ No architecture found. Specify --spec or ensure examples/ is available.")
        return

    print(f"\n◬ ─── Architecture: {arch.name} ───")
    print(f"  Modules: {len(arch.modules)}")
    print()

    for mod in arch.modules:
        deps = getattr(mod, "allowed_dependencies", []) or []
        forbidden = getattr(mod, "forbidden", []) or []
        constraints = getattr(mod, "constraints", []) or []
        methods = getattr(mod, "methods", []) or []

        print(f"  ┌─ {mod.name}")
        print(f"  │  {mod.description}")
        if methods:
            print(f"  │  Methods: {', '.join(m.name for m in methods)}")
        if deps:
            print(f"  │  Dependencies: {', '.join(deps)}")
        if forbidden:
            print(f"  │  Forbidden: {', '.join(forbidden)}")
        if constraints:
            for c in constraints[:2]:  # Show first 2 constraints
                print(f"  │  Constraint: {c}")
        print(f"  └──────")
        print()


def cmd_generate(args):
    """Generate code for one or all modules."""
    arch = load_architecture(getattr(args, "spec", None))
    if not arch:
        print("✘ No architecture found.")
        return

    try:
        provider = create_provider(args)
    except (ValueError, ImportError) as e:
        print(f"✘ Provider error: {e}")
        return

    # Create session and bridge
    session = SessionState(architecture=arch)

    bridge = None
    try:
        from scp_bridge import SCPBridge
        bridge = SCPBridge(arch)
    except ImportError:
        pass

    conductor = Conductor(session, provider, bridge=bridge)

    module_name = getattr(args, "module", None)
    all_modules = getattr(args, "all", False)

    if all_modules:
        target = None  # All modules
    elif module_name:
        target = [module_name]
    else:
        print("✘ Specify a module name or use --all")
        return

    request = getattr(args, "request", "") or f"Implement the module(s) following SCP constraints"
    results = conductor.handle_request(request, target_modules=target)

    # Save session
    session_dir = getattr(args, "output", ".nexus_session")
    session.session_dir = session_dir
    session.save()
    print(f"\n  Session saved to: {session_dir}/")


def cmd_context(args):
    """Show context report for a module."""
    arch = load_architecture(getattr(args, "spec", None))
    if not arch:
        print("✘ No architecture found.")
        return

    session = SessionState(architecture=arch)

    bridge = None
    try:
        from scp_bridge import SCPBridge
        bridge = SCPBridge(arch)
    except ImportError:
        pass

    retriever = SCPRetriever(architecture=arch, bridge=bridge)
    scorer = EntropyScorer(active_module=args.module)
    pruner = ContextPruner()

    # Build a mock conductor just for the report
    from nexus.providers.base import BaseProvider, ProviderResponse

    class MockProvider(BaseProvider):
        def generate(self, prompt, system_instruction=""):
            return ProviderResponse(content="", provider="mock")
        def is_available(self):
            return True

    conductor = Conductor(
        session,
        MockProvider(ProviderConfig(provider_name="mock")),
        bridge=bridge,
    )

    report = conductor.get_context_report(args.module)

    print(f"\n◬ ─── Context Report: {args.module} ───")
    print(f"  Full context: {report['full_context_tokens']} tokens")
    print(f"  After pruning: {report['pruned_context_tokens']} tokens")
    print(f"  Compression: {report['compression_ratio']}")
    print(f"  Entropy: {report['entropy_bits']} bits")
    print(f"  Forbidden: {', '.join(report['forbidden_modules']) or 'none'}")
    print()

    if report["items"]:
        print("  Context items (by relevance):")
        for item in report["items"]:
            print(f"    [{item['score']}] {item['category']:15s} "
                  f"{item['tokens']:5d} tok  {item['source']}")


def cmd_health(args):
    """Show session health metrics."""
    session_dir = getattr(args, "session", ".nexus_session")
    session = SessionState.load(session_dir)

    health = session.session_health
    print(f"\n𓂀 ─── Session Health ───")
    for key, val in health.items():
        print(f"  {key}: {val}")


def cmd_providers(args):
    """List available AI providers."""
    providers = list_providers()
    print(f"\n☤ ─── Available Providers ───")
    if providers:
        for p in providers:
            print(f"  • {p}")
    else:
        print("  No providers available. Install an AI SDK:")
        print("    pip install google-genai     # Gemini")
        print("    pip install openai           # OpenAI")
        print("    pip install anthropic        # Anthropic")
        print("    # Ollama — no pip needed, just install from ollama.com")


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="nexus",
        description="Nexus — SCP-Powered AI IDE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  nexus overview\n"
            "  nexus generate Transcriber --provider gemini --key YOUR_KEY\n"
            "  nexus generate --all --provider gemini --key YOUR_KEY\n"
            "  nexus context Transcriber\n"
            "  nexus health\n"
            "  nexus providers\n"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # overview
    p_overview = subparsers.add_parser("overview", help="Show architecture overview")
    p_overview.add_argument("--spec", help="Path to .chevron spec file")

    # generate
    p_gen = subparsers.add_parser("generate", help="Generate code for module(s)")
    p_gen.add_argument("module", nargs="?", help="Module name to generate")
    p_gen.add_argument("--all", action="store_true", help="Generate all modules")
    p_gen.add_argument("--provider", default="gemini", help="AI provider (gemini/openai/anthropic/ollama)")
    p_gen.add_argument("--model", default="", help="Model name")
    p_gen.add_argument("--key", default="", help="API key")
    p_gen.add_argument("--base-url", default="", help="Custom API base URL")
    p_gen.add_argument("--request", default="", help="Custom request description")
    p_gen.add_argument("--output", default=".nexus_session", help="Session output directory")
    p_gen.add_argument("--spec", help="Path to .chevron spec file")

    # context
    p_ctx = subparsers.add_parser("context", help="Show context report for a module")
    p_ctx.add_argument("module", help="Module name")
    p_ctx.add_argument("--spec", help="Path to .chevron spec file")

    # health
    p_health = subparsers.add_parser("health", help="Show session health")
    p_health.add_argument("--session", default=".nexus_session", help="Session directory")

    # providers
    subparsers.add_parser("providers", help="List available AI providers")

    args = parser.parse_args()

    commands = {
        "overview": cmd_overview,
        "generate": cmd_generate,
        "context": cmd_context,
        "health": cmd_health,
        "providers": cmd_providers,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
