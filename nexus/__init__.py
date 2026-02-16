"""
Nexus — Next-Gen AI IDE powered by Spatial Constraint Protocol
===============================================================
Consumes Project Chevron's SCP framework to build an entropy-aware,
contract-driven AI development environment.

Architecture:
    Layer 1: Context Kernel    — Entropy scoring, pruning, SCP retrieval
    Layer 2: Session Protocol  — Persistent state, edit ledger, contract cache
    Layer 3: Orchestrator      — Conductor, Planner, Executor, Verifier
    Providers: AI Abstraction  — Gemini, OpenAI, Anthropic, Ollama
"""

__version__ = "0.1.0"

from nexus.context_kernel import ContextItem, EntropyScorer, ContextPruner, SCPRetriever
from nexus.session_protocol import SessionState, EditEntry, EditLedger, ContractCache
from nexus.conductor import Conductor, Planner, Executor, Verifier

__all__ = [
    "ContextItem", "EntropyScorer", "ContextPruner", "SCPRetriever",
    "SessionState", "EditEntry", "EditLedger", "ContractCache",
    "Conductor", "Planner", "Executor", "Verifier",
]
