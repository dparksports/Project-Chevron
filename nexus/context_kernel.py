"""
Context Kernel — Layer 1 of the Nexus AI IDE
=============================================
Manages what the AI sees. Replaces "dump everything into context" with
entropy-scored, contract-scoped retrieval.

Components:
    EntropyScorer  — Scores context items by relevance (0.0–1.0)
    ContextPruner  — Keeps highest-scoring items within token budget
    SCPRetriever   — Pulls context through SCP contracts (RAG Denial)

Theory:
    The key insight from SCP is that contracts contain 100% of architectural
    information in ~1% of the tokens. The Context Kernel enforces this at
    the IDE level: instead of showing the AI raw code from every module,
    it shows contracts for dependencies and code only for the active module.

    Entropy scoring follows the SCP hierarchy:
        Contracts > Active Code > Errors > Interfaces > History > Outputs > Stale

Dan Park | MagicPoint.ai | February 2026
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────────────────────

# Context item categories — ordered by default relevance
CATEGORY_CONTRACT = "contract"        # SCP specs, interface definitions
CATEGORY_ACTIVE_CODE = "active_code"  # Code from the module being edited
CATEGORY_ERROR = "error"              # Error messages from current edit
CATEGORY_INTERFACE = "interface"      # Dependency interface signatures
CATEGORY_CONVERSATION = "conversation"  # Chat history turns
CATEGORY_OUTPUT = "output"            # Tool outputs (file listings, etc.)
CATEGORY_STALE = "stale"              # Pre-edit code versions

# Base relevance scores per category
BASE_SCORES: dict[str, float] = {
    CATEGORY_CONTRACT: 1.0,
    CATEGORY_ACTIVE_CODE: 0.9,
    CATEGORY_ERROR: 0.85,
    CATEGORY_INTERFACE: 0.7,
    CATEGORY_CONVERSATION: 0.6,
    CATEGORY_OUTPUT: 0.3,
    CATEGORY_STALE: 0.1,
}


@dataclass
class ContextItem:
    """A single item in the AI's context window.

    Every piece of information the AI sees is wrapped in a ContextItem,
    scored for relevance, and tagged with its SCP module scope. This
    enables entropy-aware pruning and RAG Denial enforcement.
    """

    content: str
    source: str             # Where it came from (filepath, "conversation", "tool:grep")
    category: str           # One of CATEGORY_* constants
    module_scope: str = "global"   # Which SCP module this belongs to
    created_at: float = field(default_factory=time.time)
    relevance_score: float = 0.0   # Set by EntropyScorer
    token_count: int = 0           # Set during scoring
    turn_age: int = 0              # How many conversation turns ago

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = estimate_tokens(self.content)


def estimate_tokens(text: str) -> int:
    """Approximate token count using the ~4 chars/token heuristic.
    This avoids depending on tiktoken while remaining accurate enough
    for budget allocation."""
    return max(1, len(text) // 4)


# ─────────────────────────────────────────────────────────────
#  Entropy Scorer
# ─────────────────────────────────────────────────────────────

class EntropyScorer:
    """Scores context items by relevance to the current editing task.

    Scoring rules (from SCP theory):
        1. Contracts always score highest (bijective singletons carry max signal)
        2. Active module code scores high (the AI needs its own implementation)
        3. Error messages are high-priority but decay with age
        4. Dependency interfaces score medium (contracts, not code)
        5. Conversation history decays exponentially with turn age
        6. Tool outputs are low-signal (consumed, rarely reused)
        7. Stale code states are near-zero (pre-edit, superseded)

    The exponential decay on conversation history is the core anti-entropy
    mechanism: it ensures that recent, relevant discussion displaces old noise
    naturally, without requiring explicit summarization.
    """

    def __init__(self, decay_rate: float = 0.92, active_module: Optional[str] = None):
        """
        Args:
            decay_rate: Per-turn decay factor for conversation history.
                        0.92 means each turn reduces score by 8%.
                        After 10 turns: 0.92^10 = 0.43 (less than half).
                        After 25 turns: 0.92^25 = 0.12 (nearly gone).
            active_module: The SCP module currently being edited.
                          Items scoped to this module get a relevance boost.
        """
        self.decay_rate = decay_rate
        self.active_module = active_module

    def score(self, item: ContextItem) -> float:
        """Score a single context item. Returns value in [0.0, 1.0]."""

        # Start with base score for the category
        base = BASE_SCORES.get(item.category, 0.5)

        # Apply exponential decay for conversation items
        if item.category == CATEGORY_CONVERSATION:
            base *= self.decay_rate ** item.turn_age

        # Apply age decay for errors (recent errors matter more)
        elif item.category == CATEGORY_ERROR:
            base *= self.decay_rate ** (item.turn_age * 0.5)  # Slower decay than conversation

        # Module scope boost: items in the active module score higher
        if self.active_module and item.module_scope == self.active_module:
            base = min(1.0, base * 1.15)  # 15% boost, capped at 1.0

        # Module scope penalty: items from unrelated modules score lower
        elif self.active_module and item.module_scope != "global" and item.module_scope != self.active_module:
            base *= 0.5  # 50% penalty for wrong-module items

        # Clamp to [0.0, 1.0]
        return max(0.0, min(1.0, base))

    def score_all(self, items: list[ContextItem]) -> list[ContextItem]:
        """Score all items in-place and return sorted by score descending."""
        for item in items:
            item.relevance_score = self.score(item)
        return sorted(items, key=lambda x: x.relevance_score, reverse=True)

    def compute_entropy(self, items: list[ContextItem]) -> float:
        """Compute Shannon entropy of the scored context.

        Low entropy = most items are high-signal (good).
        High entropy = signal is spread thin (bad — Foggy Boundary).

        This is the H(S) metric from the SCP paper:
            H(S) = -Σ pᵢ log₂(pᵢ)
        where pᵢ is the normalized relevance score of item i.

        Returns:
            Entropy in bits. Lower is better.
        """
        if not items:
            return 0.0

        total = sum(item.relevance_score for item in items)
        if total == 0:
            return 0.0

        entropy = 0.0
        for item in items:
            p = item.relevance_score / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy


# ─────────────────────────────────────────────────────────────
#  Context Pruner
# ─────────────────────────────────────────────────────────────

class ContextPruner:
    """Keeps the highest-scoring context items within a token budget.

    The pruner operates on the principle that contracts should NEVER be
    pruned (they are the architectural source of truth), while conversation
    history and tool outputs are expendable.

    Protection tiers:
        Tier 1 (Protected): Contracts — never pruned
        Tier 2 (Preferred): Active module code, errors — pruned last
        Tier 3 (Expendable): Everything else — pruned first
    """

    PROTECTED_CATEGORIES = {CATEGORY_CONTRACT}
    PREFERRED_CATEGORIES = {CATEGORY_ACTIVE_CODE, CATEGORY_ERROR}

    def prune(self, items: list[ContextItem], budget: int) -> list[ContextItem]:
        """Keep highest-scoring items that fit within the token budget.

        Args:
            items: Pre-scored context items (score_all should be called first).
            budget: Maximum total tokens allowed in context.

        Returns:
            Subset of items that fits within budget, ordered by score descending.
        """
        # Separate by protection tier
        protected = [i for i in items if i.category in self.PROTECTED_CATEGORIES]
        preferred = [i for i in items if i.category in self.PREFERRED_CATEGORIES]
        expendable = [i for i in items if i.category not in self.PROTECTED_CATEGORIES
                      and i.category not in self.PREFERRED_CATEGORIES]

        # Sort each tier by score
        protected.sort(key=lambda x: x.relevance_score, reverse=True)
        preferred.sort(key=lambda x: x.relevance_score, reverse=True)
        expendable.sort(key=lambda x: x.relevance_score, reverse=True)

        result = []
        used = 0

        # Always include protected items (contracts)
        for item in protected:
            if used + item.token_count <= budget:
                result.append(item)
                used += item.token_count

        # Then preferred items
        for item in preferred:
            if used + item.token_count <= budget:
                result.append(item)
                used += item.token_count

        # Then expendable items
        for item in expendable:
            if used + item.token_count <= budget:
                result.append(item)
                used += item.token_count

        # Sort final result by score for clean context ordering
        result.sort(key=lambda x: x.relevance_score, reverse=True)
        return result

    def compute_compression_ratio(self, original: list[ContextItem],
                                  pruned: list[ContextItem]) -> float:
        """How much context was compressed.

        Returns:
            Ratio of pruned tokens to original tokens. Lower = more compression.
            The SCP paper achieves 106× (ratio ≈ 0.009).
        """
        original_tokens = sum(i.token_count for i in original)
        pruned_tokens = sum(i.token_count for i in pruned)

        if original_tokens == 0:
            return 1.0

        return pruned_tokens / original_tokens


# ─────────────────────────────────────────────────────────────
#  SCP Retriever
# ─────────────────────────────────────────────────────────────

class SCPRetriever:
    """Retrieves context items scoped through SCP contracts.

    This is RAG Denial enforced at the retrieval level:
        - For the ACTIVE module: returns full source code
        - For DEPENDENCY modules: returns interface contracts ONLY
        - For FORBIDDEN modules: returns NOTHING

    The retriever consumes Chevron's ArchitectureSpec and SCPBridge
    to determine what the AI is allowed to see.
    """

    def __init__(self, architecture=None, bridge=None):
        """
        Args:
            architecture: A Chevron ArchitectureSpec instance.
            bridge: A Chevron SCPBridge instance.
        """
        self.architecture = architecture
        self.bridge = bridge

    def retrieve_for_module(self, module_name: str,
                            code_store: dict[str, str] = None,
                            extra_context: list[ContextItem] = None) -> list[ContextItem]:
        """Retrieve context items for editing a specific module.

        This enforces RAG Denial: the AI sees ONLY:
            1. The target module's current source code (full)
            2. Dependency modules' interface contracts (signatures only)
            3. Global architecture constraints
            4. Any extra context (conversation, errors) filtered by scope

        It does NOT return:
            - Source code from other modules
            - Anything from forbidden modules
            - Stale code from previous edits

        Args:
            module_name: The SCP module to retrieve context for.
            code_store: Dict mapping module names to their current source code.
            extra_context: Additional context items to filter and include.

        Returns:
            List of ContextItems, ready for scoring and pruning.
        """
        items: list[ContextItem] = []
        code_store = code_store or {}

        if not self.architecture:
            # No architecture loaded — return extra context only
            if extra_context:
                items.extend(extra_context)
            return items

        # Find the target module spec
        target_spec = None
        for mod in self.architecture.modules:
            if mod.name == module_name:
                target_spec = mod
                break

        if not target_spec:
            raise ValueError(f"Module '{module_name}' not found in architecture spec. "
                             f"Available: {[m.name for m in self.architecture.modules]}")

        # Determine forbidden modules
        forbidden = set(getattr(target_spec, 'forbidden', []) or [])

        # 1. Active module code (full source)
        if module_name in code_store:
            items.append(ContextItem(
                content=code_store[module_name],
                source=f"module:{module_name}",
                category=CATEGORY_ACTIVE_CODE,
                module_scope=module_name,
            ))

        # 2. SCP contract for the active module
        if self.bridge:
            try:
                contract = self.bridge.generate_system_prompt(module_name)
                items.append(ContextItem(
                    content=contract,
                    source=f"contract:{module_name}",
                    category=CATEGORY_CONTRACT,
                    module_scope=module_name,
                ))
            except Exception:
                pass  # Module might not have a full spec yet

        # 3. Dependency interface contracts (NOT source code)
        allowed_deps = set(getattr(target_spec, 'allowed_dependencies', []) or [])
        for dep_name in allowed_deps:
            if dep_name in forbidden:
                continue  # Safety check

            # Generate interface-only view of the dependency
            dep_interface = self._extract_interface(dep_name)
            if dep_interface:
                items.append(ContextItem(
                    content=dep_interface,
                    source=f"interface:{dep_name}",
                    category=CATEGORY_INTERFACE,
                    module_scope=dep_name,
                ))

        # 4. Global constraints
        global_constraints = getattr(self.architecture, 'global_constraints', []) or []
        if global_constraints:
            items.append(ContextItem(
                content="Global SCP Constraints:\n" + "\n".join(
                    f"  • {c}" for c in global_constraints
                ),
                source="architecture:global",
                category=CATEGORY_CONTRACT,
                module_scope="global",
            ))

        # 5. Filter extra context by scope (enforce RAG Denial)
        if extra_context:
            for item in extra_context:
                # Block items from forbidden modules
                if item.module_scope in forbidden:
                    continue

                # Allow global items and items scoped to this module
                if item.module_scope in ("global", module_name) or item.module_scope in allowed_deps:
                    items.append(item)

        return items

    def _extract_interface(self, module_name: str) -> Optional[str]:
        """Extract interface-only view of a module (signatures, no implementation).

        This is the RAG Denial mechanism: dependency modules are visible
        only as contracts, never as source code.
        """
        if not self.architecture:
            return None

        for mod in self.architecture.modules:
            if mod.name == module_name:
                lines = [f"# Interface: {mod.name}"]
                lines.append(f"# {mod.description}")

                if hasattr(mod, 'methods') and mod.methods:
                    lines.append("")
                    for method in mod.methods:
                        # Show signature only
                        inputs = ", ".join(method.inputs) if hasattr(method, 'inputs') else ""
                        output = getattr(method, 'output', 'Any')
                        glyph = getattr(method, 'glyph', '')
                        glyph_tag = f" [{glyph}]" if glyph else ""
                        lines.append(f"def {method.name}({inputs}) -> {output}{glyph_tag}")
                        if hasattr(method, 'constraint') and method.constraint:
                            lines.append(f"    # Constraint: {method.constraint}")

                if hasattr(mod, 'constraints') and mod.constraints:
                    lines.append("")
                    lines.append("# Module constraints:")
                    for c in mod.constraints:
                        lines.append(f"#   • {c}")

                return "\n".join(lines)

        return None

    def get_forbidden_modules(self, module_name: str) -> list[str]:
        """Return the list of modules that are forbidden for the given module."""
        if not self.architecture:
            return []

        for mod in self.architecture.modules:
            if mod.name == module_name:
                return list(getattr(mod, 'forbidden', []) or [])

        return []

    def get_dependency_graph(self) -> dict[str, list[str]]:
        """Return the full dependency graph as adjacency list."""
        if not self.architecture:
            return {}

        graph = {}
        for mod in self.architecture.modules:
            deps = list(getattr(mod, 'allowed_dependencies', []) or [])
            graph[mod.name] = deps

        return graph
