"""
Chevron Operator Registry — Non-Polysemic Topological DSL
==========================================================
Maps each Topo-Categorical mathematical operator to its semantic
primitive, physics rationale, and verification behavior.

These operators leverage "arXiv latent anchors" — mathematical symbols
(⊗, ⊕, ↦, ∂, ∅) that possess deep, pristine, zero-polysemy embeddings
in foundational LLMs trained on millions of LaTeX papers.

Operator Reference:
    Hom(A,B) ≅ 0   — Null Morphism (Strict Isolation)
    A ↦ B           — Morphism / Functor (Directed Data Flow)
    A ⊕ B           — Direct Sum (Decoupled Coexistence)
    A ⊗ B           — Tensor Product (State Entanglement)
    ∂A ∩ ∂B = ∅     — Topological Boundary (Interface Encapsulation)
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class OperatorType(Enum):
    """The five Topo-Categorical constraint operators."""
    NULL_MORPHISM   = auto()  # Hom(A,B) ≅ 0  — Strict Isolation
    MORPHISM        = auto()  # A ↦ B          — Directed Data Flow
    DIRECT_SUM      = auto()  # A ⊕ B          — Decoupled Coexistence
    TENSOR_PRODUCT  = auto()  # A ⊗ B          — State Entanglement
    TOPO_BOUNDARY   = auto()  # ∂A ∩ ∂B = ∅    — Interface Encapsulation


@dataclass(frozen=True)
class OperatorInfo:
    """
    A Topo-Categorical operator — a non-polysemic structural primitive.

    Each operator carries:
      - symbol:      The mathematical notation
      - name:        Human-readable name
      - op_type:     The operator type
      - category:    Mathematical domain (Category Theory, Topology, Tensor)
      - intent:      Why this operator exists
      - contract:    What it enforces
      - constraint:  What it must NEVER allow
    """
    symbol: str
    name: str
    op_type: OperatorType
    category: str
    intent: str
    contract: str
    constraint: str
    description: Optional[str] = None


# ─────────────────────────────────────────────────────────────
#  THE OPERATOR REGISTRY — The Five Topo-Categorical Primitives
# ─────────────────────────────────────────────────────────────

OPERATOR_REGISTRY: dict[str, OperatorInfo] = {

    "Hom≅0": OperatorInfo(
        symbol="Hom(A,B) ≅ 0",
        name="Null Morphism",
        op_type=OperatorType.NULL_MORPHISM,
        category="Category Theory",
        intent="Strict isolation. Module A is mathematically forbidden from "
               "importing, calling, or sharing state with Module B.",
        contract="Accepts (source, target) → Enforces zero coupling",
        constraint="No import, call, or shared state between source and target.",
        description="Null Morphism — the morphism space between A and B is empty.",
    ),

    "↦": OperatorInfo(
        symbol="↦",
        name="Morphism",
        op_type=OperatorType.MORPHISM,
        category="Category Theory",
        intent="Directed data flow. Data or control strictly flows from A to B.",
        contract="Accepts (source, target) → Enforces unidirectional flow",
        constraint="Reverse flow (B → A) is forbidden. Must be acyclic (DAG).",
        description="Morphism / Functor — a structure-preserving directed map.",
    ),

    "⊕": OperatorInfo(
        symbol="⊕",
        name="Direct Sum",
        op_type=OperatorType.DIRECT_SUM,
        category="Category Theory",
        intent="Decoupled coexistence. A and B exist in the same environment "
               "but maintain mutually exclusive state spaces.",
        contract="Accepts (left, right) → Enforces state independence",
        constraint="No shared state, singletons, or global mutation between A and B.",
        description="Direct Sum — orthogonal state spaces, zero shared mutable state.",
    ),

    "⊗": OperatorInfo(
        symbol="⊗",
        name="Tensor Product",
        op_type=OperatorType.TENSOR_PRODUCT,
        category="Tensor Mathematics",
        intent="State entanglement. A and B are tightly coupled; a change in A "
               "structurally mutates B.",
        contract="Accepts (left, right) → Documents tight structural coupling",
        constraint="Changes to either side must propagate. Cannot be decoupled "
                   "without breaking the contract.",
        description="Tensor Product — entangled state spaces, structural co-mutation.",
    ),

    "∂∩∅": OperatorInfo(
        symbol="∂A ∩ ∂B = ∅",
        name="Topological Boundary",
        op_type=OperatorType.TOPO_BOUNDARY,
        category="Topology",
        intent="Interface encapsulation. A and B share no global state and must "
               "communicate through a defined Abstract Interface.",
        contract="Accepts (left, right) → Enforces abstract interface communication",
        constraint="Zero direct concrete references. All communication via "
                   "declared abstract interface only.",
        description="Topological Boundary — disjoint boundaries, interface-only coupling.",
    ),
}

# Quick-access sets for the lexer
OPERATOR_CHARS = {"↦", "⊕", "⊗", "∂", "∩", "∅", "≅"}
OPERATOR_NAMES = {info.name: key for key, info in OPERATOR_REGISTRY.items()}

# Backward-compat aliases used by other modules
GLYPH_REGISTRY = OPERATOR_REGISTRY
GLYPH_CHARS = OPERATOR_CHARS


def lookup(symbol: str) -> OperatorInfo | None:
    """Look up an operator by its registry key."""
    return OPERATOR_REGISTRY.get(symbol)


def describe_all() -> str:
    """Return a formatted table of all operators for REPL help."""
    lines = [
        "╔══════════════════════════════════════════════════════════════════════╗",
        "║     PROJECT CHEVRON — TOPO-CATEGORICAL OPERATOR REGISTRY           ║",
        "╠══════════════════╦═══════════════════╦══════════════════════════════╣",
        "║ Operator         ║ Category          ║ Intent                       ║",
        "╠══════════════════╬═══════════════════╬══════════════════════════════╣",
    ]
    for key, info in OPERATOR_REGISTRY.items():
        sym = info.symbol[:16].ljust(16)
        cat = info.category.ljust(17)
        intent = info.intent[:28].ljust(28)
        lines.append(f"║ {sym} ║ {cat} ║ {intent} ║")
    lines.append("╚══════════════════╩═══════════════════╩══════════════════════════════╝")
    return "\n".join(lines)
