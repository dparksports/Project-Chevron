"""
Chevron Glyph Registry
======================
Maps each Unicode glyph to its semantic primitive, origin lore,
and execution behavior. This is the bijective singleton map:
    ∀ l ∈ L, ∃! v ∈ V_L : f(l) = v
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class GlyphType(Enum):
    """The five fundamental Chevron primitives."""
    ORIGIN   = auto()  # ◬  — The Origin (Rendlesham)
    FOLD     = auto()  # ☾  — Fold Time (Roswell)
    FILTER   = auto()  # Ө  — The Filter / Gate (Roswell)
    WITNESS  = auto()  # 𓂀 — The Witness (Egyptian)
    WEAVER   = auto()  # ☤  — The Weaver (Generic)


@dataclass(frozen=True)
class GlyphInfo:
    """
    A Chevron glyph — a bijective singleton primitive.

    Each glyph carries:
      - symbol:      The Unicode character
      - name:        Human-readable name
      - glyph_type:  The primitive type
      - origin:      Lore origin (Rendlesham, Roswell, Egyptian, Generic)
      - intent:      Why this glyph exists
      - contract:    What it accepts and produces
      - constraint:  What it must NEVER do
    """
    symbol: str
    name: str
    glyph_type: GlyphType
    origin: str
    intent: str
    contract: str
    constraint: str
    description: Optional[str] = None


# ─────────────────────────────────────────────────────────────
#  THE GLYPH REGISTRY — The Five Primitives of Chevron
# ─────────────────────────────────────────────────────────────

GLYPH_REGISTRY: dict[str, GlyphInfo] = {

    "◬": GlyphInfo(
        symbol="◬",
        name="The Origin",
        glyph_type=GlyphType.ORIGIN,
        origin="Rendlesham",
        intent="Program entry point. All threads spawn from here.",
        contract="Accepts initial data → Produces a data stream",
        constraint="Must appear exactly once per program. Must not be nested.",
        description="Triangle with 3 Dots — The root from which all computation flows.",
    ),

    "☾": GlyphInfo(
        symbol="☾",
        name="Fold Time",
        glyph_type=GlyphType.FOLD,
        origin="Roswell",
        intent="Recursion. Feeds output back into input until base case.",
        contract="Accepts (predicate, transform, value) → Produces final value",
        constraint="Must always have a reachable base case. Must not mutate external state.",
        description="Violet Crescent — Folds time by looping output to input.",
    ),

    "Ө": GlyphInfo(
        symbol="Ө",
        name="The Filter",
        glyph_type=GlyphType.FILTER,
        origin="Roswell",
        intent="Conditional gate. Only data matching the shape passes through.",
        contract="Accepts (predicate, data) → Produces filtered data",
        constraint="Must never modify data that passes through. Reject, don't transform.",
        description="Circle with Bar — The Gate that judges what may pass.",
    ),

    "𓂀": GlyphInfo(
        symbol="𓂀",
        name="The Witness",
        glyph_type=GlyphType.WITNESS,
        origin="Egyptian",
        intent="Observe the data stream without altering it.",
        contract="Accepts any data → Logs it → Passes it through unchanged",
        constraint="Must NEVER modify the data. Pure observation only.",
        description="Eye of Horus — Watches the stream, bearing witness.",
    ),

    "☤": GlyphInfo(
        symbol="☤",
        name="The Weaver",
        glyph_type=GlyphType.WEAVER,
        origin="Generic",
        intent="Merge/join two independent streams into one braided result.",
        contract="Accepts list of values → Produces single merged value",
        constraint="Must preserve all input data. Nothing may be lost in the weaving.",
        description="Double Helix — Braids separate realities into one thread.",
    ),
}

# Quick-access sets for the lexer
GLYPH_CHARS = set(GLYPH_REGISTRY.keys())
GLYPH_NAMES = {info.name: symbol for symbol, info in GLYPH_REGISTRY.items()}


def lookup(symbol: str) -> GlyphInfo | None:
    """Look up a glyph by its Unicode symbol."""
    return GLYPH_REGISTRY.get(symbol)


def describe_all() -> str:
    """Return a formatted table of all glyphs for REPL help."""
    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║              PROJECT CHEVRON — GLYPH REGISTRY               ║",
        "╠══════╦════════════════╦════════════╦════════════════════════╣",
        "║ Glyph║ Name           ║ Origin     ║ Intent                 ║",
        "╠══════╬════════════════╬════════════╬════════════════════════╣",
    ]
    for symbol, info in GLYPH_REGISTRY.items():
        name = info.name.ljust(14)
        origin = info.origin.ljust(10)
        intent = info.intent[:22].ljust(22)
        lines.append(f"║  {symbol}   ║ {name} ║ {origin} ║ {intent} ║")
    lines.append("╚══════╩════════════════╩════════════╩════════════════════════╝")
    return "\n".join(lines)
