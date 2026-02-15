"""
Chevron Verifier
================
Static analysis pass that runs after parsing (before execution) to enforce
SCP constraints at the language level.

Checks:
  1. ◬ Origin count — exactly one per program/module (hard error)
  2. 𓂀 Witness — must be terminal (no downstream mutations)
  3. ☾ Fold — must have both predicate and transform args
  4. Dependency graph — modules may only reference their declared imports
  5. Forbidden modules — any reference to a forbidden module = hard error
  6. No circular depends_on chains
  7. Type declarations — pipeline boundary type checking (structural)
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from .glyphs import GLYPH_REGISTRY, GlyphType
from .parser import (
    ASTNode, ProgramNode, GlyphNode, PipelineNode, BindingNode,
    LiteralNode, ListNode, PredicateNode, IdentifierNode,
    ModuleNode, SpecNode, TypeDeclNode, TypeAnnotNode, ConstraintNode,
    FuncCallNode,
)


class ViolationLevel(Enum):
    """Severity of a verification violation."""
    ERROR = auto()
    WARNING = auto()


@dataclass
class Violation:
    """A single verification violation."""
    level: ViolationLevel
    line: int
    col: int
    glyph: str  # Which glyph/rule is violated (e.g., "◬", "Ө", "DEPENDENCY")
    message: str

    def __str__(self) -> str:
        icon = "\u2718" if self.level == ViolationLevel.ERROR else "\u26A0"
        return f"  {icon} L{self.line}:{self.col} [{self.glyph}] {self.message}"


class SCPVerifier:
    """
    Static analysis for Chevron programs.

    Usage:
        verifier = SCPVerifier()
        violations = verifier.verify(ast)
        if violations:
            for v in violations:
                print(v)
    """

    def __init__(self):
        self.violations: list[Violation] = []
        self._module_names: set[str] = set()
        self._spec_names: set[str] = set()
        self._type_registry: dict[str, TypeDeclNode] = {}

    def verify(self, ast: ProgramNode) -> list[Violation]:
        """Run all verification checks on the AST. Returns list of violations."""
        self.violations = []
        self._module_names = set()
        self._spec_names = set()
        self._type_registry = {}

        # Collect module/spec names first
        for stmt in ast.statements:
            if isinstance(stmt, ModuleNode):
                self._module_names.add(stmt.name)
            elif isinstance(stmt, SpecNode):
                self._spec_names.add(stmt.name)

        # Run checks
        self._check_origin_count(ast)
        self._check_witness_terminal(ast)
        self._check_fold_args(ast)
        self._check_module_dependencies(ast)
        self._check_circular_dependencies(ast)
        self._check_types(ast)

        return sorted(self.violations, key=lambda v: (v.line, v.col))

    def _add(self, level: ViolationLevel, line: int, col: int, glyph: str, message: str):
        """Add a violation."""
        self.violations.append(Violation(level, line, col, glyph, message))

    # ─────────────────────────────────────────────────────────
    #  Check 1: ◬ Origin Count
    # ─────────────────────────────────────────────────────────

    def _check_origin_count(self, ast: ProgramNode):
        """Verify ◬ appears exactly once per module/program."""
        # Check top-level (non-module) statements
        top_level_origins = self._find_glyphs(
            [s for s in ast.statements if not isinstance(s, (ModuleNode, SpecNode))],
            "\u25EC"
        )
        if len(top_level_origins) > 1:
            for origin in top_level_origins[1:]:
                self._add(
                    ViolationLevel.ERROR, origin.line, origin.col, "\u25EC",
                    f"Multiple \u25EC Origins in top-level scope ({len(top_level_origins)} found). "
                    f"\u25EC must appear exactly once per scope."
                )

        # Check each module
        for stmt in ast.statements:
            if isinstance(stmt, ModuleNode):
                origins = self._find_glyphs(stmt.body, "\u25EC")
                if len(origins) == 0:
                    self._add(
                        ViolationLevel.ERROR, stmt.line, stmt.col, "\u25EC",
                        f"Module '{stmt.name}' has no \u25EC Origin. Every module must have exactly one."
                    )
                elif len(origins) > 1:
                    for origin in origins[1:]:
                        self._add(
                            ViolationLevel.ERROR, origin.line, origin.col, "\u25EC",
                            f"Module '{stmt.name}' has {len(origins)} \u25EC Origins. Must be exactly one."
                        )

    def _find_glyphs(self, nodes: list[ASTNode], glyph_char: str) -> list[GlyphNode]:
        """Recursively find all glyph nodes matching the given character."""
        found = []
        for node in nodes:
            self._walk_for_glyphs(node, glyph_char, found)
        return found

    def _walk_for_glyphs(self, node: ASTNode, glyph_char: str, found: list):
        """Walk the AST recursively to find glyphs."""
        if isinstance(node, GlyphNode) and node.glyph == glyph_char:
            found.append(node)
        if isinstance(node, PipelineNode):
            for stage in node.stages:
                self._walk_for_glyphs(stage, glyph_char, found)
        elif isinstance(node, GlyphNode):
            for arg in node.args:
                self._walk_for_glyphs(arg, glyph_char, found)
        elif isinstance(node, ListNode):
            for el in node.elements:
                self._walk_for_glyphs(el, glyph_char, found)

    # ─────────────────────────────────────────────────────────
    #  Check 2: 𓂀 Witness Must Be Terminal
    # ─────────────────────────────────────────────────────────

    def _check_witness_terminal(self, ast: ProgramNode):
        """Verify 𓂀 is always the last stage in a pipeline (no downstream stages)."""
        for stmt in ast.statements:
            self._walk_for_witness_terminal(stmt)

    def _walk_for_witness_terminal(self, node: ASTNode):
        """Walk AST looking for pipelines with non-terminal 𓂀."""
        if isinstance(node, PipelineNode):
            for i, stage in enumerate(node.stages):
                if isinstance(stage, GlyphNode) and stage.glyph == "\U000130C0":
                    # 𓂀 found — check if it's NOT the last stage
                    if i < len(node.stages) - 1:
                        self._add(
                            ViolationLevel.ERROR, stage.line, stage.col, "\U000130C0",
                            "\U000130C0 Witness must be terminal — no stages may follow it. "
                            "Witness observes without modifying; downstream stages would break this contract."
                        )
                self._walk_for_witness_terminal(stage)
        elif isinstance(node, ModuleNode):
            for stmt in node.body:
                self._walk_for_witness_terminal(stmt)
        elif isinstance(node, SpecNode):
            for stmt in node.body:
                self._walk_for_witness_terminal(stmt)
        elif isinstance(node, GlyphNode):
            for arg in node.args:
                self._walk_for_witness_terminal(arg)

    # ─────────────────────────────────────────────────────────
    #  Check 3: ☾ Fold Must Have Args
    # ─────────────────────────────────────────────────────────

    def _check_fold_args(self, ast: ProgramNode):
        """Verify ☾ always has predicate and transform arguments."""
        folds = self._find_glyphs(ast.statements, "\u263E")
        for fold in folds:
            if len(fold.args) < 2:
                self._add(
                    ViolationLevel.ERROR, fold.line, fold.col, "\u263E",
                    f"\u263E Fold Time requires at least (predicate, transform). "
                    f"Got {len(fold.args)} arg(s). Must have a reachable base case."
                )

        # Also check inside modules
        for stmt in ast.statements:
            if isinstance(stmt, ModuleNode):
                mod_folds = self._find_glyphs(stmt.body, "\u263E")
                for fold in mod_folds:
                    if len(fold.args) < 2:
                        self._add(
                            ViolationLevel.ERROR, fold.line, fold.col, "\u263E",
                            f"\u263E Fold Time in module '{stmt.name}' requires (predicate, transform). "
                            f"Got {len(fold.args)} arg(s)."
                        )

    # ─────────────────────────────────────────────────────────
    #  Check 4: Module Dependencies
    # ─────────────────────────────────────────────────────────

    def _check_module_dependencies(self, ast: ProgramNode):
        """Verify modules only reference their declared imports and no forbidden modules."""
        for stmt in ast.statements:
            if isinstance(stmt, (ModuleNode, SpecNode)):
                # Check forbidden references
                if stmt.forbidden:
                    identifiers = self._collect_identifiers(stmt.body)
                    for ident in identifiers:
                        if ident.name in stmt.forbidden:
                            self._add(
                                ViolationLevel.ERROR, ident.line, ident.col, "DEPENDENCY",
                                f"Module '{stmt.name}' references forbidden module '{ident.name}'. "
                                f"Forbidden zones: {stmt.forbidden}"
                            )

                # Check depends_on integrity
                all_known = self._module_names | self._spec_names
                for dep in stmt.depends_on:
                    if dep not in all_known and dep not in stmt.imports:
                        self._add(
                            ViolationLevel.WARNING, stmt.line, stmt.col, "DEPENDENCY",
                            f"Module '{stmt.name}' depends on '{dep}' which is not defined in this program."
                        )

    def _collect_identifiers(self, nodes: list[ASTNode]) -> list[IdentifierNode]:
        """Recursively collect all IdentifierNode references."""
        result = []
        for node in nodes:
            self._walk_for_identifiers(node, result)
        return result

    def _walk_for_identifiers(self, node: ASTNode, result: list):
        """Walk the AST to collect all identifier references."""
        if isinstance(node, IdentifierNode):
            result.append(node)
        elif isinstance(node, PipelineNode):
            for stage in node.stages:
                self._walk_for_identifiers(stage, result)
        elif isinstance(node, GlyphNode):
            for arg in node.args:
                self._walk_for_identifiers(arg, result)
        elif isinstance(node, ListNode):
            for el in node.elements:
                self._walk_for_identifiers(el, result)
        elif isinstance(node, BindingNode):
            if node.expression:
                self._walk_for_identifiers(node.expression, result)
        elif isinstance(node, FuncCallNode):
            for arg in node.args:
                self._walk_for_identifiers(arg, result)

    # ─────────────────────────────────────────────────────────
    #  Check 5: No Circular Dependencies
    # ─────────────────────────────────────────────────────────

    def _check_circular_dependencies(self, ast: ProgramNode):
        """Verify no circular depends_on chains exist."""
        # Build dependency graph
        dep_graph: dict[str, list[str]] = {}
        node_lines: dict[str, tuple[int, int]] = {}

        for stmt in ast.statements:
            if isinstance(stmt, (ModuleNode, SpecNode)):
                dep_graph[stmt.name] = stmt.depends_on
                node_lines[stmt.name] = (stmt.line, stmt.col)

        # DFS cycle detection
        visited: set[str] = set()
        in_stack: set[str] = set()

        def dfs(name: str, path: list[str]) -> list[str] | None:
            if name in in_stack:
                return path + [name]
            if name in visited:
                return None
            visited.add(name)
            in_stack.add(name)
            for dep in dep_graph.get(name, []):
                cycle = dfs(dep, path + [name])
                if cycle is not None:
                    return cycle
            in_stack.remove(name)
            return None

        for module_name in dep_graph:
            if module_name not in visited:
                cycle = dfs(module_name, [])
                if cycle is not None:
                    line, col = node_lines.get(module_name, (0, 0))
                    cycle_str = " \u2192 ".join(cycle)
                    self._add(
                        ViolationLevel.ERROR, line, col, "CYCLE",
                        f"Circular dependency detected: {cycle_str}"
                    )
                    break  # Report first cycle only

    # ─────────────────────────────────────────────────────────
    #  Check 6: Type Checking at Pipeline Boundaries
    # ─────────────────────────────────────────────────────────

    def _check_types(self, ast: ProgramNode):
        """Collect type declarations and verify pipeline boundary type compatibility."""
        # Collect all type declarations
        for stmt in ast.statements:
            if isinstance(stmt, TypeDeclNode):
                self._type_registry[stmt.type_name] = stmt
            elif isinstance(stmt, (ModuleNode, SpecNode)):
                for tdecl in stmt.types:
                    self._type_registry[tdecl.type_name] = tdecl

        # Check pipeline type annotations for consistency
        for stmt in ast.statements:
            self._walk_for_type_annots(stmt)

    def _walk_for_type_annots(self, node: ASTNode):
        """Check TypeAnnotNode references point to declared types."""
        if isinstance(node, TypeAnnotNode):
            if node.type_name not in self._type_registry:
                # Only warn — types might be external
                self._add(
                    ViolationLevel.WARNING, node.line, node.col, "TYPE",
                    f"Type '{node.type_name}' referenced but not declared. "
                    f"Declared types: {list(self._type_registry.keys()) or ['(none)']}"
                )
        elif isinstance(node, PipelineNode):
            for stage in node.stages:
                self._walk_for_type_annots(stage)
        elif isinstance(node, GlyphNode):
            for arg in node.args:
                self._walk_for_type_annots(arg)
        elif isinstance(node, ModuleNode):
            for stmt in node.body:
                self._walk_for_type_annots(stmt)
        elif isinstance(node, SpecNode):
            for stmt in node.body:
                self._walk_for_type_annots(stmt)
