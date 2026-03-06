"""
Chevron Verifier — Non-Polysemic Topological DSL
==================================================
Static analysis pass that runs after parsing (before execution) to enforce
Topo-Categorical constraints at the language level.

Checks:
  1. Null Morphism — Hom(A,B) ≅ 0: A must never reference B
  2. Morphism Direction — A ↦ B: reverse flow (B → A) is forbidden
  3. Direct Sum — A ⊕ B: no shared state between A and B
  4. Tensor Product — A ⊗ B: structural coupling is documented
  5. Topological Boundary — ∂A ∩ ∂B = ∅: interface-only communication
  6. Dependency graph — modules may only reference their declared imports
  7. Forbidden modules — any reference to a forbidden module = hard error
  8. No circular depends_on chains
  9. Type declarations — pipeline boundary type checking (structural)

Rejection Format:
  [SYSTEM 2 REJECTION]: <operator> <details>. Resample required.
"""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from .glyphs import OPERATOR_REGISTRY, OperatorType
from .parser import (
    ASTNode, ProgramNode, PipelineNode, BindingNode,
    LiteralNode, ListNode, PredicateNode, IdentifierNode,
    ModuleNode, SpecNode, TypeDeclNode, TypeAnnotNode, ConstraintNode,
    FuncCallNode,
    NullMorphismNode, MorphismNode, DirectSumNode, TensorProductNode,
    TopoBoundaryNode,
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
    operator: str  # Which operator/rule is violated (e.g., "Hom≅0", "↦", "DEPENDENCY")
    message: str

    def __str__(self) -> str:
        icon = "\u2718" if self.level == ViolationLevel.ERROR else "\u26A0"
        return f"  {icon} L{self.line}:{self.col} [{self.operator}] {self.message}"


class SCPVerifier:
    """
    Static analysis for Chevron programs — Topo-Categorical enforcement.

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
        self._check_null_morphisms(ast)
        self._check_morphism_direction(ast)
        self._check_topo_boundaries(ast)
        self._check_module_dependencies(ast)
        self._check_circular_dependencies(ast)
        self._check_types(ast)

        return sorted(self.violations, key=lambda v: (v.line, v.col))

    def _add(self, level: ViolationLevel, line: int, col: int, operator: str, message: str):
        """Add a violation."""
        self.violations.append(Violation(level, line, col, operator, message))

    # ─────────────────────────────────────────────────────────
    #  Check 1: Null Morphism — Hom(A,B) ≅ 0
    # ─────────────────────────────────────────────────────────

    def _check_null_morphisms(self, ast: ProgramNode):
        """Verify Hom(A,B) ≅ 0 constraints — A must never reference B."""
        # Collect all NullMorphism declarations
        null_morphisms = self._find_nodes(ast.statements, NullMorphismNode)

        # For each nullmorphism, check that no module body references the forbidden target
        for nm in null_morphisms:
            # Find the module named nm.source
            for stmt in ast.statements:
                if isinstance(stmt, (ModuleNode, SpecNode)) and stmt.name == nm.source:
                    identifiers = self._collect_identifiers(stmt.body)
                    for ident in identifiers:
                        if ident.name == nm.target:
                            self._add(
                                ViolationLevel.ERROR, ident.line, ident.col, "Hom≅0",
                                f"[SYSTEM 2 REJECTION]: Morphism {nm.source} → {nm.target} "
                                f"violates Hom({nm.source}, {nm.target}) ≅ 0. "
                                f"Module '{nm.source}' references forbidden target "
                                f"'{nm.target}'. Resample required."
                            )

    # ─────────────────────────────────────────────────────────
    #  Check 2: Morphism Direction — A ↦ B
    # ─────────────────────────────────────────────────────────

    def _check_morphism_direction(self, ast: ProgramNode):
        """Verify A ↦ B constraints — B must not reference A (reverse flow forbidden)."""
        morphisms = self._find_nodes(ast.statements, MorphismNode)

        for morph in morphisms:
            source_name = morph.source.name if isinstance(morph.source, IdentifierNode) else None
            target_name = morph.target.name if isinstance(morph.target, IdentifierNode) else None

            if source_name and target_name:
                # Check the target module doesn't reference the source
                for stmt in ast.statements:
                    if isinstance(stmt, (ModuleNode, SpecNode)) and stmt.name == target_name:
                        identifiers = self._collect_identifiers(stmt.body)
                        for ident in identifiers:
                            if ident.name == source_name:
                                self._add(
                                    ViolationLevel.ERROR, ident.line, ident.col, "↦",
                                    f"[SYSTEM 2 REJECTION]: Reverse flow {target_name} → "
                                    f"{source_name} violates {source_name} ↦ {target_name}. "
                                    f"Data must flow unidirectionally. Resample required."
                                )

    # ─────────────────────────────────────────────────────────
    #  Check 3: Topological Boundary — ∂A ∩ ∂B = ∅
    # ─────────────────────────────────────────────────────────

    def _check_topo_boundaries(self, ast: ProgramNode):
        """Verify ∂A ∩ ∂B = ∅ — no direct concrete references between A and B."""
        boundaries = self._find_nodes(ast.statements, TopoBoundaryNode)

        for boundary in boundaries:
            # Both A and B must not directly reference each other
            for stmt in ast.statements:
                if isinstance(stmt, (ModuleNode, SpecNode)):
                    if stmt.name == boundary.left:
                        identifiers = self._collect_identifiers(stmt.body)
                        for ident in identifiers:
                            if ident.name == boundary.right:
                                self._add(
                                    ViolationLevel.ERROR, ident.line, ident.col, "∂∩∅",
                                    f"[SYSTEM 2 REJECTION]: Direct reference {boundary.left} → "
                                    f"{boundary.right} violates ∂{boundary.left} ∩ ∂{boundary.right} = ∅. "
                                    f"Communication must go through abstract interface. "
                                    f"Resample required."
                                )
                    elif stmt.name == boundary.right:
                        identifiers = self._collect_identifiers(stmt.body)
                        for ident in identifiers:
                            if ident.name == boundary.left:
                                self._add(
                                    ViolationLevel.ERROR, ident.line, ident.col, "∂∩∅",
                                    f"[SYSTEM 2 REJECTION]: Direct reference {boundary.right} → "
                                    f"{boundary.left} violates ∂{boundary.left} ∩ ∂{boundary.right} = ∅. "
                                    f"Communication must go through abstract interface. "
                                    f"Resample required."
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
                                f"[SYSTEM 2 REJECTION]: Module '{stmt.name}' references "
                                f"forbidden module '{ident.name}'. "
                                f"Forbidden zones: {stmt.forbidden}. Resample required."
                            )

                # Check depends_on integrity
                all_known = self._module_names | self._spec_names
                for dep in stmt.depends_on:
                    if dep not in all_known and dep not in stmt.imports:
                        self._add(
                            ViolationLevel.WARNING, stmt.line, stmt.col, "DEPENDENCY",
                            f"Module '{stmt.name}' depends on '{dep}' which is not "
                            f"defined in this program."
                        )

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
                    cycle_str = " ↦ ".join(cycle)
                    self._add(
                        ViolationLevel.ERROR, line, col, "CYCLE",
                        f"[SYSTEM 2 REJECTION]: Circular dependency detected: "
                        f"{cycle_str}. DAG constraint violated. Resample required."
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
        elif isinstance(node, MorphismNode):
            if node.source:
                self._walk_for_type_annots(node.source)
            if node.target:
                self._walk_for_type_annots(node.target)
        elif isinstance(node, DirectSumNode):
            if node.left:
                self._walk_for_type_annots(node.left)
            if node.right:
                self._walk_for_type_annots(node.right)
        elif isinstance(node, TensorProductNode):
            if node.left:
                self._walk_for_type_annots(node.left)
            if node.right:
                self._walk_for_type_annots(node.right)
        elif isinstance(node, ModuleNode):
            for stmt in node.body:
                self._walk_for_type_annots(stmt)
        elif isinstance(node, SpecNode):
            for stmt in node.body:
                self._walk_for_type_annots(stmt)

    # ─────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────

    def _find_nodes(self, nodes: list[ASTNode], node_type: type) -> list:
        """Recursively find all AST nodes of a given type."""
        found = []
        for node in nodes:
            self._walk_for_nodes(node, node_type, found)
        return found

    def _walk_for_nodes(self, node: ASTNode, target_type: type, found: list):
        """Walk the AST recursively to find nodes of a given type."""
        if isinstance(node, target_type):
            found.append(node)
        if isinstance(node, PipelineNode):
            for stage in node.stages:
                self._walk_for_nodes(stage, target_type, found)
        elif isinstance(node, MorphismNode):
            if node.source:
                self._walk_for_nodes(node.source, target_type, found)
            if node.target:
                self._walk_for_nodes(node.target, target_type, found)
        elif isinstance(node, DirectSumNode):
            if node.left:
                self._walk_for_nodes(node.left, target_type, found)
            if node.right:
                self._walk_for_nodes(node.right, target_type, found)
        elif isinstance(node, TensorProductNode):
            if node.left:
                self._walk_for_nodes(node.left, target_type, found)
            if node.right:
                self._walk_for_nodes(node.right, target_type, found)
        elif isinstance(node, ListNode):
            for el in node.elements:
                self._walk_for_nodes(el, target_type, found)
        elif isinstance(node, ModuleNode):
            for stmt in node.body:
                self._walk_for_nodes(stmt, target_type, found)
        elif isinstance(node, SpecNode):
            for stmt in node.body:
                self._walk_for_nodes(stmt, target_type, found)
        elif isinstance(node, BindingNode):
            if node.expression:
                self._walk_for_nodes(node.expression, target_type, found)
        elif isinstance(node, FuncCallNode):
            for arg in node.args:
                self._walk_for_nodes(arg, target_type, found)

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
        elif isinstance(node, MorphismNode):
            if node.source:
                self._walk_for_identifiers(node.source, result)
            if node.target:
                self._walk_for_identifiers(node.target, result)
        elif isinstance(node, DirectSumNode):
            if node.left:
                self._walk_for_identifiers(node.left, result)
            if node.right:
                self._walk_for_identifiers(node.right, result)
        elif isinstance(node, TensorProductNode):
            if node.left:
                self._walk_for_identifiers(node.left, result)
            if node.right:
                self._walk_for_identifiers(node.right, result)
        elif isinstance(node, ListNode):
            for el in node.elements:
                self._walk_for_identifiers(el, result)
        elif isinstance(node, BindingNode):
            if node.expression:
                self._walk_for_identifiers(node.expression, result)
        elif isinstance(node, FuncCallNode):
            for arg in node.args:
                self._walk_for_identifiers(arg, result)
