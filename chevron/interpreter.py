"""
Chevron Interpreter — Non-Polysemic Topological DSL
=====================================================
Tree-walking interpreter that executes the AST produced by the Parser.
Implements Topo-Categorical constraint operators and standard pipeline
execution.

New Operators (structural constraints — recorded as metadata):
    Hom(A,B) ≅ 0   — Records strict isolation constraint
    A ↦ B           — Records directed data flow / acts as pipeline
    A ⊕ B           — Records decoupled coexistence
    A ⊗ B           — Records state entanglement (returns merged value)
    ∂A ∩ ∂B = ∅     — Records interface encapsulation constraint

Extensions:
  - Module scope isolation (bindings are local unless exported)
  - Spec mode (parsed but not executed — metadata recorded for verifier)
  - Function calls in predicates
  - Type annotations (pass-through in execution)
"""
import sys
from typing import Any, Callable

from .glyphs import OPERATOR_REGISTRY, OperatorType
from .parser import (
    ASTNode, ProgramNode, PipelineNode, BindingNode,
    LiteralNode, ListNode, PredicateNode, IdentifierNode, PlaceholderNode,
    ModuleNode, SpecNode, TypeDeclNode, TypeAnnotNode, ConstraintNode,
    FuncCallNode,
    NullMorphismNode, MorphismNode, DirectSumNode, TensorProductNode,
    TopoBoundaryNode,
)


class ChevronError(Exception):
    """Runtime error during Chevron execution."""
    pass


class ModuleScope:
    """An isolated namespace for a module.

    Bindings inside a module are only visible within the module
    unless explicitly listed in exports. Imports bring in bindings
    from other modules or the global scope.
    """
    def __init__(self, name: str, parent_env: dict[str, Any],
                 imports: list[str], exports: list[str]):
        self.name = name
        self.exports = exports
        self.local_env: dict[str, Any] = {}
        # Import bindings from parent
        for imp in imports:
            if imp in parent_env:
                self.local_env[imp] = parent_env[imp]

    def get(self, name: str) -> Any:
        if name in self.local_env:
            return self.local_env[name]
        raise ChevronError(f"Undefined binding: '{name}' in module '{self.name}'")

    def set(self, name: str, value: Any):
        self.local_env[name] = value

    def get_exported(self) -> dict[str, Any]:
        """Return only the exported bindings."""
        if not self.exports:
            return dict(self.local_env)
        return {k: v for k, v in self.local_env.items() if k in self.exports}


class Interpreter:
    """
    Tree-walking interpreter for Chevron programs.

    Usage:
        interp = Interpreter()
        result = interp.execute(ast)
    """

    def __init__(self, output_fn: Callable[[str], None] | None = None):
        self.env: dict[str, Any] = {}
        self.constraint_log: list[str] = []
        self.pipe_value: Any = None  # Current value in pipeline
        self.output_fn = output_fn or (lambda s: print(s))
        self.specs: dict[str, SpecNode] = {}  # Collected spec metadata
        self.types: dict[str, TypeDeclNode] = {}  # Collected type declarations
        self.modules: dict[str, ModuleScope] = {}  # Active module scopes
        self._current_scope: ModuleScope | None = None  # Active module scope
        # Topo-Categorical constraint records
        self.null_morphisms: list[dict] = []
        self.morphisms: list[dict] = []
        self.direct_sums: list[dict] = []
        self.tensor_products: list[dict] = []
        self.topo_boundaries: list[dict] = []

    def execute(self, node: ASTNode) -> Any:
        """Execute an AST node and return the result."""
        method = f"_exec_{node.node_type.lower()}"
        executor = getattr(self, method, None)
        if executor is None:
            raise ChevronError(f"Unknown node type: {node.node_type}")
        return executor(node)

    # ─────────────────────────────────────────────────────────
    #  Program & Statements
    # ─────────────────────────────────────────────────────────

    def _exec_program(self, node: ProgramNode) -> Any:
        """Execute all statements in a program."""
        result = None
        for stmt in node.statements:
            result = self.execute(stmt)
        return result

    def _exec_binding(self, node: BindingNode) -> Any:
        """Execute a binding: Name ← expression."""
        value = self.execute(node.expression)
        if self._current_scope is not None:
            self._current_scope.set(node.name, value)
        else:
            self.env[node.name] = value
        return value

    def _exec_identifier(self, node: IdentifierNode) -> Any:
        """Look up a named binding."""
        if self._current_scope is not None:
            try:
                return self._current_scope.get(node.name)
            except ChevronError:
                pass  # Fall through to global
        if node.name in self.env:
            return self.env[node.name]
        raise ChevronError(f"Undefined binding: '{node.name}' at L{node.line}:{node.col}")

    # ─────────────────────────────────────────────────────────
    #  Pipeline
    # ─────────────────────────────────────────────────────────

    def _exec_pipeline(self, node: PipelineNode) -> Any:
        """Execute a pipeline: stage1 → stage2 → stage3."""
        value = self.execute(node.stages[0])
        for stage in node.stages[1:]:
            self.pipe_value = value
            if isinstance(stage, IdentifierNode):
                # Look up binding — if it's callable, apply it
                if self._current_scope is not None:
                    try:
                        bound = self._current_scope.get(stage.name)
                    except ChevronError:
                        bound = self.env.get(stage.name)
                else:
                    bound = self.env.get(stage.name)
                if callable(bound):
                    value = bound(value)
                else:
                    value = bound
            elif isinstance(stage, TypeAnnotNode):
                # Type annotations are pass-through in execution
                pass
            elif isinstance(stage, FuncCallNode):
                # Function call in pipeline — execute with piped value
                value = self._exec_func_call_with_pipe(stage, value)
            elif isinstance(stage, PredicateNode):
                # Predicate in pipeline — filter the piped value
                pred_fn = self._build_predicate(stage)
                if isinstance(value, list):
                    value = [item for item in value if pred_fn(item)]
                else:
                    value = value if pred_fn(value) else None
            else:
                value = self.execute(stage)
        self.pipe_value = None
        return value

    # ─────────────────────────────────────────────────────────
    #  Topo-Categorical Operator Execution
    # ─────────────────────────────────────────────────────────

    def _exec_nullmorphism(self, node: NullMorphismNode) -> Any:
        """Hom(A, B) ≅ 0 — Record strict isolation constraint."""
        record = {"source": node.source, "target": node.target,
                  "line": node.line, "col": node.col}
        self.null_morphisms.append(record)
        msg = f"Hom({node.source}, {node.target}) ≅ 0  [Strict Isolation]"
        self.constraint_log.append(msg)
        self.output_fn(f"⊘ {msg}")
        return record

    def _exec_morphism(self, node: MorphismNode) -> Any:
        """A ↦ B — Directed data flow. Evaluates both sides, returns target."""
        source_val = self.execute(node.source) if node.source else None
        target_val = self.execute(node.target) if node.target else None
        # Record the morphism
        source_name = node.source.name if isinstance(node.source, IdentifierNode) else str(source_val)
        target_name = node.target.name if isinstance(node.target, IdentifierNode) else str(target_val)
        record = {"source": source_name, "target": target_name,
                  "line": node.line, "col": node.col}
        self.morphisms.append(record)
        # In pipeline context, pass through the target value
        return target_val if target_val is not None else source_val

    def _exec_directsum(self, node: DirectSumNode) -> Any:
        """A ⊕ B — Decoupled coexistence. Evaluates both, returns tuple."""
        left_val = self.execute(node.left) if node.left else None
        right_val = self.execute(node.right) if node.right else None
        left_name = node.left.name if isinstance(node.left, IdentifierNode) else str(left_val)
        right_name = node.right.name if isinstance(node.right, IdentifierNode) else str(right_val)
        record = {"left": left_name, "right": right_name,
                  "line": node.line, "col": node.col}
        self.direct_sums.append(record)
        msg = f"{left_name} ⊕ {right_name}  [Decoupled Coexistence]"
        self.constraint_log.append(msg)
        # Return both values as a list (orthogonal state spaces)
        result = []
        if isinstance(left_val, list):
            result.extend(left_val)
        elif left_val is not None:
            result.append(left_val)
        if isinstance(right_val, list):
            result.extend(right_val)
        elif right_val is not None:
            result.append(right_val)
        return result

    def _exec_tensorproduct(self, node: TensorProductNode) -> Any:
        """A ⊗ B — State entanglement. Evaluates both, returns merged value."""
        left_val = self.execute(node.left) if node.left else None
        right_val = self.execute(node.right) if node.right else None
        left_name = node.left.name if isinstance(node.left, IdentifierNode) else str(left_val)
        right_name = node.right.name if isinstance(node.right, IdentifierNode) else str(right_val)
        record = {"left": left_name, "right": right_name,
                  "line": node.line, "col": node.col}
        self.tensor_products.append(record)
        msg = f"{left_name} ⊗ {right_name}  [State Entanglement]"
        self.constraint_log.append(msg)
        # Tensor product merges the values
        if isinstance(left_val, list) and isinstance(right_val, list):
            return left_val + right_val
        if isinstance(left_val, str) and isinstance(right_val, str):
            return f"{left_val} {right_val}"
        if isinstance(left_val, list):
            return left_val + [right_val]
        if isinstance(right_val, list):
            return [left_val] + right_val
        return f"{left_val} {right_val}"

    def _exec_topoboundary(self, node: TopoBoundaryNode) -> Any:
        """∂A ∩ ∂B = ∅ — Record interface encapsulation constraint."""
        record = {"left": node.left, "right": node.right,
                  "line": node.line, "col": node.col}
        self.topo_boundaries.append(record)
        msg = f"∂{node.left} ∩ ∂{node.right} = ∅  [Interface Encapsulation]"
        self.constraint_log.append(msg)
        self.output_fn(f"∅ {msg}")
        return record

    # ─────────────────────────────────────────────────────────
    #  Predicate & Transform Builders
    # ─────────────────────────────────────────────────────────

    def _build_predicate(self, node: ASTNode) -> Callable[[Any], bool]:
        """Build a predicate function from a PredicateNode or FuncCallNode."""
        if isinstance(node, PredicateNode):
            op = node.operator
            if node.operand is not None:
                operand = self.execute(node.operand) if isinstance(node.operand, ASTNode) else node.operand
            else:
                operand = None

            match op:
                case ">":
                    return lambda x, o=operand: x > o
                case "<":
                    return lambda x, o=operand: x < o
                case "=":
                    return lambda x, o=operand: x == o
                case "!=":
                    return lambda x, o=operand: x != o
                case ">=":
                    return lambda x, o=operand: x >= o
                case "<=":
                    return lambda x, o=operand: x <= o
                case _:
                    # Named predicate — look up in environment
                    bound = self.env.get(op)
                    if callable(bound):
                        return bound
                    raise ChevronError(f"Unknown predicate operator: {op}")

        if isinstance(node, FuncCallNode):
            # Function call as predicate — look up and call with args
            bound = self.env.get(node.func_name)
            if callable(bound):
                eval_args = [self.execute(a) for a in node.args]
                return lambda x, fn=bound, args=eval_args: fn(x, *args)
            raise ChevronError(f"Unknown predicate function: {node.func_name}")

        raise ChevronError(f"Expected predicate, got {node.node_type}")

    def _build_transform(self, node: ASTNode) -> Callable[[Any], Any]:
        """Build a transform function from a PredicateNode (used as math op)."""
        if isinstance(node, PredicateNode):
            op = node.operator
            if node.operand is not None:
                operand = self.execute(node.operand) if isinstance(node.operand, ASTNode) else node.operand
            else:
                operand = None

            match op:
                case "+":
                    return lambda x, o=operand: x + o
                case "-":
                    return lambda x, o=operand: x - o
                case "*":
                    return lambda x, o=operand: x * o
                case "/":
                    return lambda x, o=operand: x / o
                case _:
                    # Named transform — look up in environment
                    bound = self.env.get(op)
                    if callable(bound):
                        return bound
                    raise ChevronError(f"Unknown transform operator: {op}")

        if isinstance(node, FuncCallNode):
            # Function call as transform
            bound = self.env.get(node.func_name)
            if callable(bound):
                eval_args = [self.execute(a) for a in node.args]
                return lambda x, fn=bound, args=eval_args: fn(x, *args)
            raise ChevronError(f"Unknown transform function: {node.func_name}")

        raise ChevronError(f"Expected transform, got {node.node_type}")

    # ─────────────────────────────────────────────────────────
    #  Module & Spec Execution
    # ─────────────────────────────────────────────────────────

    def _exec_module(self, node: ModuleNode) -> Any:
        """Execute a module block with isolated scope."""
        scope = ModuleScope(
            name=node.name,
            parent_env=self.env,
            imports=node.imports,
            exports=node.exports,
        )
        self.modules[node.name] = scope

        # Register types
        for type_decl in node.types:
            self.types[type_decl.type_name] = type_decl

        # Execute body in isolated scope
        prev_scope = self._current_scope
        self._current_scope = scope

        result = None
        for stmt in node.body:
            result = self.execute(stmt)

        self._current_scope = prev_scope

        # Merge exported bindings into global env
        for name, value in scope.get_exported().items():
            self.env[name] = value

        return result

    def _exec_spec(self, node: SpecNode) -> Any:
        """Record a spec block — parse but do NOT execute."""
        self.specs[node.name] = node

        # Register types from spec
        for type_decl in node.types:
            self.types[type_decl.type_name] = type_decl

        # Return metadata dict for inspection
        return {
            "spec": node.name,
            "imports": node.imports,
            "exports": node.exports,
            "depends_on": node.depends_on,
            "forbidden": node.forbidden,
            "constraints": node.constraints,
            "methods": len(node.body),
        }

    def _exec_typedecl(self, node: TypeDeclNode) -> Any:
        """Register a type declaration."""
        self.types[node.type_name] = node
        return None

    def _exec_typeannot(self, node: TypeAnnotNode) -> Any:
        """Type annotation pass-through — documentation only."""
        return self.pipe_value

    def _exec_constraint(self, node: ConstraintNode) -> Any:
        """Constraint pass-through — recorded for verifier."""
        return None

    def _exec_funccall(self, node: FuncCallNode) -> Any:
        """Execute a function call."""
        bound = self.env.get(node.func_name)
        if callable(bound):
            eval_args = [self.execute(a) for a in node.args]
            return bound(*eval_args)
        raise ChevronError(f"Unknown function: {node.func_name} at L{node.line}:{node.col}")

    def _exec_func_call_with_pipe(self, node: FuncCallNode, piped: Any) -> Any:
        """Execute a function call in pipeline context, injecting piped value."""
        bound = self.env.get(node.func_name)
        if callable(bound):
            eval_args = [self.execute(a) for a in node.args]
            return bound(piped, *eval_args)
        raise ChevronError(f"Unknown function: {node.func_name}")

    # ─────────────────────────────────────────────────────────
    #  Literals & Helpers
    # ─────────────────────────────────────────────────────────

    def _exec_literal(self, node: LiteralNode) -> Any:
        """Return the literal value."""
        return node.value

    def _exec_list(self, node: ListNode) -> list:
        """Evaluate all list elements."""
        return [self.execute(el) for el in node.elements]

    def _exec_predicate(self, node: PredicateNode) -> Any:
        """Predicates are handled by the glyph that consumes them."""
        return node  # Return the node itself for the glyph to interpret

    def _exec_placeholder(self, node: PlaceholderNode) -> Any:
        """Return the current piped value."""
        return self.pipe_value

    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        if isinstance(value, list):
            formatted = [self._format_value(v) for v in value]
            return f"[{', '.join(formatted)}]"
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, float):
            if value == int(value):
                return str(int(value))
            return str(value)
        if value is None:
            return "∅"
        if isinstance(value, dict):
            # Spec metadata dict
            items = [f"{k}: {self._format_value(v)}" for k, v in value.items()]
            return "{" + ", ".join(items) + "}"
        return str(value)
