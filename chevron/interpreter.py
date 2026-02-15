"""
Chevron Interpreter
===================
Tree-walking interpreter that executes the AST produced by the Parser.
Implements the 5 Chevron primitives with full semantic behavior.

Extensions:
  - Module scope isolation (bindings are local unless exported)
  - Spec mode (parsed but not executed — metadata recorded for verifier)
  - Function calls in predicates
  - Type annotations (pass-through in execution)
"""
import sys
from typing import Any, Callable

from .glyphs import GLYPH_REGISTRY, GlyphType
from .parser import (
    ASTNode, ProgramNode, GlyphNode, PipelineNode, BindingNode,
    LiteralNode, ListNode, PredicateNode, IdentifierNode, PlaceholderNode,
    ModuleNode, SpecNode, TypeDeclNode, TypeAnnotNode, ConstraintNode,
    FuncCallNode,
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
        self.witness_log: list[str] = []
        self.pipe_value: Any = None  # Current value in pipeline
        self.output_fn = output_fn or (lambda s: print(s))
        self.specs: dict[str, SpecNode] = {}  # Collected spec metadata
        self.types: dict[str, TypeDeclNode] = {}  # Collected type declarations
        self.modules: dict[str, ModuleScope] = {}  # Active module scopes
        self._current_scope: ModuleScope | None = None  # Active module scope

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
            if isinstance(stage, GlyphNode):
                # Inject piped value as first arg if glyph has no explicit data arg
                value = self._exec_glyph_with_pipe(stage, value)
            elif isinstance(stage, IdentifierNode):
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
            else:
                value = self.execute(stage)
        self.pipe_value = None
        return value

    # ─────────────────────────────────────────────────────────
    #  Glyph Execution
    # ─────────────────────────────────────────────────────────

    def _exec_glyph(self, node: GlyphNode) -> Any:
        """Execute a glyph invocation."""
        glyph_info = GLYPH_REGISTRY.get(node.glyph)
        if glyph_info is None:
            raise ChevronError(f"Unknown glyph: {node.glyph} at L{node.line}:{node.col}")

        match glyph_info.glyph_type:
            case GlyphType.ORIGIN:
                return self._exec_origin(node)
            case GlyphType.WITNESS:
                return self._exec_witness(node)
            case GlyphType.WEAVER:
                return self._exec_weaver(node)
            case GlyphType.FILTER:
                return self._exec_filter(node)
            case GlyphType.FOLD:
                return self._exec_fold(node)
            case _:
                raise ChevronError(f"Unimplemented glyph type: {glyph_info.glyph_type}")

    def _exec_glyph_with_pipe(self, node: GlyphNode, piped: Any) -> Any:
        """Execute a glyph in pipeline context, injecting the piped value."""
        glyph_info = GLYPH_REGISTRY.get(node.glyph)
        if glyph_info is None:
            raise ChevronError(f"Unknown glyph: {node.glyph}")

        match glyph_info.glyph_type:
            case GlyphType.WITNESS:
                return self._do_witness(piped)
            case GlyphType.WEAVER:
                if node.args:
                    # ☤ with args in pipeline: weave piped with args
                    other = self.execute(node.args[0])
                    return self._do_weave(piped, other)
                return piped
            case GlyphType.FILTER:
                if node.args:
                    predicate = node.args[0]
                    return self._do_filter(predicate, piped)
                return piped
            case GlyphType.FOLD:
                if len(node.args) >= 2:
                    pred = node.args[0]
                    transform = node.args[1]
                    return self._do_fold(pred, transform, piped)
                return piped
            case GlyphType.ORIGIN:
                # Origin in pipeline just passes through
                return piped
            case _:
                return self.execute(node)

    # ─────────────────────────────────────────────────────────
    #  Primitive Implementations
    # ─────────────────────────────────────────────────────────

    def _exec_origin(self, node: GlyphNode) -> Any:
        """◬ The Origin — produce the initial data stream."""
        if not node.args:
            return None
        if len(node.args) == 1:
            return self.execute(node.args[0])
        return [self.execute(a) for a in node.args]

    def _exec_witness(self, node: GlyphNode) -> Any:
        """𓂀 The Witness — observe without modifying."""
        if node.args:
            value = self.execute(node.args[0])
        elif self.pipe_value is not None:
            value = self.pipe_value
        else:
            value = None
        return self._do_witness(value)

    def _do_witness(self, value: Any) -> Any:
        """Core Witness behavior: log and pass through."""
        display = self._format_value(value)
        log_line = f"\U000130C0 \u27EB {display}"
        self.witness_log.append(log_line)
        self.output_fn(log_line)
        return value

    def _exec_weaver(self, node: GlyphNode) -> Any:
        """☤ The Weaver — merge values."""
        if not node.args:
            if self.pipe_value is not None:
                return self.pipe_value
            return None

        arg = self.execute(node.args[0])

        if isinstance(arg, list):
            # Weave a list of items together
            return self._do_weave_list(arg)

        # If two separate args
        if len(node.args) >= 2:
            other = self.execute(node.args[1])
            return self._do_weave(arg, other)

        return arg

    def _do_weave(self, a: Any, b: Any) -> Any:
        """Core Weaver behavior: merge two values."""
        if isinstance(a, str) and isinstance(b, str):
            return f"{a} {b}"
        if isinstance(a, list) and isinstance(b, list):
            return a + b
        if isinstance(a, list):
            return a + [b]
        if isinstance(b, list):
            return [a] + b
        return f"{a} {b}"

    def _do_weave_list(self, items: list) -> Any:
        """Weave a list of items into one."""
        if all(isinstance(x, str) for x in items):
            return " ".join(items)
        if all(isinstance(x, list) for x in items):
            result = []
            for sub in items:
                result.extend(sub)
            return result
        return " ".join(str(x) for x in items)

    def _exec_filter(self, node: GlyphNode) -> Any:
        """Ө The Filter — conditional gate."""
        if len(node.args) < 2:
            if len(node.args) == 1 and self.pipe_value is not None:
                return self._do_filter(node.args[0], self.pipe_value)
            raise ChevronError(f"Ө requires (predicate, data) at L{node.line}:{node.col}")

        predicate = node.args[0]
        data = self.execute(node.args[1])
        return self._do_filter(predicate, data)

    def _do_filter(self, predicate_node: ASTNode, data: Any) -> Any:
        """Core Filter behavior: apply predicate, pass only matching data."""
        pred_fn = self._build_predicate(predicate_node)

        if isinstance(data, list):
            return [item for item in data if pred_fn(item)]

        # Single value: pass or reject
        return data if pred_fn(data) else None

    def _exec_fold(self, node: GlyphNode) -> Any:
        """☾ Fold Time — recursion."""
        if len(node.args) < 3:
            raise ChevronError(f"☾ requires (predicate, transform, value) at L{node.line}:{node.col}")

        pred = node.args[0]
        transform = node.args[1]
        value = self.execute(node.args[2])
        return self._do_fold(pred, transform, value)

    def _do_fold(self, pred_node: ASTNode, transform_node: ASTNode, value: Any) -> Any:
        """Core Fold Time behavior: recursive application."""
        pred_fn = self._build_predicate(pred_node)
        transform_fn = self._build_transform(transform_node)

        results = [value]
        max_iterations = 10000  # Safety limit
        i = 0

        while pred_fn(value) and i < max_iterations:
            value = transform_fn(value)
            results.append(value)
            i += 1

        if i >= max_iterations:
            raise ChevronError("☾ Fold Time exceeded maximum iterations (possible infinite loop)")

        return value

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
            return "\u2205"
        if isinstance(value, dict):
            # Spec metadata dict
            items = [f"{k}: {self._format_value(v)}" for k, v in value.items()]
            return "{" + ", ".join(items) + "}"
        return str(value)
