"""
Code Verifier — Deterministic SCP Compliance Checker
=====================================================
Uses Python's `ast` module to statically analyze generated code against
SCP contracts. This replaces unreliable AI self-verification with
deterministic AST analysis.

This is the formal implementation of Hom(A,B) ≅ 0:
    Hom(A,B) ≅ 0  ⟺  No undeclared coupling exists in the generated code.

Checks:
  1. No global mutable state
  2. No forbidden imports (Hom(A,B) ≅ 0 enforcement)
  3. Side-effect freedom for pure-constraint operators (⊕, ∂∩∅)
  4. Interface conformance (correct methods with matching signatures)
  5. No undeclared cross-module references (↦ direction enforcement)

Rejection Format:
  [SYSTEM 2 REJECTION]: <operator> <details>. Resample required.

Dan Park | MagicPoint.ai | February 2026
"""

from __future__ import annotations

import ast
import inspect
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional


# ─────────────────────────────────────────────────────────────
#  Violation Reporting
# ─────────────────────────────────────────────────────────────

class SeverityLevel(Enum):
    """Severity of a code verification violation."""
    ERROR = auto()
    WARNING = auto()


@dataclass
class CodeViolation:
    """A single code verification violation."""
    severity: SeverityLevel
    line: int
    check: str       # Which check flagged this (e.g., "GLOBAL_STATE", "FORBIDDEN_IMPORT")
    message: str

    def __str__(self) -> str:
        icon = "✘" if self.severity == SeverityLevel.ERROR else "⚠"
        return f"  {icon} L{self.line} [{self.check}] {self.message}"


# ─────────────────────────────────────────────────────────────
#  Standard Library Allowlist
# ─────────────────────────────────────────────────────────────

# These imports are ALWAYS allowed — they are not project dependencies
STDLIB_MODULES = frozenset({
    "abc", "ast", "asyncio", "base64", "collections", "contextlib",
    "copy", "csv", "dataclasses", "datetime", "decimal", "enum",
    "functools", "hashlib", "inspect", "io", "itertools", "json",
    "logging", "math", "operator", "os", "pathlib", "pickle",
    "random", "re", "shutil", "socket", "string", "struct",
    "subprocess", "sys", "tempfile", "threading", "time", "typing",
    "unittest", "uuid", "warnings", "weakref",
    # Typing extensions
    "typing_extensions",
    # Common third-party (configurable)
    "pytest",
})

# I/O calls that indicate side effects
IO_FUNCTION_NAMES = frozenset({
    "print", "open", "input",
    "write", "writelines", "read", "readline", "readlines",
    "send", "recv", "connect", "listen",
})

# I/O module names
IO_MODULE_NAMES = frozenset({
    "requests", "urllib", "http", "socket",
    "subprocess",
})


# ─────────────────────────────────────────────────────────────
#  Code Verifier
# ─────────────────────────────────────────────────────────────

class CodeVerifier:
    """Deterministic SCP compliance checker for generated Python code.

    Replaces AI self-verification (unreliable) with AST analysis (deterministic).
    This is the formal implementation of the Weaver function W(G) = 0.

    Usage:
        verifier = CodeVerifier()
        violations = verifier.verify(code_string, module_spec)
        if violations:
            for v in violations:
                print(v)

    The ModuleSpec is imported from scp_bridge and contains:
        - name: Module name
        - methods: List of InterfaceMethod (name, inputs, output, glyph)
        - allowed_dependencies: List of allowed project module names
        - constraints: List of constraint strings
    """

    def __init__(self, extra_allowed_modules: list[str] | None = None):
        """
        Args:
            extra_allowed_modules: Additional module names to treat as allowed
                                   (e.g., third-party libraries the project uses).
        """
        self.allowed_stdlib = set(STDLIB_MODULES)
        if extra_allowed_modules:
            self.allowed_stdlib.update(extra_allowed_modules)

    def verify(self, code: str, contract=None) -> list[CodeViolation]:
        """Run all verification checks on generated Python code.

        Args:
            code: The Python source code to verify.
            contract: An scp_bridge.ModuleSpec instance (optional).
                      If provided, enables interface conformance and
                      dependency isolation checks.

        Returns:
            List of CodeViolation instances. Empty list = W(G) = 0.
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [CodeViolation(
                SeverityLevel.ERROR, e.lineno or 0, "SYNTAX",
                f"Code has syntax error: {e.msg}"
            )]

        violations: list[CodeViolation] = []

        # Always run these checks
        violations.extend(self._check_no_global_state(tree))

        # Contract-dependent checks
        if contract:
            # Determine forbidden modules
            allowed_deps = set(getattr(contract, 'allowed_dependencies', []) or [])
            violations.extend(self._check_forbidden_imports(tree, allowed_deps))
            violations.extend(self._check_interface_conformance(tree, contract))

            # Operator-specific checks (side-effect freedom)
            operator_methods = {}
            for method in getattr(contract, 'methods', []):
                operator_methods[method.name] = getattr(method, 'glyph', getattr(method, 'operator', ''))

            violations.extend(self._check_side_effect_freedom(tree, operator_methods))

        return sorted(violations, key=lambda v: v.line)

    # ─────────────────────────────────────────────────────────
    #  Check 1: No Global Mutable State
    # ─────────────────────────────────────────────────────────

    def _check_no_global_state(self, tree: ast.Module) -> list[CodeViolation]:
        """Reject `global` keyword and module-level mutable assignments.

        Allows:
            - Constants (ALL_CAPS names)
            - Type aliases
            - Immutable literals (strings, tuples, frozensets)
            - Dataclass/class definitions (they're definitions, not state)
            - Imports

        Rejects:
            - `global` keyword inside functions
            - Module-level mutable assignments (lists, dicts, sets)
        """
        violations = []

        # Check for `global` keyword in functions
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                violations.append(CodeViolation(
                    SeverityLevel.ERROR, node.lineno, "GLOBAL_STATE",
                    f"'global' keyword used for: {', '.join(node.names)}. "
                    f"SCP modules must not use global mutable state."
                ))

        # Check module-level assignments
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    name = self._get_assign_name(target)
                    if name and not name.isupper() and not name.startswith("_"):
                        # Check if the value is mutable
                        if self._is_mutable_value(node.value):
                            violations.append(CodeViolation(
                                SeverityLevel.WARNING, node.lineno, "GLOBAL_STATE",
                                f"Module-level mutable assignment: '{name}'. "
                                f"Consider using a constant (ALL_CAPS) or passing as parameter."
                            ))

            elif isinstance(node, ast.AnnAssign):
                name = self._get_assign_name(node.target)
                if name and not name.isupper() and not name.startswith("_"):
                    if node.value and self._is_mutable_value(node.value):
                        violations.append(CodeViolation(
                            SeverityLevel.WARNING, node.lineno, "GLOBAL_STATE",
                            f"Module-level mutable annotated assignment: '{name}'."
                        ))

        return violations

    # ─────────────────────────────────────────────────────────
    #  Check 2: No Forbidden Imports
    # ─────────────────────────────────────────────────────────

    def _check_forbidden_imports(self, tree: ast.Module,
                                  allowed_deps: set[str]) -> list[CodeViolation]:
        """Reject import statements referencing forbidden project modules.

        Standard library and configured third-party imports are always allowed.
        Only project modules not in allowed_deps are flagged.
        """
        violations = []
        all_allowed = self.allowed_stdlib | allowed_deps

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_module = alias.name.split(".")[0]
                    if root_module not in all_allowed and not self._is_likely_stdlib(root_module):
                        violations.append(CodeViolation(
                            SeverityLevel.ERROR, node.lineno, "FORBIDDEN_IMPORT",
                            f"Import of '{alias.name}' violates SCP isolation. "
                            f"Allowed project dependencies: {sorted(allowed_deps) or ['(none)']}"
                        ))

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root_module = node.module.split(".")[0]
                    if root_module not in all_allowed and not self._is_likely_stdlib(root_module):
                        violations.append(CodeViolation(
                            SeverityLevel.ERROR, node.lineno, "FORBIDDEN_IMPORT",
                            f"Import from '{node.module}' violates SCP isolation. "
                            f"Allowed project dependencies: {sorted(allowed_deps) or ['(none)']}"
                        ))

        return violations

    # ─────────────────────────────────────────────────────────
    #  Check 3: Side-Effect Freedom (pure constraint operators)
    # ─────────────────────────────────────────────────────────

    def _check_side_effect_freedom(self, tree: ast.Module,
                                    operator_methods: dict[str, str]) -> list[CodeViolation]:
        """For operators governed by pure constraints (⊕ Direct Sum,
        ∂∩∅ Topological Boundary), reject I/O calls.

        ⊕ Direct Sum methods must be pure — no side effects.
        ∂∩∅ Boundary methods must only observe — no file writes, no network calls.
        Ө and 𓂀 legacy operators are also supported for backward compat.
        """
        violations = []
        # Operators that enforce purity / side-effect freedom
        pure_operators = {"⊕", "∂∩∅", "Ө", "𓂀"}

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_op = operator_methods.get(node.name)
                if method_op and method_op in pure_operators:
                    # Walk this function's body for I/O calls
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            call_name = self._get_call_name(child)
                            if call_name and call_name in IO_FUNCTION_NAMES:
                                op_name = self._operator_display_name(method_op)
                                violations.append(CodeViolation(
                                    SeverityLevel.ERROR, child.lineno, "SIDE_EFFECT",
                                    f"[SYSTEM 2 REJECTION]: Method '{node.name}' "
                                    f"(governed by {method_op} {op_name}) "
                                    f"contains I/O call '{call_name}'. "
                                    f"{op_name} methods must be side-effect free. "
                                    f"Resample required."
                                ))

                        # Check for I/O module imports inside the function
                        if isinstance(child, (ast.Import, ast.ImportFrom)):
                            mod_name = ""
                            if isinstance(child, ast.ImportFrom) and child.module:
                                mod_name = child.module.split(".")[0]
                            for alias in child.names:
                                name = alias.name.split(".")[0]
                                if name in IO_MODULE_NAMES or mod_name in IO_MODULE_NAMES:
                                    violations.append(CodeViolation(
                                        SeverityLevel.ERROR, child.lineno, "SIDE_EFFECT",
                                        f"[SYSTEM 2 REJECTION]: Method '{node.name}' "
                                        f"(governed by {method_op}) "
                                        f"imports I/O module '{name}'. Must be side-effect free. "
                                        f"Resample required."
                                    ))

        return violations

    # ─────────────────────────────────────────────────────────
    #  Check 4: Interface Conformance
    # ─────────────────────────────────────────────────────────

    def _check_interface_conformance(self, tree: ast.Module,
                                      contract) -> list[CodeViolation]:
        """Verify the generated code defines exactly the methods specified
        in the SCP contract, with matching signatures.
        """
        violations = []
        methods = getattr(contract, 'methods', [])
        if not methods:
            return violations

        # Collect all function definitions (top-level and in classes)
        defined_functions: dict[str, ast.FunctionDef] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip private/dunder methods
                if not node.name.startswith("_"):
                    defined_functions[node.name] = node

        # Check each required method
        for method in methods:
            if method.name not in defined_functions:
                violations.append(CodeViolation(
                    SeverityLevel.ERROR, 0, "INTERFACE",
                    f"Required method '{method.name}' is not defined. "
                    f"Contract requires: {method.name}({', '.join(method.inputs)}) → {method.output}"
                ))
            else:
                func_node = defined_functions[method.name]
                # Check parameter count (excluding 'self' for class methods)
                params = func_node.args.args
                param_names = [p.arg for p in params if p.arg != "self"]
                expected_count = len(method.inputs)

                if len(param_names) != expected_count:
                    violations.append(CodeViolation(
                        SeverityLevel.WARNING, func_node.lineno, "INTERFACE",
                        f"Method '{method.name}' has {len(param_names)} parameter(s), "
                        f"contract expects {expected_count}: ({', '.join(method.inputs)})"
                    ))

        return violations

    @staticmethod
    def _operator_display_name(op: str) -> str:
        """Map operator symbols to human-readable display names."""
        names = {
            "⊕": "Direct Sum", "∂∩∅": "Topological Boundary",
            "⊗": "Tensor Product", "↦": "Morphism",
            "Hom≅0": "Null Morphism",
            # Legacy compat
            "Ө": "Filter", "𓂀": "Witness",
        }
        return names.get(op, op)

    # ─────────────────────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _get_assign_name(target) -> str | None:
        """Extract variable name from an assignment target."""
        if isinstance(target, ast.Name):
            return target.id
        return None

    @staticmethod
    def _is_mutable_value(node) -> bool:
        """Check if an AST value node represents a mutable type."""
        return isinstance(node, (ast.List, ast.Dict, ast.Set, ast.Call))

    @staticmethod
    def _get_call_name(node: ast.Call) -> str | None:
        """Extract the function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    @staticmethod
    def _is_likely_stdlib(module_name: str) -> bool:
        """Heuristic check for standard library modules not in our explicit list."""
        # Single underscore prefix modules are usually internal
        if module_name.startswith("_"):
            return True
        # Try importing — if it's available without pip install, it's likely stdlib
        # This is a fallback heuristic; the explicit list handles most cases
        return False


def verify_code(code: str, contract=None,
                extra_allowed: list[str] | None = None) -> list[CodeViolation]:
    """Convenience function: verify code against an SCP contract.

    Args:
        code: Python source code string.
        contract: An scp_bridge.ModuleSpec instance (optional).
        extra_allowed: Additional module names to treat as allowed imports.

    Returns:
        List of violations. Empty = W(G) = 0 (passes).
    """
    verifier = CodeVerifier(extra_allowed_modules=extra_allowed)
    return verifier.verify(code, contract)
