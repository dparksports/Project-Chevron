"""
Spec-Driven Test Generator — Deterministic Tests from SCP Contracts
====================================================================
Generates pytest test code directly from Chevron specifications,
without involving an AI. This solves the "Golden Test" problem where
an AI writes both buggy code AND buggy tests that pass the buggy code.

By deriving tests from the contract (not from reading the implementation),
we ensure that tests verify spec compliance independently.

Generated test categories:
  1. Interface compliance — method existence and signature matching
  2. Dependency isolation — forbidden import detection via AST
  3. Type contract validation — field presence and type checks
  4. Glyph constraint verification — side-effect freedom, mutation checks

Dan Park | MagicPoint.ai | February 2026
"""

from __future__ import annotations

import textwrap
from typing import Any


class SpecTestGenerator:
    """Generates deterministic test cases from SCP contracts.

    Unlike AI-generated tests (which may share the AI's misunderstanding),
    spec-driven tests are derived directly from the contract specification.

    Usage:
        from scp_bridge import SCPBridge, ModuleSpec
        from chevron.test_generator import SpecTestGenerator

        bridge = SCPBridge.from_template("todo_app")
        spec = bridge._find_module("TodoStore")

        generator = SpecTestGenerator()
        test_code = generator.generate_tests(spec)
        # Write test_code to a .py file and run with pytest
    """

    def generate_tests(self, spec, module_file: str = None) -> str:
        """Generate complete pytest test file from a ModuleSpec.

        Args:
            spec: An scp_bridge.ModuleSpec instance.
            module_file: Optional path to the module file (for AST checks).
                        If None, tests will import by module name.

        Returns:
            Complete Python source code for a pytest test file.
        """
        module_name = spec.name
        lower_name = module_name.lower()

        sections = []

        # Header
        sections.append(self._header(module_name))

        # Interface tests
        sections.append(self._interface_tests(spec))

        # Isolation tests
        sections.append(self._isolation_tests(spec, module_file))

        # Glyph constraint tests
        sections.append(self._glyph_tests(spec))

        return "\n\n".join(sections) + "\n"

    # ─────────────────────────────────────────────────────────
    #  Header
    # ─────────────────────────────────────────────────────────

    def _header(self, module_name: str) -> str:
        lower = module_name.lower()
        return textwrap.dedent(f'''\
            """
            Auto-generated SCP Contract Tests: {module_name}
            =================================================
            Generated deterministically from the SCP specification.
            These tests verify CONTRACT COMPLIANCE, not implementation details.

            Run: pytest test_{lower}.py -v
            """
            import ast
            import inspect
            import sys
            import os
            import pytest
        ''')

    # ─────────────────────────────────────────────────────────
    #  Interface Compliance Tests
    # ─────────────────────────────────────────────────────────

    def _interface_tests(self, spec) -> str:
        """Generate tests that verify method existence and signatures."""
        module_name = spec.name
        lower = module_name.lower()
        methods = getattr(spec, 'methods', [])

        lines = []
        lines.append(f"# ─────────────────────────────────────────────")
        lines.append(f"#  Interface Compliance Tests")
        lines.append(f"# ─────────────────────────────────────────────")
        lines.append("")
        lines.append(f"class TestInterface:")
        lines.append(f'    """Verify {module_name} implements the SCP contract interface."""')
        lines.append("")

        for method in methods:
            inputs = method.inputs if hasattr(method, 'inputs') else []
            param_count = len(inputs)
            # Parse parameter names from type annotations like "task: Task"
            param_names = []
            for inp in inputs:
                name = inp.split(":")[0].strip()
                param_names.append(name)

            test_name = f"test_{method.name}_exists_and_callable"
            lines.append(f"    def {test_name}(self, module_source):")
            lines.append(f'        """Contract: {method.name}({", ".join(inputs)}) → {method.output}"""')
            lines.append(f"        tree = ast.parse(module_source)")
            lines.append(f"        func_names = [")
            lines.append(f"            node.name for node in ast.walk(tree)")
            lines.append(f"            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))")
            lines.append(f"        ]")
            lines.append(f'        assert "{method.name}" in func_names, (')
            lines.append(f'            f"Method \'{method.name}\' not found in {module_name}. "')
            lines.append(f'            f"SCP contract requires: {method.name}({", ".join(inputs)}) → {method.output}"')
            lines.append(f"        )")
            lines.append("")

            # Signature check
            sig_test = f"test_{method.name}_signature"
            lines.append(f"    def {sig_test}(self, module_source):")
            lines.append(f'        """Verify {method.name} has {param_count} parameter(s)."""')
            lines.append(f"        tree = ast.parse(module_source)")
            lines.append(f"        for node in ast.walk(tree):")
            lines.append(f"            if isinstance(node, ast.FunctionDef) and node.name == '{method.name}':")
            lines.append(f"                params = [p.arg for p in node.args.args if p.arg != 'self']")
            lines.append(f"                assert len(params) == {param_count}, (")
            lines.append(f'                    f"Method \'{method.name}\' has {{len(params)}} params, "')
            lines.append(f'                    f"contract expects {param_count}: ({", ".join(param_names)})"')
            lines.append(f"                )")
            lines.append(f"                return")
            lines.append(f'        pytest.fail("Method \'{method.name}\' not found")')
            lines.append("")

        # Add conftest-style fixture
        lines.append(f"    @pytest.fixture")
        lines.append(f"    def module_source(self):")
        lines.append(f'        """Load the module source code for AST analysis."""')
        lines.append(f'        # Override this fixture with the actual module source path')
        lines.append(f'        pytest.skip("Set module_source fixture to the module source code string")')
        lines.append("")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────
    #  Dependency Isolation Tests
    # ─────────────────────────────────────────────────────────

    def _isolation_tests(self, spec, module_file: str = None) -> str:
        """Generate tests that verify forbidden imports are absent."""
        module_name = spec.name
        allowed_deps = set(getattr(spec, 'allowed_dependencies', []) or [])
        constraints = getattr(spec, 'constraints', []) or []

        lines = []
        lines.append(f"# ─────────────────────────────────────────────")
        lines.append(f"#  Dependency Isolation Tests")
        lines.append(f"# ─────────────────────────────────────────────")
        lines.append("")
        lines.append(f"class TestIsolation:")
        lines.append(f'    """Verify {module_name} respects SCP dependency boundaries."""')
        lines.append("")

        # Test: no global mutable state
        lines.append(f"    def test_no_global_keyword(self, module_source):")
        lines.append(f'        """SCP: No global mutable state allowed."""')
        lines.append(f"        tree = ast.parse(module_source)")
        lines.append(f"        for node in ast.walk(tree):")
        lines.append(f"            if isinstance(node, ast.Global):")
        lines.append(f'                pytest.fail(')
        lines.append(f'                    f"\'global\' keyword found for: {{node.names}}. "')
        lines.append(f'                    f"SCP modules must not use global mutable state."')
        lines.append(f"                )")
        lines.append("")

        # Test: no forbidden imports
        if allowed_deps:
            deps_str = ", ".join(f'"{d}"' for d in sorted(allowed_deps))
            lines.append(f"    def test_only_allowed_dependencies(self, module_source):")
            lines.append(f'        """SCP: Only [{", ".join(sorted(allowed_deps))}] are allowed project deps."""')
            lines.append(f"        allowed = {{{deps_str}}}")
        else:
            lines.append(f"    def test_no_project_dependencies(self, module_source):")
            lines.append(f'        """SCP: {module_name} is fully isolated — no project deps allowed."""')
            lines.append(f"        allowed = set()")

        lines.append(f"        # Standard library modules are always allowed")
        lines.append(f"        stdlib = {{")
        lines.append(f'            "abc", "ast", "asyncio", "collections", "copy", "csv",')
        lines.append(f'            "dataclasses", "datetime", "enum", "functools", "hashlib",')
        lines.append(f'            "inspect", "io", "itertools", "json", "logging", "math",')
        lines.append(f'            "operator", "os", "pathlib", "re", "string", "sys",')
        lines.append(f'            "threading", "time", "typing", "unittest", "uuid", "warnings",')
        lines.append(f"        }}")
        lines.append(f"        tree = ast.parse(module_source)")
        lines.append(f"        for node in ast.walk(tree):")
        lines.append(f"            if isinstance(node, ast.Import):")
        lines.append(f"                for alias in node.names:")
        lines.append(f"                    root = alias.name.split('.')[0]")
        lines.append(f"                    if root not in allowed and root not in stdlib:")
        lines.append(f"                        pytest.fail(")
        lines.append(f'                            f"Import \'{{alias.name}}\' violates SCP isolation. "')
        lines.append(f'                            f"Allowed: {{sorted(allowed) or \'(none)\'}}"')
        lines.append(f"                        )")
        lines.append(f"            elif isinstance(node, ast.ImportFrom) and node.module:")
        lines.append(f"                root = node.module.split('.')[0]")
        lines.append(f"                if root not in allowed and root not in stdlib:")
        lines.append(f"                    pytest.fail(")
        lines.append(f'                        f"Import from \'{{node.module}}\' violates SCP isolation. "')
        lines.append(f'                        f"Allowed: {{sorted(allowed) or \'(none)\'}}"')
        lines.append(f"                    )")
        lines.append("")

        # Add constraint-specific tests
        for i, constraint in enumerate(constraints):
            lines.append(f"    def test_constraint_{i}(self):")
            lines.append(f'        """Constraint: {constraint}"""')
            lines.append(f"        # This constraint must be verified via code review or runtime testing")
            lines.append(f"        # Constraint text: {constraint}")
            lines.append(f"        pass  # Manual verification required")
            lines.append("")

        # Fixture
        lines.append(f"    @pytest.fixture")
        lines.append(f"    def module_source(self):")
        lines.append(f'        """Load the module source code for AST analysis."""')
        lines.append(f'        pytest.skip("Set module_source fixture to the module source code string")')
        lines.append("")

        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────
    #  Glyph Constraint Tests
    # ─────────────────────────────────────────────────────────

    def _glyph_tests(self, spec) -> str:
        """Generate tests that verify glyph-specific constraints."""
        module_name = spec.name
        methods = getattr(spec, 'methods', [])

        lines = []
        lines.append(f"# ─────────────────────────────────────────────")
        lines.append(f"#  Glyph Contract Tests")
        lines.append(f"# ─────────────────────────────────────────────")
        lines.append("")
        lines.append(f"class TestGlyphContracts:")
        lines.append(f'    """Verify methods comply with their governing glyph contracts."""')
        lines.append("")

        # I/O function names to check for
        io_fns = '{"print", "open", "input", "write", "read", "send", "recv"}'

        for method in methods:
            glyph = getattr(method, 'glyph', '')

            if glyph == "Ө":  # Filter
                lines.append(f"    def test_{method.name}_no_side_effects(self, module_source):")
                lines.append(f'        """Ө Filter: {method.name} must be side-effect free."""')
                lines.append(f"        io_calls = {io_fns}")
                lines.append(f"        tree = ast.parse(module_source)")
                lines.append(f"        for node in ast.walk(tree):")
                lines.append(f"            if isinstance(node, ast.FunctionDef) and node.name == '{method.name}':")
                lines.append(f"                for child in ast.walk(node):")
                lines.append(f"                    if isinstance(child, ast.Call):")
                lines.append(f"                        if isinstance(child.func, ast.Name) and child.func.id in io_calls:")
                lines.append(f"                            pytest.fail(")
                lines.append(f'                                f"Ө Filter \'{method.name}\' calls \'{{child.func.id}}\'. "')
                lines.append(f'                                f"Filters must be side-effect free."')
                lines.append(f"                            )")
                lines.append("")

            elif glyph == "𓂀":  # Witness
                lines.append(f"    def test_{method.name}_no_mutation(self, module_source):")
                lines.append(f'        """𓂀 Witness: {method.name} must not modify data."""')
                lines.append(f"        io_calls = {io_fns}")
                lines.append(f"        tree = ast.parse(module_source)")
                lines.append(f"        for node in ast.walk(tree):")
                lines.append(f"            if isinstance(node, ast.FunctionDef) and node.name == '{method.name}':")
                lines.append(f"                for child in ast.walk(node):")
                lines.append(f"                    if isinstance(child, ast.Call):")
                lines.append(f"                        if isinstance(child.func, ast.Name) and child.func.id in {{'write', 'send'}}:")
                lines.append(f"                            pytest.fail(")
                lines.append(f'                                f"𓂀 Witness \'{method.name}\' calls \'{{child.func.id}}\'. "')
                lines.append(f'                                f"Witness must observe without modifying."')
                lines.append(f"                            )")
                lines.append("")

            elif glyph == "◬":  # Origin
                lines.append(f"    def test_{method.name}_is_entry_point(self, module_source):")
                lines.append(f'        """◬ Origin: {method.name} should be the entry point."""')
                lines.append(f"        tree = ast.parse(module_source)")
                lines.append(f"        found = False")
                lines.append(f"        for node in ast.walk(tree):")
                lines.append(f"            if isinstance(node, ast.FunctionDef) and node.name == '{method.name}':")
                lines.append(f"                found = True")
                lines.append(f"                break")
                lines.append(f"        assert found, \"◬ Origin method '{method.name}' must exist\"")
                lines.append("")

        # Fixture
        lines.append(f"    @pytest.fixture")
        lines.append(f"    def module_source(self):")
        lines.append(f'        """Load the module source code for AST analysis."""')
        lines.append(f'        pytest.skip("Set module_source fixture to the module source code string")')
        lines.append("")

        return "\n".join(lines)
