"""
Tests for Chevron Code Verifier
================================
Unit tests for the AST-based deterministic code verification system.
Updated for Topo-Categorical DSL v2.0.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chevron.code_verifier import CodeVerifier, CodeViolation, SeverityLevel


class TestGlobalState(unittest.TestCase):
    """Test detection of global mutable state."""

    def setUp(self):
        self.verifier = CodeVerifier()

    def test_global_keyword_detected(self):
        code = """
def foo():
    global x
    x = 10
"""
        violations = self.verifier.verify(code)
        errors = [v for v in violations if v.check == "GLOBAL_STATE"
                  and v.severity == SeverityLevel.ERROR]
        self.assertTrue(len(errors) > 0)

    def test_no_global_keyword_passes(self):
        code = """
def foo():
    x = 10
    return x
"""
        violations = self.verifier.verify(code)
        errors = [v for v in violations if v.check == "GLOBAL_STATE"
                  and v.severity == SeverityLevel.ERROR]
        self.assertEqual(len(errors), 0)

    def test_module_level_mutable_warned(self):
        code = """
config = {"key": "value"}
"""
        violations = self.verifier.verify(code)
        warnings = [v for v in violations if v.check == "GLOBAL_STATE"
                    and v.severity == SeverityLevel.WARNING]
        self.assertTrue(len(warnings) > 0)

    def test_constant_uppercase_allowed(self):
        code = """
MAX_SIZE = 100
DEFAULT_NAME = "test"
"""
        violations = self.verifier.verify(code)
        warnings = [v for v in violations if v.check == "GLOBAL_STATE"
                    and v.severity == SeverityLevel.WARNING]
        self.assertEqual(len(warnings), 0)


class TestForbiddenImports(unittest.TestCase):
    """Test detection of forbidden project imports."""

    def setUp(self):
        self.verifier = CodeVerifier()

    def _make_contract(self, allowed_deps):
        class FakeContract:
            def __init__(self, deps):
                self.allowed_dependencies = deps
                self.methods = []
                self.constraints = []
        return FakeContract(allowed_deps)

    def test_forbidden_import_detected(self):
        code = "import my_forbidden_module"
        contract = self._make_contract(["allowed_module"])
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "FORBIDDEN_IMPORT"]
        self.assertTrue(len(errors) > 0)

    def test_allowed_import_passes(self):
        code = "import allowed_module"
        contract = self._make_contract(["allowed_module"])
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "FORBIDDEN_IMPORT"]
        self.assertEqual(len(errors), 0)

    def test_stdlib_always_allowed(self):
        code = """
import os
import sys
import json
from typing import Any
from dataclasses import dataclass
"""
        contract = self._make_contract([])
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "FORBIDDEN_IMPORT"]
        self.assertEqual(len(errors), 0)

    def test_from_import_forbidden(self):
        code = "from forbidden_pkg import something"
        contract = self._make_contract(["ok_module"])
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "FORBIDDEN_IMPORT"]
        self.assertTrue(len(errors) > 0)


class TestSideEffects(unittest.TestCase):
    """Test side-effect detection for pure-constraint operators (⊕, ∂∩∅)."""

    def setUp(self):
        self.verifier = CodeVerifier()

    def _make_contract(self, method_name, operator):
        """Create a minimal contract with one method governed by an operator."""
        class FakeMethod:
            def __init__(self, name, op):
                self.name = name
                self.operator = op
                self.glyph = op  # Backward compat
                self.inputs = ["data: list"]
                self.output = "list"
                self.constraint = ""
        class FakeContract:
            def __init__(self):
                self.allowed_dependencies = []
                self.methods = [FakeMethod(method_name, operator)]
                self.constraints = []
        return FakeContract()

    def test_direct_sum_with_print_rejected(self):
        """⊕ Direct Sum methods must not contain I/O calls."""
        code = """
def transform(data):
    print("side effect!")
    return data
"""
        contract = self._make_contract("transform", "⊕")
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "SIDE_EFFECT"
                  and v.severity == SeverityLevel.ERROR]
        self.assertTrue(len(errors) > 0)
        self.assertIn("SYSTEM 2 REJECTION", errors[0].message)

    def test_direct_sum_without_io_passes(self):
        """⊕ Direct Sum methods without I/O should pass."""
        code = """
def transform(data):
    return [x * 2 for x in data]
"""
        contract = self._make_contract("transform", "⊕")
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "SIDE_EFFECT"]
        self.assertEqual(len(errors), 0)

    def test_topo_boundary_with_write_rejected(self):
        """∂∩∅ Boundary methods must not write files."""
        code = """
def observe(data):
    with open("output.txt", "w") as f:
        f.write(str(data))
    return data
"""
        contract = self._make_contract("observe", "∂∩∅")
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "SIDE_EFFECT"]
        self.assertTrue(len(errors) > 0)
        self.assertIn("SYSTEM 2 REJECTION", errors[0].message)

    def test_morphism_io_allowed(self):
        """↦ Morphism operators are NOT restricted from I/O."""
        code = """
def process(data):
    print("Processing...")
    return data
"""
        contract = self._make_contract("process", "↦")
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "SIDE_EFFECT"]
        self.assertEqual(len(errors), 0)

    def test_legacy_filter_still_checked(self):
        """Ө (legacy Filter) should still be checked for side-effects."""
        code = """
def filter_items(data):
    print("filtering...")
    return data
"""
        contract = self._make_contract("filter_items", "Ө")
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "SIDE_EFFECT"]
        self.assertTrue(len(errors) > 0)


class TestInterfaceConformance(unittest.TestCase):
    """Test method existence and signature checking."""

    def setUp(self):
        self.verifier = CodeVerifier()

    def _make_contract(self, methods):
        class FakeMethod:
            def __init__(self, name, inputs, output, operator="↦"):
                self.name = name
                self.inputs = inputs
                self.output = output
                self.operator = operator
                self.glyph = operator
                self.constraint = ""
        class FakeContract:
            def __init__(self):
                self.allowed_dependencies = []
                self.methods = [FakeMethod(*m) for m in methods]
                self.constraints = []
        return FakeContract()

    def test_missing_method_error(self):
        code = """
def wrong_name(data):
    return data
"""
        contract = self._make_contract([("process", ["data: str"], "str")])
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "INTERFACE"
                  and v.severity == SeverityLevel.ERROR]
        self.assertTrue(len(errors) > 0)

    def test_correct_interface_passes(self):
        code = """
def process(data):
    return data
"""
        contract = self._make_contract([("process", ["data: str"], "str")])
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "INTERFACE"
                  and v.severity == SeverityLevel.ERROR]
        self.assertEqual(len(errors), 0)

    def test_wrong_param_count_warning(self):
        code = """
def process(a, b, c):
    return a
"""
        contract = self._make_contract([("process", ["data: str"], "str")])
        violations = self.verifier.verify(code, contract)
        warnings = [v for v in violations if v.check == "INTERFACE"
                    and v.severity == SeverityLevel.WARNING]
        self.assertTrue(len(warnings) > 0)


class TestConvenienceFunction(unittest.TestCase):
    """Test the verify_code convenience function."""

    def test_clean_code_passes(self):
        from chevron.code_verifier import verify_code
        code = """
def add(a, b):
    return a + b
"""
        violations = verify_code(code)
        errors = [v for v in violations if v.severity == SeverityLevel.ERROR]
        self.assertEqual(len(errors), 0)

    def test_syntax_error_detected(self):
        from chevron.code_verifier import verify_code
        code = "def foo(:\n    pass"
        violations = verify_code(code)
        self.assertTrue(any(v.check == "SYNTAX" for v in violations))


if __name__ == "__main__":
    unittest.main()
