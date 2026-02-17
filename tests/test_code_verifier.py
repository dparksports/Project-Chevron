"""
Tests for Chevron Code Verifier
================================
Unit tests for the AST-based deterministic code verification system.
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
        code = '''
counter = 0
def increment():
    global counter
    counter += 1
'''
        violations = self.verifier.verify(code)
        errors = [v for v in violations if v.check == "GLOBAL_STATE" and v.severity == SeverityLevel.ERROR]
        self.assertGreater(len(errors), 0, "Should detect 'global' keyword")

    def test_no_global_keyword_passes(self):
        code = '''
def add(a, b):
    return a + b
'''
        violations = self.verifier.verify(code)
        errors = [v for v in violations if v.check == "GLOBAL_STATE" and v.severity == SeverityLevel.ERROR]
        self.assertEqual(len(errors), 0)

    def test_module_level_mutable_warned(self):
        code = '''
cache = {}
def get(key):
    return cache.get(key)
'''
        violations = self.verifier.verify(code)
        warnings = [v for v in violations if v.check == "GLOBAL_STATE" and v.severity == SeverityLevel.WARNING]
        self.assertGreater(len(warnings), 0, "Should warn about mutable module-level dict")

    def test_constant_uppercase_allowed(self):
        code = '''
MAX_SIZE = 100
DEFAULTS = {"key": "value"}
def get_max():
    return MAX_SIZE
'''
        violations = self.verifier.verify(code)
        warnings = [v for v in violations if v.check == "GLOBAL_STATE"]
        self.assertEqual(len(warnings), 0, "ALL_CAPS constants should be allowed")


class TestForbiddenImports(unittest.TestCase):
    """Test detection of forbidden project imports."""

    def setUp(self):
        self.verifier = CodeVerifier()

    def _make_contract(self, allowed_deps):
        """Create a minimal contract-like object."""
        class FakeContract:
            def __init__(self, deps):
                self.allowed_dependencies = deps
                self.methods = []
                self.constraints = []
        return FakeContract(allowed_deps)

    def test_forbidden_import_detected(self):
        code = 'import secret_module\n'
        contract = self._make_contract(["allowed_a"])
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "FORBIDDEN_IMPORT"]
        self.assertGreater(len(errors), 0, "Should flag import of undeclared module")

    def test_allowed_import_passes(self):
        code = 'import allowed_a\n'
        contract = self._make_contract(["allowed_a"])
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "FORBIDDEN_IMPORT"]
        self.assertEqual(len(errors), 0)

    def test_stdlib_always_allowed(self):
        code = '''
import typing
import dataclasses
import json
from collections import defaultdict
'''
        contract = self._make_contract([])
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "FORBIDDEN_IMPORT"]
        self.assertEqual(len(errors), 0, "Standard library imports should always pass")

    def test_from_import_forbidden(self):
        code = 'from secret_module import something\n'
        contract = self._make_contract([])
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "FORBIDDEN_IMPORT"]
        self.assertGreater(len(errors), 0)


class TestSideEffects(unittest.TestCase):
    """Test side-effect detection for Filter (Ө) and Witness (𓂀)."""

    def setUp(self):
        self.verifier = CodeVerifier()

    def _make_contract(self, method_name, glyph):
        class FakeMethod:
            def __init__(self, name, g):
                self.name = name
                self.glyph = g
                self.inputs = ["data: list"]
                self.output = "list"
                self.constraint = ""
        class FakeContract:
            def __init__(self):
                self.allowed_dependencies = []
                self.methods = [FakeMethod(method_name, glyph)]
                self.constraints = []
        return FakeContract()

    def test_filter_with_print_rejected(self):
        code = '''
def validate(data):
    print("checking")
    return [x for x in data if x > 0]
'''
        contract = self._make_contract("validate", "Ө")
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "SIDE_EFFECT"]
        self.assertGreater(len(errors), 0, "Ө Filter should not call print()")

    def test_filter_without_io_passes(self):
        code = '''
def validate(data):
    return [x for x in data if x > 0]
'''
        contract = self._make_contract("validate", "Ө")
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "SIDE_EFFECT"]
        self.assertEqual(len(errors), 0)

    def test_witness_with_write_rejected(self):
        code = '''
def audit(data):
    with open("log.txt", "w") as f:
        f.write(str(data))
    return data
'''
        contract = self._make_contract("audit", "𓂀")
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "SIDE_EFFECT"]
        self.assertGreater(len(errors), 0, "𓂀 Witness should not call open/write")

    def test_weaver_io_allowed(self):
        """☤ Weaver is NOT restricted from I/O — only Ө and 𓂀 are."""
        code = '''
def merge(data):
    print("merging")
    return sum(data, [])
'''
        contract = self._make_contract("merge", "☤")
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "SIDE_EFFECT"]
        self.assertEqual(len(errors), 0, "☤ Weaver should allow I/O")


class TestInterfaceConformance(unittest.TestCase):
    """Test method existence and signature checking."""

    def setUp(self):
        self.verifier = CodeVerifier()

    def _make_contract(self, methods):
        class FakeMethod:
            def __init__(self, name, inputs, output, glyph):
                self.name = name
                self.inputs = inputs
                self.output = output
                self.glyph = glyph
                self.constraint = ""
        class FakeContract:
            def __init__(self):
                self.allowed_dependencies = []
                self.methods = [FakeMethod(*m) for m in methods]
                self.constraints = []
        return FakeContract()

    def test_missing_method_error(self):
        code = '''
def wrong_name(x):
    return x
'''
        contract = self._make_contract([("add", ["task: Task"], "Store", "☤")])
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "INTERFACE"]
        self.assertGreater(len(errors), 0, "Should flag missing 'add' method")

    def test_correct_interface_passes(self):
        code = '''
def add(task):
    return {"task": task}

def remove(task_id):
    return {}
'''
        contract = self._make_contract([
            ("add", ["task: Task"], "Store", "☤"),
            ("remove", ["task_id: str"], "Store", "Ө"),
        ])
        violations = self.verifier.verify(code, contract)
        errors = [v for v in violations if v.check == "INTERFACE" and v.severity == SeverityLevel.ERROR]
        self.assertEqual(len(errors), 0)

    def test_wrong_param_count_warning(self):
        code = '''
def add(task, extra_param):
    return {}
'''
        contract = self._make_contract([("add", ["task: Task"], "Store", "☤")])
        violations = self.verifier.verify(code, contract)
        warnings = [v for v in violations if v.check == "INTERFACE" and v.severity == SeverityLevel.WARNING]
        self.assertGreater(len(warnings), 0, "Should warn about parameter count mismatch")


class TestSyntaxError(unittest.TestCase):
    """Test handling of unparseable code."""

    def setUp(self):
        self.verifier = CodeVerifier()

    def test_syntax_error_reported(self):
        code = 'def broken(:\n'
        violations = self.verifier.verify(code)
        errors = [v for v in violations if v.check == "SYNTAX"]
        self.assertGreater(len(errors), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
