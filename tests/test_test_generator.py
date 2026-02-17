"""
Tests for Spec-Driven Test Generator
======================================
Tests that verify the SpecTestGenerator produces valid, runnable pytest code
from SCP contracts.
"""
import sys
import os
import unittest
import ast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chevron.test_generator import SpecTestGenerator


def _make_method(name, inputs, output, glyph, constraint=""):
    """Create a fake InterfaceMethod."""
    class FakeMethod:
        pass
    m = FakeMethod()
    m.name = name
    m.inputs = inputs
    m.output = output
    m.glyph = glyph
    m.constraint = constraint
    return m


def _make_spec(name, methods, allowed_deps=None, constraints=None):
    """Create a fake ModuleSpec."""
    class FakeSpec:
        pass
    s = FakeSpec()
    s.name = name
    s.description = f"Test {name} module"
    s.methods = methods
    s.allowed_dependencies = allowed_deps or []
    s.constraints = constraints or []
    return s


class TestGenerateTests(unittest.TestCase):
    """Test that generated test code is valid Python."""

    def setUp(self):
        self.generator = SpecTestGenerator()
        self.spec = _make_spec(
            "TodoStore",
            methods=[
                _make_method("add", ["task: Task"], "Store", "☤", "Weaves task"),
                _make_method("remove", ["task_id: str"], "Store", "Ө", "Filters out task"),
                _make_method("list", ["store: Store"], "list[Task]", "𓂀", "Witnesses all tasks"),
            ],
            allowed_deps=["API"],
            constraints=["All functions must be pure"],
        )

    def test_generates_valid_python(self):
        """Generated code must parse without syntax errors."""
        code = self.generator.generate_tests(self.spec)
        try:
            ast.parse(code)
        except SyntaxError as e:
            self.fail(f"Generated test code has syntax error: {e}")

    def test_contains_interface_tests(self):
        code = self.generator.generate_tests(self.spec)
        self.assertIn("test_add_exists_and_callable", code)
        self.assertIn("test_remove_exists_and_callable", code)
        self.assertIn("test_list_exists_and_callable", code)

    def test_contains_signature_tests(self):
        code = self.generator.generate_tests(self.spec)
        self.assertIn("test_add_signature", code)
        self.assertIn("test_remove_signature", code)
        self.assertIn("test_list_signature", code)

    def test_contains_isolation_tests(self):
        code = self.generator.generate_tests(self.spec)
        self.assertIn("test_no_global_keyword", code)
        self.assertIn("test_only_allowed_dependencies", code)

    def test_contains_glyph_tests(self):
        code = self.generator.generate_tests(self.spec)
        # Filter glyph test
        self.assertIn("test_remove_no_side_effects", code)
        # Witness glyph test
        self.assertIn("test_list_no_mutation", code)

    def test_contains_constraint_tests(self):
        code = self.generator.generate_tests(self.spec)
        self.assertIn("test_constraint_0", code)
        self.assertIn("All functions must be pure", code)


class TestIsolatedSpec(unittest.TestCase):
    """Test generation for a fully isolated module (no dependencies)."""

    def setUp(self):
        self.generator = SpecTestGenerator()
        self.spec = _make_spec(
            "Ingest",
            methods=[
                _make_method("read_source", ["source: str"], "RawData", "◬"),
            ],
            allowed_deps=[],
            constraints=["Must not transform data"],
        )

    def test_generates_no_project_deps_test(self):
        code = self.generator.generate_tests(self.spec)
        self.assertIn("test_no_project_dependencies", code)
        self.assertIn("fully isolated", code)

    def test_generates_origin_glyph_test(self):
        code = self.generator.generate_tests(self.spec)
        self.assertIn("test_read_source_is_entry_point", code)


class TestGeneratedCodeStructure(unittest.TestCase):
    """Test that generated code has correct class structure."""

    def setUp(self):
        self.generator = SpecTestGenerator()
        self.spec = _make_spec(
            "Transform",
            methods=[
                _make_method("clean", ["data: RawData"], "CleanData", "Ө"),
            ],
        )

    def test_has_test_classes(self):
        code = self.generator.generate_tests(self.spec)
        tree = ast.parse(code)
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        self.assertIn("TestInterface", class_names)
        self.assertIn("TestIsolation", class_names)
        self.assertIn("TestGlyphContracts", class_names)

    def test_has_pytest_import(self):
        code = self.generator.generate_tests(self.spec)
        self.assertIn("import pytest", code)

    def test_has_ast_import(self):
        code = self.generator.generate_tests(self.spec)
        self.assertIn("import ast", code)


if __name__ == "__main__":
    unittest.main(verbosity=2)
