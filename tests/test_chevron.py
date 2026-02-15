"""
Chevron Test Suite
==================
Tests for the Chevron language extensions: lexer, parser, interpreter, and verifier.
Uses Python's built-in unittest — no external dependencies required.

Usage:
    python -m unittest tests.test_chevron -v
    python tests/test_chevron.py
"""
import sys
import os
import unittest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chevron.lexer import Lexer, TokenType
from chevron.parser import (
    Parser, ProgramNode, GlyphNode, PipelineNode, IdentifierNode,
    ModuleNode, SpecNode, TypeDeclNode, TypeAnnotNode, ConstraintNode,
    FuncCallNode, PredicateNode, LiteralNode,
)
from chevron.interpreter import Interpreter, ChevronError
from chevron.verifier import SCPVerifier, ViolationLevel


# ─────────────────────────────────────────────
#  Lexer Tests
# ─────────────────────────────────────────────

class TestLexer(unittest.TestCase):
    """Tests for lexer extensions."""

    def test_snake_case_identifier(self):
        """snake_case names should tokenize as single IDENTIFIER."""
        tokens = Lexer("find_media").tokenize()
        self.assertEqual(tokens[0].type, TokenType.IDENTIFIER)
        self.assertEqual(tokens[0].value, "find_media")

    def test_double_underscore_identifier(self):
        tokens = Lexer("batch_vad_scan").tokenize()
        self.assertEqual(tokens[0].type, TokenType.IDENTIFIER)
        self.assertEqual(tokens[0].value, "batch_vad_scan")

    def test_standalone_underscore_is_placeholder(self):
        tokens = Lexer("_ ").tokenize()
        self.assertEqual(tokens[0].type, TokenType.UNDERSCORE)

    def test_underscore_in_pipeline(self):
        tokens = Lexer('◬ "hello" → _').tokenize()
        underscore = [t for t in tokens if t.type == TokenType.UNDERSCORE]
        self.assertEqual(len(underscore), 1)

    def test_keyword_module(self):
        tokens = Lexer("module").tokenize()
        self.assertEqual(tokens[0].type, TokenType.KW_MODULE)

    def test_keyword_spec(self):
        tokens = Lexer("spec").tokenize()
        self.assertEqual(tokens[0].type, TokenType.KW_SPEC)

    def test_keyword_type(self):
        tokens = Lexer("type").tokenize()
        self.assertEqual(tokens[0].type, TokenType.KW_TYPE)

    def test_keyword_end(self):
        tokens = Lexer("end").tokenize()
        self.assertEqual(tokens[0].type, TokenType.KW_END)

    def test_keyword_constraint(self):
        tokens = Lexer("constraint").tokenize()
        self.assertEqual(tokens[0].type, TokenType.KW_CONSTRAINT)

    def test_keyword_imports_exports(self):
        tokens = Lexer("imports exports").tokenize()
        self.assertEqual(tokens[0].type, TokenType.KW_IMPORTS)
        self.assertEqual(tokens[1].type, TokenType.KW_EXPORTS)

    def test_keyword_depends_on(self):
        tokens = Lexer("depends_on").tokenize()
        self.assertEqual(tokens[0].type, TokenType.KW_DEPENDS_ON)

    def test_keyword_forbidden(self):
        tokens = Lexer("forbidden").tokenize()
        self.assertEqual(tokens[0].type, TokenType.KW_FORBIDDEN)

    def test_colon_token(self):
        tokens = Lexer("name: str").tokenize()
        colon = [t for t in tokens if t.type == TokenType.COLON]
        self.assertEqual(len(colon), 1)

    def test_identifier_not_keyword(self):
        tokens = Lexer("modular").tokenize()
        self.assertEqual(tokens[0].type, TokenType.IDENTIFIER)
        self.assertEqual(tokens[0].value, "modular")


# ─────────────────────────────────────────────
#  Parser Tests
# ─────────────────────────────────────────────

class TestParser(unittest.TestCase):
    """Tests for parser extensions."""

    def _parse(self, source: str) -> ProgramNode:
        tokens = Lexer(source).tokenize()
        return Parser(tokens).parse()

    def test_module_node(self):
        ast = self._parse('module Foo\n◬ "hello" → 𓂀\nend')
        self.assertEqual(len(ast.statements), 1)
        self.assertIsInstance(ast.statements[0], ModuleNode)
        self.assertEqual(ast.statements[0].name, "Foo")

    def test_module_imports_exports(self):
        ast = self._parse('module Foo\nimports A, B\nexports C\n◬ "x" → 𓂀\nend')
        mod = ast.statements[0]
        self.assertEqual(mod.imports, ["A", "B"])
        self.assertEqual(mod.exports, ["C"])

    def test_module_depends_on_forbidden(self):
        ast = self._parse('module Foo\ndepends_on [A, B]\nforbidden [C]\n◬ "x" → 𓂀\nend')
        mod = ast.statements[0]
        self.assertEqual(mod.depends_on, ["A", "B"])
        self.assertEqual(mod.forbidden, ["C"])

    def test_module_constraints(self):
        ast = self._parse('module Foo\nconstraint "No side effects"\n◬ "x" → 𓂀\nend')
        self.assertEqual(ast.statements[0].constraints, ["No side effects"])

    def test_spec_node(self):
        ast = self._parse('spec Bar\nexports hello\nconstraint "Pure"\nend')
        spec = ast.statements[0]
        self.assertIsInstance(spec, SpecNode)
        self.assertEqual(spec.name, "Bar")
        self.assertEqual(spec.exports, ["hello"])

    def test_type_decl(self):
        ast = self._parse('type MediaFile = { path: str, size: int }')
        td = ast.statements[0]
        self.assertIsInstance(td, TypeDeclNode)
        self.assertEqual(td.type_name, "MediaFile")
        self.assertEqual(td.fields, [("path", "str"), ("size", "int")])

    def test_func_call_in_predicate(self):
        ast = self._parse('◬ [1, 2, 3] → Ө {is_even 2} → 𓂀')
        pipeline = ast.statements[0]
        filter_node = pipeline.stages[1]
        pred = filter_node.args[0]
        self.assertIsInstance(pred, FuncCallNode)
        self.assertEqual(pred.func_name, "is_even")

    def test_type_annot_in_pipeline(self):
        ast = self._parse('◬ "hello" → Transcript → 𓂀')
        pipeline = ast.statements[0]
        self.assertIsInstance(pipeline.stages[1], TypeAnnotNode)
        self.assertEqual(pipeline.stages[1].type_name, "Transcript")

    def test_constraint_standalone(self):
        ast = self._parse('constraint "No mutation"')
        c = ast.statements[0]
        self.assertIsInstance(c, ConstraintNode)
        self.assertEqual(c.text, "No mutation")

    def test_error_accumulation(self):
        tokens = Lexer('◬ ]\n◬ }').tokenize()
        parser = Parser(tokens)
        with self.assertRaises(SyntaxError) as ctx:
            parser.parse()
        self.assertIn("parse error", str(ctx.exception).lower())

    def test_snake_case_in_pipeline(self):
        ast = self._parse('◬ "test" → find_media → 𓂀')
        pipeline = ast.statements[0]
        self.assertIsInstance(pipeline.stages[1], IdentifierNode)
        self.assertEqual(pipeline.stages[1].name, "find_media")


# ─────────────────────────────────────────────
#  Interpreter Tests
# ─────────────────────────────────────────────

class TestInterpreter(unittest.TestCase):
    """Tests for interpreter extensions."""

    def _run(self, source: str):
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
        interp = Interpreter(output_fn=lambda s: None)
        result = interp.execute(ast)
        return result, interp

    def test_module_exports_all_by_default(self):
        result, interp = self._run('module A\nx ← ◬ 42\nend')
        self.assertIn("x", interp.env)

    def test_module_exports_filtering(self):
        source = (
            'module A\n'
            'exports public_val\n'
            'public_val ← 42\n'
            'private_val ← 99\n'
            '◬ "init" → 𓂀\n'
            'end'
        )
        result, interp = self._run(source)
        self.assertIn("public_val", interp.env)
        self.assertNotIn("private_val", interp.env)

    def test_spec_not_executed(self):
        result, interp = self._run(
            'spec Foo\nexports bar\nconstraint "No side effects"\nend'
        )
        self.assertIn("Foo", interp.specs)
        self.assertEqual(interp.specs["Foo"].exports, ["bar"])

    def test_type_registration(self):
        result, interp = self._run('type MediaFile = { path: str, size: int }')
        self.assertIn("MediaFile", interp.types)

    def test_basic_pipeline(self):
        result, _ = self._run('◬ [10, 25, 3, 47] → Ө {> 10} → 𓂀')
        self.assertEqual(result, [25, 47])

    def test_fold(self):
        result, _ = self._run('◬ 5 → ☾ {> 0} {- 1} → 𓂀')
        self.assertEqual(result, 0)


# ─────────────────────────────────────────────
#  Verifier Tests
# ─────────────────────────────────────────────

class TestVerifier(unittest.TestCase):
    """Tests for SCP verifier."""

    def _verify(self, source: str):
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
        return SCPVerifier().verify(ast)

    def test_single_origin_passes(self):
        violations = self._verify('◬ "hello" → 𓂀')
        errors = [v for v in violations if v.level == ViolationLevel.ERROR]
        self.assertEqual(len(errors), 0)

    def test_multiple_origins_fails(self):
        violations = self._verify('◬ "a" → 𓂀\n◬ "b" → 𓂀')
        errors = [v for v in violations if v.level == ViolationLevel.ERROR]
        self.assertGreater(len(errors), 0)

    def test_module_missing_origin_error(self):
        violations = self._verify('module A\n𓂀 "x"\nend')
        errors = [v for v in violations if v.level == ViolationLevel.ERROR]
        self.assertTrue(any("◬" in v.glyph for v in errors))

    def test_forbidden_dependency_error(self):
        source = (
            'module A\n'
            'forbidden [badmod]\n'
            'x ← badmod\n'
            '◬ x → 𓂀\n'
            'end'
        )
        violations = self._verify(source)
        errors = [v for v in violations if v.level == ViolationLevel.ERROR]
        self.assertTrue(any("forbidden" in v.message.lower() for v in errors))

    def test_circular_dependency_error(self):
        violations = self._verify('spec A\ndepends_on [B]\nend\nspec B\ndepends_on [A]\nend')
        errors = [v for v in violations if v.level == ViolationLevel.ERROR]
        self.assertTrue(any("circular" in v.message.lower() for v in errors))

    def test_undeclared_type_warns(self):
        violations = self._verify('◬ "x" → UnknownType → 𓂀')
        warnings = [v for v in violations if v.level == ViolationLevel.WARNING]
        self.assertTrue(any("UnknownType" in v.message for v in warnings))

    def test_declared_type_no_warning(self):
        violations = self._verify('type Transcript = { text: str }\n◬ "x" → Transcript → 𓂀')
        warnings = [v for v in violations if v.level == ViolationLevel.WARNING and "Transcript" in v.message]
        self.assertEqual(len(warnings), 0)

    def test_spec_no_origin_required(self):
        violations = self._verify('spec Foo\nexports bar\nend')
        errors = [v for v in violations if v.level == ViolationLevel.ERROR]
        self.assertEqual(len(errors), 0)


# ─────────────────────────────────────────────
#  Integration (example files)
# ─────────────────────────────────────────────

class TestExampleFiles(unittest.TestCase):
    """All example .chevron files must parse, verify, and execute."""

    EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")

    def _run_example(self, filename: str):
        filepath = os.path.join(self.EXAMPLES_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()

        verifier = SCPVerifier()
        violations = verifier.verify(ast)
        errors = [v for v in violations if v.level == ViolationLevel.ERROR]
        self.assertEqual(len(errors), 0, f"{filename} has SCP violations: {errors}")

        interp = Interpreter(output_fn=lambda s: None)
        return interp.execute(ast)

    def test_hello(self):
        self._run_example("hello.chevron")

    def test_pipeline(self):
        self.assertEqual(self._run_example("pipeline.chevron"), [25, 47, 92])

    def test_recursion(self):
        self.assertEqual(self._run_example("recursion.chevron"), 0)

    def test_weave_filter(self):
        self.assertEqual(self._run_example("weave_filter.chevron"), [8, 9, 7])

    def test_todo(self):
        self.assertIsInstance(self._run_example("todo.chevron"), list)

    def test_turboscribe(self):
        self.assertIsInstance(self._run_example("turboscribe.chevron"), list)


if __name__ == "__main__":
    unittest.main(verbosity=2)
