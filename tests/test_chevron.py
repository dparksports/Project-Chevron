"""
Chevron Test Suite
==================
Tests for the Chevron v2.0 Non-Polysemic Topological DSL.
Covers: lexer, parser, interpreter, and verifier.

Usage:
    python -m unittest tests.test_chevron -v
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chevron.lexer import Lexer, TokenType
from chevron.parser import (
    Parser, ProgramNode, ModuleNode, SpecNode,
    NullMorphismNode, MorphismNode, DirectSumNode, TensorProductNode,
    TopoBoundaryNode, BindingNode, LiteralNode, IdentifierNode,
)
from chevron.interpreter import Interpreter
from chevron.verifier import SCPVerifier, ViolationLevel


# ─────────────────────────────────────────────
#  Lexer Tests
# ─────────────────────────────────────────────

class TestLexer(unittest.TestCase):
    """Tests for lexer — Topo-Categorical token types."""

    def test_morphism_token(self):
        """↦ should tokenize as MORPHISM."""
        tokens = Lexer("A ↦ B").tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        self.assertIn(TokenType.MORPHISM, types)

    def test_direct_sum_token(self):
        """⊕ should tokenize as DIRECT_SUM."""
        tokens = Lexer("A ⊕ B").tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        self.assertIn(TokenType.DIRECT_SUM, types)

    def test_tensor_product_token(self):
        """⊗ should tokenize as TENSOR_PRODUCT."""
        tokens = Lexer("A ⊗ B").tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        self.assertIn(TokenType.TENSOR_PRODUCT, types)

    def test_partial_token(self):
        """∂ should tokenize as PARTIAL."""
        tokens = Lexer("∂A").tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        self.assertIn(TokenType.PARTIAL, types)

    def test_intersection_token(self):
        """∩ should tokenize as INTERSECTION."""
        tokens = Lexer("∂A ∩ ∂B").tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        self.assertIn(TokenType.INTERSECTION, types)

    def test_empty_set_token(self):
        """∅ should tokenize as EMPTY_SET."""
        tokens = Lexer("= ∅").tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        self.assertIn(TokenType.EMPTY_SET, types)

    def test_isomorphic_token(self):
        """≅ should tokenize as ISOMORPHIC."""
        tokens = Lexer("Hom(A, B) ≅ 0").tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        self.assertIn(TokenType.ISOMORPHIC, types)

    def test_hom_keyword(self):
        """Hom should tokenize as KW_HOM."""
        tokens = Lexer("Hom(A, B) ≅ 0").tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        self.assertIn(TokenType.KW_HOM, types)

    def test_snake_case_identifier(self):
        """snake_case names should tokenize as single IDENTIFIER."""
        tokens = Lexer("my_var_name").tokenize()
        idents = [t for t in tokens if t.type == TokenType.IDENTIFIER]
        self.assertEqual(len(idents), 1)
        self.assertEqual(idents[0].value, "my_var_name")

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

    def test_pipeline_arrow(self):
        """→ should tokenize as PIPELINE."""
        tokens = Lexer("A → B").tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        self.assertIn(TokenType.PIPELINE, types)

    def test_bind_arrow(self):
        """← should tokenize as BIND."""
        tokens = Lexer("x ← 5").tokenize()
        types = [t.type for t in tokens if t.type != TokenType.EOF]
        self.assertIn(TokenType.BIND, types)


# ─────────────────────────────────────────────
#  Parser Tests
# ─────────────────────────────────────────────

class TestParser(unittest.TestCase):
    """Tests for parser — Topo-Categorical AST node types."""

    def _parse(self, source: str):
        tokens = Lexer(source).tokenize()
        return Parser(tokens).parse()

    def test_null_morphism_node(self):
        """Hom(A, B) ≅ 0 should parse as NullMorphismNode."""
        ast = self._parse("Hom(Foo, Bar) ≅ 0")
        stmts = ast.statements
        self.assertTrue(any(isinstance(s, NullMorphismNode) for s in stmts))
        nm = [s for s in stmts if isinstance(s, NullMorphismNode)][0]
        self.assertEqual(nm.source, "Foo")
        self.assertEqual(nm.target, "Bar")

    def test_morphism_node(self):
        """A ↦ B should parse as MorphismNode."""
        ast = self._parse("Alpha ↦ Beta")
        stmts = ast.statements
        self.assertTrue(any(isinstance(s, MorphismNode) for s in stmts))

    def test_direct_sum_node(self):
        """A ⊕ B should parse as DirectSumNode."""
        ast = self._parse("X ⊕ Y")
        stmts = ast.statements
        self.assertTrue(any(isinstance(s, DirectSumNode) for s in stmts))

    def test_tensor_product_node(self):
        """A ⊗ B should parse as TensorProductNode."""
        ast = self._parse("M ⊗ N")
        stmts = ast.statements
        self.assertTrue(any(isinstance(s, TensorProductNode) for s in stmts))

    def test_topo_boundary_node(self):
        """∂A ∩ ∂B = ∅ should parse as TopoBoundaryNode."""
        ast = self._parse("∂Frontend ∩ ∂Database = ∅")
        stmts = ast.statements
        self.assertTrue(any(isinstance(s, TopoBoundaryNode) for s in stmts))
        tb = [s for s in stmts if isinstance(s, TopoBoundaryNode)][0]
        self.assertEqual(tb.left, "Frontend")
        self.assertEqual(tb.right, "Database")

    def test_module_node(self):
        ast = self._parse("module Foo\nend")
        self.assertTrue(any(isinstance(s, ModuleNode) for s in ast.statements))

    def test_spec_node(self):
        ast = self._parse("spec Bar\n  exports baz\nend")
        self.assertTrue(any(isinstance(s, SpecNode) for s in ast.statements))

    def test_binding_node(self):
        ast = self._parse('x ← "hello"')
        self.assertTrue(any(isinstance(s, BindingNode) for s in ast.statements))

    def test_spec_with_constraints(self):
        ast = self._parse("""spec MyMod
    constraint "No side effects"
    constraint "Pure functions only"
end""")
        spec = [s for s in ast.statements if isinstance(s, SpecNode)][0]
        self.assertEqual(len(spec.constraints), 2)

    def test_spec_with_depends_forbidden(self):
        ast = self._parse("""spec Processor
    depends_on [Loader]
    forbidden [Renderer]
end""")
        spec = [s for s in ast.statements if isinstance(s, SpecNode)][0]
        self.assertIn("Loader", spec.depends_on)
        self.assertIn("Renderer", spec.forbidden)


# ─────────────────────────────────────────────
#  Interpreter Tests
# ─────────────────────────────────────────────

class TestInterpreter(unittest.TestCase):
    """Tests for interpreter — Topo-Categorical execution."""

    def _run(self, source: str):
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
        interp = Interpreter(output_fn=lambda s: None)
        result = interp.execute(ast)
        return result, interp

    def test_tensor_product_merge_strings(self):
        """A ⊗ B with strings should concatenate them."""
        result, _ = self._run('"Hello" ⊗ "World"')
        self.assertEqual(result, "Hello World")

    def test_tensor_product_merge_lists(self):
        """A ⊗ B with lists should concatenate them."""
        result, _ = self._run("[1, 2] ⊗ [3, 4]")
        self.assertEqual(result, [1, 2, 3, 4])

    def test_direct_sum_returns_list(self):
        """A ⊕ B should return a list of both values."""
        result, _ = self._run("[1, 2] ⊕ [3, 4]")
        self.assertEqual(result, [1, 2, 3, 4])

    def test_null_morphism_records_constraint(self):
        """Hom(A,B) ≅ 0 should record in constraint_log."""
        _, interp = self._run("Hom(Src, Tgt) ≅ 0")
        self.assertEqual(len(interp.null_morphisms), 1)
        self.assertEqual(interp.null_morphisms[0]["source"], "Src")
        self.assertEqual(interp.null_morphisms[0]["target"], "Tgt")

    def test_topo_boundary_records_constraint(self):
        """∂A ∩ ∂B = ∅ should record in topo_boundaries."""
        _, interp = self._run("∂Left ∩ ∂Right = ∅")
        self.assertEqual(len(interp.topo_boundaries), 1)
        self.assertEqual(interp.topo_boundaries[0]["left"], "Left")
        self.assertEqual(interp.topo_boundaries[0]["right"], "Right")

    def test_basic_pipeline(self):
        """Pipeline should filter values through predicates."""
        result, _ = self._run("[1, 5, 10, 15, 20] → {> 8}")
        self.assertEqual(result, [10, 15, 20])

    def test_binding_and_pipeline(self):
        result, _ = self._run('data ← [10, 25, 3, 47]\ndata → {> 20}')
        self.assertEqual(result, [25, 47])

    def test_module_exports(self):
        """Module exports should merge into global env."""
        result, interp = self._run("""module Greeter
    exports msg
    msg ← "hello"
end""")
        self.assertEqual(interp.env.get("msg"), "hello")

    def test_spec_not_executed(self):
        """Spec blocks are parsed but not executed."""
        result, interp = self._run("""spec NotRun
    exports nothing
end""")
        self.assertIn("NotRun", interp.specs)
        self.assertNotIn("nothing", interp.env)


# ─────────────────────────────────────────────
#  Verifier Tests
# ─────────────────────────────────────────────

class TestVerifier(unittest.TestCase):
    """Tests for SCPVerifier — Topo-Categorical constraint enforcement."""

    def _verify(self, source: str):
        tokens = Lexer(source).tokenize()
        ast = Parser(tokens).parse()
        verifier = SCPVerifier()
        return verifier.verify(ast)

    def test_null_morphism_violation(self):
        """Hom(A,B) ≅ 0 should flag A referencing B."""
        violations = self._verify("""
module src
    imports tgt
    run ← tgt
end
Hom(src, tgt) ≅ 0
""")
        errors = [v for v in violations if v.level == ViolationLevel.ERROR]
        self.assertTrue(any("Hom" in v.operator for v in errors),
                        f"Expected Hom≅0 violation, got: {errors}")

    def test_null_morphism_clean(self):
        """No violation when A doesn't reference B."""
        violations = self._verify("""
spec Src
    exports run
    run ← "clean"
end
spec Tgt
    exports result
end
Hom(Src, Tgt) ≅ 0
""")
        errors = [v for v in violations if v.level == ViolationLevel.ERROR
                  and v.operator == "Hom≅0"]
        self.assertEqual(len(errors), 0)

    def test_topo_boundary_violation(self):
        """∂A ∩ ∂B = ∅ should flag direct cross-reference."""
        violations = self._verify("""
module left
    imports right
    run ← right
end
∂left ∩ ∂right = ∅
""")
        errors = [v for v in violations if v.level == ViolationLevel.ERROR]
        self.assertTrue(any("∂∩∅" in v.operator for v in errors),
                        f"Expected ∂∩∅ violation, got: {errors}")

    def test_forbidden_dependency(self):
        """Referencing a forbidden module should be flagged."""
        violations = self._verify("""
module worker
    imports forbiddenmod
    forbidden [forbiddenmod]
    run ← forbiddenmod
end
""")
        errors = [v for v in violations if v.level == ViolationLevel.ERROR
                  and v.operator == "DEPENDENCY"]
        self.assertTrue(len(errors) > 0,
                        f"Expected DEPENDENCY violation, got: {violations}")

    def test_circular_dependency(self):
        """Circular depends_on should be flagged."""
        violations = self._verify("""
spec A
    depends_on [B]
end
spec B
    depends_on [A]
end
""")
        errors = [v for v in violations if v.operator == "CYCLE"]
        self.assertTrue(len(errors) > 0,
                        f"Expected CYCLE violation, got: {violations}")

    def test_clean_architecture(self):
        """A well-structured program should have no errors."""
        violations = self._verify("""
spec ModA
    exports run
    run ← "hello"
end
spec ModB
    depends_on [ModA]
    imports ModA
    exports process
    process ← ModA
end
ModA ↦ ModB
""")
        errors = [v for v in violations if v.level == ViolationLevel.ERROR]
        self.assertEqual(len(errors), 0)


if __name__ == "__main__":
    unittest.main()
