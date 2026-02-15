"""
Chevron Parser
==============
Recursive-descent parser that builds an Abstract Syntax Tree (AST)
from the token stream produced by the Lexer.

Supports:
  - Pipelines, glyphs, predicates, bindings, literals, lists
  - module / spec blocks with imports, exports, depends_on, forbidden, constraint
  - Type declarations
  - Function calls in predicates
  - Error accumulation (collects all parse errors, reports at end)
"""
from dataclasses import dataclass, field
from typing import Any

from .lexer import Token, TokenType


# ─────────────────────────────────────────────────────────────
#  AST Node Types
# ─────────────────────────────────────────────────────────────

@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    node_type: str = ""
    line: int = 0
    col: int = 0


@dataclass
class LiteralNode(ASTNode):
    """A literal value: string, number, or boolean."""
    value: Any = None

    def __post_init__(self):
        self.node_type = "Literal"


@dataclass
class ListNode(ASTNode):
    """A list of values: [a, b, c]."""
    elements: list[ASTNode] = field(default_factory=list)

    def __post_init__(self):
        self.node_type = "List"


@dataclass
class GlyphNode(ASTNode):
    """A glyph invocation with optional arguments."""
    glyph: str = ""
    args: list[ASTNode] = field(default_factory=list)

    def __post_init__(self):
        self.node_type = "Glyph"


@dataclass
class PredicateNode(ASTNode):
    """A predicate expression: {> 3}, {= 'yes'}, {- 1}."""
    operator: str = ""
    operand: Any = None

    def __post_init__(self):
        self.node_type = "Predicate"


@dataclass
class FuncCallNode(ASTNode):
    """A function call inside a predicate block: {func_name arg1 arg2}.
    
    Represents a named operation with arguments, used when predicates
    contain domain-specific function calls rather than simple comparisons.
    """
    func_name: str = ""
    args: list[ASTNode] = field(default_factory=list)

    def __post_init__(self):
        self.node_type = "FuncCall"


@dataclass
class PipelineNode(ASTNode):
    """A pipeline: expr → expr → expr."""
    stages: list[ASTNode] = field(default_factory=list)

    def __post_init__(self):
        self.node_type = "Pipeline"


@dataclass
class BindingNode(ASTNode):
    """A named binding: Name ← expression."""
    name: str = ""
    expression: ASTNode | None = None

    def __post_init__(self):
        self.node_type = "Binding"


@dataclass
class IdentifierNode(ASTNode):
    """A reference to a named binding."""
    name: str = ""

    def __post_init__(self):
        self.node_type = "Identifier"


@dataclass
class PlaceholderNode(ASTNode):
    """The _ placeholder for the current piped value."""

    def __post_init__(self):
        self.node_type = "Placeholder"


@dataclass
class ProgramNode(ASTNode):
    """Root node containing all top-level statements."""
    statements: list[ASTNode] = field(default_factory=list)

    def __post_init__(self):
        self.node_type = "Program"


# ─────────────────────────────────────────────────────────────
#  New AST Nodes for Module/Spec/Type System
# ─────────────────────────────────────────────────────────────

@dataclass
class TypeDeclNode(ASTNode):
    """A type declaration: type MediaFile = { path: str, size: int }."""
    type_name: str = ""
    fields: list[tuple[str, str]] = field(default_factory=list)  # [(name, type)]

    def __post_init__(self):
        self.node_type = "TypeDecl"


@dataclass
class TypeAnnotNode(ASTNode):
    """A type annotation in a pipeline: → TypeName.
    
    Used as documentation — pass-through in execution, but the
    verifier uses these for type checking at pipeline boundaries.
    """
    type_name: str = ""

    def __post_init__(self):
        self.node_type = "TypeAnnot"


@dataclass
class ConstraintNode(ASTNode):
    """A constraint declaration: constraint 'Must not do X'."""
    text: str = ""

    def __post_init__(self):
        self.node_type = "Constraint"


@dataclass
class ModuleNode(ASTNode):
    """A module block with isolated scope.
    
    module Name
        imports A, B
        exports C, D
        depends_on [E, F]
        forbidden [G, H]
        constraint "..."
        ... statements ...
    end
    """
    name: str = ""
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    types: list[TypeDeclNode] = field(default_factory=list)
    body: list[ASTNode] = field(default_factory=list)

    def __post_init__(self):
        self.node_type = "Module"


@dataclass
class SpecNode(ASTNode):
    """A spec block — like ModuleNode but for specification-only (no execution).
    
    Records contracts/interfaces without generating executable code.
    The verifier consumes spec nodes for type checking and dependency validation.
    """
    name: str = ""
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    types: list[TypeDeclNode] = field(default_factory=list)
    body: list[ASTNode] = field(default_factory=list)

    def __post_init__(self):
        self.node_type = "Spec"


# ─────────────────────────────────────────────────────────────
#  Parse Error (for accumulation)
# ─────────────────────────────────────────────────────────────

@dataclass
class ParseError:
    """A single parse error with location."""
    message: str
    line: int
    col: int


# ─────────────────────────────────────────────────────────────
#  Parser
# ─────────────────────────────────────────────────────────────

class Parser:
    """
    Recursive-descent parser for Chevron source.

    Usage:
        parser = Parser(tokens)
        ast = parser.parse()

    Features:
        - Error accumulation: collects all parse errors, reports at end
        - Module/spec blocks with imports/exports/dependencies
        - Type declarations and annotations
        - Function calls in predicates
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0
        self.errors: list[ParseError] = []

    def _current(self) -> Token:
        return self.tokens[self.pos]

    def _peek(self, offset: int = 1) -> Token:
        idx = self.pos + offset
        if idx >= len(self.tokens):
            return self.tokens[-1]  # EOF
        return self.tokens[idx]

    def _advance(self) -> Token:
        token = self.tokens[self.pos]
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return token

    def _expect(self, token_type: TokenType) -> Token:
        token = self._current()
        if token.type != token_type:
            self._record_error(
                f"Expected {token_type.name}, got {token.type.name} ({token.value!r})"
            )
            return token  # Return current token but don't advance
        return self._advance()

    def _skip_newlines(self):
        while self._current().type == TokenType.NEWLINE:
            self._advance()

    def _record_error(self, message: str):
        """Record a parse error with location, continue parsing."""
        token = self._current()
        self.errors.append(ParseError(message, token.line, token.col))

    def _synchronize(self):
        """Skip tokens until we reach a sync point (newline, EOF, end, glyph)."""
        while self._current().type not in (
            TokenType.NEWLINE, TokenType.EOF,
            TokenType.KW_END, TokenType.GLYPH,
        ):
            self._advance()

    # ─────────────────────────────────────────────────────────
    #  Top-Level Parsing
    # ─────────────────────────────────────────────────────────

    def parse(self) -> ProgramNode:
        """Parse the token stream into a ProgramNode."""
        program = ProgramNode(line=1, col=1)
        self._skip_newlines()

        while self._current().type != TokenType.EOF:
            try:
                stmt = self._parse_statement()
                if stmt is not None:
                    program.statements.append(stmt)
            except SyntaxError:
                # Error already recorded; synchronize and continue
                self._synchronize()
            self._skip_newlines()

        # If we accumulated errors, raise them all
        if self.errors:
            msgs = [f"  L{e.line}:{e.col} — {e.message}" for e in self.errors]
            raise SyntaxError(
                f"{len(self.errors)} parse error(s):\n" + "\n".join(msgs)
            )

        return program

    def _parse_statement(self) -> ASTNode | None:
        """Parse a single statement (binding, module, spec, type, constraint, or expression)."""
        token = self._current()

        # Module block
        if token.type == TokenType.KW_MODULE:
            return self._parse_module()

        # Spec block
        if token.type == TokenType.KW_SPEC:
            return self._parse_spec()

        # Type declaration
        if token.type == TokenType.KW_TYPE:
            return self._parse_type_decl()

        # Constraint
        if token.type == TokenType.KW_CONSTRAINT:
            return self._parse_constraint()

        # Check for binding: IDENTIFIER ← expr
        if (token.type == TokenType.IDENTIFIER
                and self._peek().type == TokenType.BIND):
            return self._parse_binding()

        return self._parse_pipeline()

    # ─────────────────────────────────────────────────────────
    #  Pipeline and Expression Parsing
    # ─────────────────────────────────────────────────────────

    def _parse_binding(self) -> BindingNode:
        """Parse: Name ← expression."""
        name_token = self._advance()
        self._expect(TokenType.BIND)
        self._skip_newlines()
        expr = self._parse_pipeline()
        return BindingNode(
            name=name_token.value, expression=expr,
            line=name_token.line, col=name_token.col,
        )

    def _parse_pipeline(self) -> ASTNode:
        """Parse: expr → expr → expr ..."""
        left = self._parse_expression()

        if self._current().type == TokenType.PIPELINE:
            stages = [left]
            while self._current().type == TokenType.PIPELINE:
                self._advance()  # consume →
                self._skip_newlines()
                stage = self._parse_expression()
                stages.append(stage)
            return PipelineNode(stages=stages, line=left.line, col=left.col)

        return left

    def _parse_expression(self) -> ASTNode:
        """Parse a single expression (glyph call, literal, list, predicate, etc.)."""
        token = self._current()

        # Glyph invocation
        if token.type == TokenType.GLYPH:
            return self._parse_glyph()

        # Grouped expression
        if token.type == TokenType.LPAREN:
            return self._parse_group()

        # List
        if token.type == TokenType.LBRACKET:
            return self._parse_list()

        # Predicate / function call block
        if token.type == TokenType.LBRACE:
            return self._parse_predicate()

        # Literals
        if token.type == TokenType.STRING:
            self._advance()
            return LiteralNode(value=token.value, line=token.line, col=token.col)

        if token.type == TokenType.NUMBER:
            self._advance()
            value = float(token.value) if "." in token.value else int(token.value)
            return LiteralNode(value=value, line=token.line, col=token.col)

        if token.type == TokenType.BOOLEAN:
            self._advance()
            return LiteralNode(value=token.value == "true", line=token.line, col=token.col)

        # Identifier (reference to binding) — also handles type annotations
        if token.type == TokenType.IDENTIFIER:
            self._advance()
            # If this looks like a type annotation (capitalized, in pipeline context),
            # return a TypeAnnotNode so the verifier can use it
            if token.value[0].isupper() and self._current().type in (
                TokenType.PIPELINE, TokenType.NEWLINE, TokenType.EOF,
                TokenType.RPAREN, TokenType.RBRACKET, TokenType.COMMA,
            ):
                return TypeAnnotNode(type_name=token.value, line=token.line, col=token.col)
            return IdentifierNode(name=token.value, line=token.line, col=token.col)

        # Placeholder
        if token.type == TokenType.UNDERSCORE:
            self._advance()
            return PlaceholderNode(line=token.line, col=token.col)

        self._record_error(
            f"Unexpected token {token.type.name} ({token.value!r})"
        )
        raise SyntaxError(
            f"Unexpected token {token.type.name} ({token.value!r}) "
            f"at line {token.line}, col {token.col}"
        )

    # ─────────────────────────────────────────────────────────
    #  Glyph, Group, List
    # ─────────────────────────────────────────────────────────

    def _parse_glyph(self) -> GlyphNode:
        """Parse a glyph with optional arguments."""
        glyph_token = self._advance()
        args = []

        # Collect arguments: anything that isn't a pipeline, newline, or EOF
        while self._current().type not in (
            TokenType.PIPELINE, TokenType.NEWLINE, TokenType.EOF,
            TokenType.RPAREN, TokenType.RBRACKET, TokenType.COMMA,
        ):
            arg = self._parse_expression()
            args.append(arg)

        return GlyphNode(
            glyph=glyph_token.value,
            args=args,
            line=glyph_token.line,
            col=glyph_token.col,
        )

    def _parse_group(self) -> ASTNode:
        """Parse a parenthesized expression."""
        self._advance()  # consume (
        self._skip_newlines()
        inner = self._parse_pipeline()
        self._skip_newlines()
        self._expect(TokenType.RPAREN)
        return inner

    def _parse_list(self) -> ListNode:
        """Parse a list: [a, b, c]."""
        token = self._advance()  # consume [
        self._skip_newlines()
        elements = []

        while self._current().type != TokenType.RBRACKET:
            if self._current().type == TokenType.EOF:
                self._record_error("Unterminated list")
                raise SyntaxError("Unterminated list")
            elements.append(self._parse_pipeline())
            self._skip_newlines()
            if self._current().type == TokenType.COMMA:
                self._advance()
                self._skip_newlines()

        self._expect(TokenType.RBRACKET)
        return ListNode(elements=elements, line=token.line, col=token.col)

    # ─────────────────────────────────────────────────────────
    #  Predicate (extended with function calls)
    # ─────────────────────────────────────────────────────────

    def _parse_predicate(self) -> ASTNode:
        """Parse a predicate: {> 3}, {= 'yes'}, {- 1}, or {func_name arg1 arg2}."""
        token = self._advance()  # consume {
        self._skip_newlines()

        # Read operator
        op_token = self._current()

        # Comparison/arithmetic operators → simple predicate
        if op_token.type in (TokenType.GT, TokenType.LT, TokenType.EQ,
                             TokenType.NEQ, TokenType.GTE, TokenType.LTE,
                             TokenType.PLUS, TokenType.MINUS, TokenType.STAR,
                             TokenType.SLASH):
            operator = op_token.value
            self._advance()
            self._skip_newlines()

            # Read operand
            operand = self._parse_expression()
            self._skip_newlines()
            self._expect(TokenType.RBRACE)

            return PredicateNode(
                operator=operator, operand=operand,
                line=token.line, col=token.col,
            )

        # Identifier → could be: named predicate {even}, or function call {func arg1 arg2}
        if op_token.type == TokenType.IDENTIFIER:
            func_name = op_token.value
            self._advance()
            self._skip_newlines()

            # If immediately followed by RBRACE → simple named predicate
            if self._current().type == TokenType.RBRACE:
                self._advance()
                return PredicateNode(
                    operator=func_name, operand=None,
                    line=token.line, col=token.col,
                )

            # Otherwise, collect arguments → function call
            args = []
            while self._current().type not in (TokenType.RBRACE, TokenType.EOF):
                args.append(self._parse_expression())
                self._skip_newlines()

            self._expect(TokenType.RBRACE)
            return FuncCallNode(
                func_name=func_name, args=args,
                line=token.line, col=token.col,
            )

        # Exclamation mark for != operator
        if op_token.type == TokenType.NEQ:
            operator = op_token.value
            self._advance()
            self._skip_newlines()
            operand = self._parse_expression()
            self._skip_newlines()
            self._expect(TokenType.RBRACE)
            return PredicateNode(
                operator=operator, operand=operand,
                line=token.line, col=token.col,
            )

        self._record_error(f"Expected operator or function name in predicate, got {op_token.type.name}")
        raise SyntaxError(
            f"Expected operator or function name in predicate at line {token.line}, col {token.col}"
        )

    # ─────────────────────────────────────────────────────────
    #  Module and Spec Blocks
    # ─────────────────────────────────────────────────────────

    def _parse_identifier_list(self) -> list[str]:
        """Parse a comma-separated list of identifiers (with optional [ ] brackets).
        
        For unbracketed lists (e.g., `exports A, B`), identifiers must be on
        the same line — newlines terminate the list. For bracketed lists
        (e.g., `depends_on [A, B]`), newlines are allowed between items.
        """
        names = []
        bracketed = False

        if self._current().type == TokenType.LBRACKET:
            bracketed = True
            self._advance()
            self._skip_newlines()

        while self._current().type == TokenType.IDENTIFIER:
            names.append(self._current().value)
            self._advance()
            if bracketed:
                self._skip_newlines()
            if self._current().type == TokenType.COMMA:
                self._advance()
                if bracketed:
                    self._skip_newlines()
            elif not bracketed:
                # Unbracketed: stop at newline or non-identifier
                break

        if bracketed:
            self._expect(TokenType.RBRACKET)

        return names

    def _parse_module_body(self) -> dict:
        """Parse the interior of a module/spec block.
        
        Returns dict with keys: imports, exports, depends_on, forbidden,
        constraints, types, body.
        """
        result = {
            "imports": [],
            "exports": [],
            "depends_on": [],
            "forbidden": [],
            "constraints": [],
            "types": [],
            "body": [],
        }

        self._skip_newlines()

        while self._current().type not in (TokenType.KW_END, TokenType.EOF):
            token = self._current()

            if token.type == TokenType.KW_IMPORTS:
                self._advance()
                self._skip_newlines()
                result["imports"] = self._parse_identifier_list()

            elif token.type == TokenType.KW_EXPORTS:
                self._advance()
                self._skip_newlines()
                result["exports"] = self._parse_identifier_list()

            elif token.type == TokenType.KW_DEPENDS_ON:
                self._advance()
                self._skip_newlines()
                result["depends_on"] = self._parse_identifier_list()

            elif token.type == TokenType.KW_FORBIDDEN:
                self._advance()
                self._skip_newlines()
                result["forbidden"] = self._parse_identifier_list()

            elif token.type == TokenType.KW_CONSTRAINT:
                self._advance()
                self._skip_newlines()
                if self._current().type == TokenType.STRING:
                    result["constraints"].append(self._current().value)
                    self._advance()
                else:
                    self._record_error("Expected string after constraint")
                    self._synchronize()

            elif token.type == TokenType.KW_TYPE:
                type_decl = self._parse_type_decl()
                if type_decl:
                    result["types"].append(type_decl)

            else:
                # Regular statement (pipeline, binding, etc.)
                try:
                    stmt = self._parse_statement()
                    if stmt is not None:
                        result["body"].append(stmt)
                except SyntaxError:
                    self._synchronize()

            self._skip_newlines()

        return result

    def _parse_module(self) -> ModuleNode:
        """Parse: module Name ... end"""
        token = self._advance()  # consume 'module'
        self._skip_newlines()

        # Module name
        name_token = self._expect(TokenType.IDENTIFIER)
        module_name = name_token.value if name_token.type == TokenType.IDENTIFIER else "unnamed"
        self._skip_newlines()

        # Parse body
        body = self._parse_module_body()

        # Expect 'end'
        self._expect(TokenType.KW_END)

        return ModuleNode(
            name=module_name,
            imports=body["imports"],
            exports=body["exports"],
            depends_on=body["depends_on"],
            forbidden=body["forbidden"],
            constraints=body["constraints"],
            types=body["types"],
            body=body["body"],
            line=token.line,
            col=token.col,
        )

    def _parse_spec(self) -> SpecNode:
        """Parse: spec Name ... end"""
        token = self._advance()  # consume 'spec'
        self._skip_newlines()

        # Spec name
        name_token = self._expect(TokenType.IDENTIFIER)
        spec_name = name_token.value if name_token.type == TokenType.IDENTIFIER else "unnamed"
        self._skip_newlines()

        # Parse body (same structure as module)
        body = self._parse_module_body()

        # Expect 'end'
        self._expect(TokenType.KW_END)

        return SpecNode(
            name=spec_name,
            imports=body["imports"],
            exports=body["exports"],
            depends_on=body["depends_on"],
            forbidden=body["forbidden"],
            constraints=body["constraints"],
            types=body["types"],
            body=body["body"],
            line=token.line,
            col=token.col,
        )

    # ─────────────────────────────────────────────────────────
    #  Type Declarations
    # ─────────────────────────────────────────────────────────

    def _parse_type_decl(self) -> TypeDeclNode:
        """Parse: type MediaFile = { path: str, size: int }"""
        token = self._advance()  # consume 'type'
        self._skip_newlines()

        # Type name
        name_token = self._expect(TokenType.IDENTIFIER)
        type_name = name_token.value if name_token.type == TokenType.IDENTIFIER else "Unknown"
        self._skip_newlines()

        # Expect '='
        self._expect(TokenType.EQ)
        self._skip_newlines()

        # Expect '{'
        self._expect(TokenType.LBRACE)
        self._skip_newlines()

        # Parse field: name pairs
        fields = []
        while self._current().type not in (TokenType.RBRACE, TokenType.EOF):
            # field_name
            if self._current().type == TokenType.IDENTIFIER:
                field_name = self._current().value
                self._advance()
                self._skip_newlines()

                # Expect ':'
                self._expect(TokenType.COLON)
                self._skip_newlines()

                # field_type (read as identifier)
                if self._current().type == TokenType.IDENTIFIER:
                    field_type = self._current().value
                    self._advance()
                else:
                    field_type = "any"

                fields.append((field_name, field_type))
                self._skip_newlines()

                # Optional comma
                if self._current().type == TokenType.COMMA:
                    self._advance()
                    self._skip_newlines()
            else:
                self._record_error("Expected field name in type declaration")
                self._synchronize()
                break

        self._expect(TokenType.RBRACE)

        return TypeDeclNode(
            type_name=type_name, fields=fields,
            line=token.line, col=token.col,
        )

    # ─────────────────────────────────────────────────────────
    #  Constraint
    # ─────────────────────────────────────────────────────────

    def _parse_constraint(self) -> ConstraintNode:
        """Parse: constraint 'Must not do X'."""
        token = self._advance()  # consume 'constraint'
        self._skip_newlines()

        if self._current().type == TokenType.STRING:
            text = self._current().value
            self._advance()
        else:
            self._record_error("Expected string after constraint")
            text = ""

        return ConstraintNode(text=text, line=token.line, col=token.col)
