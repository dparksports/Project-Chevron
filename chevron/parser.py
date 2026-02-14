"""
Chevron Parser
==============
Recursive-descent parser that builds an Abstract Syntax Tree (AST)
from the token stream produced by the Lexer.
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
#  Parser
# ─────────────────────────────────────────────────────────────

class Parser:
    """
    Recursive-descent parser for Chevron source.

    Usage:
        parser = Parser(tokens)
        ast = parser.parse()
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

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
            raise SyntaxError(
                f"Expected {token_type.name}, got {token.type.name} "
                f"({token.value!r}) at line {token.line}, col {token.col}"
            )
        return self._advance()

    def _skip_newlines(self):
        while self._current().type == TokenType.NEWLINE:
            self._advance()

    def parse(self) -> ProgramNode:
        """Parse the token stream into a ProgramNode."""
        program = ProgramNode(line=1, col=1)

        while self._current().type != TokenType.EOF:
            self._skip_newlines()
            if self._current().type == TokenType.EOF:
                break

            stmt = self._parse_statement()
            if stmt is not None:
                program.statements.append(stmt)

        return program

    def _parse_statement(self) -> ASTNode | None:
        """Parse a single statement (binding or expression)."""
        # Check for binding: IDENTIFIER ← expr
        if (self._current().type == TokenType.IDENTIFIER
                and self._peek().type == TokenType.BIND):
            return self._parse_binding()

        return self._parse_pipeline()

    def _parse_binding(self) -> BindingNode:
        """Parse: Name ← expression."""
        name_token = self._advance()  # identifier
        self._advance()               # ←
        expr = self._parse_pipeline()
        return BindingNode(
            name=name_token.value,
            expression=expr,
            line=name_token.line,
            col=name_token.col,
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

        # Predicate
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

        # Identifier (reference to binding)
        if token.type == TokenType.IDENTIFIER:
            self._advance()
            return IdentifierNode(name=token.value, line=token.line, col=token.col)

        # Placeholder
        if token.type == TokenType.UNDERSCORE:
            self._advance()
            return PlaceholderNode(line=token.line, col=token.col)

        raise SyntaxError(
            f"Unexpected token {token.type.name} ({token.value!r}) "
            f"at line {token.line}, col {token.col}"
        )

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
        expr = self._parse_pipeline()
        self._skip_newlines()
        self._expect(TokenType.RPAREN)
        return expr

    def _parse_list(self) -> ListNode:
        """Parse a list: [a, b, c]."""
        token = self._advance()  # consume [
        elements = []

        self._skip_newlines()
        if self._current().type != TokenType.RBRACKET:
            elements.append(self._parse_pipeline())
            while self._current().type == TokenType.COMMA:
                self._advance()  # consume ,
                self._skip_newlines()
                elements.append(self._parse_pipeline())

        self._skip_newlines()
        self._expect(TokenType.RBRACKET)
        return ListNode(elements=elements, line=token.line, col=token.col)

    def _parse_predicate(self) -> PredicateNode:
        """Parse a predicate: {> 3}, {= 'yes'}, {- 1}."""
        token = self._advance()  # consume {
        self._skip_newlines()

        # Read operator
        op_token = self._current()
        operator = ""
        if op_token.type in (TokenType.GT, TokenType.LT, TokenType.EQ,
                             TokenType.NEQ, TokenType.GTE, TokenType.LTE,
                             TokenType.PLUS, TokenType.MINUS, TokenType.STAR,
                             TokenType.SLASH):
            operator = op_token.value
            self._advance()
        elif op_token.type == TokenType.IDENTIFIER:
            # Named predicate like {even}
            operator = op_token.value
            self._advance()
            self._skip_newlines()
            self._expect(TokenType.RBRACE)
            return PredicateNode(operator=operator, operand=None,
                                 line=token.line, col=token.col)

        self._skip_newlines()

        # Read operand
        operand = self._parse_expression()

        self._skip_newlines()
        self._expect(TokenType.RBRACE)

        return PredicateNode(
            operator=operator, operand=operand,
            line=token.line, col=token.col,
        )
