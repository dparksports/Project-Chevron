"""
Chevron Lexer
=============
Tokenizes Chevron source code into a stream of typed tokens.
Handles Unicode glyphs, operators, string/number literals, and identifiers.
"""
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator

from .glyphs import GLYPH_CHARS


class TokenType(Enum):
    """All token types in the Chevron language."""
    # Glyphs (the 5 primitives)
    GLYPH       = auto()

    # Operators
    PIPELINE    = auto()   # →
    BIND        = auto()   # ←
    LPAREN      = auto()   # (
    RPAREN      = auto()   # )
    LBRACKET    = auto()   # [
    RBRACKET    = auto()   # ]
    LBRACE      = auto()   # {
    RBRACE      = auto()   # }
    COMMA       = auto()   # ,
    UNDERSCORE  = auto()   # _ (placeholder for piped value)

    # Literals
    STRING      = auto()   # "..."
    NUMBER      = auto()   # 42, 3.14
    BOOLEAN     = auto()   # true, false
    IDENTIFIER  = auto()   # variable/function names

    # Comparison operators inside predicates
    GT          = auto()   # >
    LT          = auto()   # <
    EQ          = auto()   # =
    NEQ         = auto()   # !=
    GTE         = auto()   # >=
    LTE         = auto()   # <=

    # Arithmetic operators inside predicates
    PLUS        = auto()   # +
    MINUS       = auto()   # -
    STAR        = auto()   # *
    SLASH       = auto()   # /

    # Special
    NEWLINE     = auto()
    EOF         = auto()
    COMMENT     = auto()   # # ...


@dataclass
class Token:
    """A single token from the Chevron source."""
    type: TokenType
    value: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:{self.col})"


# Operator mapping (single and multi-char)
SINGLE_CHAR_TOKENS = {
    "→": TokenType.PIPELINE,
    "←": TokenType.BIND,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "[": TokenType.LBRACKET,
    "]": TokenType.RBRACKET,
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
    ",": TokenType.COMMA,
    "_": TokenType.UNDERSCORE,
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.STAR,
    "/": TokenType.SLASH,
    ">": TokenType.GT,
    "<": TokenType.LT,
    "=": TokenType.EQ,
}

KEYWORDS = {
    "true": TokenType.BOOLEAN,
    "false": TokenType.BOOLEAN,
}


class Lexer:
    """
    Tokenizes Chevron source code.

    Usage:
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
    """

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1

    def _current(self) -> str | None:
        if self.pos >= len(self.source):
            return None
        return self.source[self.pos]

    def _peek(self, offset: int = 1) -> str | None:
        idx = self.pos + offset
        if idx >= len(self.source):
            return None
        return self.source[idx]

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _skip_whitespace(self):
        """Skip spaces and tabs, but NOT newlines."""
        while self.pos < len(self.source) and self.source[self.pos] in (" ", "\t", "\r"):
            self._advance()

    def _read_string(self) -> Token:
        """Read a double-quoted string literal."""
        start_line, start_col = self.line, self.col
        self._advance()  # consume opening "
        chars = []
        while self.pos < len(self.source):
            ch = self._advance()
            if ch == '"':
                return Token(TokenType.STRING, "".join(chars), start_line, start_col)
            if ch == "\\":
                next_ch = self._advance()
                escape_map = {"n": "\n", "t": "\t", "\\": "\\", '"': '"'}
                chars.append(escape_map.get(next_ch, next_ch))
            else:
                chars.append(ch)
        raise SyntaxError(f"Unterminated string at line {start_line}, col {start_col}")

    def _read_number(self) -> Token:
        """Read a numeric literal (int or float)."""
        start_line, start_col = self.line, self.col
        chars = []
        has_dot = False
        # Handle negative sign
        if self._current() == "-":
            chars.append(self._advance())
        while self.pos < len(self.source):
            ch = self._current()
            if ch is not None and ch.isdigit():
                chars.append(self._advance())
            elif ch == "." and not has_dot:
                has_dot = True
                chars.append(self._advance())
            else:
                break
        return Token(TokenType.NUMBER, "".join(chars), start_line, start_col)

    def _read_identifier(self) -> Token:
        """Read an identifier or keyword."""
        start_line, start_col = self.line, self.col
        chars = []
        while self.pos < len(self.source):
            ch = self._current()
            if ch is not None and (ch.isalnum() or ch == "_"):
                chars.append(self._advance())
            else:
                break
        word = "".join(chars)
        token_type = KEYWORDS.get(word, TokenType.IDENTIFIER)
        return Token(token_type, word, start_line, start_col)

    def _read_comment(self) -> Token:
        """Read a line comment starting with #."""
        start_line, start_col = self.line, self.col
        chars = []
        self._advance()  # consume #
        while self.pos < len(self.source) and self._current() != "\n":
            chars.append(self._advance())
        return Token(TokenType.COMMENT, "".join(chars).strip(), start_line, start_col)

    def tokenize(self) -> list[Token]:
        """Tokenize the entire source into a list of tokens."""
        tokens = []
        for token in self._iter_tokens():
            if token.type not in (TokenType.COMMENT,):
                tokens.append(token)
        tokens.append(Token(TokenType.EOF, "", self.line, self.col))
        return tokens

    def _iter_tokens(self) -> Iterator[Token]:
        """Generate tokens one at a time."""
        while self.pos < len(self.source):
            self._skip_whitespace()

            if self.pos >= len(self.source):
                break

            ch = self._current()

            # Newlines
            if ch == "\n":
                yield Token(TokenType.NEWLINE, "\\n", self.line, self.col)
                self._advance()
                continue

            # Comments
            if ch == "#":
                yield self._read_comment()
                continue

            # String literals
            if ch == '"':
                yield self._read_string()
                continue

            # Number literals (including negative)
            if ch is not None and ch.isdigit():
                yield self._read_number()
                continue

            # Negative numbers: - followed by digit
            if ch == "-" and self._peek() is not None and self._peek().isdigit():
                yield self._read_number()
                continue

            # Multi-char operators
            if ch == "!" and self._peek() == "=":
                line, col = self.line, self.col
                self._advance()
                self._advance()
                yield Token(TokenType.NEQ, "!=", line, col)
                continue

            if ch == ">" and self._peek() == "=":
                line, col = self.line, self.col
                self._advance()
                self._advance()
                yield Token(TokenType.GTE, ">=", line, col)
                continue

            if ch == "<" and self._peek() == "=":
                line, col = self.line, self.col
                self._advance()
                self._advance()
                yield Token(TokenType.LTE, "<=", line, col)
                continue

            # Glyph characters
            if ch in GLYPH_CHARS:
                yield Token(TokenType.GLYPH, ch, self.line, self.col)
                self._advance()
                continue

            # Single-char operators
            if ch in SINGLE_CHAR_TOKENS:
                yield Token(SINGLE_CHAR_TOKENS[ch], ch, self.line, self.col)
                self._advance()
                continue

            # Identifiers and keywords
            if ch is not None and (ch.isalpha() or ch == "_"):
                yield self._read_identifier()
                continue

            # Unknown character — skip with warning
            self._advance()
