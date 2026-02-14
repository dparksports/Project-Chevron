# Project Chevron — SCP Reference Implementation
"""
Project Chevron: A glyph-based programming language.
Reference implementation of the Spatial Constraint Protocol (SCP).
"""
from .glyphs import GLYPH_REGISTRY, GlyphInfo
from .lexer import Lexer, Token, TokenType
from .parser import Parser, ASTNode
from .interpreter import Interpreter

__version__ = "0.1.0"
__all__ = [
    "GLYPH_REGISTRY", "GlyphInfo",
    "Lexer", "Token", "TokenType",
    "Parser", "ASTNode",
    "Interpreter",
]
