# Project Chevron — SCP Reference Implementation
"""
Project Chevron: A glyph-based programming language.
Reference implementation of the Spatial Constraint Protocol (SCP).
"""
from .glyphs import GLYPH_REGISTRY, GlyphInfo
from .lexer import Lexer, Token, TokenType
from .parser import (
    Parser, ASTNode, ProgramNode,
    ModuleNode, SpecNode, TypeDeclNode, TypeAnnotNode, ConstraintNode, FuncCallNode,
)
from .interpreter import Interpreter
from .verifier import SCPVerifier, Violation, ViolationLevel

__version__ = "0.2.0"
__all__ = [
    "GLYPH_REGISTRY", "GlyphInfo",
    "Lexer", "Token", "TokenType",
    "Parser", "ASTNode", "ProgramNode",
    "ModuleNode", "SpecNode", "TypeDeclNode", "TypeAnnotNode",
    "ConstraintNode", "FuncCallNode",
    "Interpreter",
    "SCPVerifier", "Violation", "ViolationLevel",
]
