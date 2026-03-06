# Project Chevron — SCP Reference Implementation
"""
Project Chevron: A Non-Polysemic Topological DSL.
Reference implementation of the Spatial Constraint Protocol (SCP)
using Category Theory, Topology, and Tensor Mathematics.
"""
from .glyphs import OPERATOR_REGISTRY, OperatorInfo, OperatorType
from .lexer import Lexer, Token, TokenType
from .parser import (
    Parser, ASTNode, ProgramNode,
    ModuleNode, SpecNode, TypeDeclNode, TypeAnnotNode, ConstraintNode, FuncCallNode,
    NullMorphismNode, MorphismNode, DirectSumNode, TensorProductNode, TopoBoundaryNode,
)
from .interpreter import Interpreter
from .verifier import SCPVerifier, Violation, ViolationLevel
from .code_verifier import CodeVerifier, CodeViolation, verify_code
from .decorators import (
    ChevronContractError,
    origin, filter, fold, witness, weaver,
)
from .test_generator import SpecTestGenerator

__version__ = "2.0.0"
__all__ = [
    "OPERATOR_REGISTRY", "OperatorInfo", "OperatorType",
    "Lexer", "Token", "TokenType",
    "Parser", "ASTNode", "ProgramNode",
    "ModuleNode", "SpecNode", "TypeDeclNode", "TypeAnnotNode",
    "ConstraintNode", "FuncCallNode",
    "NullMorphismNode", "MorphismNode", "DirectSumNode",
    "TensorProductNode", "TopoBoundaryNode",
    "Interpreter",
    "SCPVerifier", "Violation", "ViolationLevel",
    "CodeVerifier", "CodeViolation", "verify_code",
    "ChevronContractError",
    "origin", "filter", "fold", "witness", "weaver",
    "SpecTestGenerator",
]
