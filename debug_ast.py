import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chevron.lexer import Lexer
from chevron.parser import Parser
from chevron.verifier import SCPVerifier

src = '''
module Src
    imports Tgt
    run ← Tgt
end
Hom(Src, Tgt) ≅ 0
'''
try:
    ast = Parser(Lexer(src).tokenize()).parse()
    v = SCPVerifier()
    stmt = ast.statements[0]
    print('Body:', stmt.body)
    idents = v._collect_identifiers(stmt.body)
    print('Idents:', [getattr(i, 'name', '?') for i in idents])
    violations = v.verify(ast)
    print('Violations:', [v.message for v in violations])
except Exception as e:
    print('Error:', e)
