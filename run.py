"""
Chevron File Runner
===================
Execute .chevron source files from the command line.

Usage:
    python run.py <filename.chevron>
    python run.py examples/hello.chevron
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chevron.lexer import Lexer
from chevron.parser import Parser
from chevron.interpreter import Interpreter, ChevronError


def run_file(filepath: str) -> int:
    """
    Execute a .chevron source file.

    Returns:
        0 on success, 1 on error
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return 1

    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    print(f"◬ ─── Running: {os.path.basename(filepath)} ───")
    print()

    try:
        # Lex
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        # Parse
        parser = Parser(tokens)
        ast = parser.parse()

        # Execute
        interp = Interpreter()
        result = interp.execute(ast)

        # Print final result if not already witnessed
        if result is not None:
            formatted = interp._format_value(result)
            last_log = interp.witness_log[-1] if interp.witness_log else ""
            if f"𓂀 ⟫ {formatted}" != last_log:
                print(f"⟹ {formatted}")

        print()
        print(f"☾ ─── Complete ───")
        return 0

    except SyntaxError as e:
        print(f"⚠ Syntax Error: {e}")
        return 1
    except ChevronError as e:
        print(f"⚠ Chevron Error: {e}")
        return 1
    except Exception as e:
        print(f"⚠ Error: {type(e).__name__}: {e}")
        return 1


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <filename.chevron>")
        print("       python run.py examples/hello.chevron")
        sys.exit(1)

    filepath = sys.argv[1]
    exit_code = run_file(filepath)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
