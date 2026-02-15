"""
Chevron File Runner
===================
Execute .chevron source files from the command line.

Usage:
    python run.py <filename.chevron>
    python run.py <filename.chevron> --verify
    python run.py examples/hello.chevron
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chevron.lexer import Lexer
from chevron.parser import Parser
from chevron.interpreter import Interpreter, ChevronError
from chevron.verifier import SCPVerifier, ViolationLevel


def run_file(filepath: str, verify: bool = False) -> int:
    """
    Execute a .chevron source file.

    Args:
        filepath: Path to the .chevron file
        verify: If True, run SCP verifier before execution

    Returns:
        0 on success, 1 on error
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return 1

    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    print(f"\u25EC \u2500\u2500\u2500 Running: {os.path.basename(filepath)} \u2500\u2500\u2500")
    print()

    try:
        # Lex
        lexer = Lexer(source)
        tokens = lexer.tokenize()

        # Parse
        parser = Parser(tokens)
        ast = parser.parse()

        # Verify (optional but recommended)
        if verify:
            verifier = SCPVerifier()
            violations = verifier.verify(ast)
            if violations:
                errors = [v for v in violations if v.level == ViolationLevel.ERROR]
                warnings = [v for v in violations if v.level == ViolationLevel.WARNING]

                if warnings:
                    print(f"\u26A0 {len(warnings)} warning(s):")
                    for v in warnings:
                        print(v)
                    print()

                if errors:
                    print(f"\u2718 {len(errors)} SCP violation(s):")
                    for v in errors:
                        print(v)
                    print()
                    print(f"\u263E \u2500\u2500\u2500 Verification FAILED \u2500\u2500\u2500")
                    return 1
                else:
                    print(f"\u2714 SCP verification passed (W(G) = 0)")
                    print()
            else:
                print(f"\u2714 SCP verification passed (W(G) = 0)")
                print()

        # Execute
        interp = Interpreter()
        result = interp.execute(ast)

        # Print final result if not already witnessed
        if result is not None:
            formatted = interp._format_value(result)
            last_log = interp.witness_log[-1] if interp.witness_log else ""
            if f"\U000130C0 \u27EB {formatted}" != last_log:
                print(f"\u27F9 {formatted}")

        print()
        print(f"\u263E \u2500\u2500\u2500 Complete \u2500\u2500\u2500")
        return 0

    except SyntaxError as e:
        print(f"\u26A0 Syntax Error: {e}")
        return 1
    except ChevronError as e:
        print(f"\u26A0 Chevron Error: {e}")
        return 1
    except Exception as e:
        print(f"\u26A0 Error: {type(e).__name__}: {e}")
        return 1


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <filename.chevron> [--verify]")
        print("       python run.py examples/hello.chevron")
        print("       python run.py examples/turboscribe.chevron --verify")
        sys.exit(1)

    filepath = sys.argv[1]
    verify = "--verify" in sys.argv
    exit_code = run_file(filepath, verify=verify)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
