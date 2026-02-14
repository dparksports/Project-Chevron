"""
Chevron REPL
============
Interactive Read-Eval-Print Loop for Project Chevron.
Type glyph expressions and see them execute in real time.
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chevron.lexer import Lexer
from chevron.parser import Parser
from chevron.interpreter import Interpreter, ChevronError
from chevron.glyphs import describe_all, GLYPH_REGISTRY


BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     ◬ ─── PROJECT CHEVRON ─── ◬                              ║
║                                                              ║
║     SCP Reference Implementation v0.1.0                      ║
║     Spatial Constraint Protocol — Glyph-Based Language       ║
║                                                              ║
║     Glyphs:  ◬ ☾ Ө 𓂀 ☤                                     ║
║     Type 'help' for glyph reference                          ║
║     Type 'exit' or Ctrl+C to quit                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
╔══════════════════════════════════════════════════════════════╗
║                  CHEVRON GLYPH REFERENCE                     ║
╠══════╦════════════════╦══════════════════════════════════════╣
║ ◬    ║ The Origin     ║ Program entry — initial data         ║
║ ☾    ║ Fold Time      ║ Recursion — output feeds to input    ║
║ Ө    ║ The Filter     ║ Conditional — only matching passes   ║
║ 𓂀    ║ The Witness    ║ Observe — log without altering       ║
║ ☤    ║ The Weaver     ║ Merge — braid streams together       ║
╠══════╬════════════════╬══════════════════════════════════════╣
║  →   ║ Pipeline       ║ Chain: ◬ data → Ө pred → 𓂀          ║
║  ←   ║ Binding        ║ Name ← expression                   ║
║ [ ]  ║ List           ║ [1, 2, 3]                            ║
║ { }  ║ Predicate      ║ {> 3}  {!= "no"}  {- 1}             ║
╚══════╩════════════════╩══════════════════════════════════════╝

Examples:
  𓂀 "Hello, Chevron!"
  𓂀 (☤ ["Hello", "World"])
  ◬ [1, 2, 3, 4, 5] → Ө {> 3} → 𓂀
  ◬ 10 → ☾ {> 0} {- 1} → 𓂀

Commands: help, env, log, clear, exit
"""


def run_repl():
    """Run the interactive Chevron REPL."""
    print(BANNER)

    interp = Interpreter()

    while True:
        try:
            # Prompt with chevron symbol
            line = input("  ◬⟩ ")
        except (EOFError, KeyboardInterrupt):
            print("\n  ☾ Folding time... Goodbye.")
            break

        line = line.strip()
        if not line:
            continue

        # Special commands
        if line.lower() == "exit" or line.lower() == "quit":
            print("  ☾ Folding time... Goodbye.")
            break

        if line.lower() == "help":
            print(HELP_TEXT)
            continue

        if line.lower() == "env":
            if interp.env:
                print("  ─── Bindings ───")
                for name, value in interp.env.items():
                    print(f"    {name} = {interp._format_value(value)}")
            else:
                print("  (no bindings)")
            continue

        if line.lower() == "log":
            if interp.witness_log:
                print("  ─── Witness Log ───")
                for entry in interp.witness_log:
                    print(f"    {entry}")
            else:
                print("  (no observations)")
            continue

        if line.lower() == "clear":
            interp.witness_log.clear()
            interp.env.clear()
            print("  ∅ State cleared.")
            continue

        # Tokenize → Parse → Execute
        try:
            lexer = Lexer(line)
            tokens = lexer.tokenize()

            parser = Parser(tokens)
            ast = parser.parse()

            result = interp.execute(ast)

            # Print result if it wasn't already printed by Witness
            if result is not None and not any(
                isinstance(stmt, (type(None),)) for stmt in [result]
            ):
                # Check if the last witness already printed it
                formatted = interp._format_value(result)
                last_log = interp.witness_log[-1] if interp.witness_log else ""
                if f"𓂀 ⟫ {formatted}" != last_log:
                    print(f"  ⟹ {formatted}")

        except SyntaxError as e:
            print(f"  ⚠ Syntax Error: {e}")
        except ChevronError as e:
            print(f"  ⚠ Chevron Error: {e}")
        except Exception as e:
            print(f"  ⚠ Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    run_repl()
