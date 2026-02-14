"""
Example: Using SCP Bridge with Google Gemini
=============================================
This shows how to use Project Chevron with Gemini (or any AI)
to generate Python code that conforms to SCP architecture specs.

Prerequisites:
    pip install google-genai

Usage:
    # Set your API key
    set GEMINI_API_KEY=your-api-key-here

    # Generate a module
    python examples/gemini_example.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scp_bridge import SCPBridge


def main():
    # ─────────────────────────────────────────────────────────
    #  STEP 1: Choose your architecture template
    # ─────────────────────────────────────────────────────────
    bridge = SCPBridge.from_template("todo_app")

    # ─────────────────────────────────────────────────────────
    #  STEP 2: Generate the SCP system prompt for ONE module
    # ─────────────────────────────────────────────────────────
    #
    #  This is the key insight: instead of pasting 128K tokens of
    #  your entire codebase, you give the AI a ~1,200 token spec
    #  that precisely defines:
    #    - What this module does (contract)
    #    - What it can see (dependency interfaces only)
    #    - What it CANNOT see (RAG Denial)
    #    - What rules it must follow (glyph constraints)
    #
    system_prompt = bridge.generate_system_prompt("TodoStore", language="python")

    print("=" * 70)
    print("◬  SCP System Prompt Generated")
    print(f"   Prompt size: {len(system_prompt)} chars (~{len(system_prompt.split())} tokens)")
    print("=" * 70)

    # ─────────────────────────────────────────────────────────
    #  STEP 3: Send to Gemini (or any AI)
    # ─────────────────────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY")

    if api_key:
        try:
            from google import genai

            client = genai.Client(api_key=api_key)

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents="Implement the TodoStore module now.",
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,  # Low temperature = more deterministic
                ),
            )

            print("\n◬ ─── Gemini Output ───\n")
            print(response.text)

            # ─────────────────────────────────────────────────
            #  STEP 4: Verify with the Weaver (☤)
            # ─────────────────────────────────────────────────
            verification_prompt = bridge.generate_verification_prompt(
                "TodoStore", response.text
            )
            print("\n☤ ─── Weaver Verification ───\n")

            verify_response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=verification_prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.0,
                ),
            )
            print(verify_response.text)

        except ImportError:
            print("\n⚠ google-genai not installed. Run: pip install google-genai")
            print("\nHere is the system prompt you would send to Gemini:\n")
            print(system_prompt)
    else:
        print("\n⚠ No GEMINI_API_KEY found in environment.")
        print()
        print("To use with Gemini:")
        print("  1. pip install google-genai")
        print("  2. set GEMINI_API_KEY=your-api-key")
        print("  3. python examples/gemini_example.py")
        print()
        print("Or copy the system prompt below into any AI chat:\n")
        print("─" * 70)
        print(system_prompt)
        print("─" * 70)
        print()
        print("Then tell the AI: 'Implement the TodoStore module now.'")
        print()
        print("The AI will generate Python code constrained by the SCP spec —")
        print("it physically cannot see other module implementations (RAG Denial),")
        print("and each method follows its governing glyph contract.")


if __name__ == "__main__":
    main()
