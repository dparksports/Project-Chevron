# Apply SCP to Your Project

> **Your AI keeps breaking things? Point this at your codebase and fix that.**

---

## What Is This?

Project Chevron is a tool that **stops AI-generated code from regressing**. It works with any AI (Gemini, GPT, Claude) by constraining what the AI can see and produce.

Instead of dumping your entire codebase into a prompt (128K tokens of noise), Chevron:

1. **Decomposes** your project into isolated modules with strict contracts
2. **Generates constrained prompts** — the AI sees only ONE module at a time, with only the interfaces of its dependencies (never their source code)
3. **Verifies** that generated code follows the rules — no global state, no forbidden imports, no undeclared coupling

**Result:** Regressions drop from 14.3% to <0.1% per commit.

---

## Why Does This Exist?

AI coding assistants break code because of **emergent coupling** — the AI sees too much context, creates invisible dependencies between modules, and each fix breaks something else. This is called **Regression Hell**.

Chevron solves this with the **Spatial Constraint Protocol (SCP)**:
- **RAG Denial** — the AI physically cannot see other modules' implementations
- **Contract-scoped prompts** — 128K tokens compressed to ~1,200 high-density tokens
- **Deterministic verification** — AST-based checks, not "ask the AI if it did a good job"

For the full theory, see the [README.md](README.md) or the [research paper (PDF)](https://github.com/dparksports/dparksports/raw/main/SCP%20II%20-%20Neuro-Symbolic%20Resolution.pdf).

---

## Prerequisites

- **Python 3.10+**
- **Gemini API key** (for AI-powered features — Forge, code generation, AI verification)

```bash
# Clone the repository
git clone https://github.com/dparksports/Project-Chevron.git
cd Project-Chevron
```

No `pip install` required for the core language. For AI features, you need the `google-genai` package:

```bash
pip install google-genai
```

Set your API key:

```bash
# Linux/Mac
export GEMINI_API_KEY="your-key-here"

# Windows PowerShell
$env:GEMINI_API_KEY = "your-key-here"
```

---

## Step 1: Convert Your Codebase → SCP Architecture

> **Start here.** The Forge scans any existing codebase and automatically decomposes it into an SCP architecture using Gemini.

### Basic Usage

```bash
# Point it at your project — that's it
python forge.py /path/to/your/project

# Specify output directory
python forge.py /path/to/project --output ./scp_output

# Choose a different Gemini model
python forge.py /path/to/project --model gemini-2.5-flash

# Scan only (no AI call — just see file stats and token counts)
python forge.py /path/to/project --scan-only
```

### What It Does

```
1. ◬ Scan       — discovers all source files, counts tokens, extracts symbols
2. ☤ Decompose  — sends to Gemini, which proposes module boundaries
3. ☾ Generate   — outputs a .chevron spec + Python ArchitectureSpec
4. 𓂀 Report     — shows compression ratio and architecture map
```

### Output Files

The Forge generates three files:

| File | Purpose |
|------|---------|
| `{project}_spec.chevron` | Chevron spec file (verifiable with `--verify`) |
| `{project}_spec.py` | Python `ArchitectureSpec` (use with `scp_bridge.py`) |
| `{project}_cli.py` | CLI scaffold for running the SCP pipeline |

### What Happens Next

Once Forge decomposes your project, you use the generated spec to drive AI code generation:

```python
# Load the generated spec
from your_project_spec import ARCHITECTURE

# Create a bridge
from scp_bridge import SCPBridge
bridge = SCPBridge(ARCHITECTURE)

# Generate a constrained prompt for one module at a time
prompt = bridge.generate_system_prompt("ModuleName")
# → Feed this to Gemini / GPT / Claude
```

---

## Step 2: Generate Code with AI (SCP Bridge)

The SCP Bridge generates **constrained system prompts** that force any AI to write code that follows your architecture.

### Quick Start (No API Key Needed)

Generate the prompt and paste it into any AI chat:

```bash
# List available templates
python scp_bridge.py

# Generate a constrained prompt for one module
python scp_bridge.py todo_app TodoStore python
```

Copy the output into ChatGPT, Claude, Gemini, etc. Then tell the AI:

> "Implement the TodoStore module now."

The AI will only see the TodoStore contract and the interfaces of its dependencies — never the implementations.

### Available Built-In Templates

```bash
python scp_bridge.py
# Output:
#   todo_app         — Todo Application (modules: TodoStore, API)
#   data_pipeline    — Data Processing Pipeline (modules: Ingest, Transform, Load)
```

### Programmatic Usage (with API Key)

```python
from scp_bridge import SCPBridge

# Step 1: Load a template (or your Forge-generated spec)
bridge = SCPBridge.from_template("todo_app")

# Step 2: Generate constrained prompt for ONE module
system_prompt = bridge.generate_system_prompt("TodoStore", language="python")

# Step 3: Feed to Gemini
from google import genai

client = genai.Client(api_key="your-key")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Implement the TodoStore module now.",
    config=genai.types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.1,
    ),
)
code = response.text
print(code)
```

### What the AI Sees vs. Doesn't See

| ✅ Visible to AI | 🚫 Hidden from AI |
|-------------------|---------------------|
| Module's contract & method signatures | Other modules' source code |
| Glyph constraint per method | Internal implementation details |
| Dependency interface signatures | Database schemas, file paths |
| Global architecture rules | Shared state, global variables |

### Generating All Modules at Once

```python
# Generate prompts for every module in the architecture
workspace = bridge.generate_full_workspace(language="python")

for module_name, prompt in workspace.items():
    print(f"\n{'='*60}")
    print(f"Module: {module_name}")
    print(f"Prompt length: {len(prompt)} chars")
```

### Structured Output (JSON Schema)

For Gemini's `response_schema` feature, generate a JSON schema that constrains the AI's output format at the grammar level:

```python
schema = bridge.generate_structured_schema("TodoStore")

# Use with Gemini
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Implement the TodoStore module.",
    config=genai.types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=schema,
    ),
)
```

---

## Step 3: Verify Generated Code

### Option A: Deterministic Verification (Recommended)

Uses Python's `ast` module — no AI involved, no hallucination possible:

```python
# Verify generated code deterministically
violations = bridge.verify_generated_code("TodoStore", code)

if not violations:
    print("✔ W(G) = 0 — code passes all SCP checks")
else:
    for v in violations:
        print(v)  # Line number, check name, and violation message
```

Checks performed:
- No global mutable state
- No forbidden imports (only declared dependencies allowed)
- Side-effect freedom for Filter (Ө) and Witness (𓂀) methods
- Interface conformance (correct methods, correct signatures)

### Option B: AI-Powered Verification

Ask the AI to verify its own output against the contract:

```python
# Generate a verification prompt
verify_prompt = bridge.generate_verification_prompt("TodoStore", code)

# Ask AI to check
verify = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=verify_prompt,
    config=genai.types.GenerateContentConfig(temperature=0.0),
)
print(verify.text)  # → PASS or FAIL with violations
```

### Standalone Code Verification

```python
from chevron.code_verifier import verify_code

code = '''
import os
data = []

class TodoStore:
    def add(self, item):
        data.append(item)  # Uses global state!
'''

violations = verify_code(code)
for v in violations:
    print(v)
# ERROR [line 3] no_global_state: Module-level mutable assignment: 'data' (list)
```

---

## Step 4: Generate & Run Tests

Generate **pytest tests from contracts**, not from implementation. Tests verify the specification, not the code's internals.

### Command Line (with AI generation)

```bash
# Generate all modules + tests for TurboScribe
python examples/turboscribe_example.py --all --with-tests
```

### Programmatic Usage

```python
from chevron.test_generator import SpecTestGenerator
from scp_bridge import SCPBridge

bridge = SCPBridge.from_template("todo_app")
module_spec = bridge._find_module("TodoStore")

generator = SpecTestGenerator()
test_code = generator.generate_tests(module_spec, module_file="todo_store.py")

# Write test file
with open("test_todo_store.py", "w") as f:
    f.write(test_code)
```

### What Tests Are Generated

| Category | What It Tests |
|----------|--------------|
| **Interface** | Methods exist, correct parameter counts |
| **Isolation** | No forbidden imports, no global mutable state |
| **Glyph contracts** | Side-effect freedom (Filter), immutability (Witness) |

### Running Tests

```bash
pytest test_todo_store.py -v
```

---

## Additional Features

### Runtime Decorators

Move glyph contracts from "prompt suggestions" to **runtime guarantees** in your Python code:

```python
import chevron

@chevron.origin
def main(data):
    """Entry point — tracked, raises on double invocation."""
    return process(data)

@chevron.filter
def is_valid(item):
    """Conditional gate — must not perform I/O."""
    return item.score > 0.5

@chevron.fold
def countdown(n):
    """Recursion — bounded depth, no external mutation."""
    if n <= 0:
        return 0
    return countdown(n - 1)

@chevron.fold(max_depth=100)
def fibonacci(n):
    """Fold with custom depth limit."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

@chevron.witness
def log_step(data):
    """Observe only — returns input unchanged, deep-copy protected."""
    print(f"Processing: {data}")
    return data

@chevron.weaver
def merge_results(a, b):
    """Merge streams — must preserve all input data."""
    return {**a, **b}
```

Contract violations raise `ChevronContractError` at runtime.

### Custom Architecture Specs

Define your own architecture specs programmatically instead of using Forge:

```python
from scp_bridge import SCPBridge, ArchitectureSpec, ModuleSpec, InterfaceMethod

spec = ArchitectureSpec(
    name="MyProject",
    modules=[
        ModuleSpec(
            name="UserStore",
            description="Manages user data storage",
            methods=[
                InterfaceMethod(
                    name="create_user",
                    inputs=["username: str", "email: str"],
                    output="User",
                    glyph="◬",
                    constraint="Must validate email format"
                ),
                InterfaceMethod(
                    name="find_user",
                    inputs=["user_id: str"],
                    output="User | None",
                    glyph="Ө",
                    constraint="Must not modify the data store"
                ),
            ],
            allowed_dependencies=["Logger"],
            constraints=[
                "No direct database access — use the Repository interface",
                "All methods must be idempotent"
            ],
        ),
    ],
    global_constraints=[
        "No module may access the file system directly",
        "All inter-module communication through declared interfaces",
    ],
)

bridge = SCPBridge(spec)
prompt = bridge.generate_system_prompt("UserStore", language="python")
```

### The Chevron Language

Chevron also includes a standalone glyph-based language you can use to write and verify architecture specs:

| Glyph | Name | What It Does |
|:-----:|------|--------------|
| `◬` | Origin | Entry point — produces initial data |
| `☾` | Fold | Recursion — output feeds back into input |
| `Ө` | Filter | Conditional gate — only matching data passes |
| `𓂀` | Witness | Observe — logs data without changing it |
| `☤` | Weaver | Merge — braids two streams into one |

```bash
# Run a Chevron program
python run.py examples/hello.chevron

# Run with SCP verification
python run.py examples/turboscribe.chevron --verify

# Interactive REPL
python repl.py
```

---

## End-to-End Examples

### TurboScribe (9-Module Real-World Demo)

```bash
# Generate a single module with Gemini
python examples/turboscribe_example.py Transcriber --gemini

# Generate ALL 9 modules, verify each, run tests
python examples/turboscribe_example.py --all --with-tests

# List all modules
python examples/turboscribe_example.py --list
```

### Gemini Example (Complete 4-Step Workflow)

```bash
python examples/gemini_example.py
```

Runs: template → prompt → generate → verify.

---

## Typical Workflow

```
┌──────────────────────────┐
│  1. Decompose Project     │  python forge.py /path/to/your/project
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  2. Generate AI Prompts   │  scp_bridge.py → constrained prompt (~1,200 tokens)
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  3. AI Generates Code     │  Feed prompt to Gemini / GPT / Claude
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  4. Verify                │  CodeVerifier (deterministic) + AI Weaver (optional)
└────────┬─────────────────┘
         ▼
┌──────────────────────────┐
│  5. Test                  │  SpecTestGenerator → pytest, confirm W(G) = 0
└──────────────────────────┘
```

---

## Quick Reference: Which File Does What?

| File | Purpose | When to Use |
|------|---------|-------------|
| `forge.py` | **Decompose existing codebases** | **Start here** — converts your project to SCP |
| `scp_bridge.py` | Generate AI prompts | AI code generation with any LLM |
| `chevron/code_verifier.py` | Verify generated code | Checking AI output deterministically |
| `chevron/test_generator.py` | Generate tests from specs | Automated testing of contracts |
| `chevron/decorators.py` | Runtime glyph contracts | Adding SCP enforcement to Python |
| `run.py` | Execute `.chevron` files | Running specs and language examples |
| `repl.py` | Interactive REPL | Experimenting with glyphs |
| `chevron/verifier.py` | Verify `.chevron` specs | Checking spec correctness |

---

## Troubleshooting

### "Module not found" errors
Make sure you're running from the project root:

```bash
cd Project-Chevron
python run.py examples/hello.chevron
```

### Gemini API key not set

```bash
# Check if it's set
echo $env:GEMINI_API_KEY  # PowerShell
echo $GEMINI_API_KEY       # Bash

# Set it
$env:GEMINI_API_KEY = "your-key"  # PowerShell
export GEMINI_API_KEY="your-key"   # Bash
```

### Forge fails with "No source files found"
The Forge skips common non-source directories (node_modules, .git, __pycache__). Make sure your source files are in the project root or standard source directories.

### Verification shows violations
Read the violation message — it tells you the exact line, check name, and what went wrong:

```
ERROR [line 15] no_global_state: Module-level mutable assignment: 'cache' (dict)
```

Fix the violation and re-run verification.

---

## Further Reading

- [README.md](README.md) — Full theory and research background
- [SPEC.md](SPEC.md) — Formal Chevron language specification
- [EXTENSIONS.md](EXTENSIONS.md) — Module system, spec mode, type declarations
- [SCP_TESTING.md](SCP_TESTING.md) — Auto-test generation details
- [Research Paper (PDF)](https://github.com/dparksports/dparksports/raw/main/SCP%20II%20-%20Neuro-Symbolic%20Resolution.pdf) — SCP II: Neuro-Symbolic Resolution
