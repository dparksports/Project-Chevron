# CLI Guide — Forge + HL Bridge (Headless Workflow)

> **No dashboard needed. Point the CLI at your project, generate code, verify, ship.**

This guide covers the headless workflow using the standalone Forge and HL Bridge tools. For the visual IDE with dashboard, see the [Nexus Guide](NEXUS_GUIDE.md).

---

## Prerequisites

- **Python 3.10+**
- **Gemini API key**

```bash
git clone https://github.com/dparksports/Project-Chevron.git
cd Project-Chevron
pip install google-genai
```

Set your API key:
```bash
export GEMINI_API_KEY="your-key-here"           # Mac/Linux
$env:GEMINI_API_KEY = "your-key-here"           # Windows PowerShell
```

---

## Step 1: Decompose Your Codebase

The Forge scans any existing codebase and automatically decomposes it into an HL architecture using Gemini.

```bash
# Point it at your project — that's it
python forge.py /path/to/your/project

# Scan only (no AI call — just see file stats and token counts)
python forge.py /path/to/project --scan-only

# Choose a different Gemini model
python forge.py /path/to/project --model gemini-2.5-flash
```

### What Happens

```
1. ◬ Scan       — discovers all source files, counts tokens, extracts symbols
2. ☤ Decompose  — sends to Gemini, which proposes module boundaries
3. ☾ Generate   — outputs spec files
4. 𓂀 Report     — shows compression ratio and next-step commands
```

### Output Files

| File | Purpose |
|------|---------|
| `{project}_scp.py` | Python spec — run this to generate code with Gemini |
| `{project}_architecture.chevron` | Chevron spec (verifiable with `--verify`) |
| `{project}_decomposition.json` | Raw decomposition data |

### Example: Decomposing a 45K-Token Project

```bash
$ python forge.py nexus/

◬ ─── Step 1: Scanning codebase ───
  Files:     17
  Lines:     4,704
  Tokens:    ~45,469

☤ ─── Step 2: Analyzing with Gemini ───
  Model: gemini-3-pro-preview
  Sending 17 files (45,469 tokens) for analysis...

☾ ─── Step 3: Generating HL spec files ───

𓂀  HL Init Complete!

  Generated files:
    📄 nexus_scp.py                   Python HL spec (runnable)
    📄 nexus_architecture.chevron     Chevron architecture spec
    📄 nexus_decomposition.json       Raw decomposition JSON

  Compression:
    Codebase:          ~45,469 tokens
    Per-module prompt:  ~720 tokens (estimated)
    Compression ratio: 63×
```

---

## Step 2: Generate Code with AI

### From a Forge-Generated Spec

```bash
# See all modules in the generated architecture
python nexus_scp.py

# Generate one module with Gemini
python nexus_scp.py ModuleName --gemini

# Generate ALL modules, verify each
python nexus_scp.py --all

# Generate all modules + auto-generate and run tests
python nexus_scp.py --all --with-tests
```

### From a Built-In Template

```bash
# List available templates
python scp_bridge.py

# Generate a constrained prompt for one module
python scp_bridge.py todo_app TodoStore python
```

Copy the output into ChatGPT, Claude, Gemini, etc. Then tell the AI:

> "Implement the TodoStore module now."

The AI will only see the TodoStore contract and dependency interfaces — never other modules' source code.

### Programmatic Usage

```python
from scp_bridge import SCPBridge

# Load a template (or import your Forge-generated spec)
bridge = SCPBridge.from_template("todo_app")

# Generate constrained prompt for ONE module
system_prompt = bridge.generate_system_prompt("TodoStore", language="python")

# Feed to Gemini
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
```

### What the AI Sees vs. Doesn't See

| ✅ Visible to AI | 🚫 Hidden from AI |
|-------------------|---------------------|
| Module's contract & method signatures | Other modules' source code |
| Glyph constraint per method | Internal implementation details |
| Dependency interface signatures | Database schemas, file paths |
| Global architecture rules | Shared state, global variables |

---

## Step 3: Verify Generated Code

### Deterministic Verification (Recommended)

Uses Python's `ast` module — no AI involved, no confabulation possible:

```python
violations = bridge.verify_generated_code("TodoStore", code)

if not violations:
    print("✔ W(G) = 0 — code passes all HL checks")
else:
    for v in violations:
        print(v)
```

Checks:
- No global mutable state
- No forbidden imports (only declared dependencies allowed)
- Topo-Categorical constraint conformance (isolation, directed flow, boundaries)
- Interface conformance (correct methods, correct signatures)

### AI-Powered Verification

```python
verify_prompt = bridge.generate_verification_prompt("TodoStore", code)
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

violations = verify_code(some_code)
for v in violations:
    print(v)
# ERROR [line 3] no_global_state: Module-level mutable assignment: 'data' (list)
```

---

## Step 4: Generate & Run Tests

Generate **pytest tests from contracts**, not from implementation:

```bash
# Full pipeline: generate all modules + tests for TurboScribe
python examples/turboscribe_example.py --all --with-tests
```

```python
from chevron.test_generator import SpecTestGenerator
from scp_bridge import SCPBridge

bridge = SCPBridge.from_template("todo_app")
module_spec = bridge._find_module("TodoStore")

generator = SpecTestGenerator()
test_code = generator.generate_tests(module_spec, module_file="todo_store.py")

with open("test_todo_store.py", "w") as f:
    f.write(test_code)
```

```bash
pytest test_todo_store.py -v
```

---

## Additional Features

### Runtime Decorators

Enforce glyph contracts at runtime in your Python code:

```python
import chevron

@chevron.origin      # Entry point — tracked, raises on double invocation
@chevron.filter      # Must not perform I/O
@chevron.fold        # Bounded recursion depth
@chevron.witness     # Returns input unchanged, deep-copy protected
@chevron.weaver      # Must preserve all input data
```

Violations raise `ChevronContractError`.

### Custom Architecture Specs

Define specs programmatically instead of using Forge:

```python
from scp_bridge import SCPBridge, ArchitectureSpec, ModuleSpec, InterfaceMethod

spec = ArchitectureSpec(
    name="MyProject",
    modules=[
        ModuleSpec(
            name="UserStore",
            description="Manages user data",
            methods=[
                InterfaceMethod("create_user", ["name: str", "email: str"],
                                "User", "◬", "Must validate email format"),
            ],
            allowed_dependencies=["Logger"],
            constraints=["No direct database access"],
        ),
    ],
    global_constraints=["All inter-module communication through declared interfaces"],
)

bridge = SCPBridge(spec)
prompt = bridge.generate_system_prompt("UserStore", language="python")
```

### Structured Output (JSON Schema)

For Gemini's `response_schema` feature:

```python
schema = bridge.generate_structured_schema("TodoStore")
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

### Chevron Language & REPL

```bash
python run.py examples/hello.chevron           # Run a .chevron file
python run.py examples/turboscribe.chevron --verify  # With HL verification
python repl.py                                 # Interactive REPL
```

---

## Quick Reference

| File | Purpose |
|------|---------|
| `forge.py` | **Decompose existing codebases** → HL architecture |
| `scp_bridge.py` | Generate AI-constrained prompts for any LLM |
| `chevron/code_verifier.py` | Deterministic AST-based verification |
| `chevron/test_generator.py` | Generate pytest tests from specs |
| `chevron/decorators.py` | Runtime glyph contract enforcement |
| `run.py` | Execute `.chevron` files |
| `repl.py` | Interactive Chevron REPL |

---

## Troubleshooting

### Gemini API key not set
```bash
echo $env:GEMINI_API_KEY        # PowerShell
echo $GEMINI_API_KEY             # Bash
$env:GEMINI_API_KEY = "your-key" # Set in PowerShell
```

### Forge: "No source files found"
The Forge skips `node_modules`, `.git`, `__pycache__`. Ensure your source files are in standard directories.

### Verification shows violations
The message tells you the exact line, check name, and what went wrong:
```
ERROR [line 15] no_global_state: Module-level mutable assignment: 'cache' (dict)
```

---

## Further Reading

- [Nexus Guide](NEXUS_GUIDE.md) — Visual dashboard + Nexus CLI workflow
- [README.md](README.md) — Full HL theory and research background
- [SPEC.md](SPEC.md) — Formal Chevron language specification
- [EXTENSIONS.md](EXTENSIONS.md) — Module system, spec mode, type declarations
- [SCP_TESTING.md](SCP_TESTING.md) — Auto-test generation details
- [Research Paper (PDF)](https://github.com/dparksports/dparksports/raw/main/SCP%20II%20-%20Neuro-Symbolic%20Resolution.pdf) — The Partition Function Explosion: An Energy-Based Analysis of Attention Decay
