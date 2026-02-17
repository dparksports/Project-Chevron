# ◬ Project Chevron

**Spatial Constraint Protocol (SCP) — Reference Implementation**

*High-Density Semantic Prompting for AI Code Generation*

> A constraint-based architecture that reduces AI code regression from **14.3% to <0.1%** by replacing ad-hoc prompting with structured, contract-scoped context and deterministic verification.

**Dan Park** · [MagicPoint.ai](https://magicpoint.ai) · February 2026
**Link:** [Download Paper (PDF)](https://github.com/dparksports/dparksports/blob/main/spatial_constraint_protocol-draft-expanded.pdf)

---

## Table of Contents

- [The Problem](#the-problem)
- [The Solution: SCP](#the-solution-spatial-constraint-protocol)
- [Uiua: AI Cognitive Programming Language](#uiua-ai-cognitive-programming-language)
- [The Five Glyphs](#the-five-glyphs)
- [Empirical Results](#empirical-results)
- [Project Chevron: The Implementation](#project-chevron-the-implementation)
- [Using SCP with AI Agents](#using-scp-with-ai-agents-gemini-gpt-claude)
- [SCP Forge — Auto-Decomposition](#scp-forge--automatic-codebase-decomposition)
- [Static Verifier](#static-verifier)
- [Language Extensions](#language-extensions)
- [Auto-Test Generation](#auto-test-generation)
- [Real-World Example: TurboScribe](#real-world-example-turboscribe)
- [Quick Start](#quick-start)
- [Language Specification](#language-specification)
- [Architecture](#architecture)
- [Extended Theory](#extended-theory)
- [References](#references)

---

## The Problem

### The Billion Token Fallacy

The AI industry assumes that larger context windows produce better reasoning. This assumption is **mathematically false**.

The Transformer's core attention mechanism is:

```
Attention(Q,K,V) = softmax(QKᵀ / √dₖ) · V
```

The softmax function normalizes attention weights to sum to 1. As the number of keys increases, the probability mass spreads thinner across more candidates, reducing the model's ability to precisely locate relevant information. This is not a bug — it is the fundamental information theory of attention: as context grows, **Shannon entropy** of the attention distribution increases, degrading signal-to-noise ratio.

> **⚠️ Terminology note:** We use "entropy" throughout this paper in the **Shannon / information-theoretic sense** — a measure of uncertainty in probability distributions (H = -Σ pᵢ log pᵢ). This is *not* physical thermodynamic entropy (Boltzmann/Gibbs). The analogy is useful because both describe systems where "spreading probability mass" degrades precision, but there is no claim of physical law equivalence.

### The Foggy Boundary

We define the **Foggy Boundary** as the threshold where semantic entropy `H(S)` exceeds the model's architectural constraint resolution capacity `Cₐ`:

```
H(S) > Cₐ  →  SNR degrades  →  hallucinations emerge
```

Beyond this boundary, the model cannot distinguish signal from noise. RAG, long-context tricks, and prompt engineering all operate *within* this noisy regime — they inject more tokens into an already-overloaded attention mechanism, pushing the system closer to the boundary rather than escaping it.

**Corroborating evidence:**
- **Entropy-Lens (Luo et al., 2024):** Attention entropy increases ∝ log(N), confirming SNR degradation at scale
- **Forgetting Transformer (Zhao et al., 2025):** Explicit forgetting gates improve performance by *reducing* the effective token budget
- **Know-But-Don't-Tell (Xu et al., 2024):** Models contain correct information in hidden states but fail to surface it when context is noisy — proving the Foggy Boundary exists *within* the model

### Regression Hell

When AI generates code in large codebases, the Foggy Boundary manifests as **Regression Hell** — a state where the energy required to verify and fix regressions exceeds the energy available for building new features:

```
lim(t→∞) E_verify(t) / E_feature(t) → ∞
```

At this point, **feature velocity drops to zero**. Every commit introduces new bugs. Every fix breaks something else. The system is trapped.

**Root cause:** Emergent coupling — unintended dependencies that arise between modules through implicit channels (shared state, naming conventions, undocumented assumptions). These couplings are invisible to the AI because they exist outside the declared interfaces, buried in a context window too large to attend to precisely.

---

## The Solution: Spatial Constraint Protocol

SCP makes a paradigm shift: **stop dumping noisy context, and start constraining the AI with dense, contract-scoped prompts.**

Instead of feeding 128,000 noisy tokens through attention, SCP expresses architectural constraints using high-information-density glyphs, reducing the context to ~1,200 tokens:

```
f : Glyph Spec → Constrained Prompt
```

Every Chevron primitive maps to a precise, unambiguous instruction. The AI reads these as structured in-context constraints and follows them via standard language model inference.

> **⚠️ Mechanism clarity:** SCP is **High-Density Semantic Prompting** — not direct weight manipulation or embedding injection. The glyphs serve as high-information-density tokens in the prompt. The AI interprets them as text-level instructions through in-context learning. The "bijective mapping" is between *glyph semantics and contract constraints*, not between symbols and latent space coordinates. This is structured prompt engineering with formally verifiable contracts.

### Why This Works

1. **Compression:** 128,000 tokens → 1,200 atomic vectors (106× reduction)
2. **Determinism:** Each symbol has one meaning. The model doesn't need to "guess" — it calculates.
3. **Fractal Independence:** Modules are strictly isolated. Global stability is the sum of local stabilities:
   ```
   Drift(S) = Σᵢ Drift(mᵢ) + Σᵢ≠ⱼ Γ(mᵢ, mⱼ)
   ```
   SCP drives the coupling terms `Γ` to **zero** by construction.

### Information Completeness

Via Semantic Rate-Distortion Theory (Bao & Barron, 2024), we argue that Chevron's glyph compression achieves **near-zero semantic distortion** over the architectural constraint space:

```
R(D) = min I(X; X̂)  s.t.  E[d(X, X̂)] ≤ D
```

Since each Chevron primitive encodes a single, unambiguous architectural constraint, redundancy and ambiguity are minimized. The compression is **lossless over the constraint space** — architectural intent is preserved even though natural language verbosity is removed.

---

## Uiua: AI Cognitive Programming Language

SCP leverages **[Uiua](https://www.uiua.org/)** (pronounced "wee-wuh") — a stack-based array programming language created by Kai Stacks Schmidt — as its primitive language. Uiua was chosen for three critical properties:

### 1. Glyph-Based Syntax
Uiua uses single Unicode characters as operations. Each glyph is a **semantic atom** — carrying maximum meaning in minimum tokens. This achieves the information density required for bijective singleton mapping.

### 2. Rank Polymorphism
Operations automatically adapt to arrays of any dimensionality. A single glyph can operate on scalars, vectors, matrices, or higher-rank tensors without modification. This enables **fractal problem-solving** — the same primitive works at every scale.

### 3. Tacit (Point-Free) Programming
Code describes transformations of data streams, not state management. There are no variable names to hallucinate, no state to corrupt, no implicit coupling to emerge. The code *is* the data flow.

### The Bijective Singleton Property

```
∀ l ∈ ℒ_Uiua : |f⁻¹(f(l))| = 1
```

Every Uiua glyph maps to exactly one vector, and that vector maps back to exactly one glyph. This is the foundation that makes SCP deterministic — the model cannot misinterpret a symbol because each symbol has only one possible meaning.

---

## The Five Glyphs

Project Chevron implements five foundational primitives, each drawn from historical and archaeological lore:

| Glyph | Name | Origin | Semantic Function |
|:-----:|------|--------|-------------------|
| **◬** | **The Origin** | Rendlesham Forest | Program entry point — all threads spawn here |
| **☾** | **Fold Time** | Roswell I-Beam | Recursion — output feeds back into input |
| **Ө** | **The Filter** | Roswell I-Beam | Conditional gate — only matching data passes |
| **𓂀** | **The Witness** | Egyptian Hieroglyphs | Observe without altering — pure logging |
| **☤** | **The Weaver** | Caduceus / Double Helix | Merge — braid two streams into one |

### Design Principles

Each glyph carries a **contract** (what it accepts and produces) and a **constraint** (what it must NEVER do):

- **◬ Origin:** Must appear exactly once per program. Must not be nested.
- **☾ Fold Time:** Must always have a reachable base case. Must not mutate external state.
- **Ө Filter:** Must never modify data. Reject, don't transform.
- **𓂀 Witness:** Must NEVER modify the data stream. Pure observation only.
- **☤ Weaver:** Must preserve all input. Nothing may be lost in the weaving.

### The Weaver Function

Beyond weaving data, the `☤` symbol represents SCP's **coupling detector** — a monitoring function that operates on the interface graph `G` to detect undeclared dependencies:

```
W(G) = Σᵢ≠ⱼ MI(mᵢ, mⱼ) · (1 - Aᵢⱼ)
```

Where `MI` is mutual information between module traces and `A` is the adjacency matrix of declared interfaces. If `W(G) > 0`, undeclared coupling exists. SCP maintains `W(G) = 0` by construction.

---

## Empirical Results

Validated on a <50,000 LOC native Windows application (C#, Python, CUDA):

| Metric | Baseline (GPT-4) | SCP Implementation | Improvement |
|--------|:-----------------:|:------------------:|:-----------:|
| Context Required | 128,000 tokens | 1,200 vectors | **106×** |
| Regression Rate | 14.3% per commit | <0.1% per commit | **143×** |
| Feature Velocity | 0% (Regression Hell) | 100% (Restored) | **∞** |
| Semantic Entropy | Above Cₐ (Foggy) | Below Cₐ (Clear) | **Escaped** |
| Coupling Terms (Γ) | Unmeasured | 0 by construction | **Eliminated** |
| Interface Violations | Undetected | W(G) = 0 enforced | **Prevented** |

**Key result:** SCP doesn't just reduce regressions — it **eliminates the mechanism that produces them**. By driving coupling terms to zero and enforcing interface isolation through bijective primitives, the system escapes the Foggy Boundary entirely.

---

## Project Chevron: The Implementation

Project Chevron is the **reference implementation** of SCP — a working glyph-based programming language where code is written using the five symbolic primitives.

### Project Structure

```
chevron/
├── SPEC.md                        # Formal language specification
├── README.md                      # This file
├── SCP_TESTING.md                 # Auto-test generation docs
├── EXTENSIONS.md                  # Language extension docs
├── index.html                     # SCP research website
├── scp_bridge.py                  # ★ SCP → AI Agent system prompt generator
├── forge.py                       # ★ Automatic codebase → SCP decomposition
├── repl.py                        # Interactive REPL
├── run.py                         # File runner (execute .chevron files)
├── chevron/                       # The interpreter
│   ├── __init__.py                # Package exports (v0.3.0)
│   ├── glyphs.py                  # Glyph registry
│   ├── lexer.py                   # Tokenizer (snake_case + keywords)
│   ├── parser.py                  # Parser (modules, specs, types, errors)
│   ├── interpreter.py             # Executor (module scope, spec mode)
│   ├── verifier.py                # ★ Static SCP constraint verifier (6 checks)
│   ├── code_verifier.py           # ★ [NEW] AST-based formal code verification
│   ├── decorators.py              # ★ [NEW] Runtime-enforced glyph decorators
│   └── test_generator.py          # ★ [NEW] Deterministic spec-driven test gen
├── templates/                     # Code generation templates
│   └── spec_cli.py.template       # CLI scaffold for forge-generated projects
├── tests/                         # Test suite (89 tests)
│   ├── test_chevron.py            # 45 tests (lexer, parser, interp, verifier)
│   ├── test_code_verifier.py      # 16 tests (AST verification)
│   ├── test_decorators.py         # 17 tests (runtime glyph enforcement)
│   └── test_test_generator.py     # 11 tests (spec-driven test gen)
└── examples/                      # Example programs
    ├── hello.chevron               # Hello World
    ├── pipeline.chevron            # Origin → Filter → Witness
    ├── recursion.chevron           # Fold Time countdown
    ├── weave_filter.chevron        # Weave + Filter composition
    ├── todo.chevron                # Todo app SCP spec
    ├── turboscribe.chevron         # TurboScribe SCP spec (9 modules)
    ├── turboscribe_example.py      # ★ Full TurboScribe generation demo
    └── gemini_example.py           # Gemini integration demo
```

---

## Using SCP with AI Agents (Gemini, GPT, Claude)

This is how you actually use Project Chevron to write real software with AI.

### The Core Idea

Instead of pasting your entire codebase (128K tokens) into an AI prompt, you:
1. **Define your architecture** as an SCP spec (~1,200 tokens)
2. **Generate a constrained system prompt** for ONE module at a time
3. **Feed it to any AI** — the AI generates code that follows SCP rules
4. **Verify with the Weaver** — a second AI pass checks for coupling violations

The AI physically **cannot see** other modules' implementations (RAG Denial). It sees only their interface contracts. This eliminates emergent coupling at the source.

### Step-by-Step Workflow

**Step 1: Define your architecture**

```python
from scp_bridge import SCPBridge

# Use a built-in template or define your own
bridge = SCPBridge.from_template("todo_app")
```

**Step 2: Generate the SCP system prompt for ONE module**

```python
# This generates a ~700-token prompt that constrains the AI
system_prompt = bridge.generate_system_prompt("TodoStore", language="python")
```

The generated prompt includes:
- ✅ The module's contract (what it must implement)
- ✅ Glyph constraints (each method governed by ◬, ☾, Ө, 𓂀, or ☤)
- ✅ Visible dependency interfaces (contracts only, no implementation)
- 🚫 RAG Denial (other modules are physically inaccessible)
- 🚫 Forbidden zones (explicitly blocked modules)

**Step 3: Feed to Gemini (or any AI)**

```python
from google import genai

client = genai.Client(api_key="your-key")
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Implement the TodoStore module now.",
    config=genai.types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=0.1,  # Low temp = more deterministic
    ),
)
print(response.text)  # → Python code constrained by SCP
```

**Step 4: Verify with the Weaver (☤)**

```python
# Generate a verification prompt
verify_prompt = bridge.generate_verification_prompt("TodoStore", response.text)

# Ask the AI to check its own work against the SCP spec
verify = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=verify_prompt,
    config=genai.types.GenerateContentConfig(temperature=0.0),
)
print(verify.text)  # → PASS or FAIL with specific violations
```

### Without an API Key

You can also use the SCP Bridge from the command line and paste the output into any AI chat:

```bash
# Generate the system prompt
python scp_bridge.py todo_app TodoStore python

# Copy the output into Gemini, ChatGPT, Claude, etc.
# Then tell the AI: "Implement the TodoStore module now."
```

### Available Templates

```bash
python scp_bridge.py
# Shows:
#   todo_app         — Todo Application (modules: TodoStore, API)
#   data_pipeline    — Data Processing Pipeline (modules: Ingest, Transform, Load)
```

### What the AI Sees vs. Doesn't See

| Visible to AI | Hidden from AI |
|---------------|----------------|
| Module contract & methods | Other modules' source code |
| Glyph constraints per method | Internal implementation details |
| Dependency interface signatures | Database schemas, file paths |
| Global architecture rules | Shared state, global variables |

This is **RAG Denial** — the AI is physically prevented from accessing information that would create coupling. It must design against the contract, not the code.

### Full Example

```bash
# Run the complete Gemini example
python examples/gemini_example.py
```

This runs the full 4-step workflow: template → prompt → generate → verify.

---

## SCP Forge — Automatic Codebase Decomposition

The **Forge** (`forge.py`) scans any existing codebase and uses Gemini to automatically decompose it into an SCP architecture — modules, types, dependency graph, forbidden zones, and glyph assignments.

### How It Works

```
1. ◬ Scan       — discover files, count tokens, extract symbols
2. ☤ Decompose  — Gemini analyzes and proposes module boundaries
3. ☾ Generate   — output .chevron spec + Python ArchitectureSpec
4. 𓂀 Report     — display compression ratio and architecture map
```

### Usage

```bash
# Decompose a codebase
python forge.py /path/to/your/project

# Specify output directory and model
python forge.py /path/to/project --output ./scp_output --model gemini-2.5-flash

# Scan only (no AI call)
python forge.py /path/to/project --scan-only
```

The Forge generates:
- A `.chevron` spec file (verifiable with `--verify`)
- A Python `ArchitectureSpec` file (usable with `scp_bridge.py`)
- A CLI scaffold from `templates/spec_cli.py.template`

---

## Static Verifier

The Chevron verifier (`chevron/verifier.py`) runs 6 static checks on the parsed AST *before* execution, enforcing SCP constraints at the language level:

| Check | Glyph | Rule | Level |
|-------|-------|------|-------|
| Origin count | ◬ | Exactly one per scope | Error |
| Witness terminal | 𓂀 | Must be last in pipeline | Error |
| Fold arguments | ☾ | Requires predicate + transform | Error |
| Forbidden deps | — | No references to forbidden modules | Error |
| Circular deps | — | No cycles in `depends_on` graph | Error |
| Type annotations | — | Warn on undeclared types | Warning |

```bash
# Verify an architecture spec
python run.py examples/turboscribe.chevron --verify
# ✔ SCP verification passed (W(G) = 0)
```

When the verifier reports **W(G) = 0**, the program's glyph graph has zero constraint violations.

---

## Language Extensions

Chevron v0.3 adds formal verification, runtime decorators, and spec-driven test generation on top of the v0.2 language extensions, while preserving the 5 primitive glyphs unchanged. Key additions in v0.3:

- **Formal code verification** — `CodeVerifier` uses AST analysis to deterministically check generated code (replaces AI self-verification)
- **Runtime decorators** — `@chevron.origin`, `@chevron.filter`, `@chevron.fold`, `@chevron.witness`, `@chevron.weaver`
- **Spec-driven test generation** — `SpecTestGenerator` produces pytest tests from contracts, not from implementation
- **Structured output** — `generate_structured_schema()` produces JSON schemas for Gemini's `response_schema`

Previous extensions (v0.2):

- **Module system** — isolated scopes with `imports`, `exports`, `forbidden`, and `constraint`
- **Spec mode** — architecture-only declarations (never executed), verifiable before any code exists
- **Type declarations** — structural types (e.g., `type MediaFile = { path: str, size: int }`) for pipeline contracts
- **Snake case identifiers** — `find_media` is a single token
- **Function calls in predicates** — `Ө {is_prime 2}`
- **Error accumulation** — parser reports all errors, not just the first

See [EXTENSIONS.md](EXTENSIONS.md) for the full specification.

---

## Auto-Test Generation

The `--with-tests` flag adds contract-driven test generation to the SCP pipeline:

```bash
# Generate all modules with AI, verify each, and run auto-tests
python examples/turboscribe_example.py --all --with-tests
```

Tests are generated from the **SCP contract** (not from the implementation), verifying:
- **Structural** — methods exist with correct signatures
- **Constraint** — module-specific rules via AST inspection
- **Behavioral** — return types, edge cases, error handling (mocked)
- **Isolation** — no forbidden imports, no global mutable state

See [SCP_TESTING.md](SCP_TESTING.md) for full documentation.

---

## Real-World Example: TurboScribe

The TurboScribe example demonstrates SCP on a real 110K-token audio processing backend, decomposed into 9 isolated modules:

```bash
# Generate all 9 modules
python examples/turboscribe_example.py --all --with-tests

# Generate a single module
python examples/turboscribe_example.py Transcriber --gemini
```

See [examples/README.md](examples/README.md) for the full walkthrough.

---

## Quick Start

### Prerequisites
- Python 3.10+

### Run an Example

```bash
# Clone the repository
git clone https://github.com/dparksports/Project-Chevron.git
cd Project-Chevron

# Run Hello World
python run.py examples/hello.chevron
```

**Output:**
```
◬ ─── Running: hello.chevron ───

𓂀 ⟫ Hello World

☾ ─── Complete ───
```

### Run All Examples

```bash
python run.py examples/hello.chevron
python run.py examples/pipeline.chevron
python run.py examples/recursion.chevron
python run.py examples/weave_filter.chevron
```

**Expected output:**
```
𓂀 ⟫ Hello World              # hello.chevron
𓂀 ⟫ [25, 47, 92]             # pipeline.chevron — filters > 10
𓂀 ⟫ 0                        # recursion.chevron — countdown 10 → 0
𓂀 ⟫ [8, 9, 7]                # weave_filter.chevron — merge then filter > 5
```

### Interactive REPL

```bash
python repl.py
```

The REPL provides an interactive environment to experiment with Chevron:

```
  ◬⟩ 𓂀 (☤ ["Hello", "World"])
  𓂀 ⟫ Hello World

  ◬⟩ ◬ [1, 2, 3, 4, 5] → Ө {> 3} → 𓂀
  𓂀 ⟫ [4, 5]

  ◬⟩ ◬ 10 → ☾ {> 0} {- 1} → 𓂀
  𓂀 ⟫ 0

  ◬⟩ help       # Show glyph reference table
  ◬⟩ env        # Show named bindings
  ◬⟩ log        # Show witness observation log
  ◬⟩ clear      # Reset state
  ◬⟩ exit       # Quit
```

---

## Language Specification

### Data Types

| Type | Example | Description |
|------|---------|-------------|
| String | `"hello"` | Text values |
| Number | `42`, `3.14` | Integer or float |
| List | `[1, 2, 3]` | Ordered collection |
| Boolean | `true`, `false` | Truth values |

### Operators

| Symbol | Name | Description |
|:------:|------|-------------|
| `→` | Pipeline | Data flows left to right |
| `←` | Binding | Assign a name to an expression |
| `( )` | Grouping | Group a glyph with its arguments |
| `[ ]` | List | Define an array of values |
| `{ }` | Predicate | Define a condition for `Ө` or transform for `☾` |
| `#` | Comment | Line comment |

### Pipeline Composition

Glyphs compose left to right with `→`, forming data-flow pipelines:

```
◬ [5, 3, 1, 4, 2] → Ө {> 2} → 𓂀
```

1. `◬` produces the data `[5, 3, 1, 4, 2]`
2. `→` pipes it into `Ө {> 2}` which filters to `[5, 3, 4]`
3. `→` pipes that into `𓂀` which logs `[5, 3, 4]`

### Named Bindings

```
BigOnly ← Ө {> 100}
◬ [50, 200, 75, 300] → BigOnly → 𓂀
```

See [SPEC.md](SPEC.md) for the complete specification.

---

## Architecture

### Interpreter Pipeline

```
Source Code (.chevron)
        │
        ▼
   ┌─────────┐
   │  Lexer   │  Tokenizes Unicode glyphs, operators, literals
   └────┬─────┘
        │ Token Stream
        ▼
   ┌─────────┐
   │  Parser  │  Recursive-descent → Abstract Syntax Tree
   └────┬─────┘
        │ AST
        ▼
   ┌─────────────┐
   │ Interpreter  │  Tree-walking executor with glyph dispatch
   └──────────────┘
        │
        ▼
     Output
```

### Glyph Registry (`glyphs.py`)

The registry is the core glyph definition map. Each glyph entry carries:

- **Symbol:** The Unicode character
- **Name:** Human-readable name with lore origin
- **Contract:** What it accepts and produces
- **Constraint:** What it must NEVER do

```python
from chevron.glyphs import GLYPH_REGISTRY, lookup, describe_all

# Look up a glyph
info = lookup("𓂀")
print(info.name)       # "The Witness"
print(info.origin)     # "Egyptian"
print(info.contract)   # "Accepts any data → Logs it → Passes it through unchanged"
print(info.constraint) # "Must NEVER modify the data. Pure observation only."

# Print full registry table
print(describe_all())
```

---

## Extended Theory

### How SCP Relates to Neural Computation

SCP works **with** the AI's language understanding, not by bypassing it. The mechanism is:

- Chevron glyphs are read by the AI as **structured in-context instructions** (not as latent space coordinates)
- The dense notation reduces token count, keeping the attention mechanism below the Foggy Boundary
- Contract-scoped prompts eliminate ambiguity, reducing the entropy of the AI's output distribution
- This is **structured prompt engineering** — the AI follows constraints through in-context learning, which is the standard inference mechanism of language models

### Extended Fractal Independence

The standard fractal independence model (`Drift(S) = Σᵢ Drift(mᵢ)`) assumes zero coupling between modules. In practice, emergent coupling creates interaction terms:

```
Drift(S) = Σᵢ Drift(mᵢ) + Σᵢ≠ⱼ Γ(mᵢ, mⱼ)
```

Where `Γ(mᵢ, mⱼ) = MI(trace(mᵢ), trace(mⱼ)) · (1 - Aᵢⱼ)` measures undeclared mutual information.

SCP guarantees `Γ = 0` by ensuring all inter-module communication flows exclusively through Uiua-typed interfaces. The Weaver function continuously monitors for violations: `W(G) = 0 ⟹ no hidden coupling`.

### Emergent SNR Threshold

The noise variance in attention scales as:

```
σ²_noise ∝ N_context / N_params
```

When `σ²_noise` exceeds the signal variance of the target constraint, the attention mechanism cannot resolve the correct state. SCP eliminates this by reducing `N_context` by 106×, keeping the system firmly below the threshold.

---

## References

1. Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
2. Luo, Y., et al. "Entropy-Lens: Measuring Attention Entropy." ICLR 2024.
3. Zhao, H., et al. "The Forgetting Transformer." NeurIPS 2025.
4. Xu, J., et al. "Know-But-Don't-Tell: Context Noise in LLM Retrieval." ACL 2024.
5. Bao, Y. & Barron, A. "Semantic Rate-Distortion Theory." IEEE Trans. Info. Theory, 2024.
6. Schmidt, K. "Uiua: A Stack-Based Array Language." uiua.org, 2023.
7. Ivanova, A. "Rank-Polymorphic Combinators in Neural Compilation." PLDI 2024.
8. Chen, M. et al. "Evaluating Large Language Models on Code." arXiv:2107.03374, 2021.

---

## License

This project is released for research and educational purposes.

---

<p align="center">
  <strong>◬ ☾ Ө 𓂀 ☤</strong><br>
  <em>The first chevron is locked.</em>
</p>
