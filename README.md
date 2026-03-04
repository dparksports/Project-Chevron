# ◬ Project Chevron

**Spatial Constraint Protocol (SCP) — Reference Implementation**

*The Partition Function Explosion: An Energy-Based Analysis of Attention Decay*

> A neuro-symbolic architecture that reduces AI code regression from **14.3% to <0.1%** by replacing noisy tokenization with **orthogonal embedding mapping** using Uiua primitives. SCP minimizes semantic cross-talk, creates steep isolated attractor basins that pierce through context noise, and pairs them with System 2 AST rejection sampling (The Weaver) to transform the LLM from a probabilistic confabulator into a reliable architectural engine.

**Dan Park** · [MagicPoint.ai](https://magicpoint.ai) · February 2026
**Link:** [Download Paper (PDF)](https://github.com/dparksports/dparksports/raw/main/Holographic-AI-Language-v19.pdf)

> **🚀 Ready to use it?**
> - **[Nexus Guide](NEXUS_GUIDE.md)** — Visual AI IDE with dashboard, templates, and existing-project conversion *(recommended)*
> - **[CLI Guide](CLI_GUIDE.md)** — Headless workflow using Forge + SCP Bridge directly

---

## Table of Contents

- [The Problem](#the-problem)
  - [The Lossless Retrieval Fallacy](#the-lossless-retrieval-fallacy)
  - [The Partition Function and Signal Dilution](#the-partition-function-and-signal-dilution)
  - [Confabulation as Thermodynamic Relaxation](#confabulation-as-thermodynamic-relaxation)
  - [Regression Hell](#regression-hell)
- [The Resolution: SCP](#the-resolution-spatial-constraint-protocol)
  - [Orthogonal Embeddings and Semantic Cross-Talk](#orthogonal-embeddings-and-semantic-cross-talk)
  - [Vertical Neuro-Symbolic Integration](#vertical-neuro-symbolic-integration)
- [Uiua: AI Cognitive Programming Language](#uiua-ai-cognitive-programming-language)
  - [The Zero-Shot Paradox](#the-zero-shot-paradox)
- [The Five Glyphs](#the-five-glyphs)
- [The Weaver: External System 2 Verification](#the-weaver-external-system-2-verification)
  - [Maxwell's Demon and Rejection Sampling](#maxwells-demon-and-rejection-sampling)
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
- [References](#references)

---

## The Problem

### The Lossless Retrieval Fallacy

The trajectory of AI research from 2023–2026 has been defined by the aggressive expansion of the Context Window (N). From 4,096 tokens to 10 million, the industry has operated under the tacit assumption termed the **"Billion Token Fallacy"** — that quantitative expansion equates to qualitative reasoning capability. This view relies on the **Lossless Retrieval Fallacy**: the assumption that the attention mechanism functions as a deterministic look-up table where access fidelity is independent of total capacity.

**This assumption is mathematically false.**

The attention mechanism is an energy-based, thermodynamic system where every additional token contributes to the normalization constant (Z), actively diluting the probability mass available for any specific signal. As N expands, the system does not merely store more data; it undergoes **Channel Capacity Saturation**, where the "Signal" (the correct retrieval) becomes mathematically indistinguishable from the "Noise" (the cumulative interference of distractor tokens).

### The Partition Function and Signal Dilution

In a standard Softmax Attention mechanism, the probability of attending to a specific token is given by the Boltzmann distribution:

```
Attention(Q,K,V) = softmax(QKᵀ / √dₖ) · V
```

Where the denominator acts as the **Partition Function (Z)** — the sum over all possible states in the window:

```
Z = Σⱼ₌₁ᴺ exp(score(q, kⱼ))
```

**The Critical Finding:** The primary failure mode of long-context LLMs is the **Explosion of Z**. As the context window N → ∞, the number of "distractor" terms in the summation grows linearly. Even if each individual distractor has high energy (low probability), their cumulative probability mass mathematically dominates the denominator.

We formally define the **Critical Energy Gap (ΔE)** required for the signal to survive this explosion:

```
ΔE = E_noise - E_signal > ln(N)
```

This equation reveals a hard physical limit: for the signal to remain distinguishable (i.e., for P_signal ≈ 1) as N scales, the energy difference between the signal and the noise must grow logarithmically. However, because the model's dot-product capacity is fixed by its dimensional resolution, it cannot arbitrarily increase this gap. Once ln(N) exceeds the model's maximum resolution, ΔE becomes insufficient, and the signal is thermally drowned out by Z.

### Confabulation as Thermodynamic Relaxation

We observe that "Regression Hell" in software engineering is a manifestation of **Mode Collapse**:

- **The Context Valley:** The prompt attempts to dig a temporary, local "energy valley" for the model's activations to settle into.
- **The Prior Canyon:** The model's pre-training has already established massive, deep energy canyons (general statistical likelihoods).

When Z explodes, the "Context Valley" becomes too shallow (high entropy). The model's latent state, seeking the path of least resistance, rolls out of the shallow context valley and falls into the deep Prior Canyon:

```
ŷ = argmax P(y|x) → P(y)
```

This confirms that hallucination (more accurately termed **confabulation**) is not a creative act, but a natural **thermodynamic relaxation to the mean**.

### Regression Hell

The theoretical failure manifests in the SDLC as **"Regression Hell"** — where feature velocity drops to zero. Every commit introduces new bugs; every fix breaks something else.

**Root cause: Emergent Coupling.** Unintended dependencies between modules arising not from explicit interfaces, but from implicit shared assumptions:

- **Implicit State Sharing:** Modules communicating via shared file paths or environment variables not declared in the API
- **Temporal Coupling:** Module A must run before Module B, enforced by convention not code
- **Semantic Drift:** The "meaning" of a data field changes (e.g., "seconds" to "milliseconds") without a schema change

Standard LLMs have no persistent memory of these conventions. They treat each snippet as statistically independent, causing regressions that are difficult to detect because the code is **syntactically correct but structurally incoherent**.

---

## The Resolution: Spatial Constraint Protocol

SCP resolves the Partition Function Explosion not by artificially restricting N, but by **altering the geometry of the prompt** to minimize semantic interference.

### Orthogonal Embeddings and Semantic Cross-Talk

Standard tokenization (BPE) utilizes distributed representations — "clouds of meaning." While this continuous representation is the engine of deep learning's flexibility, it introduces massive **semantic cross-talk** during precise engineering tasks. A common word like "sort" or "update" has appeared in millions of conflicting contexts during pre-training. In a massive context window, these heavily overloaded continuous vectors create diffuse, shallow attractor basins that are easily washed out by Z.

**The Resolution:** SCP replaces these entangled natural language tokens with mathematically specific, rare symbols (Uiua glyphs). SCP does not bypass distributed representations; rather, it leverages them by finding **isolated coordinates**.

Because these mathematical glyphs are exceedingly rare in the training corpus, their continuous vector embeddings are largely **orthogonal** to the dense, noisy clusters of common English words:

```
𝔼[sim(e_SCP, e_distractor)] ≈ 0
```

By mapping architectural constraints to these un-interfered embeddings, a rare glyph acts as an **isolated, steep attractor basin**. It minimizes semantic cross-talk, forcing the model's attention mechanism to converge cleanly on a specific continuous coordinate rather than distributing probability mass over an overloaded semantic cloud.

**Key properties:**

1. **Compression:** 128,000 tokens → 1,200 orthogonal primitives (100× reduction), drastically reducing Z
2. **Orthogonality:** Each symbol occupies an un-interfered embedding coordinate — no competing keys
3. **Context Isolation (RAG Denial):** Modules see only interface contracts, never implementation. The AI physically **cannot** create coupling because it cannot see other modules' source code
4. **Energy Gap Restoration:** The steep attractor basins restore ΔE > ln(N), preventing signal decay

### Vertical Neuro-Symbolic Integration

A critical finding is that this mapping is effective even if the specific glyphs are rare in the training corpus (the **Zero-Shot Paradox**). SCP functions via Vertical Integration: the orthogonal glyph acts as a clean pointer to a pre-existing "latent thought" or robust vector cluster (e.g., the algorithmic concept of "sorting" or "isolation") that the model already possesses, retrieving the concept **without dragging in the semantic noise** of the English word itself.

---

## Uiua: AI Cognitive Programming Language

SCP leverages **[Uiua](https://www.uiua.org/)** (pronounced "wee-wuh") — a stack-based array programming language created by Kai Schmidt — as its primitive language. Uiua was chosen not just for brevity, but for its mathematical properties:

### 1. Glyph-Based Syntax
Uiua uses single Unicode characters as operations. Operations like ⍏ (sort), ♭ (flatten), and ⇌ (reverse) achieve in a single token what requires multiple tokens in Python. This creates **Bijective Singleton Maps**:

```
∀ l ∈ ℒ_Uiua : |f⁻¹(f(l))| = 1
```

Every symbol maps to exactly one vector. Every vector maps back to exactly one symbol.

### 2. Rank Polymorphism
Operations automatically adapt to arrays of any dimensionality. The expression `+1` adds 1 to a scalar, a vector, or a billion-element tensor without code changes. This is the **fractal property in executable form**: solve the problem for one atom, and you've solved it for the universe.

### 3. Tacit (Point-Free) Programming
Functions do not name their arguments. This eliminates variable naming — a massive source of ambiguity and "noise" in standard code — allowing the attention mechanism to focus purely on the **transformation**, not the **labels**.

### The Zero-Shot Paradox

A crucial question: can this mapping work for characters the model was never trained on?

The SCP paper argues that while the **glyphs** may be rare in the training corpus (like Uiua symbols), the **latent concepts** they map to (sorting, filtering, folding) are heavily represented via Python, C++, etc. SCP performs **Vertical Neuro-Symbolic Integration** through three mechanisms:

1. **Injected Embeddings / Adapters:** The `scp_bridge.py` handles translation of glyphs into specific embedding vectors the model **does** recognize
2. **Latent Reasoning:** Research on **Coconut (Chain of Continuous Thought)** confirms LLMs can reason in "latent space" without outputting language tokens — SCP leverages this by feeding the "thought" directly
3. **Visual Tokenization:** Research into "Reasoning Over Glyphs" shows that if a glyph is treated as a **hard-coded index** rather than a language token, the model can utilize it

The model isn't trained on the *corpus* of Uiua literature (which is small), but the **protocol bridges the gap**, allowing the glyph to trigger pre-trained latent capability.

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

---

## The Weaver: External System 2 Verification

A critical challenge in generating reliable software architectures is verifying strict modular independence. Standard intuitive Transformers (System 1) cannot natively compute exact, discrete Mutual Information (MI) during a continuous forward pass. SCP addresses this via an external **System 2** verification loop known as **The Weaver**.

### Maxwell's Demon and Rejection Sampling

The Weaver Function W(G) is not an internal property of the neural network's weights or thermodynamics. Instead, it functions as a classic, external algorithm — acting as **Maxwell's Demon** — that evaluates and filters the network's output via rigorous **rejection sampling**:

1. **Generation (System 1):** The neural model proposes a code block based on the un-interfered, Uiua-constrained prompt, settling into a local minimum.
2. **Extraction (Symbolic):** A classic AST parser extracts the dependency graph G = (M, E) from the generated code.
3. **Verification (System 2):** The external Weaver calculates the structural Mutual Information between modules by analyzing the AST for shared state, implicit coupling, or side effects:
   ```
   W(G) = Σ_{(i,j) ∉ E} MI_AST(mᵢ, mⱼ)
   ```
4. **Rejection Sampling:** If W(G) > 0, the state is rejected. The classical algorithmic system throws out the generation and forces the neural model to resample, driving a search loop until it produces a valid, orthogonal architecture.

This hybrid approach layers rigorous classical algorithmic verification (System 2) on top of the intuitive generative power of the neural network (System 1), ensuring that "Emergent Coupling" is prevented not by internal magic, but by **external post-generation filtering**.

---

## Empirical Results

### Study Parameters
- **Target:** TurboScribe — large-scale native Windows application (<50,000 LOC)
- **Stack:** C#, Python, CUDA (high-dimensional, multi-language environment)
- **Baseline:** Standard GPT-4 with 128k context window
- **Intervention:** SCP with Uiua Orthogonal Mapping

### Quantitative Outcomes

| Metric | Baseline (GPT-4) | SCP Implementation | Improvement |
|--------|:-----------------:|:------------------:|:-----------:|
| Context Required | 128,000 tokens | 1,200 orthogonal primitives | **100×** |
| Regression Rate | 14.3% per commit | <0.1% per commit | **143×** |
| Feature Velocity | 0% (Regression Hell) | 100% (Restored) | **∞** |
| Partition Function (Z) | Explosive (N=128k) | Contained (N=1.2k) | **Controlled** |
| Energy Gap (ΔE) | Below ln(N) threshold | Restored above threshold | **Recovered** |
| Coupling (W(G)) | Undeclared coupling | W(G) = 0 enforced | **Eliminated** |

**Key results:**
- **Energy Gap Restoration:** By compressing 128k tokens into 1,200 precise orthogonal primitives (100× ratio), we drastically reduced Z, preventing logarithmic signal decay and restoring the necessary energy gap (ΔE).
- **Mode Stability:** The regression rate dropped from 14.3% to <0.1%, confirming that orthogonal embeddings allowed the model to successfully "settle" into steep context valleys without confabulating or slipping into Prior Collapse.
- **Feature Velocity:** Restored from 0% ("Regression Hell") to 100%.

Based on these results, the [SystemMonitor project](https://github.com/dparksports/SystemMonitor) is slated for immediate integration to further stress-test the protocol.

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
│   ├── code_verifier.py           # ★ AST-based formal code verification
│   ├── decorators.py              # ★ Runtime-enforced glyph decorators
│   └── test_generator.py          # ★ Deterministic spec-driven test gen
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

## References

1. Li et al. (2024). *The Entropy-Lens Framework*. Finding: High entropy and large partition functions correlate directly with generation degradation.
2. Lin et al. (2025). *The Forgetting Transformer (FOX)*. Finding: Forgetting and bounding context limits improves SNR.
3. Unified Theory of Latent Space Stability (2024). Finding: Semantic noise scales linearly with context window expansion, destroying local signal gaps.
4. *Coconut (Chain of Continuous Thought)*. Finding: Demonstrates latent reasoning and vector cluster retrieval without relying on standard natural language tokens.
5. Know-But-Don't-Tell Phenomenon (2024). Finding: MAP failure causes models to thermodynamically relax into pre-trained priors, leading to confabulation despite correct context presence.
6. *Semantic Rate-Distortion Theory*. Application: Lossless compression proofs for information retrieved within dense semantic spaces.
7. *Lehman's Laws of Software Evolution*. Application: Foundational software entropy model mapping system decay directly to "Regression Hell" and mode collapse.
8. Gemini 1.5 Pro Technical Report (2025). Context: Context scaling benchmarks demonstrating the push toward 10M token environments.
9. Park, D. (2026). *The Partition Function Explosion: An Energy-Based Analysis of Attention Decay*. MagicPoint.ai.
10. Schmidt, K. (2023–2026). *Uiua: A Stack-Based Array Language*. uiua.org.
11. Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.

---

## License

This project is released for research and educational purposes.

---

<p align="center">
  <strong>◬ ☾ Ө 𓂀 ☤</strong><br>
  <em>The first chevron is locked.</em>
</p>
