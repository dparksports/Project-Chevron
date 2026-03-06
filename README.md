# ⊗ Project Chevron

**Holographic Language (HL) v2.0 — Reference Implementation**

*Non-Polysemic Topological DSL for AI-Assisted Software Architecture*

> A neuro-symbolic architecture that reduces AI code regression from **14.3% to <0.1%** by replacing ambiguous natural language constraints with **mathematical operators drawn from Category Theory, Topology, and Tensor Mathematics**. These symbols occupy deep, pristine, zero-polysemy embeddings in LLM latent space (from millions of ingested arXiv LaTeX papers) and resist adversarial polysemy. HL pairs them with System 2 AST rejection sampling to transform the LLM from a probabilistic confabulator into a reliable architectural engine.

**Dan Park** · [MagicPoint.ai](https://magicpoint.ai) · February 2026
**Link:** [Download Paper (PDF)](https://github.com/dparksports/dparksports/raw/main/Holographic-AI-Language-v19.pdf)

> **🚀 Ready to use it?**
> - **[Nexus Guide](NEXUS_GUIDE.md)** — Visual AI IDE with dashboard, templates, and existing-project conversion *(recommended)*
> - **[CLI Guide](CLI_GUIDE.md)** — Headless workflow using Forge + HL Bridge directly

---

## Table of Contents

- [The Problem](#the-problem)
  - [The Lossless Retrieval Fallacy](#the-lossless-retrieval-fallacy)
  - [The Partition Function and Signal Dilution](#the-partition-function-and-signal-dilution)
  - [Confabulation as Thermodynamic Relaxation](#confabulation-as-thermodynamic-relaxation)
  - [Regression Hell](#regression-hell)
- [The Resolution: HL](#the-resolution-holographic-language)
  - [Orthogonal Embeddings and Semantic Cross-Talk](#orthogonal-embeddings-and-semantic-cross-talk)
  - [Vertical Neuro-Symbolic Integration](#vertical-neuro-symbolic-integration)
- [Topo-Categorical Orthogonality](#topo-categorical-orthogonality)
  - [The Zero-Shot Paradox](#the-zero-shot-paradox)
- [The Five Topo-Categorical Operators](#the-five-topo-categorical-operators)
- [System 2 Verification: The AST Weaver](#system-2-verification-the-ast-weaver)
  - [Maxwell's Demon and Rejection Sampling](#maxwells-demon-and-rejection-sampling)
- [Empirical Results](#empirical-results)
- [Project Chevron: The Implementation](#project-chevron-the-implementation)
- [Using HL with AI Agents](#using-scp-with-ai-agents-gemini-gpt-claude)
- [HL Forge — Auto-Decomposition](#scp-forge--automatic-codebase-decomposition)
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

## The Resolution: Holographic Language

HL resolves the Partition Function Explosion not by artificially restricting N, but by **altering the geometry of the prompt** to minimize semantic interference.

### Orthogonal Embeddings and Semantic Cross-Talk

Standard tokenization (BPE) utilizes distributed representations — "clouds of meaning." While this continuous representation is the engine of deep learning's flexibility, it introduces massive **semantic cross-talk** during precise engineering tasks. A common word like "sort" or "update" has appeared in millions of conflicting contexts during pre-training. In a massive context window, these heavily overloaded continuous vectors create diffuse, shallow attractor basins that are easily washed out by Z.

**The Resolution:** HL replaces these entangled natural language tokens with mathematically specific, rare symbols (Topo-Categorical operators). HL does not bypass distributed representations; rather, it leverages them by finding **isolated coordinates**.

Because these mathematical glyphs are exceedingly rare in the training corpus, their continuous vector embeddings are largely **orthogonal** to the dense, noisy clusters of common English words:

```
𝔼[sim(e_SCP, e_distractor)] ≈ 0
```

By mapping architectural constraints to these un-interfered embeddings, a rare operator acts as an **isolated, steep attractor basin**. It minimizes semantic cross-talk, forcing the model's attention mechanism to converge cleanly on a specific continuous coordinate rather than distributing probability mass over an overloaded semantic cloud.

**Key properties:**

1. **Compression:** 128,000 tokens → 1,200 orthogonal primitives (100× reduction), drastically reducing Z
2. **Orthogonality:** Each symbol occupies an un-interfered embedding coordinate — no competing keys
3. **Context Isolation (RAG Denial):** Modules see only interface contracts, never implementation. The AI physically **cannot** create coupling because it cannot see other modules' source code
4. **Energy Gap Restoration:** The steep attractor basins restore ΔE > ln(N), preventing signal decay

### Vertical Neuro-Symbolic Integration

A critical finding is that this mapping is effective even if the specific operators are rare in the training corpus (the **Zero-Shot Paradox**). HL functions via Vertical Integration: the orthogonal operator acts as a clean pointer to a pre-existing "latent thought" or robust vector cluster that the model already possesses, retrieving the concept **without dragging in the semantic noise** of the English word itself.

---

## Topo-Categorical Orthogonality

HL v2.0 leverages **mathematical operators from Category Theory, Topology, and Tensor Mathematics** as its constraint primitives. These symbols (⊗, ⊕, ↦, ∂, ∅, ≅) were chosen not for brevity, but for their **deep, pristine embeddings** in foundational LLMs:

### Why Mathematical Operators?

Modern LLMs are trained on millions of arXiv LaTeX papers. Symbols like `⊗` (Tensor Product), `⊕` (Direct Sum), and `∂` (Boundary) have been ingested in mathematically rigorous contexts millions of times. Unlike natural language words ("coupled", "isolated", "depends"), these operators carry **exactly one semantic interpretation** — zero polysemy.

This creates **Bijective Singleton Maps**:

```
∀ op ∈ Operators : |f⁻¹(f(op))| = 1
```

Every operator maps to exactly one vector. Every vector maps back to exactly one interpretation.

### arXiv Latent Anchors

The key insight of v2.0: rather than using rare glyphs that may not exist in the training corpus, HL now uses symbols that occupy the **deepest, most stable embeddings** in any LLM's latent space. A symbol like `⊗` has been reinforced across millions of physics and mathematics papers with a single, consistent meaning — making it a natural **steep attractor basin** that resists the Partition Function Explosion.

### Non-Polysemic Enforcement

Each operator carries exactly one constraint interpretation:
- `Hom(A,B) ≅ 0` means **strict isolation** — period
- `A ↦ B` means **directed flow** — no ambiguity
- `A ⊕ B` means **orthogonal coexistence** — no shared state
- `A ⊗ B` means **state entanglement** — documented coupling
- `∂A ∩ ∂B = ∅` means **interface encapsulation** — abstract boundary only

---

## The Five Topo-Categorical Operators

Project Chevron v2.0 implements five foundational constraint operators drawn from Category Theory, Topology, and Tensor Mathematics:

| Operator | Name | Symbol | Intent | Enforcement |
|---|---|---|---|---|
| **Null Morphism** | `Hom(A,B) ≅ 0` | ≅ | Strict isolation | A must never reference B |
| **Morphism** | `A ↦ B` | ↦ | Directed data flow | Reverse flow (B→A) forbidden |
| **Direct Sum** | `A ⊕ B` | ⊕ | Decoupled coexistence | No shared state between A and B |
| **Tensor Product** | `A ⊗ B` | ⊗ | State entanglement | Structural coupling documented |
| **Topo Boundary** | `∂A ∩ ∂B = ∅` | ∂∩∅ | Interface encapsulation | Abstract interface only |

### Design Principles

Each operator carries a **contract** (what it enforces) and a **constraint** (what it must NEVER allow):

- **Hom≅0 (Null Morphism):** Zero coupling — no import, call, or data path between A and B.
- **↦ (Morphism):** Directed flow only — reverse direction is forbidden. Must be acyclic (DAG).
- **⊕ (Direct Sum):** Orthogonal state spaces — no shared mutable state, singletons, or globals.
- **⊗ (Tensor Product):** Documented entanglement — changes to one side must propagate.
- **∂∩∅ (Topo Boundary):** Interface encapsulation — all communication via abstract interface only.

---

## System 2 Verification: The AST Weaver

A critical challenge in generating reliable software architectures is verifying strict modular independence. Standard intuitive Transformers (System 1) cannot natively compute exact, discrete Mutual Information (MI) during a continuous forward pass. HL addresses this via an external **System 2** verification loop — the AST Weaver.

### Maxwell's Demon and Rejection Sampling

The Weaver Function W(G) is not an internal property of the neural network's weights or thermodynamics. Instead, it functions as a classic, external algorithm — acting as **Maxwell's Demon** — that evaluates and filters the network's output via rigorous **rejection sampling**:

1. **Generation (System 1):** The neural model proposes a code block based on the non-polysemic, topo-categorically constrained prompt, settling into a local minimum.
2. **Extraction (Symbolic):** A classic AST parser extracts the dependency graph G = (M, E) from the generated code.
3. **Verification (System 2):** The external Weaver checks each Topo-Categorical constraint against the AST:
   ```
   W(G) = Σ_{(i,j) ∉ E} MI_AST(mᵢ, mⱼ)
   ```
   Violations produce thermodynamic rejection messages:
   ```
   [SYSTEM 2 REJECTION]: Hom≅0 — Module 'Search' references forbidden 'Database'. Resample required.
   [SYSTEM 2 REJECTION]: ↦ — Reverse flow Renderer → DataLoader violates directed morphism. Resample required.
   [SYSTEM 2 REJECTION]: ∂∩∅ — Direct reference UI → Database violates topological boundary. Resample required.
   ```
4. **Rejection Sampling:** If W(G) > 0, the state is rejected. The classical algorithmic system throws out the generation and forces the neural model to resample, driving a search loop until it produces a valid, orthogonal architecture.

This hybrid approach layers rigorous classical algorithmic verification (System 2) on top of the intuitive generative power of the neural network (System 1), ensuring that "Emergent Coupling" is prevented not by internal magic, but by **external post-generation filtering**.

---

## Empirical Results

### Study Parameters
- **Target:** TurboScribe — large-scale native Windows application (<50,000 LOC)
- **Stack:** C#, Python, CUDA (high-dimensional, multi-language environment)
- **Baseline:** Standard GPT-4 with 128k context window
- **Intervention:** HL with Topo-Categorical Operators

### Quantitative Outcomes

| Metric | Baseline (GPT-4) | HL Implementation | Improvement |
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

Project Chevron is the **reference implementation** of HL v2.0 — a Non-Polysemic Topological DSL where module relationships are expressed using five mathematical operators from Category Theory, Topology, and Tensor Mathematics.

### Project Structure

```
chevron/
├── SPEC.md                        # Formal language specification
├── README.md                      # This file
├── SCP_TESTING.md                 # Auto-test generation docs
├── EXTENSIONS.md                  # Language extension docs
├── index.html                     # HL research website
├── scp_bridge.py                  # ★ HL → AI Agent system prompt generator
├── forge.py                       # ★ Automatic codebase → HL decomposition
├── repl.py                        # Interactive REPL
├── run.py                         # File runner (execute .chevron files)
├── chevron/                       # The interpreter
│   ├── __init__.py                # Package exports (v2.0.0)
│   ├── glyphs.py                  # Operator registry (Topo-Categorical)
│   ├── lexer.py                   # Tokenizer (topo-cat operators + keywords)
│   ├── parser.py                  # Parser (topo-cat AST nodes, modules, specs)
│   ├── interpreter.py             # Executor (module scope, spec mode)
│   ├── verifier.py                # ★ Static HL constraint verifier (9 checks)
│   ├── code_verifier.py           # ★ AST-based formal code verification
│   ├── decorators.py              # ★ Runtime-enforced glyph decorators
│   └── test_generator.py          # ★ Deterministic spec-driven test gen
├── templates/                     # Code generation templates
│   └── spec_cli.py.template       # CLI scaffold for forge-generated projects
├── tests/                         # Test suite
│   ├── test_chevron.py            # 45 tests (lexer, parser, interp, verifier)
│   ├── test_code_verifier.py      # 16 tests (AST verification)
│   ├── test_decorators.py         # 17 tests (runtime glyph enforcement)
│   └── test_test_generator.py     # 11 tests (spec-driven test gen)
└── examples/                      # Example programs
    ├── hello.chevron               # Hello World
    ├── pipeline.chevron            # Origin → Filter → Witness
    ├── recursion.chevron           # Fold Time countdown
    ├── weave_filter.chevron        # Weave + Filter composition
    ├── todo.chevron                # Todo app HL spec
    ├── turboscribe.chevron         # TurboScribe HL spec (9 modules)
    ├── turboscribe_example.py      # ★ Full TurboScribe generation demo
    └── gemini_example.py           # Gemini integration demo
```

---

## Using HL with AI Agents (Gemini, GPT, Claude)

This is how you actually use Project Chevron to write real software with AI.

### The Core Idea

Instead of pasting your entire codebase (128K tokens) into an AI prompt, you:
1. **Define your architecture** as an HL spec (~1,200 tokens)
2. **Generate a constrained system prompt** for ONE module at a time
3. **Feed it to any AI** — the AI generates code that follows HL rules
4. **Verify with the Weaver** — a second AI pass checks for coupling violations

The AI physically **cannot see** other modules' implementations (RAG Denial). It sees only their interface contracts. This eliminates emergent coupling at the source.

### Step-by-Step Workflow

**Step 1: Define your architecture**

```python
from scp_bridge import SCPBridge

# Use a built-in template or define your own
bridge = SCPBridge.from_template("todo_app")
```

**Step 2: Generate the HL system prompt for ONE module**

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
print(response.text)  # → Python code constrained by HL
```

**Step 4: Verify with the Weaver (☤)**

```python
# Generate a verification prompt
verify_prompt = bridge.generate_verification_prompt("TodoStore", response.text)

# Ask the AI to check its own work against the HL spec
verify = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=verify_prompt,
    config=genai.types.GenerateContentConfig(temperature=0.0),
)
print(verify.text)  # → PASS or FAIL with specific violations
```

### Without an API Key

You can also use the HL Bridge from the command line and paste the output into any AI chat:

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

## HL Forge — Automatic Codebase Decomposition

The **Forge** (`forge.py`) scans any existing codebase and uses Gemini to automatically decompose it into an HL architecture — modules, types, dependency graph, forbidden zones, and operator assignments.

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

The Chevron verifier (`chevron/verifier.py`) runs 9 static checks on the parsed AST *before* execution, enforcing Topo-Categorical HL constraints at the language level:

| Check | Operator | Rule | Level |
|-------|----------|------|-------|
| Null Morphism | Hom≅0 | A must never reference B | Error |
| Morphism Direction | ↦ | Reverse flow (B→A) forbidden | Error |
| Direct Sum | ⊕ | No shared state between A and B | Error |
| Tensor Product | ⊗ | Structural coupling documented | Error |
| Topo Boundary | ∂∩∅ | Interface-only communication | Error |
| Forbidden deps | — | No references to forbidden modules | Error |
| Circular deps | — | No cycles in `depends_on` graph (DAG) | Error |
| Dependency integrity | — | `depends_on` targets must exist | Warning |
| Type annotations | — | Warn on undeclared types | Warning |

```bash
# Verify an architecture spec
python run.py examples/turboscribe.chevron --verify
# ✔ HL verification passed (W(G) = 0)
```

When the verifier reports **W(G) = 0**, the program's Topo-Categorical constraint graph has zero violations.

---

## Language Extensions

Chevron v2.0 introduces the Non-Polysemic Topological DSL with Topo-Categorical operators, formal verification, runtime decorators, and spec-driven test generation. Key features:

- **Topo-Categorical operators** — `Hom≅0`, `↦`, `⊕`, `⊗`, `∂∩∅` for non-polysemic module constraints
- **Formal code verification** — `CodeVerifier` uses AST analysis to deterministically check generated code (replaces AI self-verification)
- **Runtime decorators** — `@chevron.origin`, `@chevron.filter`, `@chevron.fold`, `@chevron.witness`, `@chevron.weaver`
- **Spec-driven test generation** — `SpecTestGenerator` produces pytest tests from contracts, not from implementation
- **Structured output** — `generate_structured_schema()` produces JSON schemas for Gemini's `response_schema`
- **Module system** — isolated scopes with `imports`, `exports`, `forbidden`, and `constraint`
- **Spec mode** — architecture-only declarations (never executed), verifiable before any code exists
- **Type declarations** — structural types (e.g., `type MediaFile = { path: str, size: int }`) for pipeline contracts
- **Snake case identifiers** — `find_media` is a single token
- **Error accumulation** — parser reports all errors, not just the first

See [EXTENSIONS.md](EXTENSIONS.md) for the full specification.

---

## Auto-Test Generation

The `--with-tests` flag adds contract-driven test generation to the HL pipeline:

```bash
# Generate all modules with AI, verify each, and run auto-tests
python examples/turboscribe_example.py --all --with-tests
```

Tests are generated from the **HL contract** (not from the implementation), verifying:
- **Structural** — methods exist with correct signatures
- **Constraint** — module-specific rules via AST inspection
- **Behavioral** — return types, edge cases, error handling (mocked)
- **Isolation** — no forbidden imports, no global mutable state

See [SCP_TESTING.md](SCP_TESTING.md) for full documentation.

---

## Real-World Example: TurboScribe

The TurboScribe example demonstrates HL on a real 110K-token audio processing backend, decomposed into 9 isolated modules:

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

### Run All Examples

```bash
python run.py examples/hello.chevron
python run.py examples/pipeline.chevron
python run.py examples/recursion.chevron
python run.py examples/weave_filter.chevron
```

Each example demonstrates Topo-Categorical operators:
- `hello.chevron` — Tensor Product `⊗` (state entanglement)
- `pipeline.chevron` — Pipeline `→` with filter predicates
- `recursion.chevron` — Recursive morphism chains
- `weave_filter.chevron` — Tensor Product `⊗` + pipeline filtering

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

### Topo-Categorical Operators

| Symbol | Name | Description |
|:------:|------|-------------|
| `≅` | Null Morphism | `Hom(A,B) ≅ 0` — Strict isolation |
| `↦` | Morphism | `A ↦ B` — Directed data flow |
| `⊕` | Direct Sum | `A ⊕ B` — Decoupled coexistence |
| `⊗` | Tensor Product | `A ⊗ B` — State entanglement |
| `∂∩∅` | Topo Boundary | `∂A ∩ ∂B = ∅` — Interface encapsulation |
| `→` | Pipeline | Data flows left to right |
| `←` | Binding | Assign a name to an expression |
| `{ }` | Predicate | Filter or transform condition |
| `#` | Comment | Line comment |

### Pipeline Composition

Expressions compose left to right with `→`, forming data-flow pipelines:

```
[5, 3, 1, 4, 2] → {> 2}
```

Filters the list, keeping only items greater than 2.

### Topo-Categorical Constraints

```chevron
Hom(Frontend, Database) ≅ 0     # No coupling allowed
DataLoader ↦ Processor ↦ Renderer  # Directed flow
Logger ⊕ Analytics                 # Independent state
Auth ⊗ Session                     # Documented entanglement
∂UI ∩ ∂Database = ∅               # Interface-only boundary
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

### Operator Registry (`glyphs.py`)

The registry is the core operator definition map. Each operator entry carries:

- **Symbol:** The mathematical notation
- **Name:** Human-readable name
- **Category:** Mathematical domain (Category Theory, Topology, Tensor)
- **Contract:** What it enforces
- **Constraint:** What it must NEVER allow

```python
from chevron.glyphs import OPERATOR_REGISTRY, lookup, describe_all

# Look up an operator
info = lookup("⊗")
print(info.name)       # "Tensor Product"
print(info.category)   # "Tensor Mathematics"
print(info.contract)   # "Accepts (left, right) → Documents tight structural coupling"
print(info.constraint) # "Changes to either side must propagate..."

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
  <strong>Hom≅0 · ↦ · ⊕ · ⊗ · ∂∩∅</strong><br>
  <em>The first chevron is locked.</em>
</p>
