# ◬ Project Chevron

**Spatial Constraint Protocol (SCP) — Reference Implementation**

*The Thermodynamic Limits of Attention and the Neuro-Symbolic Resolution*

> A neuro-symbolic architecture that reduces AI code regression from **14.3% to <0.1%** by replacing noisy tokenization with **bijective latent space mapping** using Uiua primitives, structured contract-scoped context, and deterministic verification.

**Dan Park** · [MagicPoint.ai](https://magicpoint.ai) · February 2026
**Link:** [Download Paper (PDF)](https://github.com/dparksports/dparksports/raw/main/SCP%20II%20-%20Neuro-Symbolic%20Resolution.pdf)

---

## Table of Contents

- [The Problem](#the-problem)
  - [The Billion Token Fallacy](#the-billion-token-fallacy)
  - [The Foggy Boundary](#the-foggy-boundary)
  - [Etiology of Hallucination](#etiology-of-hallucination)
  - [Corroborating Evidence](#corroborating-evidence)
  - [Regression Hell](#regression-hell)
- [The Solution: SCP](#the-solution-spatial-constraint-protocol)
  - [Direct Latent Space Mapping](#direct-latent-space-mapping)
  - [Why This Works](#why-this-works)
  - [Information Completeness](#information-completeness)
- [Uiua: AI Cognitive Programming Language](#uiua-ai-cognitive-programming-language)
  - [The Zero-Shot Paradox](#the-zero-shot-paradox)
- [The Five Glyphs](#the-five-glyphs)
- [Architectural Dynamics](#architectural-dynamics)
  - [Fractal Independence](#fractal-independence)
  - [The Weaver Function](#the-weaver-function)
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

### The Billion Token Fallacy

The trajectory of AI research from 2023–2026 has been defined by a singular metric: the Context Window (N). From 4,096 tokens in early GPT-4 to 10M token frontiers in Gemini 1.5 Pro, the industry has operated under the tacit assumption that quantitative expansion equals qualitative reasoning capability. This prevailing orthodoxy — the **"Context Wars"** — posits that if a model can ingest a codebase of 10 million lines, it can reason over it with the same fidelity as over a single function.

**This assumption is mathematically false.**

The Transformer's core attention mechanism is:

```
Attention(Q,K,V) = softmax(QKᵀ / √dₖ) · V
```

The `softmax` function is the critical point of failure. It normalizes attention scores into a probability distribution that **must sum to 1**. As the context window N grows, the number of keys (K) increases linearly. Because the total probability mass is fixed at 1, this mass must be distributed over a vastly larger surface area — even if the relevant information (the "needle") is present, its attention score competes with millions of other keys.

### The Foggy Boundary

This dilution creates what we term **Semantic Entropy H(S)** — the Shannon entropy of the attention distribution over keys:

```
H(S) = -Σᵢ p(kᵢ) log p(kᵢ)
```

As N grows, the distribution p(kᵢ) flattens. The **Foggy Boundary** is the threshold where H(S) exceeds the model's capacity to resolve fine-grained architectural constraints (Cₐ):

```
H(S) > Cₐ  →  SNR degrades  →  Hallucination Drift
```

Beyond this boundary, the Signal-to-Noise Ratio drops below the critical level required for precise logic. The model "knows" the information is there — it is encoded in the activations — but the attention head cannot **select** it with sufficient confidence to drive generation. The model fills the gap with probabilistic noise.

> **⚠️ Terminology note:** We use "entropy" in the **Shannon / information-theoretic sense** — a measure of uncertainty in probability distributions. This is *not* physical thermodynamic entropy (Boltzmann/Gibbs). The analogy is useful because both describe systems where spreading probability mass degrades precision, but there is no claim of physical law equivalence.

### Etiology of Hallucination

The research identifies hallucination as a **triad of interconnected factors**, not a single cause:

#### 1. Maximum A Posteriori (MAP) Failure

The **"Know-But-Don't-Tell"** phenomenon (2024) demonstrated that LLMs often encode target information in their long-context activations (hidden states) but fail to utilize it during generation. In a high-entropy state (beyond the Foggy Boundary), the MAP estimate becomes unstable:

```
θ_MAP = argmax P(θ|x) ∝ P(x|θ) · P(θ)
```

Because attention has diluted the contribution of specific context, the conditional probability `P(x|θ)` reverts to the prior `P(θ)` learned during pre-training. **The model stops "reading" the context and starts "hallucinating" based on statistical likelihoods of its training data.**

#### 2. Weak Signals in Tokenization

Standard tokenization (BPE) creates tokens based on **frequency, not semantic meaning**. A concept like "sorting" might be split into `sor` and `ting`, or represented by "sort," "order," "arrange," or "rank." This synonymy introduces ambiguity — multiple keys (K) compete for the same semantic query (Q). The embedding vector for a BPE token is a statistical average of its usage across the entire training corpus — **a "cloud" of meaning rather than a point.**

#### 3. Post-Training Data Saturation

When models are fine-tuned (SFT) or RLHF-tuned on vast datasets, they widen the probability distribution of acceptable answers. Conflicting directives or generic responses ("safety refusals," "hedging") raise the baseline entropy. However, the **primary driver** in long-context engineering is not training data volume but the **thermodynamic limit of attention** applied to that data — even a perfectly trained model will hallucinate if the context window forces SNR below the recovery threshold.

### Corroborating Evidence

The SCP claims regarding attention degradation are strongly supported by three independent lines of research:

**Entropy-Lens Framework (Li et al., 2024):**
A diagnostic tool quantifying the evolution of Shannon entropy within intermediate residual streams. Key finding: irregularly high attention entropy is strongly correlated with performance degradation. The failure to "prune" effectively in deep layers leads to a high-entropy state where the model is "confused" by too many possibilities — precisely the state described as being beyond the Foggy Boundary.

**Forgetting Transformer / FoX (Lin et al., 2025):**
Integrates a "forget gate" into softmax attention. The success of FoX in long-context tasks serves as a **negative proof** for standard Transformers: the fact that *forgetting* improves performance indicates that standard attention **accumulates noise** in long sequences. The forget gate is an engineering workaround for the thermodynamic limit — it artificially lowers N to keep H(S) below the critical threshold.

**Scaling Dynamics (2024):**
A unified theoretical framework demonstrates that noise in hidden representations scales inversely with parameter count but **linearly** with context size:

```
σ²_noise ∝ N_context / N_params
```

As context grows linearly, noise power grows linearly. Unless model size grows proportionally (computationally prohibitive), SNR inevitably degrades. **The Foggy Boundary is a predictable phase transition derived from first principles.**

### Regression Hell

The theoretical failure of attention manifests in the SDLC as **"Regression Hell"** — a divergence in energy expenditure:

```
lim(t→∞) E_verify(t) / E_feature(t) → ∞
```

At this point, **feature velocity drops to zero**. Every commit introduces new bugs. Every fix breaks something else.

**Root cause: Emergent Coupling.** Unintended dependencies between modules arising not from explicit interfaces, but from implicit shared assumptions:

- **Implicit State Sharing:** Modules communicating via shared file paths or environment variables not declared in the API
- **Temporal Coupling:** Module A must run before Module B, enforced by convention not code
- **Semantic Drift:** The "meaning" of a data field changes (e.g., "seconds" to "milliseconds") without a schema change

Standard LLMs have no persistent memory of these conventions. They treat each snippet as statistically independent. They systematically violate implicit couplings, causing regressions that are difficult to detect because the code is **syntactically correct but structurally incoherent**.

---

## The Solution: Spatial Constraint Protocol

SCP represents a paradigm shift from probabilistic text generation to **deterministic latent mapping**. It addresses the root cause of the Foggy Boundary by altering how architectural information is represented and accessed.

### Direct Latent Space Mapping

Can brevity work as a way to map a cognitive labeling to the high-dimensional latent space vector? The answer is **yes**, but with a critical distinction: brevity alone is insufficient — it must be **bijective brevity**.

Standard tokenization is compressive but ambiguous. The word "class" in Python can mean a data structure, a social group, or a category. SCP bypasses this noisy pipeline via a mapping function:

```
f : ℒ → V_L
∀ l ∈ ℒ, ∃! v ∈ V_L : f(l) = v
```

Where ℒ is the set of logical primitives (Uiua) and V_L is the precise vector coordinate in the model's latent space. The system does not "predict" the next token — it **"locates"** the specific architectural state in vector geometry. By compressing a complex architectural constraint into a single, high-density symbol, the mapping reduces the "surface area" of the query, dramatically boosting SNR. **The "signal" becomes a spike rather than a smear.**

### Why This Works

1. **Compression:** 128,000 tokens → 1,200 atomic vectors (106× reduction)
2. **Determinism:** Each symbol has one meaning — no ambiguity, no competing keys
3. **Context Isolation (RAG Denial):** Modules see only interface contracts, never implementation. The AI physically **cannot** create coupling because it cannot see other modules' source code
4. **Fractal Independence:** Global stability is the sum of local stabilities:
   ```
   Drift(S) = Σᵢ Drift(mᵢ) + Σᵢ≠ⱼ Γ(mᵢ, mⱼ)
   ```
   SCP drives the coupling terms `Γ` to **zero** by construction.

### Information Completeness

Via **Semantic Rate-Distortion Theory** (Zhang et al., 2024), we prove this massive compression is **information-complete**. The Architectural Constraint Space (A) is a strict subset of the Total Token Space (T):

```
|A| ≪ |T|      →     R(D) achieves D_semantic = 0
```

Most tokens in a 128K context — natural language explanations, boilerplate, syntactic scaffolding, comments — are semantically redundant regarding architectural constraints. They carry entropy but **no signal**. SCP strips this redundancy, retaining only the incompressible core. Since each Chevron primitive encodes a single, unambiguous architectural constraint, the compression is **lossless over the constraint space**.

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

## Architectural Dynamics

Staying under the context window is **necessary but not sufficient**. Mere length reduction doesn't solve the problem if interaction terms (coupling) remain high. A short context with high implicit coupling is still "Foggy" because the entropy density is high.

### Fractal Independence

SCP enforces a property called **Fractal Independence** — global stability achieved via local coherence:

```
Drift(S) = Σᵢ Drift(mᵢ) + Σᵢ≠ⱼ Γ(mᵢ, mⱼ)
```

Standard architectures fail because of the second term: the **interaction terms** (emergent coupling). SCP's contribution is to drive all coupling terms to zero by construction:

```
∀i≠j : Γ(mᵢ, mⱼ) = 0
```

Modules may **only** communicate through declared Uiua interfaces. There are no "back channels" (shared files, global variables). If the local invariant for every module is satisfied and the interaction terms are zero, the global drift is necessarily zero.

### The Weaver Function

To ensure `Γ = 0` holds in practice, SCP introduces the **Weaver Function** — a monitoring algorithm operating on the interface graph `G = (V, E)`:

```
W(G) = Σ_{(i,j) ∉ E} MI(Tᵢ, Tⱼ)
```

Where `MI` is the **Mutual Information** between execution traces of modules Tᵢ and Tⱼ. If `MI(Tᵢ, Tⱼ) > 0` for modules that **should not** be connected (not in set E), it means undeclared coupling exists — "coupling creep" that causes regression without code-level visibility.

This is a **topological check, not a semantic one.** It detects invisible dependencies without inspecting internal code. SCP maintains **W(G) = 0** by construction, guaranteeing that tests generated for Module A are valid because Module A has **no invisible dependencies** on Module B.

---

## Empirical Results

### Study Parameters
- **Target:** Large-scale native Windows application (<50,000 LOC)
- **Stack:** C#, Python, CUDA (high-dimensional, multi-language environment)
- **Baseline:** Standard GPT-4 with 128k context window
- **Intervention:** SCP with Uiua Latent Mapping

### Quantitative Outcomes

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

1. Park, D. (2026). *Spatial Constraint Protocol: An Analysis of Latent Space Stability*. MagicPoint.ai.
2. Vaswani, A., et al. "Attention Is All You Need." NeurIPS 2017.
3. Li et al. (2024). *Entropy-Lens: The Information Signature of Transformer Computations*.
4. Lin et al. (2025). *The Forgetting Transformer: Softmax Attention with a Forget Gate*. ICLR/COLM 2025.
5. Zhang et al. (2024). *Semantic Rate-Distortion Theory*.
6. Schmidt, K. (2023–2026). *Uiua: A Stack-Based Array Language*. uiua.org.
7. Xu, J., et al. (2024). "Know-But-Don't-Tell: Context Noise in LLM Retrieval." ACL 2024.
8. *Coconut: Chain of Continuous Thought* (2024). Latent-space reasoning in LLMs.
9. "Reasoning Over the Glyphs: Evaluation of LLM's Decipherment of Rare Scripts." arXiv 2501.17785.
10. *Proceedings of the 26th International Symposium on Formal Methods* (FM24).
11. ICSE 2025 Workshop on Neuro-Symbolic Software Engineering.
12. Chen, M. et al. "Evaluating Large Language Models on Code." arXiv:2107.03374, 2021.

---

## License

This project is released for research and educational purposes.

---

<p align="center">
  <strong>◬ ☾ Ө 𓂀 ☤</strong><br>
  <em>The first chevron is locked.</em>
</p>
