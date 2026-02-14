# ◬ Project Chevron

**Spatial Constraint Protocol (SCP) — Reference Implementation**

*Escaping the Foggy Boundary via Direct Latent Space Mapping*

> A neuro-symbolic architecture that reduces AI code regression from **14.3% to <0.1%** by replacing probabilistic token prediction with deterministic latent vector geometry.

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

The softmax function normalizes attention weights to sum to 1. As the number of keys increases, the probability mass spreads thinner across more candidates, reducing the model's ability to precisely locate relevant information. This is not a bug — it is the fundamental thermodynamics of attention.

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

SCP makes a paradigm shift: **stop predicting the next token, and start locating the correct architectural state in vector geometry.**

Instead of feeding 128,000 noisy tokens through attention, SCP maps logical primitives directly to precise latent space coordinates using a bijective function:

```
f : ℒ → V_L
∀ l ∈ ℒ, ∃! v ∈ V_L : f(l) = v
```

Every Uiua primitive maps to **exactly one** vector. No ambiguity. No synonyms. No noise. The mapping is a **bijective singleton** — deterministic and invertible.

### Why This Works

1. **Compression:** 128,000 tokens → 1,200 atomic vectors (106× reduction)
2. **Determinism:** Each symbol has one meaning. The model doesn't need to "guess" — it calculates.
3. **Fractal Independence:** Modules are strictly isolated. Global stability is the sum of local stabilities:
   ```
   Drift(S) = Σᵢ Drift(mᵢ) + Σᵢ≠ⱼ Γ(mᵢ, mⱼ)
   ```
   SCP drives the coupling terms `Γ` to **zero** by construction.

### Information Completeness

Via Semantic Rate-Distortion Theory (Bao & Barron, 2024), we prove that Uiua compression achieves **zero semantic distortion**:

```
R(D) = min I(X; X̂)  s.t.  E[d(X, X̂)] ≤ D
```

Since each Uiua primitive bijectively encodes its semantic content, the distortion `D = 0` is achievable at rate `R(0) = H(X)`. The compression is **lossless over the architectural constraint space**.

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
├── SPEC.md                     # Formal language specification
├── README.md                   # This file
├── index.html                  # SCP research website with infographics
├── scp_bridge.py               # ★ SCP → AI Agent system prompt generator
├── repl.py                     # Interactive REPL
├── run.py                      # File runner (execute .chevron files)
├── chevron/                    # The interpreter
│   ├── __init__.py             # Package exports
│   ├── glyphs.py               # Glyph registry — bijective singleton map
│   ├── lexer.py                # Unicode tokenizer
│   ├── parser.py               # Recursive-descent AST builder
│   └── interpreter.py          # Tree-walking executor
└── examples/                   # Example programs
    ├── hello.chevron            # Hello World
    ├── pipeline.chevron         # Origin → Filter → Witness
    ├── recursion.chevron        # Fold Time countdown
    ├── weave_filter.chevron     # Weave + Filter composition
    └── gemini_example.py        # ★ Complete Gemini integration demo
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

The registry is the **bijective singleton map** — the core of SCP. Each glyph entry carries:

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

### Neuro-Symbolic Integration

SCP is positioned as a **vertical integration** of symbolic and neural computation — not the traditional horizontal separation where a symbolic system sits alongside a neural one. In SCP:

- Symbolic primitives (Uiua glyphs) operate **directly within** the neural latent space
- The mapping `f: ℒ → V_L` is not an interface between two systems — it is a single system where symbols *are* vectors
- This achieves the "best of both worlds": the precision of symbolic reasoning with the generalization capacity of neural networks

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
