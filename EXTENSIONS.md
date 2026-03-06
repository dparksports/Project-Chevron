# Chevron Language Extensions (v2.0)

> Extending the core Topo-Categorical operators with module isolation, type safety, formal verification, runtime decorators, and spec-driven test generation.

## Why These Extensions Exist

Chevron v1.0 proved the core thesis: **orthogonal embeddings resist the Partition Function Explosion**. But real-world codebases (like TurboScribe's 110K-token backend) exposed gaps that made HL enforcement manual and error-prone:

| Problem | Example | HL Principle at Risk |
|---------|---------|----------------------|
| No module boundaries | Any glyph could reference any binding | **RAG Denial** — AI sees everything |
| No forbidden zones | A Transcriber module could import SearchEngine | **Glyph Contract** — no isolation |
| No type contracts | Pipeline stages silently pass wrong types | **Weaver Verification** — silent failures |
| No static checks | Multiple ◬ Origins went undetected | **Determinism** — undefined behavior |
| First error stops parsing | One typo hides all other errors | **Developer experience** — slow iteration |

The extensions solve each of these while building on the 5 Topo-Categorical operators (`Hom≅0`, `↦`, `⊕`, `⊗`, `∂∩∅`).

---

## The Extensions

### 1. Snake Case Identifiers

**Before:** `find_media` tokenized as `find` + `_` (placeholder) + `media`.
**After:** `find_media` is a single identifier.

```
◬ "directory" → find_media → 𓂀
```

This seems trivial, but it unblocked real-world naming conventions used in Python codebases being specified in Chevron.

---

### 2. Module System

Modules create **isolated scopes** with explicit contracts — the language-level enforcement of RAG Denial.

```
module AudioIngest
    imports DeviceConfig
    exports find_media, load_audio
    forbidden [Transcriber, SearchEngine]
    constraint "No ML imports — raw I/O only"

    ◬ "directory" → find_media → 𓂀
end
```

**Key behaviors:**
- **`exports`** — only listed bindings leak to the outer scope. Everything else is private.
- **`imports`** — explicitly declares what this module can see from outside.
- **`forbidden`** — hard error if any identifier from these modules appears in the body. This is RAG Denial enforced by the parser, not by convention.
- **`constraint`** — human-readable rules that translate directly into system prompts when used with `scp_bridge.py`.

**Why it matters for HL:** When an AI agent is given a module spec, `forbidden` guarantees it physically cannot reference code from other modules. The constraint string becomes part of the system prompt, anchoring the AI's orthogonal embedding to an isolated attractor basin — minimizing semantic cross-talk and preventing the Partition Function (Z) from drowning the signal.

---

### 3. Spec Mode

Specs look like modules but are **never executed** — they're pure architecture declarations.

```
spec VoiceDetector
    depends_on [AudioIngest]
    imports AudioIngest
    exports run_vad_scan, batch_vad_scan
    forbidden [Transcriber, SearchEngine, Analyzer]
    constraint "No transcription — detection only"
end
```

**Why spec mode exists:** In HL, the architecture specification *is* the source of truth. Specs let you define the entire dependency graph, type contracts, and forbidden zones without needing any executable code. The verifier runs all static checks on specs just like modules. You can define a complete system architecture in Chevron and verify it before writing a single line of implementation code.

**How to use:** `python run.py architecture.chevron --verify` validates the entire spec graph (circular deps, forbidden zones, type annotations) without executing anything.

---

### 4. Type Declarations

Structural types define the contracts between pipeline stages:

```
type MediaFile = { path: str, size: int, ext: str }
type AudioTensor = { data: str, sample_rate: int, device: str }
type Transcript = { text: str, segments: str, model: str }
```

When used as pipeline annotations:

```
◬ "directory" → MediaFile → AudioTensor → Transcript → 𓂀
```

**Execution behavior:** Type annotations are pass-through — they don't affect runtime. But the verifier checks that every `TypeAnnotNode` in a pipeline references a declared type, warning on undeclared types.

**Why it matters for HL:** Types make pipeline contracts explicit. When `scp_bridge.py` generates a system prompt from a spec, the type declarations tell the AI *exactly* what shape of data each operator must produce. This reduces the AI's degrees of freedom, steepening the attractor basin and increasing the Critical Energy Gap (ΔE) above ln(N).

---

### 5. Function Calls in Predicates

Predicates can now contain named function calls with arguments:

```
◬ [1, 2, 3, 4, 5] → Ө {is_prime 2} → 𓂀
```

This extends the Filter (Ө) and Fold (☾) glyphs to support domain-specific logic without breaking their contracts. The function is looked up in the environment, and the predicate still returns true/false — the glyph's semantics don't change.

---

### 6. Error Accumulation

The parser now collects all errors instead of stopping at the first one:

```
◬ ]        # Error 1: unexpected ]
◬ }        # Error 2: unexpected }
x ← ◬ ]   # Error 3
```

Output:
```
3 parse error(s):
  L1:3 — Unexpected token RBRACKET (']')
  L2:3 — Unexpected token RBRACE ('}')
  L3:7 — Unexpected token RBRACKET (']')
```

---

### 7. Static Verifier (`--verify`)

The verifier runs 6 checks on the parsed AST before execution:

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

Run with:
```bash
python run.py examples/turboscribe.chevron --verify
```

Output on success:
```
✔ HL verification passed (W(G) = 0)
```

**Why W(G) = 0:** In the HL v2.0 framework, W(G) is the Weaver Function — a System 2 rejection sampling check that computes structural Mutual Information (MI_AST) between modules. When the verifier reports W(G) = 0, it means zero undeclared coupling exists — all Topo-Categorical constraints are satisfied and all module pairs outside the declared dependency graph have MI_AST = 0.

---

### 8. Dependency Graph Validation

The verifier builds a dependency graph from `depends_on` declarations and checks for cycles:

```
spec A
    depends_on [B]
end

spec B
    depends_on [A]    # ✘ Circular!
end
```

```
✘ L1:1 [CYCLE] Circular dependency detected: A → B → A
```

It also validates that dependencies reference modules/specs that actually exist in the program.

---

## How the Extensions Support HL

The Holographic Language has three core enforcement mechanisms. Here's how each extension maps:

### RAG Denial

> *"Each module sees only its own spec, never the full codebase."*

| Extension | Enforcement |
|-----------|-------------|
| `module` blocks | Bindings are scoped — private by default |
| `exports` | Only declared bindings cross the boundary |
| `imports` | Module explicitly declares what it can see |
| `forbidden` | **Hard error** on any reference to a banned module |
| `spec` mode | Architecture can be verified without execution |

Before these extensions, RAG Denial was enforced only by `scp_bridge.py` at prompt generation time. Now the Chevron language itself prevents violations at parse time.

### Topo-Categorical Contracts

> *"Each operator has an inviolable semantic contract."*

| Operator | Contract | Verifier Check |
|----------|----------|----------------|
| Hom≅0 Null Morphism | Zero coupling between A and B | Reference detection in module bodies |
| ↦ Morphism | Directed flow only | Reverse-flow reference detection |
| ⊕ Direct Sum | Orthogonal state spaces | Shared-state detection |
| ⊗ Tensor Product | Documented entanglement | Coupling documentation |
| ∂∩∅ Topo Boundary | Interface-only communication | Direct reference detection |

### Weaver Verification — System 2 Rejection Sampling (W(G) = 0)

> *"The Weaver acts as Maxwell's Demon — rejecting any generation where undeclared coupling exists."*

The `--verify` flag automates this check. Before, you had to manually inspect glyph usage. Now:

```bash
python run.py my_architecture.chevron --verify
# ✔ HL verification passed (W(G) = 0)
```

If any contract is violated, the verifier reports exactly which glyph at which location, with an explanation of why the contract was broken.

---

## Quick Start

### Define an architecture

```chevron
# types.chevron — define your data contracts
type Request = { url: str, method: str }
type Response = { status: int, body: str }

spec HttpClient
    exports send_request
    forbidden [Database, FileSystem]
    constraint "No side effects beyond HTTP"
end

spec Database
    exports query, insert
    forbidden [HttpClient]
    constraint "No network access"
end
```

### Verify it

```bash
python run.py types.chevron --verify
# ✔ HL verification passed (W(G) = 0)
```

### Run tests

```bash
python -m unittest tests.test_chevron -v
# Ran 45 tests in 0.004s — OK
```

---

## File Reference

| File | Purpose |
|------|---------|
| `chevron/lexer.py` | Tokenizer with topo-cat operators + keywords |
| `chevron/parser.py` | Parser with topo-cat AST nodes + error accumulation |
| `chevron/interpreter.py` | Execution with module scope + spec mode |
| `chevron/verifier.py` | Static analysis (9 checks) |
| `chevron/glyphs.py` | 5 Topo-Categorical operators |
| `run.py` | CLI runner with `--verify` flag |
| `tests/test_chevron.py` | 45 tests (lexer, parser, interpreter, verifier, integration) |
| `examples/turboscribe.chevron` | Real-world spec: 8 types, 9 modules, 166× compression |
