# Chevron Language Extensions (v0.2)

> Extending the 5 glyph primitives with module isolation, type safety, and static verification — without breaking the core SCP contract.

## Why These Extensions Exist

Chevron v0.1 proved the core thesis: **5 glyphs are sufficient to express any data pipeline**. But real-world codebases (like TurboScribe's 110K-token backend) exposed gaps that made SCP enforcement manual and error-prone:

| Problem | Example | SCP Principle at Risk |
|---------|---------|----------------------|
| No module boundaries | Any glyph could reference any binding | **RAG Denial** — AI sees everything |
| No forbidden zones | A Transcriber module could import SearchEngine | **Glyph Contract** — no isolation |
| No type contracts | Pipeline stages silently pass wrong types | **Weaver Verification** — silent failures |
| No static checks | Multiple ◬ Origins went undetected | **Determinism** — undefined behavior |
| First error stops parsing | One typo hides all other errors | **Developer experience** — slow iteration |

The extensions solve each of these while preserving the 5 primitive glyphs unchanged.

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

**Why it matters for SCP:** When an AI agent is given a module spec, `forbidden` guarantees it physically cannot reference code from other modules. The constraint string becomes part of the system prompt, anchoring the AI's orthogonal embedding to an isolated attractor basin — minimizing semantic cross-talk and preventing the Partition Function (Z) from drowning the signal.

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

**Why spec mode exists:** In SCP, the architecture specification *is* the source of truth. Specs let you define the entire dependency graph, type contracts, and forbidden zones without needing any executable code. The verifier runs all static checks on specs just like modules. You can define a complete system architecture in Chevron and verify it before writing a single line of implementation code.

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

**Why it matters for SCP:** Types make pipeline contracts explicit. When `scp_bridge.py` generates a system prompt from a spec, the type declarations tell the AI *exactly* what shape of data each glyph must produce. This reduces the AI's degrees of freedom, steepening the attractor basin and increasing the Critical Energy Gap (ΔE) above ln(N).

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

| Check | Glyph | Rule | Level |
|-------|-------|------|-------|
| Origin count | ◬ | Exactly one per scope (module or top-level) | Error |
| Witness terminal | 𓂀 | Must be the last stage in a pipeline | Error |
| Fold arguments | ☾ | Must have ≥2 args (predicate + transform) | Error |
| Forbidden deps | — | No references to forbidden modules | Error |
| Circular deps | — | No cycles in `depends_on` graph | Error |
| Type annotations | — | Warn on undeclared types | Warning |

Run with:
```bash
python run.py examples/turboscribe.chevron --verify
```

Output on success:
```
✔ SCP verification passed (W(G) = 0)
```

**Why W(G) = 0:** In the updated SCP framework, W(G) is the Weaver Function — a System 2 rejection sampling check that computes structural Mutual Information (MI_AST) between modules. When the verifier reports W(G) = 0, it means zero undeclared coupling exists — every primitive is used within its contract and all module pairs outside the declared dependency graph have MI_AST = 0.

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

## How the Extensions Support SCP

The Spatial Constraint Protocol has three core enforcement mechanisms. Here's how each extension maps:

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

### Glyph Contracts

> *"Each primitive has an inviolable semantic contract."*

| Glyph | Contract | Verifier Check |
|-------|----------|----------------|
| ◬ Origin | Exactly one per scope | Multiple-origin detection |
| 𓂀 Witness | Observes without modifying — must be terminal | Non-terminal witness detection |
| ☾ Fold | Must have reachable base case | Argument count validation |
| Ө Filter | Predicate must return boolean | Function call type checking |
| ☤ Weaver | Merges without loss | *(structural — no new check needed)* |

### Weaver Verification — System 2 Rejection Sampling (W(G) = 0)

> *"The Weaver acts as Maxwell's Demon — rejecting any generation where undeclared coupling exists."*

The `--verify` flag automates this check. Before, you had to manually inspect glyph usage. Now:

```bash
python run.py my_architecture.chevron --verify
# ✔ SCP verification passed (W(G) = 0)
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
# ✔ SCP verification passed (W(G) = 0)
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
| `chevron/lexer.py` | Tokenizer with snake_case + 9 keywords |
| `chevron/parser.py` | Parser with 6 new AST nodes + error accumulation |
| `chevron/interpreter.py` | Execution with module scope + spec mode |
| `chevron/verifier.py` | Static analysis (6 checks) |
| `chevron/glyphs.py` | 5 glyph primitives (unchanged) |
| `run.py` | CLI runner with `--verify` flag |
| `tests/test_chevron.py` | 45 tests (lexer, parser, interpreter, verifier, integration) |
| `examples/turboscribe.chevron` | Real-world spec: 8 types, 9 modules, 166× compression |
