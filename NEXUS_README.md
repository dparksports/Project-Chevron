# ◬ Nexus — SCP-Powered AI IDE

**Next-Generation AI Development Environment powered by the Spatial Constraint Protocol**

*Escaping Context Window Entropy and Regression Hell via Entropy-Aware, Contract-Driven Architecture*

> A prototype AI IDE that reduces context window noise by **10–200×** and enforces code regression rates below **0.1%** — by replacing "dump everything into context" with SCP-scoped, one-module-at-a-time AI generation.

**Dan Park** · [MagicPoint.ai](https://magicpoint.ai) · February 2026
**Foundation:** [Project Chevron — SCP Reference Implementation](README.md) · [SCP Paper (PDF)](https://github.com/dparksports/dparksports/blob/main/spatial_constraint_protocol-draft-expanded.pdf)

---

## Table of Contents

- [Why Nexus Exists](#why-nexus-exists)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Programmatic API](#programmatic-api)
- [Building with Nexus — Samples](#building-with-nexus--samples)
- [AI Provider Setup](#ai-provider-setup)
- [Test Suite](#test-suite)
- [Project Structure](#project-structure)
- [Roadmap](#roadmap)

---

## Why Nexus Exists

### The Problem: Every AI IDE Hits a Wall

Current AI coding assistants (Copilot, Cursor, Windsurf, Aider) all share the same fatal flaw: **they degrade as conversations grow.** The longer you work with them, the worse they get.

```
Turn 1:   "Add a login form"       → Perfect code ✔
Turn 5:   "Now add logout"          → Breaks the auth state ✘
Turn 10:  "Fix the regression"      → Introduces 2 new bugs ✘✘
Turn 20:  "Just revert everything"  → Can't — too entangled ✘✘✘
```

This happens because of two forces:

**1. Context Window Entropy (The Foggy Boundary)**

Every AI has a context window. As you work, this window fills with conversation history, old code, stale errors, and accumulated noise. The AI's attention spreads thinner across more tokens:

```
H(S) > Cₐ  →  Signal-to-noise degrades  →  Hallucinations emerge
```

More context ≠ better results. The AI *sees* more but *understands* less.

**2. Regression Hell (Cascade Failures)**

When the AI edits Module A, it doesn't understand how Modules B, C, and D depend on A. So it introduces regressions — silent breakages that cascade through the codebase:

```
lim(t→∞)  E_verify(t) / E_feature(t)  →  ∞
```

Eventually you spend all your energy fixing regressions and zero energy building features.

### The Solution: SCP + Nexus

Project Chevron introduced the **Spatial Constraint Protocol (SCP)** — a formal system for constraining AI attention to prevent these problems. Nexus turns SCP theory into a working AI IDE:

| Problem | How Current IDEs Handle It | How Nexus Handles It |
|---------|---------------------------|---------------------|
| Context entropy | Dump everything into context | **Entropy scoring** — score, prune, compress |
| Module crosstalk | Show entire codebase | **RAG Denial** — show ONLY the active module |
| Regressions | Fix after the fact | **Contract freezing** — prevent at the source |
| No memory | Each prompt starts fresh | **Session protocol** — persist across edits |
| Verification | Trust the AI | **Weaver checks** — verify every single edit |

---

## How It Works

### The Core Insight: One Module at a Time

Nexus never lets the AI see or edit the entire codebase at once. Instead, it:

1. **Decomposes** your request into module-scoped tasks
2. **Retrieves** only the relevant context for each module (RAG Denial)
3. **Scores** every context item by relevance (entropy scoring)
4. **Prunes** low-signal context to fit a tight token budget
5. **Generates** code for ONE module at a time
6. **Verifies** the edit with SCP Weaver checks + static analysis
7. **Freezes** the module's interface contract after verification
8. **Records** everything in an append-only edit ledger

The AI sees ~700 tokens per module instead of ~110,000 tokens for the full codebase. That's a **157× compression ratio** — well below the Foggy Boundary.

### What the AI Actually Sees

For a module called `Transcriber`, the AI receives:

```
✔ INCLUDED (SCP-scoped):
  • Transcriber's full source code         (active module)
  • Transcriber's SCP contract             (constraints, methods, glyphs)
  • AudioIngest interface contract         (dependency — interface ONLY)
  • VoiceDetector interface contract       (dependency — interface ONLY)
  • Global SCP constraints                 (architectural rules)

✘ EXCLUDED (RAG Denial):
  • SearchEngine source code               (not a dependency)
  • MeetingDetector source code            (not a dependency)
  • LLMProvider source code                (not a dependency)
  • Old conversation history               (entropy-pruned)
  • Stale error messages                   (entropy-pruned)
  • All forbidden modules                  (completely blocked)
```

---

## Architecture

Nexus is built as a 4-layer stack on top of Project Chevron:

```
┌──────────────────────────────────────────────┐
│              CLI / Developer API              │  ← You interact here
├──────────────────────────────────────────────┤
│    Layer 3: Agentic Orchestrator              │
│    ◬ Conductor → ☤ Planner → ☾ Executor      │
│                → 𓂀 Verifier                   │
├──────────────────────────────────────────────┤
│    Layer 2: SCP Session Protocol              │
│    SessionState · EditLedger · ContractCache  │
├──────────────────────────────────────────────┤
│    Layer 1: Context Kernel                    │
│    EntropyScorer · ContextPruner · SCPRetriever│
├──────────────────────────────────────────────┤
│    Layer 0: Project Chevron (SCP)             │
│    scp_bridge · forge · verifier · glyphs     │
├──────────────────────────────────────────────┤
│    AI Providers (pluggable)                   │
│    Gemini · OpenAI · Anthropic · Ollama       │
└──────────────────────────────────────────────┘
```

Each layer has a single responsibility:

| Layer | Module | Purpose |
|-------|--------|---------|
| **L1** | `context_kernel.py` | Score context by relevance, prune noise, enforce RAG Denial |
| **L2** | `session_protocol.py` | Persist architecture state, track edits, freeze contracts |
| **L3** | `conductor.py` | Decompose tasks, execute one-at-a-time, verify every edit |
| **Providers** | `providers/` | Provider-agnostic AI abstraction (swap backends freely) |

---

## Quick Start

### Prerequisites

```bash
# Python 3.10+ required
python --version

# Clone the repo
git clone https://github.com/your-org/chevron.git
cd chevron

# Install at least one AI provider SDK
pip install google-genai       # Google Gemini (recommended)
# OR
pip install openai             # OpenAI GPT-4o
# OR
pip install anthropic          # Anthropic Claude
# OR install Ollama from https://ollama.com for local models
```

### Your First Nexus Session

```bash
# 1. Set your API key
set GEMINI_API_KEY=your-key-here          # Windows
export GEMINI_API_KEY=your-key-here       # Mac/Linux

# 2. See the TurboScribe architecture
python -m nexus.cli overview

# 3. See what the AI would see for a specific module
python -m nexus.cli context Transcriber

# 4. Generate code for a module
python -m nexus.cli generate Transcriber --provider gemini

# 5. Check session health after edits
python -m nexus.cli health

# 6. List available AI providers
python -m nexus.cli providers
```

---

## CLI Reference

### `nexus overview` — Show Architecture

Displays the full SCP module architecture with dependencies, constraints, and method signatures.

```bash
python -m nexus.cli overview
python -m nexus.cli overview --spec path/to/your.chevron
```

**Output:**
```
◬ ─── Architecture: TurboScribe — GPU Audio Transcription ───
  Modules: 9

  ┌─ AudioIngest
  │  Discovers and loads media files from directories.
  │  Methods: find_media, load_audio, get_device_config
  │  Dependencies: (none)
  │  Constraint: Must not transform or analyze audio
  └──────

  ┌─ Transcriber
  │  Transcribes audio to text using faster-whisper models.
  │  Methods: load_model, transcribe_file, batch_transcribe
  │  Dependencies: AudioIngest, VoiceDetector
  │  Constraint: Must save transcripts as filename_transcript_modelname.txt
  └──────
  ...
```

### `nexus generate` — Generate Code

Generate code for one or all modules with SCP constraints and Weaver verification.

```bash
# Single module
python -m nexus.cli generate AudioIngest --provider gemini --key YOUR_KEY

# All modules (processes one at a time)
python -m nexus.cli generate --all --provider gemini --key YOUR_KEY

# With a specific model
python -m nexus.cli generate Transcriber --provider openai --model gpt-4o

# Custom request
python -m nexus.cli generate API --provider gemini --request "Add rate limiting"

# Save session to custom directory
python -m nexus.cli generate Store --provider gemini --output ./my_session
```

**Output:**
```
◬ ─── Nexus: Processing Request ───
  Request: Implement the module(s) following SCP constraints
  Plan: 1 module(s) to edit

☾ ─── Module: Transcriber ───
  ✔ Code generated (1,247 chars)
  ✔ Transcriber: W(G) = 0 — all checks passed
  ☤ Contract frozen (v1)

𓂀 ─── Session Health ───
  Total edits: 1
  Regression rate: 0.0%
  Clean streak: 1
  Frozen contracts: 1/9
```

### `nexus context` — Context Report

See exactly what the AI would see for a module — with entropy scores and compression ratios.

```bash
python -m nexus.cli context Transcriber
```

**Output:**
```
◬ ─── Context Report: Transcriber ───
  Full context: 842 tokens
  After pruning: 712 tokens
  Compression: 84.56%
  Entropy: 1.23 bits
  Forbidden: none

  Context items (by relevance):
    [1.00] contract         187 tok  Transcriber-contract
    [1.00] contract          45 tok  global-constraints
    [0.90] active_code      312 tok  Transcriber.py
    [0.70] interface        124 tok  AudioIngest-interface
    [0.70] interface         44 tok  VoiceDetector-interface
```

### `nexus health` — Session Health

Check the health of your current session after a series of edits.

```bash
python -m nexus.cli health
python -m nexus.cli health --session ./my_session
```

### `nexus providers` — List AI Providers

```bash
python -m nexus.cli providers
```

---

## Programmatic API

For integration into your own tools, editors, or CI pipelines:

```python
from nexus.conductor import Conductor
from nexus.session_protocol import SessionState
from nexus.providers import get_provider, ProviderConfig
from scp_bridge import SCPBridge, ArchitectureSpec, ModuleSpec, InterfaceMethod

# ─── 1. Define your architecture ───
spec = ArchitectureSpec(
    name="MyApp",
    modules=[
        ModuleSpec(
            name="UserStore",
            description="Manages user data persistence",
            methods=[
                InterfaceMethod("create_user", ["name: str", "email: str"],
                                "User", "◬", "Origin — creates new user"),
                InterfaceMethod("find_user", ["email: str"],
                                "User | None", "Ө", "Filter — find by email"),
            ],
            allowed_dependencies=[],
            constraints=["No network access", "Pure data operations"],
        ),
        ModuleSpec(
            name="AuthService",
            description="Handles authentication and sessions",
            methods=[
                InterfaceMethod("login", ["email: str", "password: str"],
                                "Session", "☤", "Weaves credentials into session"),
                InterfaceMethod("verify_token", ["token: str"],
                                "bool", "𓂀", "Witnesses token validity"),
            ],
            allowed_dependencies=["UserStore"],
            constraints=["Must hash passwords", "Tokens expire after 24h"],
        ),
    ],
    global_constraints=["No global mutable state", "All errors must be typed"],
)

# ─── 2. Create provider ───
provider = get_provider(ProviderConfig(
    provider_name="gemini",        # or "openai", "anthropic", "ollama"
    api_key="your-key-here",
    model="gemini-2.0-flash",
))

# ─── 3. Create session + conductor ───
bridge = SCPBridge(spec)
session = SessionState(architecture=spec, session_dir=".nexus_session")
conductor = Conductor(session, provider, bridge=bridge)

# ─── 4. Generate code ───
results = conductor.handle_request(
    "Implement user registration with email validation",
    target_modules=["UserStore"],
)

# ─── 5. Check results ───
for r in results:
    print(f"{r['module']}: {'✔' if r['success'] else '✘'}")
    if r.get('verified'):
        print(f"  Verified, contract frozen")
    if r.get('violations'):
        print(f"  Violations: {r['violations']}")

# ─── 6. Check session health ───
print(session.session_health)
# {'total_edits': 1, 'regression_rate': '0.0%', 'clean_streak': 1, ...}

# ─── 7. Save session for later ───
session.save()
```

---

## Building with Nexus — Samples

### Sample 1: Todo App (Beginner)

A classic 3-module todo application demonstrating SCP isolation.

```python
from scp_bridge import ArchitectureSpec, ModuleSpec, InterfaceMethod

TODO_APP = ArchitectureSpec(
    name="TodoApp",
    modules=[
        ModuleSpec(
            name="TodoStore",
            description="Stores and retrieves todo items. Pure data layer.",
            methods=[
                InterfaceMethod("add_todo", ["text: str"], "Todo", "◬",
                                "Origin — creates a new todo item"),
                InterfaceMethod("list_todos", ["filter: str"], "list[Todo]", "Ө",
                                "Filter — returns todos matching filter"),
                InterfaceMethod("toggle_todo", ["todo_id: int"], "Todo", "☾",
                                "Fold — toggles completion state"),
            ],
            allowed_dependencies=[],
            constraints=[
                "Must not contain any UI logic",
                "Must not make network requests",
                "Data must be stored in-memory (dict or list)",
                "IDs must be auto-incrementing integers",
            ],
        ),
        ModuleSpec(
            name="TodoAPI",
            description="HTTP REST API for todo operations.",
            methods=[
                InterfaceMethod("handle_request", ["method: str", "path: str", "body: dict"],
                                "Response", "☤",
                                "Weaves HTTP request into response via TodoStore"),
            ],
            allowed_dependencies=["TodoStore"],
            constraints=[
                "Must validate all input before passing to TodoStore",
                "Must return proper HTTP status codes",
                "Must not access storage directly — delegate to TodoStore",
            ],
        ),
        ModuleSpec(
            name="TodoLogger",
            description="Logs all todo operations. Pure observation.",
            methods=[
                InterfaceMethod("log_action", ["action: str", "details: dict"],
                                "None", "𓂀",
                                "Witnesses action — logs without modifying state"),
            ],
            allowed_dependencies=[],
            constraints=[
                "Must NEVER modify todo data",
                "Must NEVER raise exceptions that halt the pipeline",
                "Safe to remove entirely without affecting correctness",
            ],
        ),
    ],
    global_constraints=[
        "No global mutable state between modules",
        "All inter-module communication through declared interfaces only",
    ],
)
```

**Generate it:**
```bash
python -c "
from nexus.conductor import Conductor
from nexus.session_protocol import SessionState
from nexus.providers import get_provider, ProviderConfig
from scp_bridge import SCPBridge

# (paste TODO_APP spec above)

bridge = SCPBridge(TODO_APP)
session = SessionState(architecture=TODO_APP)
provider = get_provider(ProviderConfig(provider_name='gemini', api_key='YOUR_KEY'))
conductor = Conductor(session, provider, bridge=bridge)

results = conductor.handle_request('Build the todo app', target_modules=['TodoStore', 'TodoAPI', 'TodoLogger'])
print(session.session_health)
"
```

---

### Sample 2: REST API with Auth (Intermediate)

A 4-module API with authentication, demonstrating forbidden dependencies.

```python
API_WITH_AUTH = ArchitectureSpec(
    name="SecureAPI",
    modules=[
        ModuleSpec(
            name="Database",
            description="PostgreSQL connection and query execution. Pure data access.",
            methods=[
                InterfaceMethod("execute", ["query: str", "params: list"], "list[dict]", "◬",
                                "Origin — executes SQL and returns rows"),
                InterfaceMethod("execute_one", ["query: str", "params: list"], "dict | None", "Ө",
                                "Filter — returns first matching row or None"),
            ],
            allowed_dependencies=[],
            forbidden=["Router", "Middleware"],  # DB must never know about HTTP
            constraints=[
                "Must use parameterized queries (no string interpolation)",
                "Must handle connection pooling",
                "Must never log query parameters (may contain secrets)",
            ],
        ),
        ModuleSpec(
            name="Auth",
            description="JWT-based authentication and authorization.",
            methods=[
                InterfaceMethod("create_token", ["user_id: int", "role: str"], "str", "☤",
                                "Weaves user identity into signed JWT"),
                InterfaceMethod("verify_token", ["token: str"], "TokenPayload | None", "𓂀",
                                "Witnesses token validity without side effects"),
                InterfaceMethod("hash_password", ["password: str"], "str", "☾",
                                "Folds password through bcrypt"),
            ],
            allowed_dependencies=["Database"],
            forbidden=["Router"],  # Auth must never know about routes
            constraints=[
                "Tokens expire after 24 hours",
                "Must use bcrypt with cost factor ≥ 12",
                "Must never store plaintext passwords",
            ],
        ),
        ModuleSpec(
            name="Router",
            description="HTTP request routing and response formatting.",
            methods=[
                InterfaceMethod("handle", ["method: str", "path: str", "headers: dict", "body: dict"],
                                "Response", "☤",
                                "Weaves HTTP request through middleware + handlers"),
            ],
            allowed_dependencies=["Auth", "Database", "Middleware"],
            constraints=[
                "Must validate Content-Type header",
                "Must return JSON responses with proper status codes",
                "Must handle all exceptions gracefully (no 500s)",
            ],
        ),
        ModuleSpec(
            name="Middleware",
            description="Request middleware: logging, rate limiting, CORS.",
            methods=[
                InterfaceMethod("apply", ["request: Request"], "Request | ErrorResponse", "Ө",
                                "Filter — passes valid requests, rejects invalid ones"),
            ],
            allowed_dependencies=["Auth"],
            forbidden=["Database"],  # Middleware must never touch the DB
            constraints=[
                "Rate limit: 100 requests per minute per IP",
                "Must add request-id header to all responses",
                "Must log request duration (but never request bodies)",
            ],
        ),
    ],
    global_constraints=[
        "No module may import another module's internals — interfaces only",
        "All errors must include a machine-readable error code",
        "No global mutable state",
    ],
)
```

**Notice the `forbidden` fields** — `Database` cannot know about `Router` or `Middleware`. This is RAG Denial in action: when the AI generates the `Database` module, it literally cannot see router code, preventing accidental coupling.

---

### Sample 3: TurboScribe (Production-Scale)

A real-world 9-module GPU transcription engine — the full example that ships with Nexus.

```bash
# See the full architecture
python examples/turboscribe_example.py

# Generate a single module with Gemini
set GEMINI_API_KEY=your-key
python examples/turboscribe_example.py Transcriber --gemini

# Generate ALL modules with verification + tests
python examples/turboscribe_example.py --all --with-tests

# Use Nexus CLI instead
python -m nexus.cli overview
python -m nexus.cli generate Transcriber --provider gemini
```

The TurboScribe spec decomposes a ~110,000 token codebase into 9 isolated modules. The AI sees ~700 tokens per module instead of 110,000 — a **157× compression ratio**.

| Module | Glyph | Role |
|--------|-------|------|
| AudioIngest | ◬ | Origin — file discovery & loading |
| VoiceDetector | Ө | Filter — speech/silence detection |
| Transcriber | ☾ | Fold Time — Whisper transcription |
| SearchEngine | Ө | Filter — keyword & semantic search |
| MeetingDetector | Ө | Filter — real vs hallucinated |
| LLMProvider | ☤ | Weaver — unified LLM interface |
| Analyzer | ☤ | Weaver — summarize/outline |
| TimestampExtractor | ☾ | Fold Time — video timestamp OCR |
| ProgressWitness | 𓂀 | Witness — pure logging |

---

## AI Provider Setup

Nexus supports 4 AI providers. Install only the one(s) you need:

### Google Gemini (Recommended)
```bash
pip install google-genai
set GEMINI_API_KEY=your-key       # Windows
export GEMINI_API_KEY=your-key    # Mac/Linux
```
```python
provider = get_provider(ProviderConfig(
    provider_name="gemini",
    api_key="your-key",
    model="gemini-2.0-flash",     # or gemini-2.5-pro
))
```

### OpenAI
```bash
pip install openai
set OPENAI_API_KEY=your-key
```
```python
provider = get_provider(ProviderConfig(
    provider_name="openai",
    api_key="your-key",
    model="gpt-4o",               # or gpt-4o-mini
))
```

### Anthropic
```bash
pip install anthropic
set ANTHROPIC_API_KEY=your-key
```
```python
provider = get_provider(ProviderConfig(
    provider_name="anthropic",
    api_key="your-key",
    model="claude-sonnet-4-20250514",
))
```

### Ollama (Local — No API Key Needed)
```bash
# Install from https://ollama.com
ollama pull llama3.1
```
```python
provider = get_provider(ProviderConfig(
    provider_name="ollama",
    model="llama3.1",
    # base_url defaults to http://localhost:11434
))
```

---

## Test Suite

64 tests across 3 test suites verify all Nexus components:

```bash
# Run all tests
python tests/test_context_kernel.py
python tests/test_session_protocol.py
python tests/test_conductor.py
```

| Suite | Tests | Coverage |
|-------|-------|----------|
| `test_context_kernel.py` | 25 | Entropy scoring, pruning tiers, RAG Denial, Shannon entropy |
| `test_session_protocol.py` | 24 | Edit ledger, contract cache, session persistence |
| `test_conductor.py` | 15 | Planner, Executor, Verifier, end-to-end pipeline |

All tests use mock providers — no API keys required to run the test suite.

---

## Project Structure

```
chevron/
├── nexus/                         # ← Nexus AI IDE
│   ├── __init__.py                #    Package root & public API
│   ├── context_kernel.py          #    Layer 1: Entropy scoring & RAG Denial
│   ├── session_protocol.py        #    Layer 2: Persistent state & contracts
│   ├── conductor.py               #    Layer 3: Orchestrator (Plan → Execute → Verify)
│   ├── cli.py                     #    CLI entry point
│   └── providers/                 #    AI provider abstraction
│       ├── __init__.py
│       ├── base.py                #      Abstract interface
│       ├── registry.py            #      Lazy-loading registry
│       ├── gemini_provider.py     #      Google Gemini
│       ├── openai_provider.py     #      OpenAI GPT-4o
│       ├── anthropic_provider.py  #      Anthropic Claude
│       └── ollama_provider.py     #      Ollama (local)
├── scp_bridge.py                  # Layer 0: SCP → AI prompt generation
├── forge.py                       # Layer 0: Auto-decompose codebases
├── chevron/                       # Chevron language (lexer, parser, verifier)
├── examples/                      # TurboScribe example
├── tests/                         # Test suites
│   ├── test_context_kernel.py
│   ├── test_session_protocol.py
│   ├── test_conductor.py
│   └── test_chevron.py
└── README.md                      # Chevron documentation
```

---

## Roadmap

### Now (v0.1 — Prototype)
- ✅ Entropy-aware context kernel
- ✅ Persistent session state with edit ledger
- ✅ One-module-at-a-time execution with verification
- ✅ 4 AI provider backends
- ✅ CLI interface

### Next (v0.2)
- [ ] VS Code extension (Language Server Protocol)
- [ ] Automatic test generation after each edit
- [ ] `forge.py` integration (auto-decompose existing codebases into SCP)
- [ ] Interactive conflict resolution when contracts break
- [ ] Streaming AI output with real-time Weaver verification

### Future (v1.0)
- [ ] Multi-agent mode (separate AI instances per module)
- [ ] Git integration (SCP-aware commits, branch-per-module)
- [ ] Dashboard UI for session health visualization
- [ ] Embedding-based semantic retrieval within RAG Denial constraints
- [ ] Team mode (shared contract cache across developers)

---

## Key Concepts Glossary

| Term | Definition |
|------|-----------|
| **SCP** | Spatial Constraint Protocol — formal system for constraining AI attention |
| **RAG Denial** | Only the active module sees full code; dependencies see contracts only |
| **Foggy Boundary** | The point where context entropy exceeds the AI's constraint resolution capacity |
| **Weaver (☤)** | Verification step that checks generated code against SCP contracts |
| **Contract Freeze** | After Weaver verification, a module's interface is frozen — other modules see this frozen snapshot |
| **Edit Ledger** | Append-only log of every AI edit with verification status |
| **Entropy Score** | 0–1 relevance score for each context item (1.0 = contracts, 0.1 = stale) |
| **Glyph** | SCP operator mapped to a neural operation: ◬ Origin, Ө Filter, ☾ Fold, ☤ Weave, 𓂀 Witness |

---

*Built on [Project Chevron](README.md) — the SCP reference implementation that reduces AI code regression from 14.3% to <0.1%.*
