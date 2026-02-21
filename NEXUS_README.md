# ◬ Nexus — SCP-Powered AI IDE

**Build software one module at a time, with AI that never loses context.**

> Nexus turns the [Spatial Constraint Protocol](README.md) into a working AI development environment. Instead of dumping your entire codebase into an AI's context window (causing the Partition Function Z to explode), Nexus decomposes your architecture into isolated modules and generates code for each one independently — using orthogonal embeddings that create steep attractor basins, System 2 rejection sampling (The Weaver), and zero cross-module confabulation.

---

## Why Nexus?

Every AI coding assistant hits the same wall:

```
Turn 1:   "Add a login form"       → ✔ Perfect
Turn 10:  "Now add logout"         → ✘ Breaks auth state
Turn 20:  "Fix the regression"     → ✘✘ Introduces 2 new bugs
```

**The root cause:** The AI's context window fills with distractor tokens. The Partition Function (Z) explodes. Attention probability mass dilutes. The signal drowns in noise. The model confabulates — relaxing into pre-trained priors instead of following your constraints. Regressions cascade.

**Nexus fixes this** with three mechanisms:

| Mechanism | What It Does |
|-----------|-------------|
| **RAG Denial** | AI sees ONLY the active module + dependency interfaces. Everything else is invisible — preventing semantic cross-talk. |
| **Entropy Scoring** | Every context item is scored 0–1 by relevance. Low-signal items are pruned, reducing Z. |
| **Contract Freezing** | After Weaver verification (W(G) = 0), a module's interface is frozen. Other modules see the frozen snapshot. |

The AI sees ~700 orthogonal tokens per module instead of ~110,000 noisy tokens for the full codebase. **157× compression — restoring the Critical Energy Gap (ΔE > ln(N)).**

---

## Getting Started

### Step 1: Install

```bash
git clone https://github.com/dparksports/Project-Chevron.git
cd Project-Chevron

# Install one AI provider (pick one)
pip install google-genai       # Google Gemini (recommended)
pip install openai             # OpenAI
pip install anthropic          # Anthropic Claude
# Or install Ollama from https://ollama.com for local models (no pip needed)
```

### Dashboard Mode (Recommended for Beginners)

```bash
# Install the dashboard dependencies
pip install fastapi uvicorn

# Launch the visual dashboard
python -m nexus.cli start
# Opens http://localhost:3000 — no CLI knowledge required!
```

The dashboard provides a visual interface with:
- **Template Gallery** — Click to create a project (no commands to memorize)
- **Architecture Map** — See your modules as an interactive graph
- **Conductor Chat** — Type what you want in plain English
- **Health Monitor** — Session metrics at a glance

---

### Step 2: Create a Project

```bash
# See available templates
python -m nexus.cli templates

# Scaffold from a template
python -m nexus.cli init myapp --template todo-app
python -m nexus.cli init myapi --template web-api
python -m nexus.cli init mypipeline --template data-pipeline

# Or decompose an existing codebase with AI
python -m nexus.cli init myapp --from ./existing_code --key YOUR_GEMINI_KEY
```

This creates:
```
myapp/
├── nexus.json              ← Architecture spec (modules, methods, constraints, deps)
├── src/                    ← Stub files per module
│   ├── todostore.py        ←   class TodoStore with method signatures
│   ├── todoapi.py          ←   class TodoAPI with method signatures
│   └── todologger.py       ←   class TodoLogger with method signatures
├── tests/                  ← Test stubs per module (from SCP contracts)
│   ├── test_todostore.py
│   ├── test_todoapi.py
│   └── test_todologger.py
├── README.md               ← Auto-generated project docs
├── .gitignore
└── .nexus_session/         ← Session state (persists across edits)
```

### Step 3: Review Your Architecture

```bash
cd myapp

# See the full module map
python -m nexus.cli overview --spec nexus.json
```

Edit `nexus.json` to customize modules, methods, constraints, and dependency rules. This is your **single source of truth** — the AI will be constrained by whatever you put here.

### Step 4: Generate Code

```bash
# Set your API key
set GEMINI_API_KEY=your-key-here          # Windows
export GEMINI_API_KEY=your-key-here       # Mac/Linux

# Generate one module at a time
python -m nexus.cli generate TodoStore --provider gemini
python -m nexus.cli generate TodoAPI --provider gemini

# Or generate all modules sequentially
python -m nexus.cli generate --all --provider gemini

# Use a different provider
python -m nexus.cli generate TodoStore --provider openai --key YOUR_OPENAI_KEY
python -m nexus.cli generate TodoStore --provider ollama --model llama3.1
```

Each module is generated in isolation: the AI sees only that module's code, its SCP contract, and the interface contracts of its declared dependencies. **Nothing else.**

### Step 5: Inspect & Verify

```bash
# See exactly what context the AI receives for a module
python -m nexus.cli context TodoStore --spec nexus.json

# Check session health after a series of edits
python -m nexus.cli health

# List available providers
python -m nexus.cli providers
```

---

## Templates

| Template | Modules | Best For |
|----------|---------|----------|
| **todo-app** | TodoStore, TodoAPI, TodoLogger | Learning Nexus basics |
| **web-api** | Database, Auth, Router, Middleware | REST APIs with auth |
| **cli-tool** | ArgParser, Core, FileIO, Reporter | Command-line tools |
| **data-pipeline** | Ingester, Transformer, Loader, Orchestrator, Monitor | ETL workflows |
| **blank** | Core | Custom projects |

---

## Programmatic Usage

```python
from nexus.conductor import Conductor
from nexus.session_protocol import SessionState
from nexus.providers import get_provider, ProviderConfig
from nexus.scaffold import load_spec
from scp_bridge import SCPBridge

# Load your architecture
spec = load_spec("nexus.json")

# Create provider + session + conductor
provider = get_provider(ProviderConfig(provider_name="gemini", api_key="..."))
session = SessionState(architecture=spec, session_dir=".nexus_session")
conductor = Conductor(session, provider, bridge=SCPBridge(spec))

# Generate code — one module at a time, verified after each edit
results = conductor.handle_request(
    "Add input validation",
    target_modules=["TodoAPI"],
)

# Check health
print(session.session_health)
# {'total_edits': 1, 'regression_rate': '0.0%', 'clean_streak': 1, ...}

# Save session for later
session.save()
```

---

## AI Providers

| Provider | Install | Model Examples |
|----------|---------|---------------|
| **Gemini** | `pip install google-genai` | `gemini-2.5-pro`, `gemini-2.0-flash` |
| **OpenAI** | `pip install openai` | `gpt-4o`, `gpt-4o-mini` |
| **Anthropic** | `pip install anthropic` | `claude-sonnet-4-20250514` |
| **Ollama** | [ollama.com](https://ollama.com) | `llama3.1`, `codellama` |

All providers use **lazy loading** — importing Nexus doesn't require all SDKs installed.

---

## The nexus.json Spec

The `nexus.json` file is your architecture definition. Here's a minimal example:

```json
{
  "name": "MyApp",
  "modules": [
    {
      "name": "UserStore",
      "description": "Manages user data persistence",
      "methods": [
        {
          "name": "create_user",
          "inputs": ["name: str", "email: str"],
          "output": "User",
          "glyph": "◬",
          "constraint": "Origin — creates new user"
        }
      ],
      "allowed_dependencies": [],
      "constraints": ["No network access", "Pure data operations"]
    }
  ],
  "global_constraints": ["No global mutable state"]
}
```

**Key fields:**
- `allowed_dependencies` — Which modules this module can see (RAG Denial enforced)
- `constraints` — Rules the AI must follow when generating code
- `methods.glyph` — SCP glyph (◬ Origin, Ө Filter, ☾ Fold, ☤ Weave, 𓂀 Witness)

---

## Architecture

```
┌──────────────────────────────────────────────┐
│              CLI / Developer API              │
├──────────────────────────────────────────────┤
│    Layer 3: Orchestrator (conductor.py)       │
│    ◬ Conductor → ☤ Planner → ☾ Executor      │
│                → 𓂀 Verifier                   │
├──────────────────────────────────────────────┤
│    Layer 2: Session Protocol                  │
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

---

## Test Suite

64 tests, all passing. No API keys needed — tests use mock providers.

```bash
python tests/test_context_kernel.py      # 25 tests — entropy, pruning, RAG Denial
python tests/test_session_protocol.py    # 24 tests — ledger, contracts, persistence
python tests/test_conductor.py           # 15 tests — planner, executor, verifier, e2e
```

---

## Project Structure

```
chevron/
├── nexus/                         # Nexus AI IDE
│   ├── __init__.py                #   Package root
│   ├── context_kernel.py          #   Layer 1: Entropy scoring & RAG Denial
│   ├── session_protocol.py        #   Layer 2: Session state & contracts
│   ├── conductor.py               #   Layer 3: Orchestrator
│   ├── scaffold.py                #   Project scaffolding & templates
│   ├── cli.py                     #   CLI (init, templates, overview, generate, ...)
│   ├── nexus.chevron              #   Self-specification in Chevron
│   └── providers/                 #   AI provider abstraction
│       ├── base.py                #     Abstract interface
│       ├── registry.py            #     Lazy-loading registry
│       ├── gemini_provider.py     #     Google Gemini
│       ├── openai_provider.py     #     OpenAI
│       ├── anthropic_provider.py  #     Anthropic Claude
│       └── ollama_provider.py     #     Ollama (local)
├── scp_bridge.py                  # SCP → AI prompt generation
├── forge.py                       # Auto-decompose existing codebases
├── chevron/                       # Chevron language runtime
├── tests/                         # Test suites
└── README.md                      # Chevron/SCP documentation
```

---

*Built on [Project Chevron](README.md) — the Spatial Constraint Protocol reference implementation.*
*Paper: [The Partition Function Explosion: An Energy-Based Analysis of Attention Decay](https://github.com/dparksports/dparksports/raw/main/SCP%20II%20-%20Neuro-Symbolic%20Resolution.pdf)*
