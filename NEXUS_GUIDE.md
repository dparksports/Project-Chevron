# Nexus — AI IDE That Won't Break Your Code

> **Build software one module at a time, with AI that never loses context.**

Nexus wraps the Spatial Constraint Protocol into a working AI development environment. Instead of dumping your entire codebase into a prompt (causing the Partition Function Z to explode and attention to decay), it decomposes your architecture into isolated modules and generates code for each one independently — using orthogonal Uiua embeddings that create steep attractor basins, with System 2 rejection sampling (The Weaver) and zero cross-module confabulation.

---

## Quick Start

```bash
# Clone and enter the project
git clone https://github.com/dparksports/Project-Chevron.git
cd Project-Chevron

# Install one AI provider (pick one)
pip install google-genai       # Google Gemini (recommended)
pip install openai             # OpenAI
pip install anthropic          # Anthropic Claude
# Or install Ollama from https://ollama.com for local models

# Set your API key
$env:GEMINI_API_KEY = "your-key"           # Windows PowerShell
# export GEMINI_API_KEY="your-key"         # Mac/Linux
```

---

## Two Ways to Start

### Option A: Convert an Existing Project

Point Nexus at your existing codebase and it will automatically decompose it into an SCP architecture using AI:

```bash
python -m nexus.cli init myapp --from ./path/to/your/project --key YOUR_GEMINI_KEY
```

This scans your codebase, sends it to Gemini for architectural analysis, and creates a fully scaffolded Nexus project:

```
myapp/
├── nexus.json              ← Architecture spec (AI-generated module boundaries)
├── src/                    ← Stub files per module (ready for code generation)
│   ├── userstore.py
│   ├── authservice.py
│   └── apirouter.py
├── tests/                  ← Test stubs per module (from SCP contracts)
│   ├── test_userstore.py
│   ├── test_authservice.py
│   └── test_apirouter.py
├── README.md               ← Auto-generated project docs
├── .gitignore
└── .nexus_session/         ← Session state (persists across edits)
```

### Option B: Start from a Template

```bash
# See available templates
python -m nexus.cli templates

# Scaffold from a template
python -m nexus.cli init myapp --template todo-app
python -m nexus.cli init myapi --template web-api
python -m nexus.cli init mypipeline --template data-pipeline
python -m nexus.cli init mytool --template cli-tool
```

| Template | Modules | Best For |
|----------|---------|----------|
| **todo-app** | TodoStore, TodoAPI, TodoLogger | Learning Nexus basics |
| **web-api** | Database, Auth, Router, Middleware | REST APIs with auth |
| **cli-tool** | ArgParser, Core, FileIO, Reporter | Command-line tools |
| **data-pipeline** | Ingester, Transformer, Loader, Orchestrator, Monitor | ETL workflows |
| **blank** | Core | Custom projects |

---

## Using the Visual Dashboard

The easiest way to use Nexus — no CLI knowledge required.

```bash
# Install dashboard dependencies (one-time)
pip install fastapi uvicorn

# Launch
python -m nexus.cli start
# Opens http://localhost:3000
```

### What You See

**Template Gallery** — Click to create a project. No commands to memorize.

**Architecture Graph** — Interactive node graph of your modules:
- Grey nodes = Stub (no code yet)
- Green nodes = Verified and frozen (contract-locked)
- Amber pulse = Currently generating
- Click a node to inspect its contract, methods, constraints, and dependencies

**Conductor Chat** — Type what you want in plain English:
```
Implement the TodoStore module
Add a delete method to TodoAPI
Make the CLI support a --verbose flag
```

The chat streams real-time progress:
1. **Planning** — Which modules will be generated
2. **Generating** — Code appears as it's written (node pulses amber)
3. **Verification** — SCP Weaver checks contracts (pass ✔ / fail ✘)
4. **Health Update** — Session metrics refresh

**Health Monitor** — Session metrics: total edits, regression rate, clean streak, frozen contracts.

---

## Using the CLI

### Review Your Architecture

```bash
cd myapp
python -m nexus.cli overview --spec nexus.json
```

Edit `nexus.json` to customize modules, methods, constraints, and dependency rules. This is your **single source of truth** — the AI will be constrained by whatever you put here.

### Generate Code (One Module at a Time)

```bash
# Generate one module
python -m nexus.cli generate TodoStore --provider gemini

# Generate all modules sequentially
python -m nexus.cli generate --all --provider gemini

# Use a different provider
python -m nexus.cli generate TodoStore --provider openai --key YOUR_OPENAI_KEY
python -m nexus.cli generate TodoStore --provider ollama --model llama3.1
```

Each module is generated in **complete isolation**: the AI sees only that module's code, its SCP contract, and the interface contracts of its declared dependencies. **Nothing else.**

### Inspect & Verify

```bash
# See exactly what context the AI receives for a module
python -m nexus.cli context TodoStore --spec nexus.json

# Check session health after edits
python -m nexus.cli health

# List available AI providers
python -m nexus.cli providers
```

---

## How It Works

| Mechanism | What It Does |
|-----------|-------------|
| **RAG Denial** | AI sees ONLY the active module + dependency interfaces. Everything else is invisible — preventing semantic cross-talk. |
| **Entropy Scoring** | Every context item is scored 0–1 by relevance. Low-signal items are pruned, reducing Z. |
| **Contract Freezing** | After Weaver verification (W(G) = 0), a module's interface is frozen. Other modules see the frozen snapshot. |

The AI sees **~700 tokens per module** instead of ~110,000 for the full codebase. **157× compression.**

### Why Build Bottom-Up?

- Each module has a **contract** (methods, types, constraints)
- When you generate module B that depends on module A, the AI only sees A's **frozen contract** — not its full source code
- This prevents the **Partition Function Explosion** (the AI's attention drowning in distractor tokens from irrelevant code)
- Orthogonal embeddings create steep attractor basins that resist confabulation
- Build leaves first, then dependents

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

Your architecture definition. Edit this to refine what the AI generates:

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
```

---

## Typical Workflow

```
┌──────────────────────────────────────────────────────────────┐
│  1. Create Project                                           │
│     nexus init myapp --from ./existing_code                  │
│     nexus init myapp --template web-api                      │
└────────┬─────────────────────────────────────────────────────┘
         ▼
┌──────────────────────────────────────────────────────────────┐
│  2. Review & Customize nexus.json                            │
│     nexus overview --spec nexus.json                         │
└────────┬─────────────────────────────────────────────────────┘
         ▼
┌──────────────────────────────────────────────────────────────┐
│  3. Generate Code (Bottom-Up)                                │
│     nexus generate TodoStore --provider gemini               │
│     nexus generate --all --provider gemini                   │
└────────┬─────────────────────────────────────────────────────┘
         ▼
┌──────────────────────────────────────────────────────────────┐
│  4. Verify & Check Health                                    │
│     nexus context TodoStore   (inspect what AI sees)         │
│     nexus health              (regression rate, streak)      │
└──────────────────────────────────────────────────────────────┘
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

## Further Reading

- [CLI & Forge Guide](CLI_GUIDE.md) — Headless workflow using Forge + SCP Bridge directly
- [Dashboard Guide](DASHBOARD_README.md) — Detailed dashboard documentation
- [README.md](README.md) — Full SCP theory and research background
- [Research Paper (PDF)](https://github.com/dparksports/dparksports/raw/main/SCP%20II%20-%20Neuro-Symbolic%20Resolution.pdf) — The Partition Function Explosion: An Energy-Based Analysis of Attention Decay
