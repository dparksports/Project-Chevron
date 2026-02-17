# Nexus Dashboard — Visual AI Development

> Build software one module at a time, without ever touching the command line.

The Nexus Dashboard is a local web interface that wraps the Nexus AI IDE into a visual, point-and-click experience. Instead of memorizing CLI commands, you interact through a **template gallery**, **architecture graph**, and **chat-driven code generation**.

---

## Quick Start

```bash
# 1. Activate the environment
chevron-env\Scripts\activate.ps1          # Windows
# source chevron-env/bin/activate         # macOS/Linux

# 2. Install dashboard dependencies (one-time)
pip install fastapi uvicorn

# 3. Set your AI provider API key
$env:GEMINI_API_KEY = "your-key-here"     # Windows PowerShell
# export GEMINI_API_KEY="your-key-here"   # macOS/Linux

# 4. Launch
cd chevron
python -m nexus.cli start
```

Your browser opens to **http://localhost:3000**. That's it.

### Supported AI Providers

| Provider   | Env Variable         | Install                          |
|------------|----------------------|----------------------------------|
| Gemini     | `GEMINI_API_KEY`     | `pip install google-genai`       |
| OpenAI     | `OPENAI_API_KEY`     | `pip install openai`             |
| Anthropic  | `ANTHROPIC_API_KEY`  | `pip install anthropic`          |
| Ollama     | *(none needed)*      | [ollama.com](https://ollama.com) |

---

## What You See

The dashboard has four main areas:

### 1. Template Gallery (Welcome Screen)

When no project is loaded, you see a gallery of starter templates:

| Template       | What It Builds                                    |
|----------------|---------------------------------------------------|
| **Todo App**   | 3-module todo list — great for beginners          |
| **Web API**    | REST API with auth, DB, routing, middleware       |
| **CLI Tool**   | Command-line app with clean I/O separation        |
| **Data Pipeline** | ETL pipeline with ingest/transform/load/monitor |
| **Blank**      | Empty project with a single Core module           |

**Click any card** → enter a project name → Nexus scaffolds the full architecture for you.

### 2. Architecture Graph (Main View)

After creating a project, you see an interactive node graph of your modules:

- **Grey nodes** = Stub (no code yet)
- **Green nodes** = Verified and frozen (contract-locked)
- **Amber pulse** = Currently generating
- **Drag** nodes to rearrange
- **Click** a node to open the Inspector
- **Double-click** a node to pre-fill the chat with "Implement the [module] module"

### 3. Inspector Panel (Right Side)

Click any module node to see:

- **Description** — What the module does
- **Status** — Stub / Has Code / Verified & Frozen
- **Methods** — Function signatures with SCP glyphs
- **Constraints** — Rules the AI must follow
- **Dependencies** — What this module can import
- **Forbidden** — What it cannot access (RAG Denial)

### 4. Conductor Chat (Bottom Panel)

This is where you talk to the AI. Type plain English:

```
Implement the TodoStore module
```
```
Add a delete method to TodoAPI
```
```
Make the CLI support a --verbose flag
```

The chat streams **real-time progress**:
1. **Planning** — Which modules will be generated
2. **Generating** — Code appears as it's written (node pulses amber)
3. **Verification** — SCP Weaver checks contracts (pass ✔ / fail ✘)
4. **Health Update** — Session metrics refresh

### 5. Header Navigation

- **Architecture** — The graph + chat workspace
- **Health** — Session metrics (regression rate, clean streak, total edits)
- **Providers** — See which AI SDKs are installed and available

---

## Building a Program: Step-by-Step

Here's how to build a todo app from scratch using only the dashboard.

### Step 1: Create the Project

1. Open `http://localhost:3000`
2. Click the **Todo App** template card
3. Enter `my-todo-app` as the project name
4. The architecture graph appears with 3 modules: `TodoStore`, `TodoAPI`, `TodoCLI`

### Step 2: Explore the Architecture

1. Click the **TodoStore** node in the graph
2. The Inspector shows its methods: `add(item)`, `remove(id)`, `list() → items`
3. Note the constraints — the AI will be forced to respect these
4. Check the **Dependencies** — TodoStore has no dependencies (it's a leaf module)

### Step 3: Generate Code (Bottom-Up)

Start with modules that have **no dependencies** and work upward:

1. **Click TodoStore** in the graph → it highlights
2. In the chat, type: `Implement the TodoStore module`
3. Watch the progress:
   - Plan: 1 module → TodoStore
   - ⚡ Generating TodoStore...
   - Code preview appears
   - ✔ Verification: TodoStore — PASSED
4. The node turns **green** — its contract is now frozen

### Step 4: Build the Next Layer

1. **Click TodoAPI** → type: `Implement the TodoAPI module`
2. The AI sees TodoStore's frozen contract and generates code that correctly imports from it
3. Verification passes → TodoAPI turns green

### Step 5: Complete the App

1. **Click TodoCLI** → type: `Implement the TodoCLI module`
2. The AI generates the CLI interface, importing from TodoAPI
3. All 3 nodes are now green

### Step 6: Check Health

1. Click **Health** in the top nav
2. You'll see:
   - **Total edits**: 3
   - **Regression rate**: 0%
   - **Clean streak**: 3
   - **Frozen contracts**: 3

### Step 7: Find Your Code

The generated files are in your project directory:

```
my-todo-app/
├── nexus.json              # Architecture spec
├── TodoStore.py            # Generated + verified
├── TodoAPI.py              # Generated + verified
├── TodoCLI.py              # Generated + verified
└── .nexus_session/         # Session state
```

---

## Key Concepts

### Why Bottom-Up?

Nexus uses **SCP (Structured Context Protocol)** which means:
- Each module has a **contract** (methods, types, constraints)
- When you generate module B that depends on module A, the AI only sees A's **frozen contract** — not its full source code
- This prevents **context entropy** (the AI getting confused by too much irrelevant code)
- Build leaves first, then dependents

### What is RAG Denial?

Modules have a **forbidden** list. If TodoStore is forbidden from accessing `TodoCLI`, the AI literally cannot see TodoCLI's code during generation. This enforces clean architecture by construction.

### What Happens on Failure?

If verification fails:
- The chat shows ✘ with the specific violations
- The node stays grey/amber (not frozen)
- Rephrase your request or add more detail, then try again
- Your regression rate goes up (visible in Health view)

---

## CLI Flags

```bash
python -m nexus.cli start                     # Default: port 3000, auto-open browser
python -m nexus.cli start --port 8080         # Custom port
python -m nexus.cli start --no-browser        # Don't auto-open browser
```

---

## Architecture

```
nexus/
├── server.py               # FastAPI backend (REST + WebSocket)
├── cli.py                  # CLI entry point (nexus start)
└── static/
    ├── index.html           # SPA shell
    ├── css/
    │   └── dashboard.css    # Dark theme design system
    └── js/
        ├── app.js           # View switching, API client, inspector
        ├── graph.js         # Canvas-based architecture graph
        └── chat.js          # WebSocket conductor chat
```

### API Endpoints

| Method | Path                    | Description                        |
|--------|-------------------------|------------------------------------|
| GET    | `/`                     | Dashboard HTML                     |
| GET    | `/api/templates`        | List project templates             |
| POST   | `/api/init`             | Create a new project               |
| GET    | `/api/architecture`     | Current architecture + status      |
| GET    | `/api/context/{module}` | Context report for a module        |
| GET    | `/api/health`           | Session health metrics             |
| GET    | `/api/providers`        | Available AI providers             |
| WS     | `/ws/generate`          | Stream generation progress         |
