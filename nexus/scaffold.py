"""
Nexus Project Scaffolding — Create New SCP Projects
=====================================================
Handles project initialization for the Nexus IDE.

Two workflows:
  1. New project from template:   nexus init myapp --template web-api
  2. Existing codebase:           nexus init myapp --from ./existing_code

Templates provide pre-built ArchitectureSpec with modules, constraints,
and dependency DAGs. Users can then generate code module-by-module.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add parent dir for chevron imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scp_bridge import ArchitectureSpec, ModuleSpec, InterfaceMethod


# ─────────────────────────────────────────────────────────────
#  Templates
# ─────────────────────────────────────────────────────────────

TEMPLATES: dict[str, dict] = {}


def _register(name: str, display: str, description: str, builder):
    """Register a project template."""
    TEMPLATES[name] = {
        "display": display,
        "description": description,
        "builder": builder,
    }


# ── Template: todo-app ──────────────────────────────────────

def _build_todo_app(project_name: str) -> ArchitectureSpec:
    return ArchitectureSpec(
        name=f"{project_name} — Todo Application",
        modules=[
            ModuleSpec(
                name="TodoStore",
                description="In-memory store for todo items. Pure data layer.",
                methods=[
                    InterfaceMethod("add_todo", ["text: str"], "Todo", "◬",
                                    "Origin — creates a new todo item"),
                    InterfaceMethod("get_todo", ["todo_id: int"], "Todo | None", "Ө",
                                    "Filter — returns todo by ID or None"),
                    InterfaceMethod("list_todos", ["filter: str"], "list[Todo]", "Ө",
                                    "Filter — returns todos matching filter (all/active/done)"),
                    InterfaceMethod("toggle_todo", ["todo_id: int"], "Todo", "☾",
                                    "Fold — toggles completion state"),
                    InterfaceMethod("delete_todo", ["todo_id: int"], "bool", "☾",
                                    "Fold — removes a todo item"),
                ],
                allowed_dependencies=[],
                constraints=[
                    "Must not contain any UI or HTTP logic",
                    "Must not make network requests",
                    "Data stored in-memory (dict keyed by auto-incrementing int ID)",
                    "Thread-safe if accessed concurrently",
                ],
            ),
            ModuleSpec(
                name="TodoAPI",
                description="REST API layer. Routes HTTP requests to TodoStore.",
                methods=[
                    InterfaceMethod("handle_request", ["method: str", "path: str", "body: dict"],
                                    "Response", "☤",
                                    "Weaves HTTP request into response via TodoStore"),
                ],
                allowed_dependencies=["TodoStore"],
                constraints=[
                    "Must validate all input before passing to TodoStore",
                    "Must return proper HTTP status codes (200, 201, 400, 404)",
                    "Must not access storage directly — delegate to TodoStore",
                    "Must not import any web framework (pure stdlib)",
                ],
            ),
            ModuleSpec(
                name="TodoLogger",
                description="Logs all operations. Pure observation — never modifies data.",
                methods=[
                    InterfaceMethod("log_action", ["action: str", "details: dict"],
                                    "None", "𓂀",
                                    "Witnesses action — logs without modifying state"),
                ],
                allowed_dependencies=[],
                constraints=[
                    "Must NEVER modify todo data or control flow",
                    "Must NEVER raise exceptions that halt the pipeline",
                    "Safe to remove entirely without affecting correctness",
                    "Output: structured JSON lines to stdout",
                ],
            ),
        ],
        global_constraints=[
            "No global mutable state between modules",
            "All inter-module communication through declared interfaces only",
            "TodoStore is the single source of truth for all data",
        ],
    )

_register("todo-app", "Todo App", "Simple 3-module todo list (beginner)", _build_todo_app)


# ── Template: web-api ───────────────────────────────────────

def _build_web_api(project_name: str) -> ArchitectureSpec:
    return ArchitectureSpec(
        name=f"{project_name} — Web API",
        modules=[
            ModuleSpec(
                name="Database",
                description="Database connection and query execution. Pure data access.",
                methods=[
                    InterfaceMethod("execute", ["query: str", "params: list"], "list[dict]", "◬",
                                    "Origin — executes parameterized SQL, returns rows"),
                    InterfaceMethod("execute_one", ["query: str", "params: list"], "dict | None", "Ө",
                                    "Filter — returns first matching row or None"),
                    InterfaceMethod("insert", ["table: str", "data: dict"], "int", "◬",
                                    "Origin — inserts row, returns new ID"),
                ],
                allowed_dependencies=[],
                constraints=[
                    "Must use parameterized queries — no string interpolation",
                    "Must handle connection pooling internally",
                    "Must never log query parameters (may contain secrets)",
                    "Must support SQLite for dev and PostgreSQL for production",
                ],
            ),
            ModuleSpec(
                name="Auth",
                description="JWT authentication and password management.",
                methods=[
                    InterfaceMethod("register", ["email: str", "password: str"], "User", "◬",
                                    "Origin — creates new user with hashed password"),
                    InterfaceMethod("login", ["email: str", "password: str"], "Token | None", "☤",
                                    "Weaves credentials into signed JWT (or None if invalid)"),
                    InterfaceMethod("verify_token", ["token: str"], "TokenPayload | None", "𓂀",
                                    "Witnesses token validity without side effects"),
                    InterfaceMethod("hash_password", ["password: str"], "str", "☾",
                                    "Folds password through bcrypt"),
                ],
                allowed_dependencies=["Database"],
                constraints=[
                    "Tokens expire after 24 hours",
                    "Must use bcrypt with cost factor >= 12",
                    "Must never store or log plaintext passwords",
                    "Must never import Router or Middleware",
                ],
            ),
            ModuleSpec(
                name="Router",
                description="HTTP request routing, input validation, response formatting.",
                methods=[
                    InterfaceMethod("handle", ["method: str", "path: str", "headers: dict", "body: dict"],
                                    "Response", "☤",
                                    "Weaves HTTP request through auth + handlers into response"),
                    InterfaceMethod("register_route", ["method: str", "path: str", "handler: callable"],
                                    "None", "◬",
                                    "Origin — registers a route handler"),
                ],
                allowed_dependencies=["Auth", "Database"],
                constraints=[
                    "Must validate Content-Type header on POST/PUT",
                    "Must return JSON responses with proper status codes",
                    "Must handle all exceptions gracefully (never return raw 500s)",
                    "Must check Auth.verify_token for protected routes",
                ],
            ),
            ModuleSpec(
                name="Middleware",
                description="Request middleware: rate limiting, logging, CORS.",
                methods=[
                    InterfaceMethod("apply", ["request: dict"], "dict | ErrorResponse", "Ө",
                                    "Filter — passes valid requests, rejects invalid ones"),
                    InterfaceMethod("add_cors_headers", ["response: dict"], "dict", "☤",
                                    "Weaves CORS headers into response"),
                ],
                allowed_dependencies=["Auth"],
                constraints=[
                    "Rate limit: 100 requests per minute per IP",
                    "Must add X-Request-ID header to all responses",
                    "Must log request duration (but never request bodies)",
                    "Must never import Database directly",
                ],
            ),
        ],
        global_constraints=[
            "Request flow: Middleware → Router → Auth → Database (DAG, not cycle)",
            "No module may import another module's internals — interfaces only",
            "All errors must include a machine-readable error code",
            "No global mutable state between modules",
        ],
    )

_register("web-api", "Web API", "REST API with auth, DB, routing, middleware (intermediate)", _build_web_api)


# ── Template: cli-tool ──────────────────────────────────────

def _build_cli_tool(project_name: str) -> ArchitectureSpec:
    return ArchitectureSpec(
        name=f"{project_name} — CLI Tool",
        modules=[
            ModuleSpec(
                name="ArgParser",
                description="Command-line argument parsing and validation.",
                methods=[
                    InterfaceMethod("parse", ["argv: list[str]"], "ParsedArgs", "◬",
                                    "Origin — parses raw argv into typed, validated args"),
                    InterfaceMethod("print_help", [], "None", "𓂀",
                                    "Witnesses — displays help text without side effects"),
                ],
                allowed_dependencies=[],
                constraints=[
                    "Must validate all inputs and return typed ParsedArgs dataclass",
                    "Must not perform any business logic — parsing only",
                    "Must support --help, --version, --verbose flags",
                    "Must use stdlib argparse or manual parsing — no pip dependencies",
                ],
            ),
            ModuleSpec(
                name="Core",
                description="Core business logic. Pure computation — no I/O.",
                methods=[
                    InterfaceMethod("process", ["input_data: InputData", "config: Config"],
                                    "Result", "☾",
                                    "Fold — transforms input through core algorithm"),
                    InterfaceMethod("validate", ["input_data: InputData"], "list[ValidationError]", "Ө",
                                    "Filter — checks input validity, returns errors"),
                ],
                allowed_dependencies=[],
                constraints=[
                    "Must not perform any file I/O or network requests",
                    "Must not read environment variables",
                    "Must be fully testable with pure unit tests (no mocks needed)",
                    "All errors returned as values, never raised as exceptions",
                ],
            ),
            ModuleSpec(
                name="FileIO",
                description="File reading and writing. All filesystem access goes here.",
                methods=[
                    InterfaceMethod("read_input", ["file_path: str"], "InputData", "◬",
                                    "Origin — reads and deserializes input file"),
                    InterfaceMethod("write_output", ["result: Result", "output_path: str"],
                                    "None", "☾",
                                    "Fold — serializes and writes result to file"),
                    InterfaceMethod("discover_files", ["directory: str", "pattern: str"],
                                    "list[str]", "◬",
                                    "Origin — finds files matching glob pattern"),
                ],
                allowed_dependencies=[],
                constraints=[
                    "Must not contain any business logic — I/O only",
                    "Must handle encoding (UTF-8 with fallback)",
                    "Must create parent directories when writing",
                    "Must never delete files unless explicitly instructed",
                ],
            ),
            ModuleSpec(
                name="Reporter",
                description="Output formatting and progress reporting.",
                methods=[
                    InterfaceMethod("format_result", ["result: Result", "format: str"],
                                    "str", "☤",
                                    "Weaves result into formatted output (text/json/csv)"),
                    InterfaceMethod("emit_progress", ["current: int", "total: int", "label: str"],
                                    "None", "𓂀",
                                    "Witnesses progress — never modifies data"),
                ],
                allowed_dependencies=[],
                constraints=[
                    "Must support output formats: text, json, csv",
                    "Must NEVER modify result data",
                    "Must NEVER raise exceptions that halt the pipeline",
                    "Progress output goes to stderr, results to stdout",
                ],
            ),
        ],
        global_constraints=[
            "Pipeline: ArgParser → FileIO (read) → Core → FileIO (write) → Reporter",
            "No global mutable state between modules",
            "Core must have zero I/O — all file access through FileIO",
            "All inter-module communication through declared interfaces only",
        ],
    )

_register("cli-tool", "CLI Tool", "Command-line tool with I/O separation (intermediate)", _build_cli_tool)


# ── Template: data-pipeline ─────────────────────────────────

def _build_data_pipeline(project_name: str) -> ArchitectureSpec:
    return ArchitectureSpec(
        name=f"{project_name} — Data Pipeline",
        modules=[
            ModuleSpec(
                name="Ingester",
                description="Data source discovery and raw ingestion. Entry point for all pipelines.",
                methods=[
                    InterfaceMethod("discover_sources", ["config: SourceConfig"], "list[DataSource]", "◬",
                                    "Origin — discovers data sources (files, APIs, DBs)"),
                    InterfaceMethod("ingest", ["source: DataSource"], "RawDataFrame", "◬",
                                    "Origin — loads raw data from a single source"),
                    InterfaceMethod("ingest_batch", ["sources: list[DataSource]"], "list[RawDataFrame]", "☾",
                                    "Fold — iterates through all sources, ingesting each"),
                ],
                allowed_dependencies=[],
                constraints=[
                    "Must not transform or analyze data — raw loading only",
                    "Must support CSV, JSON, and Parquet formats",
                    "Must handle encoding detection automatically",
                    "Must log source metadata (size, row count) without modifying data",
                ],
            ),
            ModuleSpec(
                name="Transformer",
                description="Data cleaning, transformation, and enrichment. Pure functions.",
                methods=[
                    InterfaceMethod("clean", ["data: RawDataFrame", "rules: list[CleanRule]"],
                                    "CleanDataFrame", "Ө",
                                    "Filter — removes nulls, duplicates, invalid rows"),
                    InterfaceMethod("transform", ["data: CleanDataFrame", "transforms: list[Transform]"],
                                    "TransformedDataFrame", "☾",
                                    "Fold — applies column transforms (rename, cast, derive)"),
                    InterfaceMethod("validate", ["data: TransformedDataFrame", "schema: Schema"],
                                    "ValidationReport", "𓂀",
                                    "Witness — checks data against schema, reports violations"),
                ],
                allowed_dependencies=["Ingester"],
                constraints=[
                    "Must not perform I/O — operates on in-memory data only",
                    "Must be deterministic — same input always produces same output",
                    "Must never drop rows silently — all drops recorded in report",
                    "Must not import ML libraries",
                ],
            ),
            ModuleSpec(
                name="Loader",
                description="Writes processed data to destination stores.",
                methods=[
                    InterfaceMethod("load", ["data: TransformedDataFrame", "destination: Destination"],
                                    "LoadResult", "☾",
                                    "Fold — writes data to destination (file, DB, API)"),
                    InterfaceMethod("upsert", ["data: TransformedDataFrame", "destination: Destination",
                                               "key_columns: list[str]"],
                                    "UpsertResult", "☤",
                                    "Weaves new data with existing, updating matches"),
                ],
                allowed_dependencies=["Transformer"],
                constraints=[
                    "Must support CSV, Parquet, and SQLite output",
                    "Must handle partial failures (write what succeeded, report what failed)",
                    "Must never read from Ingester directly — only accepts transformed data",
                    "Must create destination directories/tables if they don't exist",
                ],
            ),
            ModuleSpec(
                name="Orchestrator",
                description="Pipeline scheduling, retry logic, and run tracking.",
                methods=[
                    InterfaceMethod("run_pipeline", ["config: PipelineConfig"],
                                    "PipelineResult", "☤",
                                    "Weaves Ingester → Transformer → Loader into a complete run"),
                    InterfaceMethod("retry_failed", ["run_id: str"],
                                    "PipelineResult", "☾",
                                    "Fold — retries only failed stages from a previous run"),
                ],
                allowed_dependencies=["Ingester", "Transformer", "Loader"],
                constraints=[
                    "Must log every stage start/end with timing",
                    "Must support dry-run mode (validate without loading)",
                    "Must save run history for audit (append-only JSON log)",
                    "Must never modify data — delegates all work to other modules",
                ],
            ),
            ModuleSpec(
                name="Monitor",
                description="Pipeline health monitoring and alerting. Pure observation.",
                methods=[
                    InterfaceMethod("check_health", ["run: PipelineResult"], "HealthReport", "𓂀",
                                    "Witnesses pipeline health without modifying state"),
                    InterfaceMethod("emit_metrics", ["metrics: dict"], "None", "𓂀",
                                    "Witnesses metrics — logs without altering pipeline"),
                ],
                allowed_dependencies=[],
                constraints=[
                    "Must NEVER modify pipeline data or control flow",
                    "Must NEVER raise exceptions that halt the pipeline",
                    "Must be safe to remove entirely without affecting correctness",
                    "Output: structured JSON for monitoring systems",
                ],
            ),
        ],
        global_constraints=[
            "Pipeline flow: Ingester → Transformer → Loader (DAG, not cycle)",
            "Orchestrator coordinates but never manipulates data directly",
            "Monitor observes but never modifies — pure 𓂀 Witness",
            "No global mutable state between modules",
            "All inter-module communication through declared interfaces only",
        ],
    )

_register("data-pipeline", "Data Pipeline",
          "ETL pipeline with ingest/transform/load/monitor (advanced)", _build_data_pipeline)


# ── Template: blank ─────────────────────────────────────────

def _build_blank(project_name: str) -> ArchitectureSpec:
    return ArchitectureSpec(
        name=project_name,
        modules=[
            ModuleSpec(
                name="Core",
                description="Main application logic. Edit this module's spec first.",
                methods=[
                    InterfaceMethod("run", ["config: dict"], "Result", "☾",
                                    "Fold — main entry point"),
                ],
                allowed_dependencies=[],
                constraints=[
                    "Define your constraints here",
                ],
            ),
        ],
        global_constraints=[
            "No global mutable state between modules",
            "All inter-module communication through declared interfaces only",
        ],
    )

_register("blank", "Blank Project", "Empty project with a single Core module", _build_blank)


# ─────────────────────────────────────────────────────────────
#  Spec Serialization (to/from JSON)
# ─────────────────────────────────────────────────────────────

def spec_to_json(spec: ArchitectureSpec) -> dict:
    """Serialize an ArchitectureSpec to a JSON-safe dict."""
    return {
        "name": spec.name,
        "modules": [
            {
                "name": m.name,
                "description": m.description,
                "methods": [
                    {
                        "name": method.name,
                        "inputs": method.inputs,
                        "output": method.output,
                        "glyph": method.glyph,
                        "constraint": method.constraint,
                    }
                    for method in (m.methods or [])
                ],
                "allowed_dependencies": m.allowed_dependencies or [],
                "constraints": m.constraints or [],
            }
            for m in spec.modules
        ],
        "global_constraints": spec.global_constraints or [],
    }


def spec_from_json(data: dict) -> ArchitectureSpec:
    """Deserialize an ArchitectureSpec from a JSON dict."""
    modules = []
    for m in data.get("modules", []):
        methods = []
        for method in m.get("methods", []):
            methods.append(InterfaceMethod(
                name=method["name"],
                inputs=method.get("inputs", []),
                output=method.get("output", "None"),
                glyph=method.get("glyph", "☾"),
                constraint=method.get("constraint", ""),
            ))
        modules.append(ModuleSpec(
            name=m["name"],
            description=m.get("description", ""),
            methods=methods,
            allowed_dependencies=m.get("allowed_dependencies", []),
            constraints=m.get("constraints", []),
        ))
    return ArchitectureSpec(
        name=data.get("name", "Untitled"),
        modules=modules,
        global_constraints=data.get("global_constraints", []),
    )


def load_spec(spec_path: str) -> ArchitectureSpec:
    """Load an ArchitectureSpec from a nexus.json file."""
    with open(spec_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return spec_from_json(data)


# ─────────────────────────────────────────────────────────────
#  Project Scaffolding
# ─────────────────────────────────────────────────────────────

def list_templates() -> list[dict]:
    """Return available templates as a list of {name, display, description}."""
    return [
        {"name": k, "display": v["display"], "description": v["description"]}
        for k, v in TEMPLATES.items()
    ]


def init_project(
    project_dir: str,
    project_name: str,
    template: str = "blank",
    from_codebase: str = None,
    provider_name: str = "gemini",
    api_key: str = "",
    model: str = "",
) -> str:
    """
    Initialize a new Nexus project.

    Args:
        project_dir: Directory to create the project in.
        project_name: Human-readable project name.
        template: Template name (e.g., 'todo-app', 'web-api', 'cli-tool').
        from_codebase: Path to existing codebase to decompose via forge.py.
        provider_name: AI provider for forge decomposition.
        api_key: API key for forge decomposition.
        model: AI model for forge decomposition.

    Returns:
        Path to the created nexus.json spec file.
    """
    project_path = Path(project_dir).resolve()
    project_path.mkdir(parents=True, exist_ok=True)

    spec_file = project_path / "nexus.json"
    session_dir = project_path / ".nexus_session"
    src_dir = project_path / "src"
    tests_dir = project_path / "tests"

    # ── Route 1: From existing codebase (forge) ──
    if from_codebase:
        spec = _forge_decompose(from_codebase, project_name, provider_name, api_key, model)
    # ── Route 2: From template ──
    elif template in TEMPLATES:
        builder = TEMPLATES[template]["builder"]
        spec = builder(project_name)
    else:
        available = ", ".join(TEMPLATES.keys())
        raise ValueError(f"Unknown template: '{template}'. Available: {available}")

    # ── Write project files ──
    # nexus.json — the architecture spec
    with open(spec_file, "w", encoding="utf-8") as f:
        json.dump(spec_to_json(spec), f, indent=2, ensure_ascii=False)

    # Create directories
    session_dir.mkdir(exist_ok=True)
    src_dir.mkdir(exist_ok=True)
    tests_dir.mkdir(exist_ok=True)

    # .gitignore for session data
    gitignore = project_path / ".gitignore"
    if not gitignore.exists():
        with open(gitignore, "w", encoding="utf-8") as f:
            f.write("# Nexus session data\n.nexus_session/\n__pycache__/\n*.pyc\n")

    # README stub
    readme = project_path / "README.md"
    if not readme.exists():
        module_list = "\n".join(f"  - **{m.name}** — {m.description}" for m in spec.modules)
        with open(readme, "w", encoding="utf-8") as f:
            f.write(f"# {project_name}\n\n")
            f.write(f"Built with [Nexus](https://github.com/dparksports/Project-Chevron) — "
                    f"SCP-powered AI development.\n\n")
            f.write(f"## Modules\n\n{module_list}\n\n")
            f.write(f"## Getting Started\n\n")
            f.write(f"```bash\n")
            f.write(f"# See the architecture\n")
            f.write(f"python -m nexus.cli overview --spec nexus.json\n\n")
            f.write(f"# Generate code for a module\n")
            f.write(f"python -m nexus.cli generate {spec.modules[0].name} "
                    f"--provider gemini --key YOUR_KEY --spec nexus.json\n")
            f.write(f"```\n")

    # Per-module stub files in src/
    for mod in spec.modules:
        mod_file = src_dir / f"{mod.name.lower()}.py"
        if not mod_file.exists():
            methods_stub = ""
            for m in (mod.methods or []):
                params = ", ".join(m.inputs)
                methods_stub += f"\n    def {m.name}(self, {params}) -> {m.output}:\n"
                methods_stub += f"        \"{m.constraint}\"\n"
                methods_stub += f"        raise NotImplementedError\n"

            with open(mod_file, "w", encoding="utf-8") as f:
                f.write(f'"""\n{mod.name} — {mod.description}\n')
                f.write(f'SCP Glyph: {", ".join(set(m.glyph for m in mod.methods))}\n')
                deps = ", ".join(mod.allowed_dependencies) if mod.allowed_dependencies else "none"
                f.write(f'Dependencies: {deps}\n')
                f.write(f'"""\n\n')
                f.write(f"class {mod.name}:{methods_stub}\n")

    # Per-module test stubs in tests/
    for mod in spec.modules:
        test_file = tests_dir / f"test_{mod.name.lower()}.py"
        if not test_file.exists():
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(f'"""Tests for {mod.name} — generated from SCP contract."""\n\n')
                f.write(f"import unittest\n")
                f.write(f"from src.{mod.name.lower()} import {mod.name}\n\n\n")
                f.write(f"class Test{mod.name}(unittest.TestCase):\n")
                for m in (mod.methods or []):
                    f.write(f"    def test_{m.name}(self):\n")
                    f.write(f"        \"\"\"Verify {m.constraint}\"\"\"\n")
                    f.write(f"        # TODO: implement test from SCP contract\n")
                    f.write(f"        self.fail('Not yet implemented')\n\n")
                f.write(f"\nif __name__ == '__main__':\n")
                f.write(f"    unittest.main()\n")

    return str(spec_file)


def _forge_decompose(
    codebase_path: str,
    project_name: str,
    provider_name: str = "gemini",
    api_key: str = "",
    model: str = "",
) -> ArchitectureSpec:
    """Use forge.py to auto-decompose an existing codebase into SCP modules."""
    try:
        from forge import scan_codebase, build_decomposition_prompt, call_gemini, parse_decomposition
    except ImportError:
        raise ImportError(
            "forge.py not found. Make sure you're running from the chevron/ directory.\n"
            "  cd chevron && python -m nexus.cli init myapp --from ./existing_code"
        )

    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Auto-decomposition requires an API key.\n"
            "  Set GEMINI_API_KEY environment variable or pass --key YOUR_KEY"
        )

    if not model:
        model = "gemini-2.5-pro"

    print(f"◬ ─── Scanning codebase: {codebase_path} ───")
    scan = scan_codebase(codebase_path)
    print(f"  Found {len(scan.files)} files, ~{scan.total_tokens:,} tokens")

    print(f"\n☤ ─── Decomposing with {model} ───")
    prompt = build_decomposition_prompt(scan)
    response = call_gemini(prompt, model)
    decomp = parse_decomposition(response)

    print(f"  Proposed {len(decomp.get('modules', []))} modules")

    # Convert forge output to ArchitectureSpec
    modules = []
    for m in decomp.get("modules", []):
        methods = []
        for method in m.get("methods", []):
            methods.append(InterfaceMethod(
                name=method.get("name", "unknown"),
                inputs=method.get("inputs", []),
                output=method.get("output", "None"),
                glyph=method.get("glyph", "☾"),
                constraint=method.get("constraint", ""),
            ))
        modules.append(ModuleSpec(
            name=m.get("name", "Unknown"),
            description=m.get("description", ""),
            methods=methods,
            allowed_dependencies=m.get("allowed_dependencies", []),
            constraints=m.get("constraints", []),
        ))

    return ArchitectureSpec(
        name=project_name or decomp.get("name", "Decomposed Project"),
        modules=modules,
        global_constraints=decomp.get("global_constraints", []),
    )


# ─────────────────────────────────────────────────────────────
#  Display Helpers
# ─────────────────────────────────────────────────────────────

def print_templates():
    """Print available templates in a formatted table."""
    print("\n◬ ─── Available Project Templates ───\n")
    for name, info in TEMPLATES.items():
        print(f"  {name:<18} {info['description']}")
    print()
    print("  Usage:")
    print("    python -m nexus.cli init myapp --template todo-app")
    print("    python -m nexus.cli init myapp --from ./existing_code")
    print()


def print_project_summary(spec: ArchitectureSpec, project_dir: str):
    """Print a summary of the initialized project."""
    print(f"\n✔ ─── Project Initialized: {spec.name} ───\n")
    print(f"  Directory: {project_dir}/")
    print(f"  Spec:      nexus.json")
    print(f"  Modules:   {len(spec.modules)}")
    print()

    for mod in spec.modules:
        glyphs = " ".join(sorted(set(m.glyph for m in (mod.methods or []))))
        deps = ", ".join(mod.allowed_dependencies) if mod.allowed_dependencies else "—"
        print(f"    {glyphs:<4} {mod.name:<20} deps: {deps}")

    print(f"\n  Next steps:")
    first_mod = spec.modules[0].name if spec.modules else "Core"
    print(f"    1. Review the architecture:  python -m nexus.cli overview --spec {project_dir}/nexus.json")
    print(f"    2. Edit nexus.json to refine modules, methods, and constraints")
    print(f"    3. Generate code:            python -m nexus.cli generate {first_mod} --provider gemini --key YOUR_KEY")
    print(f"    4. Check session health:     python -m nexus.cli health")
    print()
