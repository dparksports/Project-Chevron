"""
Nexus Dashboard Server — Web Interface for the SCP-Powered AI IDE
==================================================================
FastAPI application that wraps the Nexus CLI into a visual dashboard.

Launch:
    python -m nexus.server          # Direct
    python -m nexus.cli start       # Via CLI

Endpoints:
    GET  /                          → Dashboard SPA
    GET  /api/templates             → Available project templates
    POST /api/init                  → Create a new project
    GET  /api/architecture          → Current architecture spec
    GET  /api/context/{module}      → Context report for a module
    GET  /api/health                → Session health metrics
    GET  /api/providers             → Available AI providers
    WS   /ws/generate               → Stream generation progress
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import webbrowser
import threading

# Add parent dir for chevron imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

from nexus.scaffold import (
    init_project, list_templates as get_templates, load_spec,
    spec_to_json, spec_from_json,
)
from nexus.session_protocol import SessionState
from nexus.conductor import Conductor, Planner
from nexus.context_kernel import EntropyScorer, ContextPruner, SCPRetriever
from nexus.providers.base import BaseProvider, ProviderConfig, ProviderResponse
from nexus.providers.registry import get_provider, list_providers


# ─────────────────────────────────────────────────────────────
#  App Setup
# ─────────────────────────────────────────────────────────────

app = FastAPI(title="Nexus Dashboard", version="1.0.0")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global state — managed per project directory
_state = {
    "project_dir": None,
    "spec": None,
    "session": None,
    "conductor": None,
}


# ─────────────────────────────────────────────────────────────
#  Request / Response Models
# ─────────────────────────────────────────────────────────────

class InitRequest(BaseModel):
    name: str
    template: str = "blank"
    output: Optional[str] = None


class GenerateRequest(BaseModel):
    request: str = "Implement the module following SCP constraints"
    module: Optional[str] = None
    all_modules: bool = False
    provider: str = "gemini"
    api_key: str = ""
    model: str = ""


# ─────────────────────────────────────────────────────────────
#  Helper — Load/Refresh State
# ─────────────────────────────────────────────────────────────

def _load_project(project_dir: str = None):
    """Load or reload the project from a directory."""
    if project_dir:
        _state["project_dir"] = os.path.abspath(project_dir)

    pdir = _state["project_dir"] or os.getcwd()
    spec_path = os.path.join(pdir, "nexus.json")

    if os.path.exists(spec_path):
        _state["spec"] = load_spec(spec_path)
        session_dir = os.path.join(pdir, ".nexus_session")
        _state["session"] = SessionState(
            architecture=_state["spec"],
            session_dir=session_dir,
        )
        return True
    return False


def _ensure_project():
    """Ensure a project is loaded; raise 404 if not."""
    if not _state["spec"]:
        if not _load_project():
            raise HTTPException(
                status_code=404,
                detail="No nexus.json found. Initialize a project first."
            )


def _get_conductor(provider_name: str = "gemini", api_key: str = "", model: str = ""):
    """Get or create a Conductor instance."""
    _ensure_project()

    config = ProviderConfig(
        provider_name=provider_name,
        api_key=api_key or os.environ.get(
            {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY",
             "anthropic": "ANTHROPIC_API_KEY"}.get(provider_name, "API_KEY"), ""
        ),
        model=model,
    )

    try:
        provider = get_provider(config)
    except (ValueError, ImportError):
        # Use a mock provider so dashboard still works without API keys
        provider = _MockProvider(config)

    bridge = None
    try:
        from scp_bridge import SCPBridge
        bridge = SCPBridge(_state["spec"])
    except ImportError:
        pass

    conductor = Conductor(
        _state["session"], provider, bridge=bridge
    )
    _state["conductor"] = conductor
    return conductor


class _MockProvider(BaseProvider):
    """Fallback provider when no AI SDK is configured."""
    def generate(self, prompt: str, system_instruction: str = ""):
        return ProviderResponse(
            content="# [Mock] No AI provider configured.\n# Install one: pip install google-genai",
            provider="mock", model="none",
        )
    def is_available(self):
        return True


# ─────────────────────────────────────────────────────────────
#  Routes — Pages
# ─────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    """Serve the dashboard SPA."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# ─────────────────────────────────────────────────────────────
#  Routes — REST API
# ─────────────────────────────────────────────────────────────

@app.get("/api/templates")
async def api_templates():
    """Return available project templates."""
    templates = get_templates()
    return JSONResponse(templates)


@app.post("/api/init")
async def api_init(req: InitRequest):
    """Initialize a new Nexus project."""
    output_dir = req.output or f"./{req.name}"
    try:
        spec_file = init_project(
            project_dir=output_dir,
            project_name=req.name,
            template=req.template,
        )
        _load_project(os.path.dirname(spec_file))
        spec = load_spec(spec_file)
        return JSONResponse({
            "success": True,
            "project_dir": os.path.abspath(output_dir),
            "spec": spec_to_json(spec),
        })
    except (ValueError, ImportError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/architecture")
async def api_architecture():
    """Return the current architecture spec as JSON."""
    _ensure_project()
    spec = _state["spec"]
    data = spec_to_json(spec)

    # Enrich with module statuses from session
    session = _state["session"]
    for mod_data in data.get("modules", []):
        name = mod_data["name"]
        mod_data["is_frozen"] = session.contract_cache.is_frozen(name)
        mod_data["has_code"] = name in session.code_store and bool(session.code_store[name])
        edits = session.ledger.edits_for_module(name)
        mod_data["edit_count"] = len(edits)
        mod_data["last_verified"] = edits[-1].verified if edits else None

    return JSONResponse(data)


@app.get("/api/context/{module_name}")
async def api_context(module_name: str):
    """Return the context report for a specific module."""
    conductor = _get_conductor()
    try:
        report = conductor.get_context_report(module_name)
        return JSONResponse(report)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/health")
async def api_health():
    """Return session health metrics."""
    _ensure_project()
    health = _state["session"].session_health
    return JSONResponse(health)


@app.get("/api/providers")
async def api_providers():
    """Return available AI providers."""
    providers = list_providers()
    return JSONResponse({"providers": providers})


# ─────────────────────────────────────────────────────────────
#  Routes — WebSocket (Generation Streaming)
# ─────────────────────────────────────────────────────────────

@app.websocket("/ws/generate")
async def ws_generate(websocket: WebSocket):
    """Stream generation progress via WebSocket.

    Client sends:
        {"request": "Add delete", "module": "TodoStore", "provider": "gemini", "api_key": "..."}

    Server streams:
        {"type": "plan", "data": {...}}
        {"type": "executing", "module": "TodoStore"}
        {"type": "code", "module": "TodoStore", "code": "..."}
        {"type": "verified", "module": "TodoStore", "passed": true, "violations": []}
        {"type": "health", "data": {...}}
        {"type": "done"}
        {"type": "error", "message": "..."}
    """
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_json()
            request_text = data.get("request", "Implement the module")
            module = data.get("module")
            all_modules = data.get("all_modules", False)
            provider_name = data.get("provider", "gemini")
            api_key = data.get("api_key", "")
            model = data.get("model", "")

            try:
                conductor = _get_conductor(provider_name, api_key, model)

                target = None
                if module:
                    target = [module]
                elif not all_modules:
                    target = None  # Let planner decide

                # Phase 1: Plan
                planner = conductor.planner
                plan = planner.decompose(request_text, target_modules=target)
                await websocket.send_json({
                    "type": "plan",
                    "data": {
                        "request": plan.request,
                        "modules": [t.module_name for t in plan.tasks],
                        "count": plan.module_count,
                    }
                })

                # Phase 2: Execute + Verify each module
                for task in plan.tasks:
                    await websocket.send_json({
                        "type": "executing",
                        "module": task.module_name,
                    })

                    # Run in thread pool (blocking AI call)
                    result = await asyncio.to_thread(
                        conductor.executor.execute, task
                    )

                    await websocket.send_json({
                        "type": "code",
                        "module": task.module_name,
                        "code": result.code_generated if result.success else "",
                        "success": result.success,
                        "error": result.error,
                    })

                    if result.success:
                        # Verify
                        report = await asyncio.to_thread(
                            conductor.verifier.verify, result
                        )
                        await websocket.send_json({
                            "type": "verified",
                            "module": task.module_name,
                            "passed": report.passed,
                            "violations": report.violations,
                        })

                # Phase 3: Final health
                health = _state["session"].session_health
                await websocket.send_json({
                    "type": "health",
                    "data": health,
                })

                await websocket.send_json({"type": "done"})

            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })

    except WebSocketDisconnect:
        pass


# ─────────────────────────────────────────────────────────────
#  Startup
# ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Try to load project from current directory on startup."""
    _load_project()


def run_server(port: int = 3000, project_dir: str = None, open_browser: bool = True):
    """Launch the Nexus Dashboard server."""
    import uvicorn

    if project_dir:
        _state["project_dir"] = os.path.abspath(project_dir)

    if open_browser:
        def _open():
            import time
            time.sleep(1.5)
            webbrowser.open(f"http://localhost:{port}")
        threading.Thread(target=_open, daemon=True).start()

    print(f"\n◬ ─── Nexus Dashboard ───")
    print(f"  http://localhost:{port}")
    print(f"  Press Ctrl+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")


if __name__ == "__main__":
    run_server()
