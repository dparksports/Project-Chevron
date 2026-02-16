"""
Nexus Test Suite — Conductor / Orchestrator (Layer 3)
======================================================
Tests for the Planner, Executor, Verifier, and Conductor.
Uses mock providers to avoid API calls during testing.

Usage:
    python -m pytest tests/test_conductor.py -v
    python tests/test_conductor.py
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus.conductor import (
    Conductor, Planner, Executor, Verifier,
    ModuleTask, EditResult, VerificationReport, Plan,
)
from nexus.context_kernel import EntropyScorer, ContextPruner, SCPRetriever, ContextItem
from nexus.session_protocol import SessionState, EditEntry
from nexus.providers.base import BaseProvider, ProviderConfig, ProviderResponse


# ─────────────────────────────────────────────
#  Mock Provider
# ─────────────────────────────────────────────

class MockProvider(BaseProvider):
    """Mock AI provider that returns predictable responses."""

    def __init__(self, response_text="def hello(): pass", should_fail=False):
        super().__init__(ProviderConfig(provider_name="mock"))
        self.response_text = response_text
        self.should_fail = should_fail
        self.call_count = 0
        self.last_prompt = ""
        self.last_system = ""

    def generate(self, prompt: str, system_instruction: str = "") -> ProviderResponse:
        self.call_count += 1
        self.last_prompt = prompt
        self.last_system = system_instruction

        if self.should_fail:
            return ProviderResponse(content="", provider="mock", error="Mock failure")

        return ProviderResponse(
            content=self.response_text,
            provider="mock",
            model="mock-model",
            tokens_used=100,
        )

    def is_available(self) -> bool:
        return True


def _make_test_architecture():
    """Create a test architecture, or skip if scp_bridge unavailable."""
    try:
        from scp_bridge import ArchitectureSpec, ModuleSpec, InterfaceMethod
    except ImportError:
        return None

    return ArchitectureSpec(
        name="TestApp",
        modules=[
            ModuleSpec(
                name="Store",
                description="Data storage module",
                methods=[
                    InterfaceMethod(name="save", inputs=["data"], output="bool",
                                    glyph="◬", constraint="No network"),
                    InterfaceMethod(name="load", inputs=["key"], output="dict",
                                    glyph="Ө", constraint="Filter by key"),
                ],
                allowed_dependencies=["Logger"],
                constraints=["No network access", "Pure data operations"],
            ),
            ModuleSpec(
                name="API",
                description="HTTP API layer",
                methods=[
                    InterfaceMethod(name="handle_request", inputs=["req"],
                                    output="response", glyph="☤"),
                ],
                allowed_dependencies=["Store"],
                constraints=["Must validate all input"],
            ),
            ModuleSpec(
                name="Logger",
                description="Logging module",
                methods=[
                    InterfaceMethod(name="log", inputs=["msg"], output="None",
                                    glyph="𓂀", constraint="Observe only"),
                ],
                allowed_dependencies=[],
                constraints=["Must NEVER modify data"],
            ),
        ],
        global_constraints=["No global mutable state"],
    )


# ─────────────────────────────────────────────
#  Planner Tests
# ─────────────────────────────────────────────

class TestPlanner(unittest.TestCase):

    def test_explicit_modules(self):
        session = SessionState()
        planner = Planner(session)
        plan = planner.decompose("Add logging", target_modules=["Store", "API"])
        self.assertEqual(plan.module_count, 2)
        self.assertEqual(plan.tasks[0].module_name, "Store")
        self.assertEqual(plan.tasks[1].module_name, "API")

    def test_no_architecture_single_task(self):
        session = SessionState()
        planner = Planner(session)
        plan = planner.decompose("Do something")
        self.assertEqual(plan.module_count, 1)
        self.assertEqual(plan.tasks[0].module_name, "default")

    def test_heuristic_decompose(self):
        arch = _make_test_architecture()
        if arch is None:
            self.skipTest("scp_bridge not available")

        session = SessionState(architecture=arch)
        planner = Planner(session)
        plan = planner.decompose("Fix the Store module save function")
        module_names = [t.module_name for t in plan.tasks]
        self.assertIn("Store", module_names)


# ─────────────────────────────────────────────
#  Executor Tests
# ─────────────────────────────────────────────

class TestExecutor(unittest.TestCase):

    def test_successful_execution(self):
        session = SessionState()
        provider = MockProvider(response_text="def save(data): return True")
        executor = Executor(session, provider)

        task = ModuleTask(module_name="Store", description="Implement save")
        result = executor.execute(task)

        self.assertTrue(result.success)
        self.assertIn("save", result.code_generated)
        self.assertEqual(provider.call_count, 1)

    def test_failed_execution(self):
        session = SessionState()
        provider = MockProvider(should_fail=True)
        executor = Executor(session, provider)

        task = ModuleTask(module_name="Store", description="Implement save")
        result = executor.execute(task)

        self.assertFalse(result.success)
        self.assertEqual(result.error, "Mock failure")

    def test_code_extraction_from_fenced(self):
        """Should extract code from markdown fences."""
        session = SessionState()
        provider = MockProvider(response_text="```python\ndef save(data):\n    return True\n```")
        executor = Executor(session, provider)

        task = ModuleTask(module_name="Store", description="Implement save")
        result = executor.execute(task)

        self.assertNotIn("```", result.code_generated)
        self.assertIn("def save", result.code_generated)

    def test_prompt_includes_forbidden_warning(self):
        """System prompt should warn about forbidden modules."""
        arch = _make_test_architecture()
        if arch is None:
            self.skipTest("scp_bridge not available")

        # Add forbidden to API module
        arch.modules[1].forbidden = ["Logger"]

        session = SessionState(architecture=arch)
        retriever = SCPRetriever(architecture=arch)
        provider = MockProvider()
        executor = Executor(session, provider, retriever=retriever)

        task = ModuleTask(module_name="API", description="Implement API")
        result = executor.execute(task)

        # System prompt should contain forbidden warning
        self.assertIn("FORBIDDEN", provider.last_system)
        self.assertIn("Logger", provider.last_system)


# ─────────────────────────────────────────────
#  Verifier Tests
# ─────────────────────────────────────────────

class TestVerifier(unittest.TestCase):

    def test_failed_edit_reports_failure(self):
        session = SessionState()
        verifier = Verifier(session)

        edit = EditResult(module_name="Store", code_generated="", success=False,
                          error="Generation failed", prompt_used="")
        report = verifier.verify(edit)

        self.assertFalse(report.passed)
        self.assertGreater(len(report.violations), 0)

    def test_clean_code_passes(self):
        session = SessionState()
        verifier = Verifier(session)  # No provider = skip AI verification

        edit = EditResult(
            module_name="Store",
            code_generated="def save(data):\n    return True",
            success=True,
            prompt_used="",
        )
        report = verifier.verify(edit)
        self.assertTrue(report.passed)

    def test_global_state_detected(self):
        session = SessionState()
        verifier = Verifier(session)

        edit = EditResult(
            module_name="Store",
            code_generated="def save(data):\n    global counter\n    counter += 1",
            success=True,
            prompt_used="",
        )
        report = verifier.verify(edit)
        self.assertFalse(report.weaver_passed)
        self.assertTrue(any("Global state" in v for v in report.violations))

    def test_forbidden_import_detected(self):
        arch = _make_test_architecture()
        if arch is None:
            self.skipTest("scp_bridge not available")

        arch.modules[0].forbidden = ["API"]  # Store cannot import API

        session = SessionState(architecture=arch)
        verifier = Verifier(session)

        edit = EditResult(
            module_name="Store",
            code_generated="from api import handle_request\ndef save(data): pass",
            success=True,
            prompt_used="",
        )
        report = verifier.verify(edit)
        self.assertFalse(report.weaver_passed)
        self.assertTrue(any("Forbidden" in v for v in report.violations))


# ─────────────────────────────────────────────
#  Conductor Tests (Integration)
# ─────────────────────────────────────────────

class TestConductor(unittest.TestCase):

    def test_full_pipeline(self):
        """End-to-end: request → plan → execute → verify → record."""
        arch = _make_test_architecture()
        if arch is None:
            self.skipTest("scp_bridge not available")

        session = SessionState(architecture=arch)
        provider = MockProvider(response_text="def save(data):\n    return True")
        output_lines = []

        conductor = Conductor(
            session, provider,
            output_fn=lambda s: output_lines.append(s),
        )

        results = conductor.handle_request(
            "Implement the Store module",
            target_modules=["Store"],
        )

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["success"])
        self.assertGreater(session.ledger.total_edits, 0)

    def test_multi_module_request(self):
        arch = _make_test_architecture()
        if arch is None:
            self.skipTest("scp_bridge not available")

        session = SessionState(architecture=arch)
        provider = MockProvider(response_text="def handler(): pass")

        conductor = Conductor(session, provider, output_fn=lambda s: None)

        results = conductor.handle_request(
            "Add error handling",
            target_modules=["Store", "API"],
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(session.ledger.total_edits, 2)

    def test_context_report(self):
        arch = _make_test_architecture()
        if arch is None:
            self.skipTest("scp_bridge not available")

        session = SessionState(architecture=arch)
        session.update_code("Store", "def save(data): return True")

        provider = MockProvider()
        conductor = Conductor(session, provider, output_fn=lambda s: None)

        report = conductor.get_context_report("Store")
        self.assertIn("module", report)
        self.assertIn("compression_ratio", report)
        self.assertEqual(report["module"], "Store")

    def test_session_health_updates(self):
        arch = _make_test_architecture()
        if arch is None:
            self.skipTest("scp_bridge not available")

        session = SessionState(architecture=arch)
        provider = MockProvider()
        conductor = Conductor(session, provider, output_fn=lambda s: None)

        conductor.handle_request("Build it", target_modules=["Store"])

        health = session.session_health
        self.assertGreater(health["total_edits"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
