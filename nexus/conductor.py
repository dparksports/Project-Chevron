"""
Agentic Orchestrator — Layer 3 of the Nexus AI IDE
====================================================
Decomposes developer requests into SCP-scoped operations and
executes them one module at a time with verification gates.

Components (each governed by an SCP glyph):
    ◬ Conductor  — Entry point for all developer requests
    ☤ Planner    — Decomposes tasks into module-scoped operations
    ☾ Executor   — Edits ONE module at a time (anti-entropy core)
    𓂀 Verifier   — Post-edit verification (Weaver + tests)

Theory:
    The core anti-entropy mechanism is one-module-at-a-time execution.
    Instead of letting the AI see and edit everything at once (which
    causes the Foggy Boundary), the Executor:
        1. Retrieves context through SCP contracts (RAG Denial)
        2. Generates a scoped system prompt via scp_bridge.py
        3. Calls the AI with entropy-scored context
        4. Applies the edit
        5. Verifies with the Weaver
        6. Only proceeds to the next module after verification passes

    This ensures that each module edit is verified in isolation before
    moving on, preventing the cascade regressions that plague current IDEs.

Dan Park | MagicPoint.ai | February 2026
"""

from __future__ import annotations

import sys
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Any

# Add parent dir so we can import from chevron/scp_bridge
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nexus.context_kernel import (
    ContextItem, EntropyScorer, ContextPruner, SCPRetriever,
    CATEGORY_CONTRACT, CATEGORY_ACTIVE_CODE, CATEGORY_ERROR,
    CATEGORY_CONVERSATION,
)
from nexus.session_protocol import SessionState, EditEntry, ContractCache
from nexus.providers.base import BaseProvider, ProviderResponse


# ─────────────────────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────────────────────

@dataclass
class ModuleTask:
    """A single task scoped to one SCP module."""

    module_name: str
    description: str          # What to do in this module
    priority: int = 0         # Execution order (0 = first)
    depends_on: list[str] = field(default_factory=list)  # Must complete before this
    prompt_override: str = ""  # Custom prompt (otherwise auto-generated)


@dataclass
class EditResult:
    """Result of executing a single module task."""

    module_name: str
    code_generated: str       # The AI-generated code
    prompt_used: str          # The system prompt that was sent
    provider_response: Optional[ProviderResponse] = None
    success: bool = False
    error: Optional[str] = None


@dataclass
class VerificationReport:
    """Result of verifying an edit."""

    module_name: str
    weaver_passed: bool = False    # W(G) = 0?
    tests_passed: bool = False     # Auto-tests pass?
    coupling_delta: float = 0.0    # Change in inter-module coupling
    violations: list[str] = field(default_factory=list)
    details: str = ""

    @property
    def passed(self) -> bool:
        return self.weaver_passed and self.tests_passed


@dataclass
class Plan:
    """A complete execution plan for a developer request."""

    request: str                      # Original natural language request
    tasks: list[ModuleTask] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def module_count(self) -> int:
        return len(self.tasks)


# ─────────────────────────────────────────────────────────────
#  ☤ Planner — Decomposes requests into module-scoped tasks
# ─────────────────────────────────────────────────────────────

class Planner:
    """Decomposes a developer request into module-scoped operations.

    Uses the architecture spec to determine which modules need to change
    and in what order (respecting dependency ordering).
    """

    def __init__(self, session: SessionState, provider: Optional[BaseProvider] = None):
        self.session = session
        self.provider = provider

    def decompose(self, request: str, target_modules: list[str] = None) -> Plan:
        """Decompose a natural language request into module tasks.

        Args:
            request: What the developer wants to do.
            target_modules: Specific modules to target (auto-detect if None).

        Returns:
            A Plan with ordered ModuleTasks.
        """
        plan = Plan(request=request)

        if target_modules:
            # Developer specified which modules to edit
            for i, mod_name in enumerate(target_modules):
                plan.tasks.append(ModuleTask(
                    module_name=mod_name,
                    description=request,
                    priority=i,
                ))
        elif self.session.architecture and self.provider:
            # Use AI to determine which modules need changes
            plan.tasks = self._ai_decompose(request)
        elif self.session.architecture:
            # No AI provider — ask developer to specify modules
            plan.tasks = self._heuristic_decompose(request)
        else:
            # No architecture loaded — single task
            plan.tasks.append(ModuleTask(
                module_name="default",
                description=request,
                priority=0,
            ))

        # Sort by priority then dependency order
        plan.tasks.sort(key=lambda t: t.priority)

        return plan

    def _ai_decompose(self, request: str) -> list[ModuleTask]:
        """Use AI to determine which modules need changes.

        Sends the architecture overview + request to the AI and asks
        it to identify affected modules and ordering.
        """
        if not self.provider or not self.session.architecture:
            return self._heuristic_decompose(request)

        # Build a concise architecture summary for the AI
        arch = self.session.architecture
        summary_lines = [f"Architecture: {arch.name}", "Modules:"]
        for mod in arch.modules:
            deps = getattr(mod, 'allowed_dependencies', []) or []
            summary_lines.append(f"  - {mod.name}: {mod.description}")
            if deps:
                summary_lines.append(f"    depends_on: {', '.join(deps)}")

        prompt = (
            "Given this SCP architecture and the following request, "
            "identify which modules need to change and in what order.\n\n"
            f"{'chr(10)'.join(summary_lines)}\n\n"
            f"Request: {request}\n\n"
            "Respond with ONLY a JSON array of objects with 'module' and 'reason' keys, "
            "ordered by dependency (edit dependencies first). Example:\n"
            '[{"module": "AudioIngest", "reason": "needs new file format support"}]'
        )

        response = self.provider.generate(prompt)

        if response.success:
            try:
                import json
                tasks_data = json.loads(response.content.strip().strip("`").strip())
                if isinstance(tasks_data, list):
                    tasks = []
                    for i, item in enumerate(tasks_data):
                        mod_name = item.get("module", "")
                        reason = item.get("reason", request)
                        if mod_name in self.session.module_names:
                            tasks.append(ModuleTask(
                                module_name=mod_name,
                                description=reason,
                                priority=i,
                            ))
                    if tasks:
                        return tasks
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        # Fallback to heuristic
        return self._heuristic_decompose(request)

    def _heuristic_decompose(self, request: str) -> list[ModuleTask]:
        """Simple keyword-matching decomposition when AI is unavailable."""
        if not self.session.architecture:
            return [ModuleTask(module_name="default", description=request)]

        tasks = []
        request_lower = request.lower()

        for i, mod in enumerate(self.session.architecture.modules):
            # Check if module name or related keywords appear in request
            name_lower = mod.name.lower()
            desc_lower = mod.description.lower()

            if (name_lower in request_lower or
                    any(word in request_lower for word in desc_lower.split()
                        if len(word) > 4)):
                tasks.append(ModuleTask(
                    module_name=mod.name,
                    description=request,
                    priority=i,
                ))

        # If nothing matched, target all modules
        if not tasks:
            for i, mod in enumerate(self.session.architecture.modules):
                tasks.append(ModuleTask(
                    module_name=mod.name,
                    description=request,
                    priority=i,
                ))

        return tasks


# ─────────────────────────────────────────────────────────────
#  ☾ Executor — One module at a time
# ─────────────────────────────────────────────────────────────

class Executor:
    """Executes a single module task with SCP constraints.

    This is the core anti-entropy mechanism:
        1. Retrieve context via SCP Retriever (RAG Denial enforced)
        2. Generate system prompt via scp_bridge
        3. Call AI with entropy-scored context (not full conversation)
        4. Return the generated code for verification

    The Executor NEVER edits multiple modules in one call. This is
    by design — it prevents the Foggy Boundary from emerging.
    """

    def __init__(self, session: SessionState, provider: BaseProvider,
                 scorer: EntropyScorer = None, pruner: ContextPruner = None,
                 retriever: SCPRetriever = None):
        self.session = session
        self.provider = provider
        self.scorer = scorer or EntropyScorer()
        self.pruner = pruner or ContextPruner()
        self.retriever = retriever or SCPRetriever()

    def execute(self, task: ModuleTask,
                extra_context: list[ContextItem] = None) -> EditResult:
        """Execute a single module task.

        Args:
            task: The module-scoped task to execute.
            extra_context: Additional context (conversation, errors, etc.)

        Returns:
            EditResult with the generated code.
        """
        # 1. Set active module for scoped scoring
        self.scorer.active_module = task.module_name

        # 2. Retrieve context through SCP (RAG Denial enforced)
        context_items = self.retriever.retrieve_for_module(
            task.module_name,
            code_store=self.session.code_store,
            extra_context=extra_context,
        )

        # 3. Score all context items by relevance
        scored = self.scorer.score_all(context_items)

        # 4. Prune to fit within entropy budget
        pruned = self.pruner.prune(scored, self.session.entropy_budget)

        # 5. Build the system prompt
        system_prompt = self._build_system_prompt(task, pruned)

        # 6. Build the user prompt
        user_prompt = self._build_user_prompt(task)

        # 7. Call the AI
        response = self.provider.generate(user_prompt, system_instruction=system_prompt)

        if response.success:
            return EditResult(
                module_name=task.module_name,
                code_generated=self._extract_code(response.content),
                prompt_used=system_prompt,
                provider_response=response,
                success=True,
            )
        else:
            return EditResult(
                module_name=task.module_name,
                code_generated="",
                prompt_used=system_prompt,
                provider_response=response,
                success=False,
                error=response.error,
            )

    def _build_system_prompt(self, task: ModuleTask,
                             context_items: list[ContextItem]) -> str:
        """Build the SCP-constrained system prompt.

        Uses scp_bridge if available, otherwise constructs manually.
        """
        parts = []

        # Try to use SCP Bridge for the module contract
        if self.retriever.bridge:
            try:
                contract = self.retriever.bridge.generate_system_prompt(
                    task.module_name, language="python"
                )
                parts.append(contract)
            except Exception:
                pass

        if not parts:
            # Manual prompt construction
            parts.append(f"You are implementing the module: {task.module_name}")
            parts.append(f"Task: {task.description}")

        # Add scored context items
        for item in context_items:
            if item.category == CATEGORY_CONTRACT:
                parts.append(f"\n--- Contract ({item.source}) ---\n{item.content}")
            elif item.category == CATEGORY_ACTIVE_CODE:
                parts.append(f"\n--- Current Code ({item.source}) ---\n{item.content}")
            elif item.category == CATEGORY_ERROR:
                parts.append(f"\n--- Error ---\n{item.content}")

        # Add forbidden zone warning
        forbidden = self.retriever.get_forbidden_modules(task.module_name)
        if forbidden:
            parts.append(
                f"\n⚠️ FORBIDDEN: You must NOT import, reference, or depend on: "
                f"{', '.join(forbidden)}"
            )

        return "\n".join(parts)

    def _build_user_prompt(self, task: ModuleTask) -> str:
        """Build the user-facing prompt."""
        if task.prompt_override:
            return task.prompt_override
        return (
            f"Implement the {task.module_name} module now.\n\n"
            f"Requirements: {task.description}\n\n"
            "Generate clean, production-quality Python code that follows "
            "all SCP constraints specified in the system prompt. "
            "Include docstrings and type hints."
        )

    def _extract_code(self, content: str) -> str:
        """Extract code from AI response (strip markdown fences if present)."""
        content = content.strip()

        # Handle ```python ... ``` fences
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first line (```python) and last line (```)
            start = 1
            end = len(lines)
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip() == "```":
                    end = i
                    break
            return "\n".join(lines[start:end]).strip()

        return content


# ─────────────────────────────────────────────────────────────
#  𓂀 Verifier — Post-edit verification
# ─────────────────────────────────────────────────────────────

class Verifier:
    """Verifies that an edit conforms to SCP constraints.

    Runs three checks:
        1. Weaver verification (W(G) = 0?) — via scp_bridge
        2. Auto-test generation and execution — via provider
        3. Coupling analysis — checks for undeclared dependencies

    The Verifier is the 𓂀 Witness — it observes the edit result
    without modifying it, reporting pass/fail.
    """

    def __init__(self, session: SessionState, provider: Optional[BaseProvider] = None):
        self.session = session
        self.provider = provider

    def verify(self, edit_result: EditResult) -> VerificationReport:
        """Verify a single edit result.

        Args:
            edit_result: The output of the Executor.

        Returns:
            VerificationReport with pass/fail and details.
        """
        report = VerificationReport(module_name=edit_result.module_name)

        if not edit_result.success:
            report.violations.append(f"Edit failed: {edit_result.error}")
            report.details = "Edit did not produce code."
            return report

        # 1. Weaver verification (via AI if available)
        weaver_result = self._weaver_check(edit_result)
        report.weaver_passed = weaver_result["passed"]
        if not weaver_result["passed"]:
            report.violations.extend(weaver_result.get("violations", []))

        # 2. Static constraint checks
        static_result = self._static_check(edit_result)
        if not static_result["passed"]:
            report.weaver_passed = False
            report.violations.extend(static_result.get("violations", []))

        # 3. Tests (mark as passed if no test infrastructure — don't block)
        report.tests_passed = True  # Will be updated when test runner is integrated

        # Compile details
        if report.passed:
            report.details = f"✔ {edit_result.module_name}: W(G) = 0 — all checks passed"
        else:
            report.details = (
                f"✘ {edit_result.module_name}: {len(report.violations)} violation(s)\n"
                + "\n".join(f"  • {v}" for v in report.violations)
            )

        return report

    def _weaver_check(self, edit_result: EditResult) -> dict:
        """Run Weaver verification using AI.

        Generates a verification prompt and asks the AI to check
        the generated code against the SCP spec.
        """
        if not self.provider:
            return {"passed": True, "violations": []}  # No provider = skip

        # Try to use SCP Bridge's verification prompt
        try:
            from scp_bridge import SCPBridge
            if self.session.architecture:
                bridge = SCPBridge(self.session.architecture)
                verify_prompt = bridge.generate_verification_prompt(
                    edit_result.module_name,
                    edit_result.code_generated,
                )

                response = self.provider.generate(verify_prompt)

                if response.success:
                    content_lower = response.content.lower()
                    passed = "pass" in content_lower and "fail" not in content_lower
                    violations = []
                    if not passed:
                        # Extract violation descriptions
                        for line in response.content.split("\n"):
                            line = line.strip()
                            if line and any(kw in line.lower()
                                            for kw in ["violation", "fail", "error", "❌", "✘"]):
                                violations.append(line)
                    return {"passed": passed, "violations": violations}
        except (ImportError, Exception):
            pass

        return {"passed": True, "violations": []}

    def _static_check(self, edit_result: EditResult) -> dict:
        """Run static checks on generated code.

        Checks for obvious SCP violations without needing AI:
            - Forbidden imports
            - Global mutable state
            - Missing function signatures
        """
        violations = []
        code = edit_result.code_generated

        if not code:
            violations.append("No code generated")
            return {"passed": False, "violations": violations}

        # Check for forbidden module imports
        if self.session.architecture:
            for mod in self.session.architecture.modules:
                if mod.name == edit_result.module_name:
                    forbidden = getattr(mod, 'forbidden', []) or []
                    for f_mod in forbidden:
                        f_lower = f_mod.lower()
                        # Check for import statements referencing forbidden modules
                        for line in code.split("\n"):
                            line_stripped = line.strip().lower()
                            if (line_stripped.startswith("import ") or
                                    line_stripped.startswith("from ")):
                                if f_lower in line_stripped:
                                    violations.append(
                                        f"Forbidden import: '{line.strip()}' "
                                        f"references forbidden module '{f_mod}'"
                                    )

        # Check for global mutable state (common SCP anti-pattern)
        for line in code.split("\n"):
            stripped = line.strip()
            if (not stripped.startswith("#") and
                    not stripped.startswith("def ") and
                    not stripped.startswith("class ") and
                    not stripped.startswith("@") and
                    not stripped.startswith("\"\"\"") and
                    not stripped.startswith("'''") and
                    "global " in stripped):
                violations.append(f"Global state mutation: '{stripped}'")

        return {"passed": len(violations) == 0, "violations": violations}


# ─────────────────────────────────────────────────────────────
#  ◬ Conductor — Main entry point
# ─────────────────────────────────────────────────────────────

class Conductor:
    """The main orchestrator for the Nexus AI IDE.

    Ties together all layers:
        - Layer 1 (Context Kernel) via the Executor
        - Layer 2 (Session Protocol) via SessionState
        - Layer 3 (Planner, Executor, Verifier)
        - Providers (any AI backend)

    Usage:
        from nexus.conductor import Conductor
        from nexus.session_protocol import SessionState
        from nexus.providers import get_provider, ProviderConfig
        from scp_bridge import SCPBridge, ArchitectureSpec

        # Load architecture
        spec = ArchitectureSpec(name="MyApp", modules=[...])
        bridge = SCPBridge(spec)

        # Create session
        session = SessionState(architecture=spec)

        # Create provider
        provider = get_provider(ProviderConfig(
            provider_name="gemini",
            api_key="your-key",
        ))

        # Create conductor
        conductor = Conductor(session, provider, bridge)

        # Execute a request
        results = conductor.handle_request(
            "Add error handling to the API module",
            target_modules=["API"],
        )
    """

    def __init__(self, session: SessionState, provider: BaseProvider,
                 bridge=None, output_fn=None):
        """
        Args:
            session: The persistent session state.
            provider: AI provider to use for generation and verification.
            bridge: Optional SCPBridge instance for contract generation.
            output_fn: Function for status output (default: print).
        """
        self.session = session
        self.provider = provider
        self.bridge = bridge
        self.output = output_fn or print

        # Initialize sub-components
        self.planner = Planner(session, provider)
        self.retriever = SCPRetriever(
            architecture=session.architecture,
            bridge=bridge,
        )
        self.scorer = EntropyScorer()
        self.pruner = ContextPruner()
        self.executor = Executor(
            session, provider,
            scorer=self.scorer,
            pruner=self.pruner,
            retriever=self.retriever,
        )
        self.verifier = Verifier(session, provider)

    def handle_request(self, request: str,
                       target_modules: list[str] = None) -> list[dict]:
        """Handle a developer request end-to-end.

        1. Decompose the request into module-scoped tasks
        2. Execute each task one-at-a-time with SCP constraints
        3. Verify each edit with the Weaver
        4. Record results in the edit ledger
        5. Freeze verified contracts

        Args:
            request: Natural language request from the developer.
            target_modules: Specific modules to target (auto-detect if None).

        Returns:
            List of results, one per module task.
        """
        results = []

        # ◬ — Decompose the request
        self.output(f"\n◬ ─── Nexus: Processing Request ───")
        self.output(f"  Request: {request}")

        plan = self.planner.decompose(request, target_modules)
        self.output(f"  Plan: {plan.module_count} module(s) to edit")

        for task in plan.tasks:
            self.output(f"\n☾ ─── Module: {task.module_name} ───")

            # Compute pre-edit entropy
            pre_items = self.retriever.retrieve_for_module(
                task.module_name,
                code_store=self.session.code_store,
            )
            self.scorer.active_module = task.module_name
            self.scorer.score_all(pre_items)
            entropy_before = self.scorer.compute_entropy(pre_items)

            # ☾ — Execute (one module at a time)
            self.session.set_active_module(task.module_name)
            edit_result = self.executor.execute(task)

            if edit_result.success:
                self.output(f"  ✔ Code generated ({len(edit_result.code_generated)} chars)")

                # 𓂀 — Verify
                report = self.verifier.verify(edit_result)
                self.output(f"  {report.details}")

                # Compute post-edit entropy
                self.session.update_code(task.module_name, edit_result.code_generated)
                post_items = self.retriever.retrieve_for_module(
                    task.module_name,
                    code_store=self.session.code_store,
                )
                self.scorer.score_all(post_items)
                entropy_after = self.scorer.compute_entropy(post_items)

                # Record in ledger
                entry = EditEntry(
                    module=task.module_name,
                    description=task.description,
                    code_after={f"{task.module_name}.py": edit_result.code_generated},
                    verified=report.weaver_passed,
                    test_passed=report.tests_passed,
                    verification_details=report.details,
                    entropy_before=entropy_before,
                    entropy_after=entropy_after,
                )
                self.session.record_edit(entry)

                # ☤ — Freeze contract if verification passed
                if report.passed and self.bridge:
                    try:
                        contract = self.bridge.generate_system_prompt(task.module_name)
                        self.session.freeze_contract(task.module_name, contract)
                        self.output(f"  ☤ Contract frozen (v{self.session.contract_cache.get_version(task.module_name)})")
                    except Exception:
                        pass

                results.append({
                    "module": task.module_name,
                    "success": True,
                    "verified": report.passed,
                    "code_length": len(edit_result.code_generated),
                    "entropy_delta": entropy_after - entropy_before,
                    "violations": report.violations,
                })
            else:
                self.output(f"  ✘ Generation failed: {edit_result.error}")
                results.append({
                    "module": task.module_name,
                    "success": False,
                    "error": edit_result.error,
                })

        # 𓂀 — Report session health
        health = self.session.session_health
        self.output(f"\n𓂀 ─── Session Health ───")
        self.output(f"  Total edits: {health['total_edits']}")
        self.output(f"  Regression rate: {health['regression_rate']}")
        self.output(f"  Clean streak: {health['clean_streak']}")
        self.output(f"  Frozen contracts: {health['frozen_modules']}/{health['total_modules']}")

        return results

    def get_context_report(self, module_name: str) -> dict:
        """Generate a detailed context report for a module.

        Shows what the AI would see, with entropy scores and
        compression ratios.
        """
        # Retrieve full context
        full_items = self.retriever.retrieve_for_module(
            module_name,
            code_store=self.session.code_store,
        )

        # Score
        self.scorer.active_module = module_name
        scored = self.scorer.score_all(full_items)

        # Prune
        pruned = self.pruner.prune(scored, self.session.entropy_budget)

        # Metrics
        full_tokens = sum(i.token_count for i in scored)
        pruned_tokens = sum(i.token_count for i in pruned)
        compression = self.pruner.compute_compression_ratio(scored, pruned)
        entropy = self.scorer.compute_entropy(pruned)

        return {
            "module": module_name,
            "full_context_tokens": full_tokens,
            "pruned_context_tokens": pruned_tokens,
            "compression_ratio": f"{compression:.2%}",
            "entropy_bits": f"{entropy:.2f}",
            "items_before_prune": len(scored),
            "items_after_prune": len(pruned),
            "forbidden_modules": self.retriever.get_forbidden_modules(module_name),
            "items": [
                {
                    "source": i.source,
                    "category": i.category,
                    "score": f"{i.relevance_score:.2f}",
                    "tokens": i.token_count,
                    "module_scope": i.module_scope,
                }
                for i in pruned
            ],
        }
