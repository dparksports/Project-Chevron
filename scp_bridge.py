"""
SCP Bridge — Chevron → AI Agent Interface (Topo-Categorical Orthogonality)
===========================================================================
Translates Chevron architecture specifications into constrained
system prompts for AI agents (Gemini, GPT, Claude, etc.).

Mechanism:
    SCP uses Topo-Categorical Orthogonality. The mathematical operators serve as
    non-polysemic, arXiv-anchored tokens in the prompt, leveraging pristine
    embedding coordinates in the LLM's latent space. The AI reads operator
    definitions as text and follows the constraints via in-context learning.
    This is prompt engineering with formally structured, contract-scoped
    context — not direct weight manipulation or embedding injection.

Usage:
    # Define your architecture in a .chevron spec
    python scp_bridge.py specs/todo_app.chevron

    # Use with Gemini API
    from scp_bridge import SCPBridge
    bridge = SCPBridge("specs/todo_app.chevron")
    system_prompt = bridge.generate_system_prompt("TodoStore")
    # Feed system_prompt to Gemini as context
"""
import sys
import os
import json
from dataclasses import dataclass, field
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chevron.glyphs import OPERATOR_REGISTRY, OperatorType
from chevron.code_verifier import CodeVerifier, CodeViolation, SeverityLevel


# ─────────────────────────────────────────────────────────────
#  Module & Interface Definitions
# ─────────────────────────────────────────────────────────────

@dataclass
class InterfaceMethod:
    """A single method on a module interface."""
    name: str
    inputs: list[str]
    output: str
    operator: str  # Which Topo-Categorical operator governs this method
    constraint: str = ""


@dataclass
class ModuleSpec:
    """A module specification in the SCP architecture."""
    name: str
    description: str
    methods: list[InterfaceMethod] = field(default_factory=list)
    allowed_dependencies: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)


@dataclass
class ArchitectureSpec:
    """Full architecture specification parsed from Chevron."""
    name: str
    modules: list[ModuleSpec] = field(default_factory=list)
    global_constraints: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────
#  Pre-built Architecture Templates
# ─────────────────────────────────────────────────────────────

# These demonstrate how Topo-Categorical operators map to real module constraints

TEMPLATES = {
    "todo_app": ArchitectureSpec(
        name="Todo Application",
        modules=[
            ModuleSpec(
                name="TodoStore",
                description="Manages the storage and retrieval of todo items",
                methods=[
                    InterfaceMethod("add", ["task: Task"], "Store", "☤",
                                    "Weaves a new task into the store"),
                    InterfaceMethod("remove", ["task_id: str"], "Store", "Ө",
                                    "Filters out the task matching the ID"),
                    InterfaceMethod("list", ["store: Store"], "list[Task]", "𓂀",
                                    "Witnesses all tasks without modification"),
                    InterfaceMethod("complete", ["task_id: str"], "Store", "☾",
                                    "Folds task state from incomplete → complete"),
                ],
                allowed_dependencies=["API"],
                constraints=[
                    "All functions must be pure — no side effects",
                    "Store is immutable — return new store, don't mutate",
                    "No direct database access — use API interface only",
                ],
            ),
            ModuleSpec(
                name="API",
                description="HTTP API layer — receives requests and dispatches to store",
                methods=[
                    InterfaceMethod("handle_request", ["request: Request"], "Response", "◬",
                                    "Origin — entry point for all external requests"),
                    InterfaceMethod("validate", ["request: Request"], "Request | None", "Ө",
                                    "Filters invalid requests"),
                    InterfaceMethod("log_request", ["request: Request", "response: Response"], "None", "𓂀",
                                    "Witnesses request/response for audit"),
                ],
                allowed_dependencies=["TodoStore"],
                constraints=[
                    "Must validate all input before passing to TodoStore",
                    "Must not contain business logic — delegate to TodoStore",
                    "Must log all requests via the Witness (𓂀) pattern",
                ],
            ),
        ],
        global_constraints=[
            "Only two modules exist: TodoStore and API",
            "Communication flows: API → TodoStore only (not reverse)",
            "No shared mutable state between modules",
            "All inter-module calls go through declared interfaces",
        ],
    ),

    "data_pipeline": ArchitectureSpec(
        name="Data Processing Pipeline",
        modules=[
            ModuleSpec(
                name="Ingest",
                description="Reads raw data from external sources",
                methods=[
                    InterfaceMethod("read_source", ["source: str"], "RawData", "◬",
                                    "Origin — entry point for data ingestion"),
                ],
                allowed_dependencies=[],
                constraints=["Must not transform data — raw pass-through only"],
            ),
            ModuleSpec(
                name="Transform",
                description="Cleans, validates, and transforms data",
                methods=[
                    InterfaceMethod("clean", ["data: RawData"], "CleanData", "Ө",
                                    "Filters out invalid/malformed records"),
                    InterfaceMethod("normalize", ["data: CleanData"], "NormalizedData", "☾",
                                    "Recursively applies normalization rules"),
                ],
                allowed_dependencies=["Ingest"],
                constraints=["Must be idempotent — same input always gives same output"],
            ),
            ModuleSpec(
                name="Load",
                description="Writes processed data to destination",
                methods=[
                    InterfaceMethod("write", ["data: NormalizedData"], "Result", "☤",
                                    "Weaves data into the destination store"),
                    InterfaceMethod("audit", ["result: Result"], "AuditLog", "𓂀",
                                    "Witnesses the write operation for compliance"),
                ],
                allowed_dependencies=["Transform"],
                constraints=["Must log every write via Witness pattern"],
            ),
        ],
        global_constraints=[
            "Pipeline flows: Ingest → Transform → Load (unidirectional)",
            "No module may reach back to a prior stage",
            "All data flows through declared interfaces only",
        ],
    ),
}


# ─────────────────────────────────────────────────────────────
#  SCP Bridge — Core
# ─────────────────────────────────────────────────────────────

class SCPBridge:
    """
    High-Density Semantic Prompting bridge.

    Translates an SCP architecture specification into AI-constraining
    system prompts. Compresses architectural context by expressing
    module contracts as dense Topo-Categorical operators rather than
    verbose natural language descriptions.

    The operators are read by the AI as in-context instructions — this is
    structured prompt engineering, not direct latent space manipulation.

    Usage:
        bridge = SCPBridge.from_template("todo_app")

        # Generate a system prompt for AI to implement a specific module
        prompt = bridge.generate_system_prompt("TodoStore", language="python")

        # Feed this to Gemini / GPT / Claude as the system message
        # The AI will generate code constrained by the SCP spec

        # Verify generated code deterministically (no AI self-check)
        violations = bridge.verify_generated_code("TodoStore", code)
    """

    def __init__(self, spec: ArchitectureSpec):
        self.spec = spec
        self._code_verifier = CodeVerifier()

    @classmethod
    def from_template(cls, template_name: str) -> "SCPBridge":
        if template_name not in TEMPLATES:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(TEMPLATES.keys())}")
        return cls(TEMPLATES[template_name])

    def generate_system_prompt(self, module_name: str, language: str = "python") -> str:
        """
        Generate a complete system prompt that constrains the AI to implement
        ONLY the specified module, seeing ONLY the interfaces of its declared
        dependencies.

        This is the SCP "RAG Denial" pattern — the AI physically cannot see
        the implementation of other modules, only their interface contracts.
        """
        module = self._find_module(module_name)
        if module is None:
            raise ValueError(f"Module '{module_name}' not found in spec '{self.spec.name}'")

        sections = []

        # Header
        sections.append(self._header(module, language))

        # Module contract
        sections.append(self._module_contract(module, language))

        # Dependency interfaces (visible) — ONLY interfaces, not implementations
        sections.append(self._dependency_interfaces(module, language))

        # Forbidden zones
        sections.append(self._forbidden_zones(module))

        # Operator constraints
        sections.append(self._operator_constraints(module))

        # Global constraints
        sections.append(self._global_constraints())

        # Output format
        sections.append(self._output_format(module, language))

        return "\n\n".join(sections)

    def generate_verification_prompt(self, module_name: str, code: str) -> str:
        """
        Generate a prompt for AI to VERIFY that generated code conforms
        to the SCP spec. This is the Weaver function (☤) in action.
        """
        module = self._find_module(module_name)
        if module is None:
            raise ValueError(f"Module '{module_name}' not found")

        return f"""# ☤ SCP Weaver Verification

You are the Weaver (☤) — the coupling detector. Your job is to verify that
the following code conforms to the SCP architecture specification.

## Module: {module.name}
{module.description}

## Code to Verify:
```
{code}
```

## Verification Checklist:

### 1. Interface Conformance
Check that the code implements EXACTLY these methods with these signatures:
{self._format_methods(module, "python")}

### 2. Dependency Isolation
The code may ONLY import/reference these PROJECT modules: {module.allowed_dependencies or ["(none — fully isolated)"]}
Standard library imports (typing, dataclasses, enum, abc, collections, etc.) are ALWAYS allowed.
Flag ANY import or reference to undeclared PROJECT modules.

### 3. Constraint Compliance
{chr(10).join(f"- [ ] {c}" for c in module.constraints)}

### 4. Coupling Detection (W(G) = 0)
Check for:
- [ ] No global mutable state
- [ ] No implicit channels (file I/O, environment variables, singletons)
- [ ] No undeclared PROJECT dependencies (stdlib imports are fine)
- [ ] All inter-module communication goes through declared interfaces

### 5. Operator Contract Compliance
{self._format_operator_checks(module)}

## Output Format:
- PASS: All checks satisfied. W(G) = 0.
- FAIL: List each violation with line number and explanation.
"""

    def generate_test_prompt(self, module_name: str, code: str, language: str = "python") -> str:
        """
        Generate a prompt for AI to create pytest tests for a module.
        Tests are derived from the SCP CONTRACT (not from reading the implementation),
        ensuring they verify spec compliance rather than implementation details.
        """
        module = self._find_module(module_name)
        if module is None:
            raise ValueError(f"Module '{module_name}' not found")

        # Build method test expectations
        method_tests = []
        for m in module.methods:
            inputs = ", ".join(m.inputs) if m.inputs else ""
            method_tests.append(
                f"- `{m.name}({inputs}) -> {m.output}` — {m.constraint or 'standard behavior'}"
            )
        methods_section = "\n".join(method_tests)

        # Build constraint test expectations
        constraint_tests = []
        for c in module.constraints:
            constraint_tests.append(f"- {c}")
        constraints_section = "\n".join(constraint_tests) if constraint_tests else "- (no specific constraints)"

        # Build dependency info
        allowed_deps = module.allowed_dependencies or ["(none — fully isolated)"]

        return f"""# 🧪 SCP Test Generation: {module.name}

You are generating pytest tests for the **{module.name}** module.

## IMPORTANT RULES
1. Tests must verify the **SCP contract** — method signatures, return types, constraints
2. Mock ALL external dependencies (no real GPU, no API calls, no file I/O)
3. Use `unittest.mock.patch` and `unittest.mock.MagicMock` for dependencies
4. Tests must be runnable with just: `pytest test_{module_name.lower()}.py -v`
5. No external packages beyond pytest and unittest.mock
6. Import the module as: `import {module_name.lower()}`

## Module Under Test
**{module.name}**: {module.description}

## Methods to Test (verify these exist and have correct signatures)
{methods_section}

## Constraints to Verify
{constraints_section}

## Dependency Isolation Rules
Allowed project dependencies: {allowed_deps}
The tests must verify that the module does NOT import any forbidden project modules.

## Operator Contract
{self._format_operator_checks(module)}

## Code Under Test
```{language}
{code}
```

## Required Test Categories

### 1. Structural Tests (`test_interface_*`)
- Verify each method exists as a callable
- Verify method signatures match the spec (parameter names and count)
- Verify the module has the expected public API (no missing/extra public methods)

### 2. Constraint Tests (`test_constraint_*`)
- For each constraint listed above, write a test that verifies it
- Use AST inspection (import ast) to check import restrictions
- Check for forbidden patterns in the source code

### 3. Behavioral Tests (`test_behavior_*`)
- Call each method with mocked dependencies
- Verify return types match the spec
- Test edge cases: empty inputs, None values, error handling
- For Fold Time (☾) methods: verify base case handling

### 4. Isolation Tests (`test_isolation_*`)
- Parse the source with `ast` to extract all imports
- Verify no forbidden project modules are imported
- Verify no global mutable state

## Output Format
Output ONLY the Python test file content. No markdown fences, no explanation.
Start with the imports, then the test classes/functions.
Use descriptive test names that reference the SCP contract.
"""

    def verify_generated_code(self, module_name: str, code: str,
                               extra_allowed: list[str] | None = None) -> list[CodeViolation]:
        """Deterministically verify generated code against the SCP contract.

        This is the formal Weaver function W(G). Unlike the AI verification
        prompt (generate_verification_prompt), this uses Python's ast module
        to perform deterministic static analysis.

        Run this BEFORE (or alongside) the AI verification prompt.

        Args:
            module_name: The SCP module the code implements.
            code: The generated Python source code.
            extra_allowed: Additional module names to treat as allowed imports.

        Returns:
            List of CodeViolation instances. Empty = W(G) = 0 (passes).
        """
        module = self._find_module(module_name)
        if module is None:
            raise ValueError(f"Module '{module_name}' not found in spec '{self.spec.name}'")

        if extra_allowed:
            verifier = CodeVerifier(extra_allowed_modules=extra_allowed)
        else:
            verifier = self._code_verifier

        return verifier.verify(code, module)

    def generate_structured_schema(self, module_name: str) -> dict:
        """Generate a JSON schema that constrains the AI's output structure.

        When used with Gemini's response_schema, this provides grammar-level
        guarantees on output structure — the closest practical approximation
        to "direct latent space mapping" available in production APIs.

        Usage:
            schema = bridge.generate_structured_schema("TodoStore")
            # Pass to Gemini as:
            #   generation_config=genai.GenerationConfig(
            #       response_mime_type="application/json",
            #       response_schema=schema
            #   )
        """
        module = self._find_module(module_name)
        if module is None:
            raise ValueError(f"Module '{module_name}' not found")

        # Build method schemas
        method_schemas = {}
        for m in module.methods:
            param_props = {}
            for inp in m.inputs:
                name = inp.split(":")[0].strip()
                type_hint = inp.split(":")[1].strip() if ":" in inp else "Any"
                param_props[name] = {
                    "type": "string",
                    "description": f"Parameter type: {type_hint}"
                }

            method_schemas[m.name] = {
                "type": "object",
                "properties": {
                    "docstring": {
                        "type": "string",
                        "description": f"Docstring referencing operator {m.operator}: {m.constraint}"
                    },
                    "parameters": {
                        "type": "object",
                        "properties": param_props,
                        "description": f"Method parameters"
                    },
                    "return_type": {
                        "type": "string",
                        "description": f"Return type: {m.output}"
                    },
                    "body": {
                        "type": "string",
                        "description": "The method implementation code"
                    }
                },
                "required": ["docstring", "body", "return_type"]
            }

        return {
            "type": "object",
            "properties": {
                "module_name": {
                    "type": "string",
                    "enum": [module.name],
                    "description": f"Must be '{module.name}'"
                },
                "imports": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": f"Allowed imports. Project deps: {module.allowed_dependencies or ['(none)']}"
                },
                "methods": {
                    "type": "object",
                    "properties": method_schemas,
                    "required": [m.name for m in module.methods]
                }
            },
            "required": ["module_name", "imports", "methods"]
        }

    def generate_full_workspace(self, language: str = "python") -> str:

        """Generate prompts for ALL modules — the complete SCP workspace."""
        output = [f"# SCP Architecture: {self.spec.name}\n"]
        output.append(f"## Global Constraints")
        for c in self.spec.global_constraints:
            output.append(f"- {c}")
        output.append("")

        for module in self.spec.modules:
            output.append(f"---\n")
            output.append(self.generate_system_prompt(module.name, language))
            output.append("")

        return "\n".join(output)

    # ─────────────────────────────────────────────────────────
    #  Private helpers
    # ─────────────────────────────────────────────────────────

    def _find_module(self, name: str) -> ModuleSpec | None:
        for m in self.spec.modules:
            if m.name == name:
                return m
        return None

    def _header(self, module: ModuleSpec, language: str) -> str:
        return f"""# ◬ SCP System Prompt — Module: {module.name}
# Architecture: {self.spec.name}
# Language: {language}
# Protocol: Spatial Constraint Protocol v1.0

You are an AI code generator operating under the Spatial Constraint Protocol (SCP).
You must implement the **{module.name}** module in **{language}**.

## CRITICAL RULES:
1. You may ONLY implement the {module.name} module
2. You may ONLY call interfaces of: {', '.join(module.allowed_dependencies) or '(no dependencies — fully isolated)'}
3. You must NOT access, import, or reference any other PROJECT module's implementation
4. You must follow all operator contracts below — each method is governed by a Topo-Categorical primitive
5. Standard library imports (typing, dataclasses, enum, etc.) are ALWAYS allowed — they are NOT project dependencies"""

    def _module_contract(self, module: ModuleSpec, language: str) -> str:
        lines = [f"## 📋 Module Contract: {module.name}", f"**Purpose:** {module.description}", ""]
        lines.append("### Methods to Implement:")
        lines.append(self._format_methods(module, language))
        if module.constraints:
            lines.append("### Module Constraints:")
            for c in module.constraints:
                lines.append(f"- ⚠️ {c}")
        return "\n".join(lines)

    def _format_methods(self, module: ModuleSpec, language: str) -> str:
        lines = []
        for m in module.methods:
            op_info = OPERATOR_REGISTRY.get(m.operator, None)
            op_name = op_info.name if op_info else m.operator
            inputs = ", ".join(m.inputs)
            lines.append(f"- `{m.operator}` **{m.name}**({inputs}) → {m.output}")
            lines.append(f"  - Operator: {op_name} — {m.constraint}")
            if op_info:
                lines.append(f"  - Contract: {op_info.contract}")
                lines.append(f"  - Must NOT: {op_info.constraint}")
        return "\n".join(lines)

    def _dependency_interfaces(self, module: ModuleSpec, language: str) -> str:
        if not module.allowed_dependencies:
            return """## 🔒 Dependencies: NONE
This module is fully isolated. It may not import or reference any other module."""

        lines = ["## 🔗 Visible Dependency Interfaces",
                  "You may call ONLY these methods. You CANNOT see their implementation.",
                  ""]

        for dep_name in module.allowed_dependencies:
            dep = self._find_module(dep_name)
            if dep:
                lines.append(f"### {dep.name} (interface only)")
                lines.append(f"*{dep.description}*")
                for m in dep.methods:
                    inputs = ", ".join(m.inputs)
                    lines.append(f"- `{m.name}({inputs}) → {m.output}`")
                lines.append("")

        lines.append("> ⚠️ RAG DENIAL: You see INTERFACES ONLY. The implementation of these")
        lines.append("> dependencies is physically inaccessible. Design against the contract, not the code.")

        return "\n".join(lines)

    def _forbidden_zones(self, module: ModuleSpec) -> str:
        all_modules = [m.name for m in self.spec.modules]
        forbidden = [m for m in all_modules if m != module.name and m not in module.allowed_dependencies]

        if not forbidden:
            return ""

        lines = ["## 🚫 Forbidden Zones",
                  "You MUST NOT reference, import, or access these modules:",
                  ""]
        for f in forbidden:
            lines.append(f"- ❌ `{f}` — inaccessible (SCP isolation)")

        return "\n".join(lines)

    def _operator_constraints(self, module: ModuleSpec) -> str:
        lines = ["## ↦ Operator Semantic Contracts",
                  "Each method is governed by a Topo-Categorical operator. You MUST follow its contract:",
                  ""]

        used_operators = set()
        for m in module.methods:
            if m.operator not in used_operators:
                used_operators.add(m.operator)
                info = OPERATOR_REGISTRY.get(m.operator)
                if info:
                    lines.append(f"### {m.operator} {info.name}")
                    lines.append(f"- **Intent:** {info.intent}")
                    lines.append(f"- **Contract:** {info.contract}")
                    lines.append(f"- **Constraint:** {info.constraint}")
                    lines.append("")

        return "\n".join(lines)

    def _global_constraints(self) -> str:
        lines = ["## 🌐 Global Architecture Constraints"]
        for c in self.spec.global_constraints:
            lines.append(f"- {c}")
        return "\n".join(lines)

    def _output_format(self, module: ModuleSpec, language: str) -> str:
        return f"""## ✅ Output Requirements
1. Implement `{module.name}` as a single {language} module/file
2. Include type hints for all method signatures
3. Include docstrings referencing the governing operator (e.g., "↦ Morphism data flow")
4. Include no imports from forbidden modules
5. Mark each method with its operator in a comment: `# ↦ Morphism`, `# ⊗ Tensor Product`, etc.
6. All inter-module calls must go through the declared interface only"""

    def _format_operator_checks(self, module: ModuleSpec) -> str:
        lines = []
        for m in module.methods:
            info = OPERATOR_REGISTRY.get(m.operator)
            if info:
                lines.append(f"- [ ] `{m.name}` governed by {m.operator} ({info.name}): {info.constraint}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("◬  SCP BRIDGE — Chevron → AI System Prompt Generator")
    print("=" * 70)
    print()

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scp_bridge.py <template> [module] [language]")
        print()
        print("Templates:")
        for name, spec in TEMPLATES.items():
            modules = ", ".join(m.name for m in spec.modules)
            print(f"  {name:20s} — {spec.name} (modules: {modules})")
        print()
        print("Examples:")
        print("  python scp_bridge.py todo_app TodoStore python")
        print("  python scp_bridge.py todo_app API python")
        print("  python scp_bridge.py data_pipeline Transform python")
        print("  python scp_bridge.py todo_app --all python")
        sys.exit(0)

    template = sys.argv[1]
    module = sys.argv[2] if len(sys.argv) > 2 else "--all"
    language = sys.argv[3] if len(sys.argv) > 3 else "python"

    bridge = SCPBridge.from_template(template)

    if module == "--all":
        print(bridge.generate_full_workspace(language))
    elif module == "--verify":
        # Read code from stdin for verification
        code = sys.stdin.read()
        module_name = language  # In verify mode, arg3 is module name
        print(bridge.generate_verification_prompt(module_name, code))
    else:
        print(bridge.generate_system_prompt(module, language))


if __name__ == "__main__":
    main()
