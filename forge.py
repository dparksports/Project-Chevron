"""
SCP Init — Automatic Codebase → SCP Architecture Decomposition
================================================================
Point this tool at any codebase and it will:

  1. ◬ Scan     — discover all source files, measure token counts
  2. Ө Filter  — extract functions, classes, imports from each file
  3. ☤ Weave   — send the summary to Gemini to propose module decomposition
  4. ☾ Fold    — iterate with the AI to refine the architecture spec
  5. 𓂀 Witness — output the final spec files + verification report

Usage:
    # Scan a codebase and generate SCP architecture
    python forge.py /path/to/your/project

    # Use a specific model
    python forge.py /path/to/your/project --model gemini-2.5-flash

    # Specify the project language
    python forge.py /path/to/your/project --language typescript

    # Just scan (no Gemini, shows what would be analyzed)
    python forge.py /path/to/your/project --scan-only
"""

import os
import sys
import json
import ast
import re
import textwrap
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────

SUPPORTED_MODELS = [
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]
DEFAULT_MODEL = "gemini-3-pro-preview"

# File extensions to scan, grouped by language
LANGUAGE_EXTENSIONS = {
    "python":     [".py"],
    "javascript": [".js", ".jsx", ".mjs"],
    "typescript": [".ts", ".tsx"],
    "go":         [".go"],
    "rust":       [".rs"],
    "java":       [".java"],
    "csharp":     [".cs"],
    "ruby":       [".rb"],
    "php":        [".php"],
    "swift":      [".swift"],
    "kotlin":     [".kt", ".kts"],
}

# Directories to always skip
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv", "env",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".next",
    ".tox", "eggs", "*.egg-info", ".idea", ".vscode", "bin", "obj",
}

# Glyph descriptions for the decomposition prompt
GLYPH_GUIDE = """
The 5 Chevron SCP Glyphs — assign one primary glyph per module:
  ◬ Origin       — Entry point / data source. Appears once per scope. (e.g., file loader, API server init)
  Ө Filter       — Pass/reject gate. Pure predicate. (e.g., search, validation, authentication check)
  ☾ Fold Time    — Recursive/iterative processing with base case. (e.g., batch processing, parsing, compilation)
  ☤ Weaver       — Merges multiple streams into one. (e.g., LLM + prompt → response, template rendering)
  𓂀 Witness      — Pure observation. Zero side effects. (e.g., logging, metrics, progress reporting)
"""

# ─────────────────────────────────────────────────────────────
#  Data Structures
# ─────────────────────────────────────────────────────────────

@dataclass
class SourceFile:
    """A single discovered source file."""
    path: str
    relative_path: str
    size_bytes: int
    lines: int
    tokens_approx: int  # ~4 bytes per token
    functions: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)

@dataclass
class ScanResult:
    """Complete scan of a codebase."""
    root_dir: str
    language: str
    files: list[SourceFile] = field(default_factory=list)
    total_bytes: int = 0
    total_lines: int = 0
    total_tokens: int = 0

# ─────────────────────────────────────────────────────────────
#  Step 1: ◬ Scanner
# ─────────────────────────────────────────────────────────────

def detect_language(root_dir: str) -> str:
    """Auto-detect the dominant language by file count."""
    counts = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            ext = os.path.splitext(f)[1].lower()
            for lang, exts in LANGUAGE_EXTENSIONS.items():
                if ext in exts:
                    counts[lang] = counts.get(lang, 0) + 1
    if not counts:
        return "python"  # default
    return max(counts, key=counts.get)


def extract_python_symbols(file_path: str, source: str) -> tuple[list[str], list[str], list[str]]:
    """Extract functions, classes, and imports from Python source."""
    functions, classes, imports = [], [], []
    try:
        tree = ast.parse(source, filename=file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                # Get signature info
                args = [a.arg for a in node.args.args if a.arg != 'self']
                ret = ""
                if node.returns and hasattr(node.returns, 'id'):
                    ret = f" -> {node.returns.id}"
                functions.append(f"{node.name}({', '.join(args)}){ret}")
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
    except SyntaxError:
        # Fallback: regex extraction
        functions = re.findall(r'def\s+(\w+)\s*\(', source)
        classes = re.findall(r'class\s+(\w+)', source)
        imports = re.findall(r'(?:import|from)\s+([\w.]+)', source)
    return functions, classes, imports


def extract_generic_symbols(source: str) -> tuple[list[str], list[str], list[str]]:
    """Fallback symbol extraction using regex for non-Python files."""
    functions = re.findall(
        r'(?:function|func|fn|def|public|private|protected|static|async)\s+(\w+)\s*\(',
        source
    )
    classes = re.findall(
        r'(?:class|struct|interface|enum|type)\s+(\w+)',
        source
    )
    imports = re.findall(
        r'(?:import|from|require|use|using)\s+["\']?([\w./@]+)',
        source
    )
    return functions, classes, imports


def scan_codebase(root_dir: str, language: str = None) -> ScanResult:
    """◬ Origin — Scan the entire codebase and extract metadata."""
    root_dir = os.path.abspath(root_dir)
    if language is None:
        language = detect_language(root_dir)

    extensions = set()
    for exts in LANGUAGE_EXTENSIONS.values():
        extensions.update(exts)  # scan all languages

    result = ScanResult(root_dir=root_dir, language=language)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for filename in sorted(filenames):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in extensions:
                continue

            filepath = os.path.join(dirpath, filename)
            relpath = os.path.relpath(filepath, root_dir)

            try:
                with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                    source = f.read()
            except (OSError, PermissionError):
                continue

            size = len(source.encode('utf-8'))
            lines = source.count('\n') + 1
            tokens = size // 4  # rough approximation

            # Extract symbols
            if ext == '.py':
                functions, classes, imports = extract_python_symbols(filepath, source)
            else:
                functions, classes, imports = extract_generic_symbols(source)

            sf = SourceFile(
                path=filepath,
                relative_path=relpath,
                size_bytes=size,
                lines=lines,
                tokens_approx=tokens,
                functions=functions,
                classes=classes,
                imports=imports,
            )
            result.files.append(sf)
            result.total_bytes += size
            result.total_lines += lines
            result.total_tokens += tokens

    return result


# ─────────────────────────────────────────────────────────────
#  Step 2: ☤ Decomposition Prompt
# ─────────────────────────────────────────────────────────────

def build_decomposition_prompt(scan: ScanResult) -> str:
    """Build the prompt that asks Gemini to decompose the codebase into SCP modules."""

    # Build file summaries
    file_summaries = []
    for sf in scan.files:
        summary = f"### {sf.relative_path}\n"
        summary += f"  Lines: {sf.lines}  |  Tokens: ~{sf.tokens_approx:,}\n"
        if sf.classes:
            summary += f"  Classes: {', '.join(sf.classes[:15])}\n"
        if sf.functions:
            # Show first 20 functions
            fns = sf.functions[:20]
            if len(sf.functions) > 20:
                fns.append(f"... +{len(sf.functions) - 20} more")
            summary += f"  Functions: {', '.join(fns)}\n"
        if sf.imports:
            # Show unique imports
            unique_imports = sorted(set(sf.imports))[:15]
            summary += f"  Imports: {', '.join(unique_imports)}\n"
        file_summaries.append(summary)

    files_section = "\n".join(file_summaries)

    prompt = f"""You are an expert software architect performing an SCP (Spatial Constraint Protocol) decomposition.

## Your Task

Analyze this {scan.language} codebase and decompose it into isolated SCP modules.

## Codebase Summary

- **Root:** {scan.root_dir}
- **Language:** {scan.language}
- **Files:** {len(scan.files)}
- **Total Lines:** {scan.total_lines:,}
- **Total Tokens:** ~{scan.total_tokens:,}

## Source Files

{files_section}

{GLYPH_GUIDE}

## Output Format

Respond with a JSON object (and ONLY the JSON, no markdown fences) with this exact structure:

{{
  "project_name": "Human-readable project name",
  "modules": [
    {{
      "name": "ModuleName",
      "description": "What this module does (1-2 sentences)",
      "primary_glyph": "◬",
      "glyph_name": "Origin",
      "source_files": ["relative/path/to/file.py"],
      "methods": [
        {{
          "name": "method_name",
          "inputs": ["param1: str", "param2: int"],
          "output": "ReturnType",
          "glyph": "◬",
          "description": "What this method does"
        }}
      ],
      "allowed_dependencies": ["OtherModule"],
      "forbidden_modules": ["ForbiddenModule"],
      "constraints": [
        "Must not do X",
        "Must always do Y"
      ]
    }}
  ],
  "global_constraints": [
    "Pipeline flows: A → B → C (DAG, not cycle)",
    "No shared mutable state between modules"
  ],
  "dependency_dag": "A -> B -> C, A -> D"
}}

## Rules for Decomposition

1. **Each module gets ONE primary glyph** that defines its core operation
2. **Exactly one ◬ Origin module** — the entry point for the whole system
3. **Dependencies must form a DAG** — no circular dependencies
4. **Forbidden zones must be explicit** — if Module A shouldn't touch Module C, say so
5. **Constraints must be enforceable** — concrete rules, not vague guidelines
6. **Methods must have typed signatures** — inputs and outputs with types
7. **Aim for 5-12 modules** — enough isolation without over-fragmentation
8. **Group by domain, not by file** — one module may span multiple files
9. **A 𓂀 Witness module is optional** but recommended for logging/metrics
10. **Minimize cross-module coupling** — each module should be independently implementable

Analyze the codebase and produce the JSON decomposition now."""

    return prompt


# ─────────────────────────────────────────────────────────────
#  Step 3: ☾ Generate Architecture
# ─────────────────────────────────────────────────────────────

def call_gemini(prompt: str, model: str, system_instruction: str = None) -> str:
    """Call Gemini API and return the response text."""
    try:
        from google import genai
    except ImportError:
        print("\n⚠ google-genai not installed. Run: pip install google-genai\n")
        sys.exit(1)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n⚠ No GEMINI_API_KEY found. Set it first:")
        print("  set GEMINI_API_KEY=your-key\n")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    config = genai.types.GenerateContentConfig(temperature=0.1)
    if system_instruction:
        config.system_instruction = system_instruction

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response.text


def parse_decomposition(response_text: str) -> dict:
    """Parse the JSON response from Gemini, handling markdown fences."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove first line (```json or ```)
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3].rstrip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Try to find JSON within the response
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        print(f"\n⚠ Failed to parse Gemini response as JSON: {e}")
        print("Raw response:")
        print(text[:2000])
        sys.exit(1)


# ─────────────────────────────────────────────────────────────
#  Step 4: 𓂀 Output Generation
# ─────────────────────────────────────────────────────────────

def generate_spec_file(decomp: dict, output_dir: str, language: str) -> str:
    """Generate the Python ArchitectureSpec file from decomposition."""
    project_name = decomp.get("project_name", "My Project")
    modules = decomp.get("modules", [])

    lines = [
        '"""',
        f'SCP Architecture Spec — {project_name}',
        f'Generated by forge.py',
        '"""',
        'import sys, os',
        'sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))',
        '',
        'from scp_bridge import SCPBridge, ArchitectureSpec, ModuleSpec, InterfaceMethod',
        '',
        '',
        f'SPEC = ArchitectureSpec(',
        f'    name="{project_name}",',
        f'    modules=[',
    ]

    for mod in modules:
        glyph_char = mod.get("primary_glyph", "☤")
        lines.append(f'        # ─── {glyph_char} {mod["name"]} ───')
        lines.append(f'        ModuleSpec(')
        lines.append(f'            name="{mod["name"]}",')
        desc = mod.get("description", "").replace('"', '\\"')
        lines.append(f'            description="{desc}",')
        lines.append(f'            methods=[')

        for method in mod.get("methods", []):
            inputs_str = json.dumps(method.get("inputs", []))
            m_glyph = method.get("glyph", glyph_char)
            m_desc = method.get("description", "").replace('"', '\\"')
            lines.append(f'                InterfaceMethod(')
            lines.append(f'                    "{method["name"]}",')
            lines.append(f'                    {inputs_str},')
            lines.append(f'                    "{method.get("output", "Any")}",')
            lines.append(f'                    "{m_glyph}",')
            lines.append(f'                    "{m_desc}",')
            lines.append(f'                ),')

        lines.append(f'            ],')

        deps = json.dumps(mod.get("allowed_dependencies", []))
        lines.append(f'            allowed_dependencies={deps},')

        constraints = mod.get("constraints", [])
        lines.append(f'            constraints=[')
        for c in constraints:
            c_escaped = c.replace('"', '\\"')
            lines.append(f'                "{c_escaped}",')
        lines.append(f'            ],')

        lines.append(f'        ),')
        lines.append('')

    lines.append(f'    ],')

    # Global constraints
    global_constraints = decomp.get("global_constraints", [])
    lines.append(f'    global_constraints=[')
    for gc in global_constraints:
        gc_escaped = gc.replace('"', '\\"')
        lines.append(f'        "{gc_escaped}",')
    lines.append(f'    ],')
    lines.append(f')')
    lines.append('')

    # CLI section — read from template file
    spec_basename = f"{os.path.basename(output_dir)}_scp.py"
    template_path = os.path.join(os.path.dirname(__file__), "templates", "spec_cli.py.template")
    with open(template_path, 'r', encoding='utf-8') as f:
        cli_code = f.read()

    # Substitute placeholders
    cli_code = cli_code.replace('{{LANGUAGE}}', language)
    cli_code = cli_code.replace('{{PROJNAME}}', project_name)
    cli_code = cli_code.replace('{{SPECBASE}}', spec_basename)

    lines.append(cli_code)


    spec_path = os.path.join(output_dir, f"{os.path.basename(output_dir)}_scp.py")

    # Actually write relative to the project — put it in CWD
    basename = os.path.basename(os.path.normpath(output_dir))
    spec_filename = f"{basename}_scp.py"


    with open(spec_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    return spec_filename


def generate_chevron_spec(decomp: dict, output_dir: str) -> str:
    """Generate a .chevron spec file from decomposition."""
    project_name = decomp.get("project_name", "My Project")
    modules = decomp.get("modules", [])

    lines = [f"# {project_name} — SCP Architecture", f"# Generated by scp_init.py", ""]

    # Type declarations (derive from method outputs)
    types_seen = set()
    for mod in modules:
        for method in mod.get("methods", []):
            output_type = method.get("output", "")
            # Extract simple type names
            for t in re.findall(r'[A-Z]\w+', output_type):
                if t not in {"Any", "None", "List", "Dict", "Optional", "Union", "str", "int", "float", "bool"}:
                    types_seen.add(t)

    if types_seen:
        for t in sorted(types_seen):
            lines.append(f"type {t} = {{ data: str }}")
        lines.append("")

    # Spec blocks for each module
    for mod in modules:
        glyph_char = mod.get("primary_glyph", "☤")
        lines.append(f"# {glyph_char} {mod['name']}")
        lines.append(f"spec {mod['name']}")

        deps = mod.get("allowed_dependencies", [])
        if deps:
            lines.append(f"    depends_on [{', '.join(deps)}]")

        # Infer imports from deps
        if deps:
            lines.append(f"    imports {', '.join(deps)}")

        # Exports from methods
        method_names = [m["name"] for m in mod.get("methods", [])]
        if method_names:
            lines.append(f"    exports {', '.join(method_names)}")

        forbidden = mod.get("forbidden_modules", [])
        if forbidden:
            lines.append(f"    forbidden [{', '.join(forbidden)}]")

        constraints = mod.get("constraints", [])
        for c in constraints:
            c_escaped = c.replace('"', '\\"')
            lines.append(f'    constraint "{c_escaped}"')

        lines.append("end")
        lines.append("")

    # Single pipeline with one Origin
    origin_mod = None
    for mod in modules:
        if mod.get("primary_glyph") == "◬":
            origin_mod = mod
            break

    if origin_mod:
        # Build a simple pipeline from the DAG
        pipeline_parts = [f'◬ "{project_name}"']
        for mod in modules:
            glyph = mod.get("primary_glyph", "☤")
            if glyph != "◬":
                pipeline_parts.append(f'{glyph} {mod["name"]}')
        pipeline_parts.append("𓂀")
        lines.append("# Pipeline")
        lines.append(" → ".join(pipeline_parts))
        lines.append("")

    basename = os.path.basename(os.path.normpath(output_dir))
    chevron_filename = f"{basename}_architecture.chevron"

    with open(chevron_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    return chevron_filename


def print_scan_report(scan: ScanResult):
    """𓂀 Witness — display the scan results."""
    line = "─" * 70
    print()
    print(line)
    print(f"◬  SCP Init — Codebase Scan")
    print(line)
    print(f"  Root:      {scan.root_dir}")
    print(f"  Language:  {scan.language}")
    print(f"  Files:     {len(scan.files)}")
    print(f"  Lines:     {scan.total_lines:,}")
    print(f"  Tokens:    ~{scan.total_tokens:,}")
    print(f"  Size:      {scan.total_bytes:,} bytes")
    print(line)
    print()

    # Top files by size
    sorted_files = sorted(scan.files, key=lambda f: f.size_bytes, reverse=True)
    print(f"{'File':<45} {'Lines':>7} {'Tokens':>8} {'Functions':>10}")
    print(line)
    for sf in sorted_files[:20]:
        fn_count = len(sf.functions)
        print(f"  {sf.relative_path:<43} {sf.lines:>7,} {sf.tokens_approx:>7,}  {fn_count:>8}")
    if len(scan.files) > 20:
        print(f"  ... +{len(scan.files) - 20} more files")
    print(line)
    print()


def print_decomposition_report(decomp: dict):
    """𓂀 Witness — display the decomposition results."""
    line = "─" * 70
    modules = decomp.get("modules", [])

    print()
    print(line)
    print(f"☤  SCP Decomposition: {decomp.get('project_name', 'Project')}")
    print(f"   {len(modules)} modules identified")
    print(line)
    print()

    print(f"{'Module':<22} {'Glyph':^7} {'Methods':>8} {'Dependencies'}")
    print(line)
    for mod in modules:
        glyph = mod.get("primary_glyph", "?")
        methods = len(mod.get("methods", []))
        deps = ", ".join(mod.get("allowed_dependencies", [])) or "none"
        print(f"  {mod['name']:<20} {glyph:^7} {methods:>6}   {deps}")
    print(line)
    print()

    # Global constraints
    global_constraints = decomp.get("global_constraints", [])
    if global_constraints:
        print("Global Constraints:")
        for gc in global_constraints:
            print(f"  • {gc}")
        print()

    # DAG
    dag = decomp.get("dependency_dag", "")
    if dag:
        print(f"Dependency DAG: {dag}")
        print()


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    # Parse arguments
    target_dir = None
    language = None
    model = DEFAULT_MODEL
    scan_only = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            i += 1
            model = args[i]
        elif args[i] == "--language" and i + 1 < len(args):
            i += 1
            language = args[i]
        elif args[i] == "--scan-only":
            scan_only = True
        elif args[i] in ("--help", "-h"):
            print(__doc__)
            return
        elif not args[i].startswith("-"):
            target_dir = args[i]
        i += 1

    if target_dir is None:
        print(__doc__)
        return

    target_dir = os.path.abspath(target_dir)
    if not os.path.isdir(target_dir):
        print(f"⚠ Not a directory: {target_dir}")
        sys.exit(1)

    # ─── Step 1: ◬ Scan ───
    print("\n◬ ─── Step 1: Scanning codebase ───\n")
    scan = scan_codebase(target_dir, language)
    print_scan_report(scan)

    if len(scan.files) == 0:
        print("⚠ No source files found. Check the directory path.")
        sys.exit(1)

    if scan_only:
        print("Scan complete. Use without --scan-only to generate SCP architecture.")
        return

    # ─── Step 2: ☤ Decompose ───
    print("☤ ─── Step 2: Analyzing with Gemini ───\n")
    print(f"  Model: {model}")
    print(f"  Sending {len(scan.files)} files ({scan.total_tokens:,} tokens) for analysis...\n")

    prompt = build_decomposition_prompt(scan)
    response_text = call_gemini(prompt, model)
    decomp = parse_decomposition(response_text)

    print_decomposition_report(decomp)

    # ─── Step 3: ☾ Generate Output ───
    print("☾ ─── Step 3: Generating SCP spec files ───\n")

    spec_file = generate_spec_file(decomp, target_dir, scan.language)
    chevron_file = generate_chevron_spec(decomp, target_dir)

    # Also save the raw JSON
    basename = os.path.basename(os.path.normpath(target_dir))
    json_file = f"{basename}_decomposition.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(decomp, f, indent=2, ensure_ascii=False)

    # ─── Step 4: 𓂀 Summary ───
    line = "─" * 70
    print(line)
    print("𓂀  SCP Init Complete!")
    print(line)
    print()
    print(f"  Generated files:")
    print(f"    📄 {spec_file:<40} Python SCP spec (runnable)")
    print(f"    📄 {chevron_file:<40} Chevron architecture spec")
    print(f"    📄 {json_file:<40} Raw decomposition JSON")
    print()
    print(f"  Next steps:")
    print(f"    # See the architecture overview")
    print(f"    python {spec_file}")
    print()
    print(f"    # Generate a module implementation")
    print(f"    python {spec_file} <ModuleName> --gemini")
    print()
    print(f"    # Verify the Chevron spec")
    print(f"    python run.py {chevron_file} --verify")
    print()

    # Compression stats
    modules = decomp.get("modules", [])
    avg_methods = sum(len(m.get("methods", [])) for m in modules) / max(len(modules), 1)
    est_prompt_tokens = int(avg_methods * 120 + 400)  # rough estimate
    if scan.total_tokens > 0:
        compression = scan.total_tokens // max(est_prompt_tokens, 1)
        print(f"  Compression:")
        print(f"    Codebase:          ~{scan.total_tokens:,} tokens")
        print(f"    Per-module prompt:  ~{est_prompt_tokens:,} tokens (estimated)")
        print(f"    Compression ratio: {compression}×")
        print()


if __name__ == "__main__":
    main()
