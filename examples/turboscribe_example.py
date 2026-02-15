"""
TurboScribe SCP Architecture — Standalone Example
===================================================
Self-contained SCP (Spatial Constraint Protocol) decomposition of the
TurboScribe GPU transcription engine into 9 isolated modules with
glyph contracts, dependency DAGs, and RAG Denial zones.

This file is fully independent — it does NOT modify scp_bridge.py.
It imports the base classes and glyph registry, then defines the
complete TurboScribe architecture as a standalone template.

TurboScribe is a GPU-accelerated transcription app with a ~1,500-line
monolithic Python backend (fast_engine.py). This SCP template decomposes
it into 9 isolated modules:

    ◬ AudioIngest        — File discovery & loading (Origin)
    Ө VoiceDetector      — Speech/silence detection (Filter)
    ☾ Transcriber        — Whisper transcription (Fold Time)
    Ө SearchEngine       — Keyword & semantic search (Filter)
    Ө MeetingDetector    — Real vs hallucinated (Filter)
    ☤ LLMProvider        — Local/cloud LLM interface (Weaver)
    ☤ Analyzer           — Summarize/outline (Weaver)
    ☾ TimestampExtractor — Video timestamp OCR (Fold Time)
    𓂀 ProgressWitness    — Pure logging (Witness)

Usage:
    # Show full architecture overview
    python examples/turboscribe_example.py

    # Generate prompt for one module
    python examples/turboscribe_example.py Transcriber

    # Use with Gemini API (default model: gemini-3-pro-preview)
    set GEMINI_API_KEY=your-key-here
    python examples/turboscribe_example.py Transcriber --gemini

    # Choose a specific model
    python examples/turboscribe_example.py Transcriber --gemini --model gemini-3-pro-preview
    python examples/turboscribe_example.py Transcriber --gemini --model gemini-2.5-flash

    Available models:
        gemini-2.5-pro           Best constraint adherence (default)
        gemini-2.5-flash         Fast & cheap, good for simple modules
        gemini-3-pro-preview     Latest pro preview
        gemini-3-flash-preview   Latest flash preview
"""
import sys
import os

# ─────────────────────────────────────────────────────────────
#  Import base infrastructure from Chevron (no modifications)
# ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scp_bridge import (
    SCPBridge,
    ArchitectureSpec,
    ModuleSpec,
    InterfaceMethod,
)


# ─────────────────────────────────────────────────────────────
#  TurboScribe Architecture Spec
# ─────────────────────────────────────────────────────────────

TURBOSCRIBE_SPEC = ArchitectureSpec(
    name="TurboScribe \u2014 GPU Audio Transcription & Analysis",
    modules=[
        # ─── ◬ AudioIngest: Origin ───
        ModuleSpec(
            name="AudioIngest",
            description="Discovers and loads media files from directories. Entry point for all pipelines.",
            methods=[
                InterfaceMethod("find_media", ["directory: str"], "list[MediaFile]", "\u25ec",
                                "Origin \u2014 discovers all media files recursively"),
                InterfaceMethod("load_audio", ["file_path: str"], "AudioTensor", "\u25ec",
                                "Origin \u2014 loads audio as 16kHz mono float32 tensor via ffmpeg"),
                InterfaceMethod("get_device_config", ["override: str | None"], "DeviceConfig", "\U000130c0",
                                "Witnesses GPU/CPU availability without side effects"),
            ],
            allowed_dependencies=[],
            constraints=[
                "Must not transform or analyze audio \u2014 raw loading only",
                "Must not import Whisper, VAD, or any ML model",
                "File discovery must be deterministic (sorted, reproducible)",
                "MEDIA_EXTENSIONS is the single source of truth for supported formats",
            ],
        ),

        # ─── Ө VoiceDetector: Filter ───
        ModuleSpec(
            name="VoiceDetector",
            description="Detects speech segments in audio using Silero VAD. Gates audio \u2014 only speech passes.",
            methods=[
                InterfaceMethod("run_vad_scan", ["file_path: str", "threshold: float"], "VadResult", "\u04e8",
                                "Filters audio into speech/silence segments"),
                InterfaceMethod("run_batch_vad_scan", ["directory: str", "threshold: float"], "VadReport", "\u04e8",
                                "Filters entire directory \u2014 only files with speech pass"),
                InterfaceMethod("cluster_segments", ["segments: list[Segment]", "gap_threshold: float"], "list[Block]", "\u2624",
                                "Weaves adjacent segments into contiguous speech blocks"),
            ],
            allowed_dependencies=["AudioIngest"],
            constraints=[
                "Must not transcribe \u2014 detection only, no Whisper",
                "Must not modify source audio files",
                "VAD model must be loaded lazily and cached (singleton)",
                "Must support skip_existing for resumable batch operations",
            ],
        ),

        # ─── ☾ Transcriber: Fold Time ───
        ModuleSpec(
            name="Transcriber",
            description="Transcribes audio to text using faster-whisper models. Recursive batch processing.",
            methods=[
                InterfaceMethod("load_model", ["model_name: str", "device: str", "compute: str"], "WhisperModel", "\u25ec",
                                "Origin \u2014 loads the Whisper model with CPU fallback"),
                InterfaceMethod("transcribe_file", ["file_path: str", "model_name: str", "beam_size: int"], "Transcript", "\u263e",
                                "Folds audio through model \u2014 iterates segments into full transcript"),
                InterfaceMethod("batch_transcribe", ["directory: str", "model_name: str", "beam_size: int"], "list[Transcript]", "\u263e",
                                "Recursively folds all files in directory through transcription"),
                InterfaceMethod("transcribe_segment", ["file_path: str", "start: float", "end: float"], "Transcript", "\u263e",
                                "Folds a specific time range through the model"),
            ],
            allowed_dependencies=["AudioIngest", "VoiceDetector"],
            constraints=[
                "Must save transcripts as filename_transcript_modelname.txt",
                "Must support skip_existing \u2014 never re-transcribe completed files",
                "Model loading must fallback CPU \u2192 GPU gracefully",
                "Must emit [PROGRESS] markers for real-time UI updates",
                "Must not perform any analysis or search on transcripts",
            ],
        ),

        # ─── Ө SearchEngine: Filter ───
        ModuleSpec(
            name="SearchEngine",
            description="Searches transcripts by keyword or semantic embedding similarity.",
            methods=[
                InterfaceMethod("keyword_search", ["directory: str", "query: str"], "list[SearchResult]", "\u04e8",
                                "Filters transcripts \u2014 only matching lines pass"),
                InterfaceMethod("semantic_search", ["directories: list[str]", "query: str", "model_name: str"], "list[SearchResult]", "\u04e8",
                                "Filters transcripts by embedding cosine similarity"),
            ],
            allowed_dependencies=["AudioIngest"],
            constraints=[
                "Must not modify transcripts \u2014 read-only search",
                "Semantic search must fallback to keyword search if sentence-transformers unavailable",
                "Must return results with context snippets and confidence scores",
                "Must not import Whisper, VAD, or LLM dependencies",
            ],
        ),

        # ─── Ө MeetingDetector: Filter ───
        ModuleSpec(
            name="MeetingDetector",
            description="Uses LLM to classify transcripts as real meetings vs hallucinated content.",
            methods=[
                InterfaceMethod("detect_meetings", ["directory: str", "provider: str"], "DetectionReport", "\u04e8",
                                "Filters transcripts \u2014 real meetings pass, hallucinations rejected"),
                InterfaceMethod("classify_transcript", ["text: str", "provider: str"], "Classification", "\u04e8",
                                "Gates a single transcript \u2014 pass/reject with confidence score"),
            ],
            allowed_dependencies=["AudioIngest", "LLMProvider"],
            constraints=[
                "Must not modify or delete transcript files",
                "Must support skip_checked for resumable batch scans",
                "Must emit [DETECT_PROGRESS] for real-time UI updates",
                "Classification prompt must be deterministic (no random sampling)",
                "Must not import Whisper or sentence-transformers",
            ],
        ),

        # ─── ☤ LLMProvider: Weaver ───
        ModuleSpec(
            name="LLMProvider",
            description="Unified interface to local (llama.cpp) and cloud (Gemini, OpenAI, Claude) LLMs.",
            methods=[
                InterfaceMethod("load_local_model", ["model_name: str"], "LLMInstance", "\u25ec",
                                "Origin \u2014 loads GGUF model with GPU layers, cached singleton"),
                InterfaceMethod("generate", ["prompt: str", "provider: str", "api_key: str | None"], "str", "\u2624",
                                "Weaves prompt + model into response \u2014 braids local/cloud into unified output"),
                InterfaceMethod("list_models", [], "list[ModelInfo]", "\U000130c0",
                                "Witnesses available models without modification"),
            ],
            allowed_dependencies=[],
            constraints=[
                "Must cache loaded local models \u2014 never reload same model",
                "Must support providers: local, gemini, openai, claude",
                "Must not contain domain logic (meeting detection, analysis, etc.)",
                "API keys must be passed in, never read from environment directly",
                "Must handle GPU OOM gracefully with CPU fallback",
            ],
        ),

        # ─── ☤ Analyzer: Weaver ───
        ModuleSpec(
            name="Analyzer",
            description="Summarizes or outlines transcripts using LLM providers.",
            methods=[
                InterfaceMethod("summarize", ["transcript_path: str", "provider: str"], "str", "\u2624",
                                "Weaves transcript + LLM into a concise summary"),
                InterfaceMethod("outline", ["transcript_path: str", "provider: str"], "str", "\u2624",
                                "Weaves transcript + LLM into structured outline"),
            ],
            allowed_dependencies=["LLMProvider"],
            constraints=[
                "Must not modify transcript files \u2014 read-only analysis",
                "Must not import Whisper, VAD, or search dependencies",
                "Must delegate all LLM calls to LLMProvider \u2014 no direct API calls",
                "Must truncate transcripts that exceed model context limits",
            ],
        ),

        # ─── ☾ TimestampExtractor: Fold Time ───
        ModuleSpec(
            name="TimestampExtractor",
            description="Extracts burned-in timestamps from video frames using Qwen2.5-VL vision model.",
            methods=[
                InterfaceMethod("extract_timestamps", ["video_path: str", "num_frames: int"], "TimestampResult", "\u263e",
                                "Folds multiple frames through VLM \u2014 iterates to consensus timestamp"),
                InterfaceMethod("batch_rename", ["folder_path: str", "crop_ratio: float"], "list[RenameResult]", "\u263e",
                                "Recursively folds all videos through timestamp extraction + rename"),
            ],
            allowed_dependencies=["AudioIngest"],
            constraints=[
                "Must use majority voting across frames for timestamp consensus",
                "Must clean up temporary frame files after processing",
                "VLM model must be loaded lazily and cached (singleton)",
                "Must not import Whisper, VAD, or LLM dependencies",
                "Must support skip-if-already-renamed for batch operations",
            ],
        ),

        # ─── 𓂀 ProgressWitness: Witness ───
        ModuleSpec(
            name="ProgressWitness",
            description="Observes and reports pipeline progress. Pure logging \u2014 never modifies data.",
            methods=[
                InterfaceMethod("emit_progress", ["current: int", "total: int", "label: str"], "None", "\U000130c0",
                                "Witnesses progress state \u2014 logs without altering pipeline"),
                InterfaceMethod("emit_result", ["result: dict"], "None", "\U000130c0",
                                "Witnesses a completed result \u2014 JSON output for UI consumption"),
                InterfaceMethod("emit_error", ["error: str", "context: str"], "None", "\U000130c0",
                                "Witnesses an error \u2014 logs without attempting recovery"),
            ],
            allowed_dependencies=[],
            constraints=[
                "Must NEVER modify pipeline data or control flow",
                "Must NEVER raise exceptions that halt the pipeline",
                "Output must be line-buffered for real-time UI streaming",
                "Must use structured markers: [PROGRESS], [RESULT], [ERROR]",
                "Must be safe to remove entirely without affecting pipeline correctness",
            ],
        ),
    ],
    global_constraints=[
        "Pipeline flows: AudioIngest \u2192 VoiceDetector \u2192 Transcriber \u2192 (SearchEngine | MeetingDetector | Analyzer) (DAG, not cycle)",
        "TimestampExtractor is an independent branch from AudioIngest \u2014 no cross-dependency with Transcriber",
        "LLMProvider is a shared utility \u2014 MeetingDetector and Analyzer may both depend on it",
        "ProgressWitness has ZERO dependencies and ZERO dependents \u2014 pure observation",
        "No shared mutable state between modules (no globals, no singletons crossing module boundaries)",
        "All inter-module communication flows through declared interfaces only",
        "All modules must support graceful GPU \u2192 CPU fallback independently",
        "Transcript file format (filename_transcript_modelname.txt) is the universal contract between Transcriber and all consumers",
    ],
)


# ─────────────────────────────────────────────────────────────
#  Visualization Helpers
# ─────────────────────────────────────────────────────────────

def show_architecture_map():
    """Print the module dependency DAG."""
    print()
    print("  ┌─────────────┐")
    print("  │ AudioIngest  │ ◬ Origin — discovers & loads media")
    print("  └──────┬───┬──┘")
    print("         │   │")
    print("    ┌────┘   └──────────────────────────┐")
    print("    ▼                                    ▼")
    print("  ┌──────────────┐              ┌──────────────────┐")
    print("  │ VoiceDetector │ Ө Filter     │ TimestampExtractor│ ☾ Fold")
    print("  └──────┬───────┘              └──────────────────┘")
    print("         │")
    print("         ▼")
    print("  ┌──────────────┐")
    print("  │  Transcriber  │ ☾ Fold Time")
    print("  └──┬─────┬──┬──┘")
    print("     │     │  │")
    print("     ▼     │  ▼")
    print("  ┌──────┐ │ ┌─────────────────┐")
    print("  │Search│ │ │ MeetingDetector  │ Ө Filter")
    print("  │Engine│ │ └────────┬────────┘")
    print("  └──────┘ │          │")
    print("     Ө     │          ▼")
    print("           │  ┌─────────────┐")
    print("           │  │ LLMProvider  │ ☤ Weaver")
    print("           │  └──────┬──────┘")
    print("           │         │")
    print("           ▼         ▼")
    print("        ┌──────────────┐")
    print("        │   Analyzer    │ ☤ Weaver")
    print("        └──────────────┘")
    print()
    print("  ╔═══════════════════╗")
    print("  ║ ProgressWitness 𓂀 ║  (observes all — modifies nothing)")
    print("  ╚═══════════════════╝")
    print()


def show_module_stats(bridge):
    """Show token counts for each module prompt."""
    line = "\u2500" * 70
    print(line)
    print(f"{'Module':<22} {'Glyph':^7} {'Prompt Size':>12} {'Methods':>9}")
    print(line)
    total_tokens = 0
    for module in bridge.spec.modules:
        prompt = bridge.generate_system_prompt(module.name, "python")
        tokens = len(prompt.split())
        total_tokens += tokens
        glyphs = set(m.glyph for m in module.methods)
        glyph_str = " ".join(sorted(glyphs))
        print(f"  {module.name:<20} {glyph_str:^7} ~{tokens:>5} tokens  {len(module.methods):>3}")
    print(line)
    total_methods = sum(len(m.methods) for m in bridge.spec.modules)
    print(f"  {'TOTAL':<20} {'':^7} ~{total_tokens:>5} tokens  {total_methods:>3}")
    print()

    # Actual TurboScribe codebase measured sizes (bytes → tokens at ~4 bytes/token)
    # fast_engine.py: 63,571  timestamp_engine.py: 15,085
    # MainWindow.xaml.cs: 163,427  PythonRunner.cs: 34,942
    # MainWindow.xaml: 99,871  App.xaml: 33,876
    # PipInstaller.cs: 8,869  AnalyticsService.cs: 9,343
    # ScanResult.cs: 2,073  App.xaml.cs: 1,376  Other: ~6,100
    # Total: ~438,533 bytes → ~109,633 tokens
    codebase_tokens = 109_633
    avg_module = total_tokens // len(bridge.spec.modules)
    print(f"  TurboScribe codebase: ~{codebase_tokens:,} tokens ({438_533:,} bytes)")
    print(f"  SCP full spec:        ~{total_tokens:,} tokens (all 9 modules)")
    print(f"  SCP per-module avg:   ~{avg_module:,} tokens (what the AI actually sees)")
    print()
    print(f"  Compression: {codebase_tokens // max(total_tokens, 1)}\u00d7 (full spec) / {codebase_tokens // max(avg_module, 1)}\u00d7 (single module)")
    print()


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

SUPPORTED_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
]
DEFAULT_MODEL = "gemini-3-pro-preview"


def generate_module(bridge, module_name, model, client, output_dir=None):
    """Generate a single module and optionally save to output_dir. Returns (code, verification) or None."""
    system_prompt = bridge.generate_system_prompt(module_name, language="python")

    print(f"\n◬ ─── Sending to {model}: Implement {module_name} ───")
    print(f"   Prompt size: {len(system_prompt)} chars (~{len(system_prompt.split())} tokens)")

    try:
        from google import genai

        response = client.models.generate_content(
            model=model,
            contents=f"Implement the {module_name} module now.",
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.1,
            ),
        )
        code = response.text

        # ─── Verify with Weaver ───
        print(f"☤ ─── Weaver Verification: {module_name} ───")
        verify_prompt = bridge.generate_verification_prompt(module_name, code)
        verify = client.models.generate_content(
            model=model,
            contents=verify_prompt,
            config=genai.types.GenerateContentConfig(temperature=0.0),
        )
        verification = verify.text

        # Determine pass/fail
        passed = "PASS" in verification.upper().split("\n")[0] if verification else False
        status = "✔ PASS" if passed else "✘ FAIL"
        print(f"   {status}")

        # Save to output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

            # Save generated code
            code_file = os.path.join(output_dir, f"{module_name.lower()}.py")
            # Extract Python code from markdown fences if present
            clean_code = code
            if "```python" in code:
                import re
                match = re.search(r'```python\s*\n(.*?)```', code, re.DOTALL)
                if match:
                    clean_code = match.group(1)

            with open(code_file, 'w', encoding='utf-8') as f:
                f.write(clean_code)

            # Save full output (code + verification)
            output_file = os.path.join(output_dir, f"{module_name.lower()}_output.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"◬ SCP Generated: {module_name}\n")
                f.write(f"Model: {model}\n")
                f.write(f"{'=' * 60}\n\n")
                f.write(code)
                f.write(f"\n\n{'=' * 60}\n")
                f.write(f"☤ Weaver Verification\n{'=' * 60}\n\n")
                f.write(verification)

            print(f"   Saved: {code_file}")

        return code, verification

    except Exception as e:
        print(f"   ⚠ Error generating {module_name}: {e}")
        return None


def generate_tests(bridge, module_name, code, model, client, output_dir):
    """Generate pytest tests from SCP contract and run them. Returns 'PASS', 'FAIL', or 'ERROR'."""
    import re
    import subprocess

    print(f"🧪 ─── Test Generation: {module_name} ───")
    test_prompt = bridge.generate_test_prompt(module_name, code, language="python")
    print(f"   Prompt size: {len(test_prompt)} chars (~{len(test_prompt.split())} tokens)")

    try:
        from google import genai
        response = client.models.generate_content(
            model=model,
            contents=f"Generate pytest tests for the {module_name} module now.",
            config=genai.types.GenerateContentConfig(
                system_instruction=test_prompt, temperature=0.1,
            ),
        )
        test_code = response.text

        # Strip markdown fences
        clean_test = test_code
        if "```python" in test_code:
            match = re.search(r'```python\s*\n(.*?)```', test_code, re.DOTALL)
            if match:
                clean_test = match.group(1)
        elif "```" in test_code:
            match = re.search(r'```\s*\n(.*?)```', test_code, re.DOTALL)
            if match:
                clean_test = match.group(1)

        # Save test file
        os.makedirs(output_dir, exist_ok=True)
        test_file = os.path.join(output_dir, f"test_{module_name.lower()}.py")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(clean_test)
        print(f"   Saved: {test_file}")

        # Run pytest
        print(f"   Running: pytest {test_file} -v --tb=short")
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short", "--no-header"],
            capture_output=True, text=True, cwd=output_dir, timeout=60,
        )
        # Print output (trim to reasonable length)
        output = result.stdout + result.stderr
        for line in output.strip().split("\n")[-20:]:
            print(f"   {line}")

        status = "PASS" if result.returncode == 0 else "FAIL"
        print(f"   pytest: {'✔ PASS' if status == 'PASS' else '✘ FAIL'}")
        return status

    except subprocess.TimeoutExpired:
        print(f"   ⚠ pytest timed out (60s)")
        return "TIMEOUT"
    except Exception as e:
        print(f"   ⚠ Error generating tests: {e}")
        return "ERROR"


def generate_main_driver(output_dir: str, module_names: list, bridge):
    """Generate a main.py driver that imports and wires all generated modules."""
    modules_lower = [m.lower() for m in module_names]

    # Build import lines
    imports = []
    for mod in module_names:
        imports.append(f"    from {mod.lower()} import *")
    imports_block = "\n".join(imports)

    # Build module descriptions for help
    help_lines = []
    for mod in bridge.spec.modules:
        if mod.name in module_names:
            glyphs = set(m.glyph for m in mod.methods)
            glyph_str = " ".join(sorted(glyphs))
            help_lines.append(f'    print("  {glyph_str:<6} {mod.name:<22} {mod.description[:50]}")')
    help_block = "\n".join(help_lines)

    driver_code = f'''"""
TurboScribe — SCP-Generated Application
=========================================
This entire application was generated module-by-module from SCP contracts.
Each module was independently produced by Gemini, verified by the Weaver,
and assembled here into a working pipeline.

Usage:
    python main.py                          # Show architecture overview
    python main.py transcribe <file>        # Transcribe a single file
    python main.py transcribe-dir <dir>     # Batch transcribe a directory
    python main.py search <dir> <query>     # Search transcripts
    python main.py analyze <transcript>     # Analyze/summarize a transcript
"""

import sys
import os

# ─── Import all generated modules ───
# Each module was independently generated from its SCP contract.
# They communicate only through their declared interfaces.
try:
{imports_block}
except ImportError as e:
    print(f"⚠ Module import error: {{e}}")
    print("  Some modules may not have been generated successfully.")
    print()


def show_help():
    print("=" * 60)
    print("◬  TurboScribe — SCP-Generated Application")
    print("=" * 60)
    print()
    print("Modules:")
{help_block}
    print()
    print("Commands:")
    print("  transcribe <file>          Transcribe an audio/video file")
    print("  transcribe-dir <dir>       Batch transcribe all files in directory")
    print("  search <dir> <query>       Search transcripts for text")
    print("  analyze <file>             Analyze/summarize a transcript")
    print()
    print("Options:")
    print("  --model <name>             Whisper model (default: base.en)")
    print("  --beam-size <n>            Beam size (default: 5)")
    print()
    print("Examples:")
    print("  python main.py transcribe meeting.wav")
    print("  python main.py transcribe-dir ./recordings --model large-v3")
    print("  python main.py search ./transcripts \\"quarterly results\\"")
    print()


def cmd_transcribe(args):
    """Transcribe a single file."""
    if not args:
        print("Usage: python main.py transcribe <file> [--model <name>]")
        return

    file_path = args[0]
    model_name = "base.en"
    beam_size = 5

    i = 1
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            i += 1
            model_name = args[i]
        elif args[i] == "--beam-size" and i + 1 < len(args):
            i += 1
            beam_size = int(args[i])
        i += 1

    if not os.path.exists(file_path):
        print(f"⚠ File not found: {{file_path}}")
        return

    try:
        result = transcribe_file(file_path, model_name, beam_size)
        print(f"\\n──── Transcript ────")
        print(result.full_text)
        print(f"\\n──── Stats ────")
        print(f"  Model: {{result.model_name}}")
        print(f"  Segments: {{len(result.segments)}}")
    except Exception as e:
        print(f"⚠ Error: {{e}}")


def cmd_transcribe_dir(args):
    """Batch transcribe a directory."""
    if not args:
        print("Usage: python main.py transcribe-dir <directory> [--model <name>]")
        return

    directory = args[0]
    model_name = "base.en"
    beam_size = 5

    i = 1
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            i += 1
            model_name = args[i]
        elif args[i] == "--beam-size" and i + 1 < len(args):
            i += 1
            beam_size = int(args[i])
        i += 1

    if not os.path.isdir(directory):
        print(f"⚠ Directory not found: {{directory}}")
        return

    try:
        results = batch_transcribe(directory, model_name, beam_size)
        print(f"\\n𓂀 Transcribed {{len(results)}} files.")
        for r in results:
            status = "✔" if r.full_text else "✘"
            print(f"  {{status}} {{os.path.basename(r.file_path)}}")
    except Exception as e:
        print(f"⚠ Error: {{e}}")


def cmd_search(args):
    """Search transcripts."""
    if len(args) < 2:
        print("Usage: python main.py search <directory> <query>")
        return

    directory = args[0]
    query = " ".join(args[1:])

    try:
        results = search_transcripts(directory, query)
        if not results:
            print(f"No results for: {{query}}")
        else:
            print(f"\\nFound {{len(results)}} matches for \\"{{query}}\\":")
            for r in results:
                print(f"  • {{r}}")
    except NameError:
        # Fallback: simple grep-like search
        print(f"Searching for \\"{{query}}\\" in {{directory}}...")
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.endswith("_transcript.txt") or f.endswith("_transcript_base.en.txt"):
                    path = os.path.join(root, f)
                    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                        text = fh.read()
                    if query.lower() in text.lower():
                        print(f"  ✔ {{os.path.relpath(path, directory)}}")
    except Exception as e:
        print(f"⚠ Error: {{e}}")


def cmd_analyze(args):
    """Analyze a transcript."""
    if not args:
        print("Usage: python main.py analyze <transcript_file>")
        return

    file_path = args[0]
    if not os.path.exists(file_path):
        print(f"⚠ File not found: {{file_path}}")
        return

    try:
        result = analyze_transcript(file_path)
        print(result)
    except NameError:
        # Fallback: show basic stats
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        words = len(text.split())
        lines = text.count("\\n") + 1
        print(f"\\n──── Transcript Analysis ────")
        print(f"  File: {{os.path.basename(file_path)}}")
        print(f"  Words: {{words:,}}")
        print(f"  Lines: {{lines:,}}")
        print(f"  Characters: {{len(text):,}}")
    except Exception as e:
        print(f"⚠ Error: {{e}}")


def main():
    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1].lower()
    args = sys.argv[2:]

    commands = {{
        "transcribe": cmd_transcribe,
        "transcribe-dir": cmd_transcribe_dir,
        "search": cmd_search,
        "analyze": cmd_analyze,
        "help": lambda a: show_help(),
        "--help": lambda a: show_help(),
        "-h": lambda a: show_help(),
    }}

    if command in commands:
        commands[command](args)
    else:
        print(f"Unknown command: {{command}}")
        show_help()


if __name__ == "__main__":
    main()
'''

    main_path = os.path.join(output_dir, "main.py")
    with open(main_path, 'w', encoding='utf-8') as f:
        f.write(driver_code)


def main():

    # Build bridge from our standalone spec (no scp_bridge.py modification)
    bridge = SCPBridge(TURBOSCRIBE_SPEC)

    # Parse args
    module_name = None
    use_gemini = False
    generate_all = False
    with_tests = False
    model = DEFAULT_MODEL
    output_dir = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--gemini":
            use_gemini = True
        elif args[i] == "--all":
            generate_all = True
            use_gemini = True  # --all implies --gemini
        elif args[i] == "--with-tests":
            with_tests = True
        elif args[i] == "--model" and i + 1 < len(args):
            i += 1
            model = args[i]
            if model not in SUPPORTED_MODELS:
                print(f"\n⚠ Unknown model: {model}")
                print(f"  Supported: {', '.join(SUPPORTED_MODELS)}")
                print(f"  Proceeding anyway (the API will reject if invalid).\n")
        elif args[i] == "--output-dir" and i + 1 < len(args):
            i += 1
            output_dir = args[i]
        elif not args[i].startswith("-"):
            module_name = args[i]
        i += 1

    # Default output directory for --all
    if generate_all and output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "generated")

    if generate_all:
        # ─── Generate ALL modules ───
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("\n⚠ No GEMINI_API_KEY found. Set it first:")
            print("  set GEMINI_API_KEY=your-key\n")
            return

        try:
            from google import genai
            client = genai.Client(api_key=api_key)
        except ImportError:
            print("\n⚠ google-genai not installed. Run: pip install google-genai\n")
            return

        modules = [m.name for m in bridge.spec.modules]
        api_calls_per_mod = 3 if with_tests else 2
        est_sec_per_call = 30  # rough estimate per Gemini API call
        est_total = len(modules) * api_calls_per_mod * est_sec_per_call
        est_min = est_total / 60
        print("=" * 70)
        print("◬  SCP Batch Generation: TurboScribe")
        print(f"   Generating {len(modules)} modules with {model}")
        if with_tests:
            print("   Tests: enabled (will generate + run pytest)")
        print(f"   Output: {output_dir}")
        print(f"   Estimated time: ~{est_min:.0f} min ({api_calls_per_mod} API calls × {len(modules)} modules)")
        print("=" * 70)

        import time
        batch_start = time.time()
        results = {}
        timings = {}
        for idx, mod_name in enumerate(modules, 1):
            mod_start = time.time()
            elapsed = mod_start - batch_start
            if idx > 1 and timings:
                avg_per_mod = elapsed / (idx - 1)
                remaining = avg_per_mod * (len(modules) - idx + 1)
                eta_min = remaining / 60
                print(f"\n{'─' * 70}")
                print(f"  [{idx}/{len(modules)}] {mod_name}  (ETA: ~{eta_min:.1f} min remaining)")
                print(f"{'─' * 70}")
            else:
                print(f"\n{'─' * 70}")
                print(f"  [{idx}/{len(modules)}] {mod_name}")
                print(f"{'─' * 70}")

            result = generate_module(bridge, mod_name, model, client, output_dir)
            if result:
                code, verification = result
                passed = "PASS" in verification.upper().split("\n")[0] if verification else False
                results[mod_name] = "PASS" if passed else "FAIL"
                # Generate and run tests if requested
                if with_tests and output_dir:
                    test_status = generate_tests(bridge, mod_name, code, model, client, output_dir)
                    results[mod_name] = f"{results[mod_name]}/{test_status}"
            else:
                results[mod_name] = "ERROR"
            timings[mod_name] = time.time() - mod_start

        # ─── Generate main.py driver ───
        print(f"\n{'─' * 70}")
        print(f"  Generating main.py driver...")
        print(f"{'─' * 70}")

        successful_modules = [m for m, s in results.items() if s != "ERROR"]
        generate_main_driver(output_dir, successful_modules, bridge)
        print(f"   Saved: {os.path.join(output_dir, 'main.py')}")

        # ─── Summary ───
        print(f"\n{'=' * 70}")
        title = "𓂀  Batch Generation Complete!" + (" (with tests)" if with_tests else "")
        print(title)
        print(f"{'=' * 70}\n")

        if with_tests:
            print(f"  {'Module':<25} {'Weaver':^8} {'Pytest':^8}")
            print(f"  {'─' * 45}")
            for mod_name, status in results.items():
                parts = status.split("/") if "/" in status else [status, "—"]
                w_icon = "✔" if parts[0] == "PASS" else "✘"
                t_icon = "✔" if len(parts) > 1 and parts[1] == "PASS" else ("✘" if len(parts) > 1 and parts[1] not in ("—", "ERROR") else "—")
                print(f"  {w_icon} {mod_name:<24} {parts[0]:^8} {parts[1] if len(parts) > 1 else '—':^8}")
        else:
            for mod_name, status in results.items():
                icon = "✔" if status == "PASS" else "✘"
                print(f"  {icon} {mod_name:<25} {status}")

        passed = sum(1 for s in results.values() if s.startswith("PASS"))
        total_elapsed = time.time() - batch_start
        print(f"\n  {passed}/{len(results)} modules passed Weaver verification")
        print(f"  Total time: {total_elapsed / 60:.1f} min ({total_elapsed:.0f}s)")
        print()
        print(f"  Per-module timing:")
        for mod_name, t in timings.items():
            print(f"    {mod_name:<25} {t:.1f}s")
        print(f"  Output saved to: {output_dir}/")
        print()
        print("  Run the generated app:")
        print(f"    python {os.path.join(output_dir, 'main.py')}")
        print(f"    python {os.path.join(output_dir, 'main.py')} transcribe meeting.wav")
        print(f"    python {os.path.join(output_dir, 'main.py')} search \"quarterly results\"")
        print()
        return


    if module_name is None:
        # ─── Full architecture overview ───
        print("=" * 70)
        print("\u25ec  SCP Architecture: TurboScribe")
        print("   GPU-Accelerated Audio Transcription & Analysis")
        print("=" * 70)

        show_architecture_map()
        show_module_stats(bridge)

        print("Generate a single module:")
        print()
        for module in bridge.spec.modules:
            print(f"  python examples/turboscribe_example.py {module.name} --gemini")
        print()
        print("Generate ALL modules at once:")
        print("  python examples/turboscribe_example.py --all")
        print()
        print("Generate ALL modules + auto-generate and run tests:")
        print("  python examples/turboscribe_example.py --all --with-tests")
        print()
        print("Choose a model (default: gemini-3-pro-preview):")
        for m in SUPPORTED_MODELS:
            tag = " ← default" if m == DEFAULT_MODEL else ""
            print(f"  python examples/turboscribe_example.py --all --model {m}{tag}")
        return

    # ─── Generate prompt for specific module ───
    system_prompt = bridge.generate_system_prompt(module_name, language="python")

    print("=" * 70)
    print(f"\u25ec  SCP System Prompt: {module_name}")
    print(f"   Size: {len(system_prompt)} chars (~{len(system_prompt.split())} tokens)")
    print("=" * 70)

    if use_gemini:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("\n\u26a0 No GEMINI_API_KEY found. Set it first:")
            print("  set GEMINI_API_KEY=your-key")
            print("\nFalling back to prompt display.\n")
        else:
            try:
                from google import genai

                client = genai.Client(api_key=api_key)

                result = generate_module(bridge, module_name, model, client, output_dir)
                if result:
                    code, verification = result
                    if not output_dir:
                        # Print to console if not saving to file
                        print(f"\n{'─' * 70}")
                        print(code)
                        print(f"\n{'─' * 70}")
                        print(verification)
                return

            except ImportError:
                print("\n\u26a0 google-genai not installed. Run: pip install google-genai\n")

    # ─── Fallback: print the prompt ───
    print()
    print("\u2500" * 70)
    print(system_prompt)
    print("\u2500" * 70)
    print()
    print(f"Copy the above into any AI chat (Gemini, ChatGPT, Claude),")
    print(f"then tell the AI: 'Implement the {module_name} module now.'")
    print()


if __name__ == "__main__":
    main()

