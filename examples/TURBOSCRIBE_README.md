# TurboScribe × Chevron — SCP Case Study

> How the Spatial Constraint Protocol reduced a 110,000-token codebase to
> ~660 tokens per AI prompt — while **eliminating hallucination-driven
> regressions** that plague traditional AI code generation.

---

## The Problem

TurboScribe is a GPU-accelerated audio transcription and meeting detection
application (~7,500 LOC across Python + C#/WPF). Its Python backend
(`fast_engine.py`) is a 1,520-line monolith handling 10 distinct modes:
VAD scanning, batch transcription, semantic search, meeting detection, LLM
analysis, and timestamp extraction.

When you ask an AI to modify this codebase, it faces a fundamental challenge:

| Approach | Tokens In | Risk |
|---|---:|---|
| Paste entire codebase | **~110,000** | Exceeds most context windows; AI loses focus |
| Paste one file | **~16,000** | AI sees internal details of unrelated modes, hallucinates cross-dependencies |
| Describe changes in English | **~500** | AI invents its own contracts, breaking existing interfaces |

Every approach leads to the same failure: **AI-generated code that silently
couples modules that were previously independent**, introducing regressions
that only surface at runtime.

## The SCP Solution

The Spatial Constraint Protocol (SCP) replaces "paste everything and hope"
with **deterministic architectural contracts**. Instead of showing the AI
your code, you show it:

1. **What this module does** (interface contract)
2. **What it may depend on** (dependency list — interfaces only, not implementations)
3. **What it must never touch** (forbidden zones via RAG Denial)
4. **How each method behaves** (glyph semantics — Origin, Filter, Fold, Witness, Weaver)

The AI never sees any other module's implementation. It **cannot** hallucinate
cross-module coupling because the coupling surfaces are simply not in context.

## TurboScribe Decomposition

The monolithic `fast_engine.py` was decomposed into 9 isolated SCP modules:

```
  ┌─────────────┐
  │ AudioIngest  │ ◬ Origin — discovers & loads media
  └──────┬───┬──┘
         │   │
    ┌────┘   └──────────────────────────┐
    ▼                                    ▼
  ┌──────────────┐              ┌──────────────────┐
  │ VoiceDetector │ Ө Filter     │ TimestampExtractor│ ☾ Fold
  └──────┬───────┘              └──────────────────┘
         │
         ▼
  ┌──────────────┐
  │  Transcriber  │ ☾ Fold Time
  └──┬─────┬──┬──┘
     │     │  │
     ▼     │  ▼
  ┌──────┐ │ ┌─────────────────┐
  │Search│ │ │ MeetingDetector  │ Ө Filter
  │Engine│ │ └────────┬────────┘
  └──────┘ │          │
     Ө     │          ▼
           │  ┌─────────────┐
           │  │ LLMProvider  │ ☤ Weaver
           │  └──────┬──────┘
           │         │
           ▼         ▼
        ┌──────────────┐
        │   Analyzer    │ ☤ Weaver
        └──────────────┘

  ╔═══════════════════╗
  ║ ProgressWitness 𓂀 ║  (observes all — modifies nothing)
  ╚═══════════════════╝
```

### Module Summary

| Module | Glyph | Role | Methods | Key Constraint |
|---|:---:|---|:---:|---|
| **AudioIngest** | ◬ | File discovery & audio loading | 3 | No ML imports — raw I/O only |
| **VoiceDetector** | Ө | Speech/silence segmentation | 3 | No transcription — detection only |
| **Transcriber** | ☾ | Whisper batch transcription | 4 | No analysis or search |
| **SearchEngine** | Ө | Keyword & semantic search | 2 | Read-only — no transcript mutation |
| **MeetingDetector** | Ө | Real vs hallucinated meetings | 2 | No file modification |
| **LLMProvider** | ☤ | Unified local/cloud LLM access | 3 | No domain logic |
| **Analyzer** | ☤ | Summarization & outlining | 2 | Delegates all LLM calls |
| **TimestampExtractor** | ☾ | Video timestamp OCR via VLM | 2 | Independent of transcription |
| **ProgressWitness** | 𓂀 | Pure progress logging | 3 | Zero side effects |

## Compression Results

```
┌──────────────────────────────────────────────────────────────────┐
│                    Context Window Usage                          │
├─────────────────────────────┬────────────────────────────────────┤
│ TurboScribe full codebase   │  ~109,633 tokens  (438,533 bytes) │
│ SCP full spec (all modules) │   ~5,920 tokens                   │
│ SCP single module prompt    │     ~660 tokens   (avg)           │
├─────────────────────────────┼────────────────────────────────────┤
│ Full-spec compression       │  18×                              │
│ Per-module compression      │  166×                             │
└─────────────────────────────┴────────────────────────────────────┘
```

**What this means in practice:**

- The AI sees **~660 tokens** of precise contracts instead of **~110,000 tokens**
  of raw code. This leaves 99.4% of the context window free for reasoning.
- A 4K context window model can implement any single TurboScribe module.
- A 128K context window model never needs to page or truncate — it processes
  the entire module spec in one shot with massive room for chain-of-thought.

## How SCP Prevents Regressions

### 1. RAG Denial

When implementing the `Transcriber` module, the AI sees:

- ✅ Transcriber's own contract (methods, constraints, glyph rules)
- ✅ AudioIngest and VoiceDetector **interfaces** (what they return)
- ❌ AudioIngest implementation (how they load files)
- ❌ SearchEngine, MeetingDetector, Analyzer (everything downstream)
- ❌ TimestampExtractor (independent branch)

The AI **cannot** accidentally import `SearchEngine` internals into
`Transcriber` because `SearchEngine` doesn't exist in its context.

### 2. Glyph Contracts

Each method is governed by a Chevron primitive that enforces behavioral rules:

| Glyph | Name | Rule | Violation Example |
|:---:|---|---|---|
| ◬ | Origin | Appears once per program, no nesting | Calling `load_model()` inside a loop |
| ☾ | Fold Time | Must have a reachable base case | Infinite recursion in `batch_transcribe()` |
| Ө | Filter | Pure gate — input passes or is rejected | `keyword_search()` modifying transcripts |
| 𓂀 | Witness | Observe-only — zero side effects | `emit_progress()` raising an exception |
| ☤ | Weaver | Braids multiple streams into one | `generate()` returning raw API response |

The AI cannot generate a `Transcriber.transcribe_file()` that also does
search, because the ☾ (Fold Time) glyph requires a pure fold operation
with a reachable base case — not a multi-purpose function.

### 3. Weaver Verification

After the AI generates code, the Weaver (☤) produces a verification
prompt that checks:

- All declared methods are implemented
- No forbidden imports are present
- Glyph contracts are satisfied
- Dependency boundaries are respected

A passing verification returns `W(G) = 0`, meaning zero coupling violations.

## How This Was Constructed

### Step 1: Analyze the Monolith

The original `fast_engine.py` was analyzed to identify natural boundaries:

```python
# 10 modes found in fast_engine.py:
#   mode 1: transcribe_file   → Transcriber
#   mode 2: batch_transcribe   → Transcriber
#   mode 3: transcribe_dir     → Transcriber
#   mode 4: vad_scan           → VoiceDetector
#   mode 5: batch_vad_scan     → VoiceDetector
#   mode 6: search             → SearchEngine
#   mode 7: semantic_search    → SearchEngine
#   mode 8: run_server         → (excluded — HTTP wrapper)
#   mode 9: analyze            → Analyzer
#   mode 10: detect_meetings   → MeetingDetector
```

Shared concerns were extracted:
- Audio loading → `AudioIngest`
- LLM calls → `LLMProvider`
- Progress output → `ProgressWitness`
- `timestamp_engine.py` → `TimestampExtractor`

### Step 2: Assign Glyphs

Each module's primary operation was mapped to a Chevron primitive:

- **Data entry points** (file discovery, model loading) → ◬ Origin
- **Batch/recursive operations** (transcription, timestamp extraction) → ☾ Fold Time
- **Pass/reject gates** (search, VAD, meeting detection) → Ө Filter
- **Pure observers** (progress logging) → 𓂀 Witness
- **Multi-stream combiners** (LLM + prompt → response) → ☤ Weaver

### Step 3: Define Contracts

For each module, we specified:
1. **Methods** — exact signatures with input/output types
2. **Glyph per method** — which primitive governs its behavior
3. **Constraints** — what the module must NOT do
4. **Dependencies** — which other modules' interfaces it may call

### Step 4: Generate & Verify

```bash
# Generate SCP prompt for any module
python examples/turboscribe_example.py Transcriber

# Generate + implement + verify with Gemini
set GEMINI_API_KEY=your-key
python examples/turboscribe_example.py Transcriber --gemini
```

The Gemini output was verified by the Weaver: **PASS: W(G) = 0**.

## Quick Start

```bash
# See the full architecture
python examples/turboscribe_example.py

# Generate a module prompt (copy into any AI chat)
python examples/turboscribe_example.py AudioIngest

# End-to-end with Gemini
set GEMINI_API_KEY=your-key
python examples/turboscribe_example.py Transcriber --gemini
```

## File Structure

```
chevron/
├── scp_bridge.py                    # Core SCP Bridge (NOT modified)
├── examples/
│   ├── turboscribe.chevron          # Architecture in Chevron glyph DSL
│   ├── turboscribe_example.py       # Standalone TurboScribe SCP spec + CLI
│   ├── TURBOSCRIBE_README.md        # This file
│   └── gemini_example.py            # Generic Gemini integration example
```

## Why This Matters

Traditional AI code generation asks the model to understand your **entire
codebase** before making a change. This is like asking a surgeon to memorize
every patient in the hospital before performing one operation.

SCP flips this: the AI sees **only the operating table** — the one module it
needs to implement, with precise contracts for everything it touches. The
result is code that fits into the existing architecture by construction,
not by coincidence.

For TurboScribe, this means:
- **18× less context** for the full architecture spec
- **166× less context** per individual module
- **Zero coupling violations** verified by the Weaver
- **Any AI model** (even 4K context) can implement a single module correctly
