# SCP Code Generation — How It Works

> An AI generates production-quality module code from a ~815 token orthogonal contract,
> then System 2 rejection sampling (The Weaver) verifies it against SCP rules. Zero human prompting required.

---

## What Just Happened

We ran one command:

```bash
python examples/turboscribe_example.py Transcriber --gemini --model gemini-3-pro-preview
```

This produced a complete, working `Transcriber` module — 260+ lines of Python
with proper error handling, GPU fallback, caching, and progress reporting —
**without the AI ever seeing the TurboScribe codebase**. By using orthogonal Uiua embeddings, SCP creates steep attractor basins that prevent the model from confabulating across module boundaries.

### The Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│  Step 1: SCP Bridge generates a constrained system prompt       │
│          (~815 tokens — what the module does, what it can see,   │
│           what it must never touch)                              │
├──────────────────────────────────────────────────────────────────┤
│  Step 2: Gemini 3 Pro receives the prompt and generates code    │
│          (the AI sees ONLY the contract — not the codebase)     │
├──────────────────────────────────────────────────────────────────┤
│  Step 3: Weaver (System 2) — AST rejection sampling audits      │
│          the generated code for coupling violations              │
├──────────────────────────────────────────────────────────────────┤
│  Result: PASS (W(G)=0) / REJECT & Resample                     │
└──────────────────────────────────────────────────────────────────┘
```

## What the AI Saw (815 tokens)

The system prompt told Gemini 3 Pro:

| Section | Content |
|---------|---------|
| **Module contract** | 4 methods with exact signatures, return types, and glyph assignments |
| **Glyph rules** | ◬ Origin = load once, singleton; ☾ Fold Time = must have base case |
| **Allowed dependencies** | `AudioIngest` and `VoiceDetector` — interfaces only, no implementations |
| **Forbidden zones** | ❌ SearchEngine, MeetingDetector, LLMProvider, Analyzer, TimestampExtractor, ProgressWitness |
| **Constraints** | Save as `filename_transcript_modelname.txt`, skip existing, GPU→CPU fallback, `[PROGRESS]` markers |

## What the AI Did NOT See

- ❌ The full TurboScribe codebase (~110,000 tokens)
- ❌ `AudioIngest` implementation (only its 3-method interface)
- ❌ Any other module's code
- ❌ The database schema, UI layer, or deployment config

This is **RAG Denial** in action — the AI cannot confabulate cross-module
coupling because the other modules don't exist in its context. The orthogonal
embeddings create steep, isolated attractor basins that pierce through context noise.

## What the AI Generated

📁 [`transcriber.py`](transcriber.py) — the full generated module.

### Key Features the AI Produced

| Feature | How It Works |
|---------|-------------|
| **GPU→CPU fallback** | `load_model()` tries CUDA first, falls back to CPU with int8 |
| **Model caching** | Singleton pattern — loads once, returns cached on subsequent calls |
| **Skip existing** | Checks for `filename_transcript_modelname.txt` before re-transcribing |
| **Progress markers** | Emits `[PROGRESS]` on every significant event |
| **Segment-level timing** | Each `TranscriptSegment` has start/end timestamps |
| **Batch processing** | `batch_transcribe()` discovers files via `AudioIngest.find_media()` |
| **Time-range transcription** | `transcribe_segment()` slices audio at 16kHz and adjusts timestamps |

### Glyph Compliance

| Method | Glyph | Contract | ✔/✘ |
|--------|:-----:|----------|:---:|
| `load_model` | ◬ | Singleton origin, loads once | ✔ |
| `transcribe_file` | ☾ | Folds audio→text, base case = existing file | ✔ |
| `batch_transcribe` | ☾ | Folds directory→transcripts, base case = empty dir | ✔ |
| `transcribe_segment` | ☾ | Folds time slice→text, base case = empty slice | ✔ |

## Weaver Verification Result (System 2 Rejection Sampling)

```
PASS — W(G) = 0 (No undeclared coupling detected)

1. Interface Conformance: All 4 methods match required signatures
2. Dependency Isolation: Only AudioIngest and VoiceDetector imported (orthogonal)
3. Constraint Compliance: Correct naming, skip_existing, fallback, [PROGRESS]
4. Coupling Detection: MI_AST = 0 for all non-edge module pairs
5. Glyph Contracts: All methods follow their assigned primitive rules
```

## Try It Yourself

### Generate a module prompt (no API key needed)

```bash
# See what the AI would receive
python examples/turboscribe_example.py AudioIngest
python examples/turboscribe_example.py VoiceDetector
python examples/turboscribe_example.py MeetingDetector
```

### Generate + verify with Gemini

```bash
# Set your API key
set GEMINI_API_KEY=your-key

# Generate any of the 9 modules
python examples/turboscribe_example.py AudioIngest --gemini
python examples/turboscribe_example.py Transcriber --gemini
python examples/turboscribe_example.py SearchEngine --gemini

# Use a different model
python examples/turboscribe_example.py Transcriber --gemini --model gemini-2.5-flash
```

### Available models

| Model | Best For |
|-------|----------|
| `gemini-3-pro-preview` | Complex modules with many constraints (default) |
| `gemini-3-flash-preview` | Fast iteration on simple modules |
| `gemini-2.5-pro` | Strong reasoning, good constraint adherence |
| `gemini-2.5-flash` | Cheapest option, good for ProgressWitness-level simplicity |

## What This Means for Your Projects

You can use this same process for **any** codebase:

1. **Decompose** your monolith into modules with clear boundaries
2. **Assign glyphs** — what is each module's primary operation?
3. **Define contracts** — methods, constraints, dependencies, forbidden zones
4. **Generate** — `python examples/turboscribe_example.py ModuleName --gemini`
5. **Verify** — The Weaver (System 2 rejection sampling) catches coupling violations automatically

The AI generates code that fits your architecture **by construction**, not by
coincidence. Orthogonal Uiua embeddings create steep attractor basins free of semantic
cross-talk. It physically cannot import forbidden modules because they aren't
in context. If the Weaver detects W(G) > 0, it rejects and resamples until orthogonality is restored.

## Files in This Directory

| File | Purpose |
|------|---------|
| [`transcriber.py`](transcriber.py) | AI-generated Transcriber module (Gemini 3 Pro) |
| [`gemini3_transcriber_output.txt`](gemini3_transcriber_output.txt) | Raw terminal output including Weaver verification |
| [`README.md`](README.md) | This file |
