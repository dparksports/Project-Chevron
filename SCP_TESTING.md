# 🧪 SCP Auto-Testing — Contract-Driven Test Generation

## What It Does

The `--with-tests` flag adds a third verification step to the SCP pipeline:

```
1. Generate module code   → from SCP contract (system prompt)
2. Weaver verification    → structural compliance check (AI-driven)
3. Auto-test generation   → pytest tests from contract, executed automatically
```

Tests are generated from the **SCP contract specification**, not from reading the implementation.
This means they verify that the generated code conforms to the *architecture spec*
rather than merely testing that the code does what it does.

## How to Use

```bash
# Generate all modules + generate and run tests
python examples/turboscribe_example.py --all --with-tests

# With a specific model and output directory
python examples/turboscribe_example.py --all --with-tests --model gemini-2.5-flash --output-dir ./generated
```

Output for each module:
```
examples/generated/
├── transcriber.py            # generated module
├── transcriber_output.txt    # full generation log
├── test_transcriber.py       # auto-generated pytest tests
├── audioingest.py
├── test_audioingest.py
├── ...
└── main.py                   # auto-generated driver
```

## How It Works

### Step 1: Module Generation (existing)
The SCP Bridge generates a **system prompt** from the architecture spec, constraining the AI to implement
only the specified module with only its declared dependencies visible (RAG Denial pattern).

### Step 2: Weaver Verification (existing)
The AI reviews the generated code against the SCP checklist: method signatures, dependency isolation,
constraint compliance, and coupling detection (W(G) = 0).

### Step 3: Test Generation (new — `--with-tests`)
A **test generation prompt** is created from the same SCP contract. The AI generates pytest tests
that verify four categories:

| Category | What It Tests | How |
|---|---|---|
| **Structural** (`test_interface_*`) | Methods exist with correct signatures | `hasattr()`, `inspect.signature()` |
| **Constraint** (`test_constraint_*`) | Module-specific rules | AST inspection, source parsing |
| **Behavioral** (`test_behavior_*`) | Return types, edge cases, error handling | Mocked calls with assertions |
| **Isolation** (`test_isolation_*`) | No forbidden imports, no global state | `ast.parse()` on source |

Tests use `unittest.mock` for all external dependencies — no GPU, no API keys, no file I/O required.

### Step 4: Pytest Execution (automatic)
Each test file is run with `pytest -v --tb=short`. Results are reported per-module:

```
𓂀  Batch Generation Complete! (with tests)
══════════════════════════════════════════════════════════════

  Module                    Weaver    Pytest
  ─────────────────────────────────────────────
  ✔ AudioIngest              PASS      PASS
  ✔ VoiceDetector            PASS      PASS
  ✘ Transcriber              PASS      FAIL
  ✔ SearchEngine             PASS      PASS
  ...

  7/9 modules passed Weaver verification
```

## Score Interpretation

| Weaver | Pytest | Meaning |
|:---:|:---:|---|
| PASS | PASS | Module conforms to spec and passes behavioral tests ✓ |
| PASS | FAIL | Structurally correct but behavioral issues (edge cases, type errors) |
| FAIL | PASS | Tests pass but Weaver found coupling or constraint violations |
| FAIL | FAIL | Both checks failed — regenerate with different model or refine spec |

## Coverage

### What Auto-Tests Cover

- ✅ **Method existence** — every method in the SCP spec exists as a callable
- ✅ **Signature conformance** — parameter names and count match spec
- ✅ **Return type checking** — output types match declared types
- ✅ **Import restrictions** — forbidden project modules are not imported
- ✅ **Constraint compliance** — module-specific rules via AST/source inspection
- ✅ **Edge cases** — empty inputs, None values, error paths
- ✅ **Glyph contract behavior** — Fold Time base cases, Filter boolean returns, etc.
- ✅ **Isolation verification** — no global mutable state

### What Auto-Tests Do NOT Cover

- ❌ **Runtime integration** — tests mock all dependencies, so cross-module wiring is not tested
- ❌ **GPU/hardware paths** — all hardware access is mocked
- ❌ **Performance** — no benchmarking or latency checks
- ❌ **API correctness** — actual Whisper/LLM outputs are mocked
- ❌ **Concurrency** — no thread safety or race condition testing
- ❌ **Data quality** — test inputs are synthetic, not real audio/transcript data
- ❌ **Deterministic test content** — tests are AI-generated and may vary between runs

## Limitations

### AI-Generated Tests Are Non-Deterministic
Each `--with-tests` run generates new test files. Test content, naming, and assertions
may vary between runs even with the same model and low temperature. To get consistent
results, save and reuse test files rather than regenerating.

### Tests May Import Unavailable Modules  
The AI may generate tests that try to import the module under test, which requires
the module's dependencies (e.g., `faster-whisper`, `torch`). If these are not installed,
tests will fail at import time rather than during assertions.

### Mocking Depth
Tests mock at the top level (e.g., `@patch("transcriber.WhisperModel")`).
If the generated code uses deeply nested calls or unusual import patterns,
mocks may not cover all code paths.

### 60-Second Timeout
Each test file gets a 60-second timeout. Complex test suites or slow
environments may hit this limit.

## Architecture

```
scp_bridge.py
  └─ generate_test_prompt()      ← builds AI prompt from SCP contract

turboscribe_example.py
  └─ generate_tests()            ← calls Gemini, saves test file, runs pytest
  └─ main() --with-tests         ← integrates into batch flow

templates/spec_cli.py.template
  └─ _generate_tests()           ← same logic for forge-generated projects
```

The test prompt includes the **full generated code** alongside the **SCP contract**,
giving the AI enough context to write tests that actually import and call the module
while verifying contract compliance.
