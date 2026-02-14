# Project Chevron — Language Specification
# Reference Implementation of the Spatial Constraint Protocol (SCP)
# Dan Park | MagicPoint.ai | February 2026

## Overview

Chevron is a glyph-based programming language where code is written using symbolic
primitives inspired by the Rendlesham Forest binary code, Roswell I-beam geometric
symbols, and Egyptian hieroglyphs. Each glyph maps to a deterministic semantic
operation, achieving the bijective singleton property defined in the SCP paper.

## Core Primitives

| Glyph | Name           | Origin      | Semantic Meaning              | Code Equivalent              |
|-------|----------------|-------------|-------------------------------|------------------------------|
| `◬`   | The Origin     | Rendlesham  | Program entry / root          | `main()` — all threads spawn here |
| `☾`   | Fold Time      | Roswell     | Recursion / temporal feedback | Output feeds back into input |
| `Ө`   | The Filter     | Roswell     | Conditional gate              | `if/else` — only matching data passes |
| `𓂀`   | The Witness    | Egyptian    | Observe without altering      | Logging / observability      |
| `☤`   | The Weaver     | Generic     | Merge / join                  | Combine two streams into one braid |

## Operators

| Symbol | Name         | Meaning                                    |
|--------|--------------|--------------------------------------------|
| `→`    | Pipeline     | Data flows left to right                   |
| `←`    | Binding      | Assign a name to a glyph expression        |
| `( )`  | Grouping     | Group a glyph with its arguments           |
| `[ ]`  | List         | Define a list / array of values            |
| `{ }`  | Predicate    | Define a filter condition for `Ө`          |
| `" "`  | String       | String literal                             |
| `#`    | Comment      | Line comment                               |

## Data Types

- **String**: `"hello"` — text values
- **Number**: `42`, `3.14` — numeric values
- **List**: `["a", "b", "c"]` — ordered collections
- **Boolean**: `true`, `false` — truth values
- **Stream**: Implicit — data flowing through a pipeline

## Syntax Rules

### 1. The Origin (◬) — Program Entry
Every Chevron program begins with `◬`. It defines the root data and spawns execution.
```
◬ "Hello, Chevron"
◬ [1, 2, 3, 4, 5]
```

### 2. The Witness (𓂀) — Observation
The Witness observes data and logs it, passing it through unchanged.
```
𓂀 "I see you"           # Logs: 𓂀 ⟫ I see you
𓂀 (☤ ["Hello", "World"]) # Logs: 𓂀 ⟫ Hello World
```

### 3. The Weaver (☤) — Merging
The Weaver braids two or more values into one.
```
☤ ["Hello", "World"]     # → "Hello World"
☤ [[1,2], [3,4]]         # → [1, 2, 3, 4]
```

### 4. The Filter (Ө) — Conditional Gate
The Filter passes only data matching a predicate.
```
Ө {> 3} [1, 2, 3, 4, 5]        # → [4, 5]
Ө {= "yes"} ["yes", "no"]      # → ["yes"]
```

### 5. Fold Time (☾) — Recursion
Fold Time feeds the output of an expression back into itself until a base case.
```
☾ {> 0} {- 1} 5     # 5 → 4 → 3 → 2 → 1 → 0 (stop)
```

### 6. Pipelines (→) — Composition
Glyphs compose left to right with `→`.
```
◬ [5, 3, 1, 4, 2] → Ө {> 2} → 𓂀
# Origin: [5,3,1,4,2] → Filter >2: [5,3,4] → Witness logs [5,3,4]
```

### 7. Bindings (←) — Named Definitions
```
Avg ← ◬ → ☤ → 𓂀
BigOnly ← Ө {> 100}
```

## Example Programs

### Hello World
```
𓂀 (☤ ["Hello", "World"])
```

### Pipeline
```
◬ [10, 25, 3, 47, 8, 92, 1] → Ө {> 10} → 𓂀
```

### Recursive Countdown
```
◬ 10 → ☾ {> 0} {- 1} → 𓂀
```

### Full Pipeline
```
# Perceive → Filter → Judge → Record
◬ ["meeting.wav", "noise.wav", "talk.wav"]
  → Ө {!= "noise.wav"}
  → ☤ ["[valid] ", _]
  → 𓂀
```
