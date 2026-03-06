# Chevron v2.0 Specification — Non-Polysemic Topological DSL

## Overview

Chevron is a Non-Polysemic Topological DSL for AI-assisted software architecture.
It replaces ambiguous natural language constraints with mathematical operators drawn
from Category Theory, Topology, and Tensor Mathematics. These symbols occupy deep,
pristine embeddings in LLM latent space (from millions of ingested arXiv LaTeX papers)
and resist adversarial polysemy.

**Protocol Name:** Holographic Language (HL) v2.0

## Core Principle: Topo-Categorical Orthogonality

Every module relationship is expressed through one of 5 mathematical operators.
Each operator carries exactly one semantic interpretation — zero ambiguity.

## The 5 Topo-Categorical Operators

| Operator | Name | Symbol | Intent | Enforcement |
|---|---|---|---|---|
| **Null Morphism** | `Hom(A,B) ≅ 0` | ≅ | Strict isolation | A must never reference B |
| **Morphism** | `A ↦ B` | ↦ | Directed data flow | Reverse flow (B→A) forbidden |
| **Direct Sum** | `A ⊕ B` | ⊕ | Decoupled coexistence | No shared state between A and B |
| **Tensor Product** | `A ⊗ B` | ⊗ | State entanglement | Structural coupling documented |
| **Topo Boundary** | `∂A ∩ ∂B = ∅` | ∂∩∅ | Interface encapsulation | Abstract interface only |

### 1. Null Morphism — `Hom(A, B) ≅ 0`
**Meaning:** The space of morphisms from A to B is trivial (zero).
No function, import, reference, or data path may exist from A to B.

```
Hom(Frontend, Database) ≅ 0
```

### 2. Morphism — `A ↦ B`
**Meaning:** A directed arrow in the category of modules. Data flows from A to B.
B may depend on A. The reverse (B → A) is forbidden.

```
DataLoader ↦ Processor ↦ Renderer
```

### 3. Direct Sum — `A ⊕ B`
**Meaning:** A and B coexist in independent, orthogonal state spaces.
They share no mutable state, no globals, and no side channels.

```
Logger ⊕ Analytics
```

### 4. Tensor Product — `A ⊗ B`
**Meaning:** A and B are entangled — they share state and are tightly coupled.
Changes to one may affect the other. This coupling must be documented.

```
Auth ⊗ Session
```

### 5. Topological Boundary — `∂A ∩ ∂B = ∅`
**Meaning:** The boundaries of A and B do not intersect. All communication
must go through an abstract interface — no direct concrete references.

```
∂UI ∩ ∂Database = ∅
```

## Pipeline Syntax

The pipeline operator `→` chains expressions from left to right:
```
source → transform → filter → output
```

### Predicates
Predicates filter or transform values in pipelines:
```
[1, 2, 3, 4, 5] → {> 3}      # Filter: keep items > 3
[10, 20, 30] → {+ 5}           # Transform: add 5 to each
```

### Bindings
```
data ← [1, 2, 3, 4, 5]
result ← data → {> 3}
```

## Module Specifications

```
spec ModuleName
    depends_on [Dep1, Dep2]
    imports Dep1, Dep2
    exports method1, method2
    forbidden [ForbiddenMod1, ForbiddenMod2]
    constraint "Description of constraint"
end
```

### Module Declarations
```
module ModuleName
    imports Dep1
    exports func1
    func1 ← "implementation"
end
```

## Type Declarations

```
type UserRecord = { name: str, age: int, email: str }
type AudioChunk = { data: str, sampleRate: int }
```

## Verification: System 2 Rejection

The AST Weaver (verifier) performs static analysis on the AST before execution.
Violations produce thermodynamic rejection messages:

```
[SYSTEM 2 REJECTION]: Hom≅0 — Module 'Search' references forbidden 'Database'. Resample required.
[SYSTEM 2 REJECTION]: ↦ — Reverse flow Renderer → DataLoader violates directed morphism. Resample required.
[SYSTEM 2 REJECTION]: ∂∩∅ — Direct reference UI → Database violates topological boundary. Resample required.
[SYSTEM 2 REJECTION]: CYCLE — Circular dependency A ↦ B ↦ A. DAG constraint violated. Resample required.
```

## Token Types

| Token | Symbol | Example |
|---|---|---|
| MORPHISM | ↦ | `A ↦ B` |
| DIRECT_SUM | ⊕ | `A ⊕ B` |
| TENSOR_PRODUCT | ⊗ | `A ⊗ B` |
| PARTIAL | ∂ | `∂A` |
| INTERSECTION | ∩ | `∂A ∩ ∂B` |
| EMPTY_SET | ∅ | `= ∅` |
| ISOMORPHIC | ≅ | `Hom(A,B) ≅ 0` |
| KW_HOM | Hom | `Hom(A, B)` |
| ARROW | → | Pipeline flow |
| BIND | ← | Binding |
| PIPE | \| | Pipe |

## Complete Example

```chevron
# Architecture for a Todo App

type Task = { id: str, title: str, done: bool }

spec TodoStore
    exports add, remove, list
    forbidden [UI, Network]
    constraint "Pure data storage — no I/O"
end

spec TodoUI
    depends_on [TodoStore]
    imports TodoStore
    exports render, handle_input
    forbidden [Network]
    constraint "No direct database access"
end

# Constraints
Hom(TodoUI, Network) ≅ 0
TodoStore ↦ TodoUI
∂TodoStore ∩ ∂TodoUI = ∅

# Pipeline demo
["Buy milk", "Write code", "Deploy app"] → {!= "Deploy app"}
```

## Version History

- **v2.0.0** — Non-Polysemic Topological DSL (Topo-Categorical operators)
- **v1.0.0** — Original Uiua-glyph system (deprecated)
