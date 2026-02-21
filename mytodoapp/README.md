# mytodoapp

Built with [Nexus](https://github.com/dparksports/Project-Chevron) — SCP-powered AI development.

> Uses orthogonal Uiua embeddings to create steep, isolated attractor basins for each module — eliminating semantic cross-talk and preventing confabulation across module boundaries. The Weaver (System 2 rejection sampling) enforces W(G) = 0.

## Modules

  - **TodoStore** — In-memory store for todo items. Pure data layer. *(◬ Origin)*
  - **TodoAPI** — REST API layer. Routes HTTP requests to TodoStore. *(Ө Filter)*
  - **TodoLogger** — Logs all operations. Pure observation — never modifies data. *(𓂀 Witness)*

## Getting Started

```bash
# See the architecture
python -m nexus.cli overview --spec nexus.json

# Generate code for a module (orthogonal contract → steep attractor basin)
python -m nexus.cli generate TodoStore --provider gemini --key YOUR_KEY --spec nexus.json
```
