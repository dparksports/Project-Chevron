# mytodoapp

Built with [Nexus](https://github.com/dparksports/Project-Chevron) — SCP-powered AI development.

## Modules

  - **TodoStore** — In-memory store for todo items. Pure data layer.
  - **TodoAPI** — REST API layer. Routes HTTP requests to TodoStore.
  - **TodoLogger** — Logs all operations. Pure observation — never modifies data.

## Getting Started

```bash
# See the architecture
python -m nexus.cli overview --spec nexus.json

# Generate code for a module
python -m nexus.cli generate TodoStore --provider gemini --key YOUR_KEY --spec nexus.json
```
