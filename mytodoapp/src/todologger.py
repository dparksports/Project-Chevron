"""
TodoLogger — Logs all operations. Pure observation — never modifies data.
SCP Glyph: 𓂀
Dependencies: none
"""

class TodoLogger:
    def log_action(self, action: str, details: dict) -> None:
        "Witnesses action — logs without modifying state"
        raise NotImplementedError

