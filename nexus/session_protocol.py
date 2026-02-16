"""
SCP Session Protocol — Layer 2 of the Nexus AI IDE
====================================================
Gives the IDE persistent architectural memory across edits.

Components:
    SessionState   — The persistent model of the project architecture
    EditEntry      — Record of a single AI edit with verification status
    EditLedger     — Append-only log of all edits (enables undo, audit)
    ContractCache  — Frozen interface snapshots after Weaver verification

Theory:
    Current AI IDEs are stateless: each prompt rediscovers architecture
    from scratch. The Session Protocol maintains:
        1. Which SCP architecture is loaded
        2. Which module is being edited
        3. What edits have been made (and whether they passed verification)
        4. Frozen interface contracts that prevent cascade regressions

    When a module passes Weaver verification (W(G) = 0), its interface
    contract is frozen in the ContractCache. Other modules see this frozen
    contract, NOT the live code. This means:
        - Changing Module A's internals doesn't break Module B's context
        - Contracts upgrade only on explicit re-verification
        - Stale couplings are impossible by construction

Dan Park | MagicPoint.ai | February 2026
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any


# ─────────────────────────────────────────────────────────────
#  Edit Entry
# ─────────────────────────────────────────────────────────────

@dataclass
class EditEntry:
    """Record of a single AI-generated edit.

    Every edit the AI makes is logged here with its verification status.
    This creates an audit trail and enables intelligent undo.
    """

    module: str                   # Which SCP module was edited
    timestamp: float = field(default_factory=time.time)
    description: str = ""         # What the edit did (from AI)
    files_changed: list[str] = field(default_factory=list)
    code_before: dict[str, str] = field(default_factory=dict)  # file -> pre-edit content
    code_after: dict[str, str] = field(default_factory=dict)   # file -> post-edit content
    verified: bool = False        # Did Weaver verification pass?
    test_passed: bool = False     # Did auto-generated tests pass?
    verification_details: str = ""  # Weaver output
    entropy_before: float = 0.0   # Context entropy before edit
    entropy_after: float = 0.0    # Context entropy after edit

    @property
    def entropy_delta(self) -> float:
        """How much entropy this edit added to context.
        Positive = increased noise, negative = reduced noise."""
        return self.entropy_after - self.entropy_before

    @property
    def is_clean(self) -> bool:
        """Edit is clean if both verified and tests passed."""
        return self.verified and self.test_passed

    def to_dict(self) -> dict:
        """Serialize to dict for JSON persistence."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> EditEntry:
        """Deserialize from dict."""
        return cls(**{k: v for k, v in data.items()
                      if k in cls.__dataclass_fields__})


# ─────────────────────────────────────────────────────────────
#  Edit Ledger
# ─────────────────────────────────────────────────────────────

class EditLedger:
    """Append-only log of all edits in a session.

    The ledger supports:
        - Adding new edits
        - Querying edit history (by module, by verification status)
        - Computing session-level metrics (regression rate, velocity)
        - Persistence to/from JSON
    """

    def __init__(self):
        self._entries: list[EditEntry] = []

    def record(self, entry: EditEntry) -> int:
        """Record a new edit. Returns the edit index."""
        self._entries.append(entry)
        return len(self._entries) - 1

    def get_entry(self, index: int) -> EditEntry:
        """Get a specific edit by index."""
        return self._entries[index]

    @property
    def entries(self) -> list[EditEntry]:
        """All edit entries (read-only view)."""
        return list(self._entries)

    def edits_for_module(self, module_name: str) -> list[EditEntry]:
        """Get all edits for a specific module."""
        return [e for e in self._entries if e.module == module_name]

    def unverified_edits(self) -> list[EditEntry]:
        """Get all edits that haven't passed Weaver verification."""
        return [e for e in self._entries if not e.verified]

    def failed_edits(self) -> list[EditEntry]:
        """Get all edits where tests failed."""
        return [e for e in self._entries if not e.test_passed and e.verified]

    # ─── Metrics ──────────────────────────────────────────

    @property
    def total_edits(self) -> int:
        return len(self._entries)

    @property
    def regression_rate(self) -> float:
        """Percentage of edits that failed verification.
        SCP target: < 0.1% (vs baseline 14.3%)."""
        if not self._entries:
            return 0.0
        failed = sum(1 for e in self._entries if not e.is_clean)
        return failed / len(self._entries)

    @property
    def average_entropy_delta(self) -> float:
        """Average entropy change per edit. Lower = better."""
        if not self._entries:
            return 0.0
        return sum(e.entropy_delta for e in self._entries) / len(self._entries)

    @property
    def clean_streak(self) -> int:
        """Number of consecutive clean edits from the end."""
        streak = 0
        for entry in reversed(self._entries):
            if entry.is_clean:
                streak += 1
            else:
                break
        return streak

    # ─── Persistence ──────────────────────────────────────

    def save(self, filepath: str):
        """Save ledger to JSON file."""
        data = {"edits": [e.to_dict() for e in self._entries]}
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> EditLedger:
        """Load ledger from JSON file."""
        ledger = cls()
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            ledger._entries = [EditEntry.from_dict(e) for e in data.get("edits", [])]
        return ledger


# ─────────────────────────────────────────────────────────────
#  Contract Cache
# ─────────────────────────────────────────────────────────────

class ContractCache:
    """Frozen interface contracts for verified modules.

    When a module passes Weaver verification (W(G) = 0), its interface
    contract is frozen here. Other modules see this frozen snapshot,
    not the live code.

    This prevents cascade regressions:
        - Module B depends on Module A
        - Developer changes Module A's internals
        - Module B still sees A's frozen contract (unchanged)
        - Only when A is re-verified does the contract update

    The cache is the implementation of SCP's "zero coupling" guarantee
    at the IDE level.
    """

    def __init__(self):
        self._contracts: dict[str, str] = {}        # module_name -> frozen contract
        self._timestamps: dict[str, float] = {}     # module_name -> freeze time
        self._versions: dict[str, int] = {}          # module_name -> version counter

    def freeze(self, module_name: str, contract: str):
        """Freeze a module's interface contract after Weaver verification."""
        self._contracts[module_name] = contract
        self._timestamps[module_name] = time.time()
        self._versions[module_name] = self._versions.get(module_name, 0) + 1

    def get(self, module_name: str) -> Optional[str]:
        """Get the frozen contract for a module. Returns None if not frozen."""
        return self._contracts.get(module_name)

    def get_version(self, module_name: str) -> int:
        """Get the contract version (how many times it's been frozen)."""
        return self._versions.get(module_name, 0)

    def is_frozen(self, module_name: str) -> bool:
        """Check if a module has a frozen contract."""
        return module_name in self._contracts

    def invalidate(self, module_name: str):
        """Invalidate a frozen contract (e.g., after interface-breaking change)."""
        self._contracts.pop(module_name, None)
        self._timestamps.pop(module_name, None)

    def invalidate_all(self):
        """Clear all frozen contracts."""
        self._contracts.clear()
        self._timestamps.clear()

    @property
    def frozen_modules(self) -> list[str]:
        """List of all modules with frozen contracts."""
        return list(self._contracts.keys())

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "contracts": dict(self._contracts),
            "timestamps": dict(self._timestamps),
            "versions": dict(self._versions),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ContractCache:
        """Deserialize from dict."""
        cache = cls()
        cache._contracts = data.get("contracts", {})
        cache._timestamps = data.get("timestamps", {})
        cache._versions = data.get("versions", {})
        return cache


# ─────────────────────────────────────────────────────────────
#  Session State
# ─────────────────────────────────────────────────────────────

class SessionState:
    """The persistent model of the project architecture.

    This is the central state object that connects all Nexus components:
        - Architecture spec (from Chevron)
        - Active module (what the AI is currently editing)
        - Edit ledger (full history with verification status)
        - Contract cache (frozen interfaces)
        - Code store (current source code per module)
        - Entropy budget (adaptive token budget)
    """

    def __init__(self, architecture=None, session_dir: str = None):
        """
        Args:
            architecture: A Chevron ArchitectureSpec instance.
            session_dir: Directory for session persistence (ledger, cache, etc.)
        """
        self.architecture = architecture
        self.active_module: Optional[str] = None
        self.ledger = EditLedger()
        self.contract_cache = ContractCache()
        self.code_store: dict[str, str] = {}  # module_name -> current source code
        self.entropy_budget: int = 8000       # Default token budget
        self.session_dir = session_dir or "."
        self._metadata: dict[str, Any] = {}

    def set_active_module(self, module_name: str):
        """Set the module currently being edited."""
        if self.architecture:
            module_names = [m.name for m in self.architecture.modules]
            if module_name not in module_names:
                raise ValueError(
                    f"Module '{module_name}' not in architecture. "
                    f"Available: {module_names}"
                )
        self.active_module = module_name

    def update_code(self, module_name: str, code: str):
        """Update the source code for a module."""
        self.code_store[module_name] = code

    def record_edit(self, entry: EditEntry) -> int:
        """Record an edit in the ledger."""
        return self.ledger.record(entry)

    def freeze_contract(self, module_name: str, contract: str):
        """Freeze a verified module's interface contract."""
        self.contract_cache.freeze(module_name, contract)

    @property
    def module_names(self) -> list[str]:
        """List of all module names in the architecture."""
        if not self.architecture:
            return []
        return [m.name for m in self.architecture.modules]

    @property
    def session_health(self) -> dict:
        """Quick health check of the session.

        Returns metrics aligned with SCP paper:
            - regression_rate: target < 0.1% (vs baseline 14.3%)
            - clean_streak: consecutive verified edits
            - frozen_modules: how many modules have stable contracts
            - entropy_budget_used: how much of the budget is consumed
        """
        return {
            "total_edits": self.ledger.total_edits,
            "regression_rate": f"{self.ledger.regression_rate:.1%}",
            "clean_streak": self.ledger.clean_streak,
            "unverified_edits": len(self.ledger.unverified_edits()),
            "frozen_modules": len(self.contract_cache.frozen_modules),
            "total_modules": len(self.module_names),
            "entropy_budget": self.entropy_budget,
        }

    # ─── Persistence ──────────────────────────────────────

    def save(self):
        """Persist session state to disk."""
        os.makedirs(self.session_dir, exist_ok=True)

        # Save ledger
        self.ledger.save(os.path.join(self.session_dir, "edit_ledger.json"))

        # Save contract cache
        cache_path = os.path.join(self.session_dir, "contract_cache.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(self.contract_cache.to_dict(), f, indent=2)

        # Save session metadata
        meta = {
            "active_module": self.active_module,
            "entropy_budget": self.entropy_budget,
            "module_names": self.module_names,
            "code_store_keys": list(self.code_store.keys()),
        }
        meta_path = os.path.join(self.session_dir, "session_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, session_dir: str, architecture=None) -> SessionState:
        """Load session state from disk."""
        session = cls(architecture=architecture, session_dir=session_dir)

        # Load ledger
        ledger_path = os.path.join(session_dir, "edit_ledger.json")
        if os.path.exists(ledger_path):
            session.ledger = EditLedger.load(ledger_path)

        # Load contract cache
        cache_path = os.path.join(session_dir, "contract_cache.json")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                session.contract_cache = ContractCache.from_dict(json.load(f))

        # Load metadata
        meta_path = os.path.join(session_dir, "session_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            session.active_module = meta.get("active_module")
            session.entropy_budget = meta.get("entropy_budget", 8000)

        return session
