"""
Nexus Test Suite — Session Protocol (Layer 2)
===============================================
Tests for session state, edit ledger, and contract cache.

Usage:
    python -m pytest tests/test_session_protocol.py -v
    python tests/test_session_protocol.py
"""
import sys
import os
import json
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus.session_protocol import (
    SessionState, EditEntry, EditLedger, ContractCache,
)


# ─────────────────────────────────────────────
#  EditEntry Tests
# ─────────────────────────────────────────────

class TestEditEntry(unittest.TestCase):

    def test_entropy_delta(self):
        entry = EditEntry(module="A", entropy_before=2.0, entropy_after=2.5)
        self.assertAlmostEqual(entry.entropy_delta, 0.5)

    def test_clean_edit(self):
        entry = EditEntry(module="A", verified=True, test_passed=True)
        self.assertTrue(entry.is_clean)

    def test_dirty_edit(self):
        entry = EditEntry(module="A", verified=True, test_passed=False)
        self.assertFalse(entry.is_clean)

    def test_serialization_roundtrip(self):
        entry = EditEntry(
            module="TestMod",
            description="Added logging",
            files_changed=["mod.py"],
            verified=True,
            test_passed=True,
        )
        data = entry.to_dict()
        restored = EditEntry.from_dict(data)
        self.assertEqual(restored.module, "TestMod")
        self.assertEqual(restored.description, "Added logging")
        self.assertTrue(restored.verified)


# ─────────────────────────────────────────────
#  EditLedger Tests
# ─────────────────────────────────────────────

class TestEditLedger(unittest.TestCase):

    def test_record_and_retrieve(self):
        ledger = EditLedger()
        idx = ledger.record(EditEntry(module="A", description="init"))
        self.assertEqual(idx, 0)
        self.assertEqual(ledger.get_entry(0).module, "A")

    def test_total_edits(self):
        ledger = EditLedger()
        ledger.record(EditEntry(module="A"))
        ledger.record(EditEntry(module="B"))
        self.assertEqual(ledger.total_edits, 2)

    def test_edits_for_module(self):
        ledger = EditLedger()
        ledger.record(EditEntry(module="A"))
        ledger.record(EditEntry(module="B"))
        ledger.record(EditEntry(module="A"))
        self.assertEqual(len(ledger.edits_for_module("A")), 2)
        self.assertEqual(len(ledger.edits_for_module("B")), 1)

    def test_regression_rate(self):
        ledger = EditLedger()
        ledger.record(EditEntry(module="A", verified=True, test_passed=True))
        ledger.record(EditEntry(module="B", verified=True, test_passed=False))
        self.assertAlmostEqual(ledger.regression_rate, 0.5)

    def test_clean_streak(self):
        ledger = EditLedger()
        ledger.record(EditEntry(module="A", verified=True, test_passed=False))
        ledger.record(EditEntry(module="B", verified=True, test_passed=True))
        ledger.record(EditEntry(module="C", verified=True, test_passed=True))
        self.assertEqual(ledger.clean_streak, 2)

    def test_unverified_edits(self):
        ledger = EditLedger()
        ledger.record(EditEntry(module="A", verified=False))
        ledger.record(EditEntry(module="B", verified=True, test_passed=True))
        self.assertEqual(len(ledger.unverified_edits()), 1)

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "ledger.json")

            # Save
            ledger = EditLedger()
            ledger.record(EditEntry(module="A", description="test", verified=True, test_passed=True))
            ledger.save(filepath)

            # Load
            loaded = EditLedger.load(filepath)
            self.assertEqual(loaded.total_edits, 1)
            self.assertEqual(loaded.get_entry(0).module, "A")
            self.assertEqual(loaded.get_entry(0).description, "test")

    def test_load_missing_file(self):
        ledger = EditLedger.load("/nonexistent/path.json")
        self.assertEqual(ledger.total_edits, 0)


# ─────────────────────────────────────────────
#  ContractCache Tests
# ─────────────────────────────────────────────

class TestContractCache(unittest.TestCase):

    def test_freeze_and_get(self):
        cache = ContractCache()
        cache.freeze("ModA", "def do_a(x) -> int")
        self.assertEqual(cache.get("ModA"), "def do_a(x) -> int")

    def test_version_increments(self):
        cache = ContractCache()
        cache.freeze("ModA", "v1")
        self.assertEqual(cache.get_version("ModA"), 1)
        cache.freeze("ModA", "v2")
        self.assertEqual(cache.get_version("ModA"), 2)
        self.assertEqual(cache.get("ModA"), "v2")

    def test_is_frozen(self):
        cache = ContractCache()
        self.assertFalse(cache.is_frozen("ModA"))
        cache.freeze("ModA", "contract")
        self.assertTrue(cache.is_frozen("ModA"))

    def test_invalidate(self):
        cache = ContractCache()
        cache.freeze("ModA", "contract")
        cache.invalidate("ModA")
        self.assertFalse(cache.is_frozen("ModA"))
        self.assertIsNone(cache.get("ModA"))

    def test_invalidate_all(self):
        cache = ContractCache()
        cache.freeze("A", "c1")
        cache.freeze("B", "c2")
        cache.invalidate_all()
        self.assertEqual(len(cache.frozen_modules), 0)

    def test_serialization_roundtrip(self):
        cache = ContractCache()
        cache.freeze("A", "contract_a")
        cache.freeze("B", "contract_b")

        data = cache.to_dict()
        restored = ContractCache.from_dict(data)

        self.assertEqual(restored.get("A"), "contract_a")
        self.assertEqual(restored.get("B"), "contract_b")
        self.assertEqual(restored.get_version("A"), 1)


# ─────────────────────────────────────────────
#  SessionState Tests
# ─────────────────────────────────────────────

class TestSessionState(unittest.TestCase):

    def test_session_health(self):
        session = SessionState()
        health = session.session_health
        self.assertEqual(health["total_edits"], 0)
        self.assertEqual(health["regression_rate"], "0.0%")

    def test_update_code(self):
        session = SessionState()
        session.update_code("ModA", "def a(): pass")
        self.assertIn("ModA", session.code_store)

    def test_record_edit(self):
        session = SessionState()
        session.record_edit(EditEntry(module="A", verified=True, test_passed=True))
        self.assertEqual(session.ledger.total_edits, 1)

    def test_freeze_contract(self):
        session = SessionState()
        session.freeze_contract("A", "interface A")
        self.assertTrue(session.contract_cache.is_frozen("A"))

    def test_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            session = SessionState(session_dir=tmpdir)
            session.active_module = "TestMod"
            session.entropy_budget = 16000
            session.record_edit(EditEntry(module="A", verified=True, test_passed=True))
            session.freeze_contract("A", "contract_A")
            session.save()

            # Load
            loaded = SessionState.load(tmpdir)
            self.assertEqual(loaded.active_module, "TestMod")
            self.assertEqual(loaded.entropy_budget, 16000)
            self.assertEqual(loaded.ledger.total_edits, 1)
            self.assertTrue(loaded.contract_cache.is_frozen("A"))

    def test_set_active_module_with_architecture(self):
        try:
            from scp_bridge import ArchitectureSpec, ModuleSpec
        except ImportError:
            self.skipTest("scp_bridge not available")

        arch = ArchitectureSpec(name="Test", modules=[
            ModuleSpec(name="A", description="mod A"),
        ])
        session = SessionState(architecture=arch)
        session.set_active_module("A")
        self.assertEqual(session.active_module, "A")

        with self.assertRaises(ValueError):
            session.set_active_module("NonExistent")


if __name__ == "__main__":
    unittest.main(verbosity=2)
