"""
Nexus Test Suite — Context Kernel (Layer 1)
=============================================
Tests for entropy scoring, context pruning, and SCP retrieval.

Usage:
    python -m pytest tests/test_context_kernel.py -v
    python tests/test_context_kernel.py
"""
import sys
import os
import unittest
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nexus.context_kernel import (
    ContextItem, EntropyScorer, ContextPruner, SCPRetriever,
    CATEGORY_CONTRACT, CATEGORY_ACTIVE_CODE, CATEGORY_ERROR,
    CATEGORY_INTERFACE, CATEGORY_CONVERSATION, CATEGORY_OUTPUT,
    CATEGORY_STALE, estimate_tokens,
)


# ─────────────────────────────────────────────
#  Token Estimation Tests
# ─────────────────────────────────────────────

class TestTokenEstimation(unittest.TestCase):
    def test_basic_estimation(self):
        self.assertEqual(estimate_tokens("1234"), 1)
        self.assertEqual(estimate_tokens("12345678"), 2)

    def test_empty_string(self):
        self.assertEqual(estimate_tokens(""), 1)  # min 1


# ─────────────────────────────────────────────
#  Entropy Scorer Tests
# ─────────────────────────────────────────────

class TestEntropyScorer(unittest.TestCase):

    def test_contracts_score_highest(self):
        """SCP contracts must always score 1.0 — they are the source of truth."""
        scorer = EntropyScorer()
        item = ContextItem(content="contract", source="spec", category=CATEGORY_CONTRACT)
        self.assertEqual(scorer.score(item), 1.0)

    def test_active_code_scores_high(self):
        scorer = EntropyScorer()
        item = ContextItem(content="code", source="file", category=CATEGORY_ACTIVE_CODE)
        self.assertEqual(scorer.score(item), 0.9)

    def test_conversation_decays_with_age(self):
        """Conversation items must decay exponentially with turn age."""
        scorer = EntropyScorer(decay_rate=0.9)

        fresh = ContextItem(content="hi", source="chat", category=CATEGORY_CONVERSATION, turn_age=0)
        old = ContextItem(content="hi", source="chat", category=CATEGORY_CONVERSATION, turn_age=10)

        fresh_score = scorer.score(fresh)
        old_score = scorer.score(old)

        self.assertGreater(fresh_score, old_score)
        self.assertAlmostEqual(old_score, 0.6 * 0.9 ** 10, places=3)

    def test_stale_items_score_lowest(self):
        scorer = EntropyScorer()
        item = ContextItem(content="old", source="prev", category=CATEGORY_STALE)
        self.assertEqual(scorer.score(item), 0.1)

    def test_active_module_boost(self):
        """Items in the active module get a 15% boost."""
        scorer = EntropyScorer(active_module="MyModule")

        in_scope = ContextItem(content="x", source="f", category=CATEGORY_ACTIVE_CODE,
                               module_scope="MyModule")
        out_scope = ContextItem(content="x", source="f", category=CATEGORY_ACTIVE_CODE,
                                module_scope="OtherModule")

        self.assertGreater(scorer.score(in_scope), scorer.score(out_scope))

    def test_wrong_module_penalty(self):
        """Items from unrelated modules get a 50% penalty."""
        scorer = EntropyScorer(active_module="MyModule")
        item = ContextItem(content="x", source="f", category=CATEGORY_ACTIVE_CODE,
                           module_scope="ForeignModule")
        # 0.9 * 0.5 = 0.45
        self.assertAlmostEqual(scorer.score(item), 0.45, places=2)

    def test_global_scope_no_penalty(self):
        """Global items should not get a wrong-module penalty."""
        scorer = EntropyScorer(active_module="MyModule")
        item = ContextItem(content="x", source="f", category=CATEGORY_INTERFACE,
                           module_scope="global")
        # Should be 0.7 (base), no penalty
        self.assertEqual(scorer.score(item), 0.7)

    def test_score_all_returns_sorted(self):
        """score_all should return items sorted by score descending."""
        scorer = EntropyScorer()
        items = [
            ContextItem(content="a", source="x", category=CATEGORY_STALE),
            ContextItem(content="b", source="x", category=CATEGORY_CONTRACT),
            ContextItem(content="c", source="x", category=CATEGORY_OUTPUT),
        ]
        sorted_items = scorer.score_all(items)
        scores = [i.relevance_score for i in sorted_items]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(sorted_items[0].category, CATEGORY_CONTRACT)

    def test_score_clamped(self):
        """Scores must be in [0.0, 1.0]."""
        scorer = EntropyScorer(active_module="M")
        item = ContextItem(content="x", source="f", category=CATEGORY_CONTRACT,
                           module_scope="M")
        # Contract (1.0) * 1.15 boost → should clamp to 1.0
        self.assertEqual(scorer.score(item), 1.0)

    def test_entropy_computation(self):
        """Shannon entropy: skewed distributions have LOWER entropy than uniform."""
        scorer = EntropyScorer()

        # All contracts (uniform scores) → MAXIMUM Shannon entropy
        uniform = [
            ContextItem(content="a", source="x", category=CATEGORY_CONTRACT),
            ContextItem(content="b", source="x", category=CATEGORY_CONTRACT),
        ]
        scorer.score_all(uniform)
        uniform_entropy = scorer.compute_entropy(uniform)

        # Skewed signal (one high, one very low) → LOWER entropy
        skewed = [
            ContextItem(content="a", source="x", category=CATEGORY_CONTRACT),
            ContextItem(content="b", source="x", category=CATEGORY_STALE),
        ]
        scorer.score_all(skewed)
        skewed_entropy = scorer.compute_entropy(skewed)

        # Uniform = max entropy, skewed = lower entropy (Shannon property)
        self.assertGreater(uniform_entropy, skewed_entropy)

    def test_entropy_empty(self):
        scorer = EntropyScorer()
        self.assertEqual(scorer.compute_entropy([]), 0.0)


# ─────────────────────────────────────────────
#  Context Pruner Tests
# ─────────────────────────────────────────────

class TestContextPruner(unittest.TestCase):

    def test_contracts_never_pruned(self):
        """Contracts must survive pruning even with tiny budgets."""
        pruner = ContextPruner()
        items = [
            ContextItem(content="contract text", source="spec",
                        category=CATEGORY_CONTRACT, relevance_score=1.0),
            ContextItem(content="conversational noise " * 100, source="chat",
                        category=CATEGORY_CONVERSATION, relevance_score=0.3),
        ]
        # Budget that fits only the contract
        pruned = pruner.prune(items, budget=10)
        categories = [i.category for i in pruned]
        self.assertIn(CATEGORY_CONTRACT, categories)
        self.assertNotIn(CATEGORY_CONVERSATION, categories)

    def test_budget_respected(self):
        """Total tokens after pruning must be within budget."""
        pruner = ContextPruner()
        items = [
            ContextItem(content="a" * 400, source="x", category=CATEGORY_ACTIVE_CODE,
                        relevance_score=0.9),
            ContextItem(content="b" * 400, source="x", category=CATEGORY_OUTPUT,
                        relevance_score=0.3),
            ContextItem(content="c" * 400, source="x", category=CATEGORY_CONVERSATION,
                        relevance_score=0.6),
        ]
        pruned = pruner.prune(items, budget=250)
        total_tokens = sum(i.token_count for i in pruned)
        self.assertLessEqual(total_tokens, 250)

    def test_preferred_before_expendable(self):
        """Active code and errors should be preferred over outputs."""
        pruner = ContextPruner()
        items = [
            ContextItem(content="x" * 100, source="x", category=CATEGORY_ACTIVE_CODE,
                        relevance_score=0.9),
            ContextItem(content="y" * 100, source="x", category=CATEGORY_OUTPUT,
                        relevance_score=0.95),  # Higher score but expendable!
        ]
        # Budget fits only one (~25 tokens each)
        pruned = pruner.prune(items, budget=30)
        self.assertEqual(len(pruned), 1)
        self.assertEqual(pruned[0].category, CATEGORY_ACTIVE_CODE)

    def test_compression_ratio(self):
        pruner = ContextPruner()
        original = [
            ContextItem(content="a" * 1000, source="x", category=CATEGORY_OUTPUT,
                        relevance_score=0.3),
            ContextItem(content="b" * 100, source="x", category=CATEGORY_CONTRACT,
                        relevance_score=1.0),
        ]
        pruned = pruner.prune(original, budget=50)
        ratio = pruner.compute_compression_ratio(original, pruned)
        self.assertLess(ratio, 1.0)

    def test_empty_input(self):
        pruner = ContextPruner()
        self.assertEqual(pruner.prune([], budget=1000), [])


# ─────────────────────────────────────────────
#  SCP Retriever Tests
# ─────────────────────────────────────────────

class TestSCPRetriever(unittest.TestCase):

    def _make_architecture(self):
        """Create a minimal architecture for testing."""
        try:
            from scp_bridge import ArchitectureSpec, ModuleSpec, InterfaceMethod
        except ImportError:
            self.skipTest("scp_bridge not available")

        return ArchitectureSpec(
            name="TestApp",
            modules=[
                ModuleSpec(
                    name="ModuleA",
                    description="First module",
                    methods=[
                        InterfaceMethod(name="do_a", inputs=["x"], output="int",
                                        glyph="◬", constraint="No side effects"),
                    ],
                    allowed_dependencies=["ModuleB"],
                    constraints=["Pure functions only"],
                ),
                ModuleSpec(
                    name="ModuleB",
                    description="Second module",
                    methods=[
                        InterfaceMethod(name="do_b", inputs=["y"], output="str",
                                        glyph="Ө", constraint="Filter only"),
                    ],
                    allowed_dependencies=[],
                    constraints=["No imports from A"],
                ),
                ModuleSpec(
                    name="ModuleC",
                    description="Forbidden module",
                    methods=[],
                    allowed_dependencies=[],
                    constraints=[],
                ),
            ],
            global_constraints=["No global mutable state"],
        )

    def test_retrieve_active_module_code(self):
        """Retrieve should include full code for the active module."""
        arch = self._make_architecture()
        retriever = SCPRetriever(architecture=arch)

        code_store = {"ModuleA": "def do_a(x): return x + 1"}
        items = retriever.retrieve_for_module("ModuleA", code_store=code_store)

        active_items = [i for i in items if i.category == CATEGORY_ACTIVE_CODE]
        self.assertEqual(len(active_items), 1)
        self.assertIn("do_a", active_items[0].content)

    def test_dependency_interface_only(self):
        """Dependencies should be visible as interfaces, not full code."""
        arch = self._make_architecture()
        retriever = SCPRetriever(architecture=arch)

        code_store = {
            "ModuleA": "def do_a(x): return x + 1",
            "ModuleB": "def do_b(y): return str(y)  # SECRET IMPLEMENTATION",
        }
        items = retriever.retrieve_for_module("ModuleA", code_store=code_store)

        # Should have ModuleB interface but NOT its source code
        interface_items = [i for i in items if i.category == CATEGORY_INTERFACE]
        self.assertTrue(len(interface_items) > 0)

        # ModuleB's source code should NOT appear
        all_content = " ".join(i.content for i in items)
        self.assertNotIn("SECRET IMPLEMENTATION", all_content)

    def test_forbidden_modules_blocked(self):
        """Items from forbidden modules must be filtered out."""
        arch = self._make_architecture()
        # Add forbidden to ModuleA
        arch.modules[0].forbidden = ["ModuleC"]
        retriever = SCPRetriever(architecture=arch)

        extra = [
            ContextItem(content="forbidden info", source="x",
                        category=CATEGORY_CONVERSATION, module_scope="ModuleC"),
            ContextItem(content="allowed info", source="x",
                        category=CATEGORY_CONVERSATION, module_scope="global"),
        ]

        items = retriever.retrieve_for_module("ModuleA", extra_context=extra)

        # Forbidden content should NOT appear
        all_content = " ".join(i.content for i in items)
        self.assertNotIn("forbidden info", all_content)
        self.assertIn("allowed info", all_content)

    def test_global_constraints_included(self):
        """Global SCP constraints should always be included."""
        arch = self._make_architecture()
        retriever = SCPRetriever(architecture=arch)

        items = retriever.retrieve_for_module("ModuleA")
        contract_items = [i for i in items if i.category == CATEGORY_CONTRACT]
        all_content = " ".join(i.content for i in contract_items)
        self.assertIn("No global mutable state", all_content)

    def test_unknown_module_raises(self):
        arch = self._make_architecture()
        retriever = SCPRetriever(architecture=arch)
        with self.assertRaises(ValueError):
            retriever.retrieve_for_module("NonExistent")

    def test_no_architecture_returns_extra_only(self):
        retriever = SCPRetriever()
        extra = [ContextItem(content="hi", source="x", category=CATEGORY_CONVERSATION)]
        items = retriever.retrieve_for_module("anything", extra_context=extra)
        self.assertEqual(len(items), 1)

    def test_get_dependency_graph(self):
        arch = self._make_architecture()
        retriever = SCPRetriever(architecture=arch)
        graph = retriever.get_dependency_graph()
        self.assertEqual(graph["ModuleA"], ["ModuleB"])
        self.assertEqual(graph["ModuleB"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
