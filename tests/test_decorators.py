"""
Tests for Chevron Decorators
==============================
Tests for runtime-enforced glyph decorators.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chevron.decorators import (
    ChevronContractError,
    origin, filter, fold, witness, weaver,
)


class TestOriginDecorator(unittest.TestCase):
    """◬ Origin — entry point enforcement."""

    def test_single_call_succeeds(self):
        @origin
        def main(data):
            return data * 2

        result = main(5)
        self.assertEqual(result, 10)

    def test_double_call_raises(self):
        @origin
        def main(data):
            return data

        main(1)  # First call OK
        with self.assertRaises(ChevronContractError) as ctx:
            main(2)  # Second call should raise
        self.assertIn("◬", str(ctx.exception))
        self.assertIn("invoked 2 times", str(ctx.exception))

    def test_glyph_attribute(self):
        @origin
        def main(data):
            return data

        self.assertEqual(main.__chevron_glyph__, "◬")


class TestFilterDecorator(unittest.TestCase):
    """Ө Filter — side-effect-free gate."""

    def test_predicate_returns_bool(self):
        @filter
        def is_positive(x):
            return x > 0

        self.assertTrue(is_positive(5))
        self.assertFalse(is_positive(-1))

    def test_none_return_raises(self):
        @filter
        def bad_filter(x):
            return None

        with self.assertRaises(ChevronContractError) as ctx:
            bad_filter(5)
        self.assertIn("Ө", str(ctx.exception))
        self.assertIn("None", str(ctx.exception))

    def test_filter_returns_list(self):
        @filter
        def positive_only(items):
            return [x for x in items if x > 0]

        result = positive_only([1, -2, 3, -4])
        self.assertEqual(result, [1, 3])

    def test_glyph_attribute(self):
        @filter
        def check(x):
            return True

        self.assertEqual(check.__chevron_glyph__, "Ө")


class TestFoldDecorator(unittest.TestCase):
    """☾ Fold — recursion with convergence."""

    def test_recursive_countdown(self):
        @fold
        def countdown(n):
            if n <= 0:
                return 0
            return countdown(n - 1)

        result = countdown(5)
        self.assertEqual(result, 0)

    def test_max_depth_exceeded(self):
        @fold(max_depth=10)
        def infinite(n):
            return infinite(n + 1)

        with self.assertRaises(ChevronContractError) as ctx:
            infinite(0)
        self.assertIn("☾", str(ctx.exception))
        self.assertIn("exceeded max recursion depth", str(ctx.exception))

    def test_custom_max_depth(self):
        @fold(max_depth=5)
        def deep(n):
            if n <= 0:
                return 0
            return deep(n - 1)

        # 3 levels deep should work
        self.assertEqual(deep(3), 0)

    def test_glyph_attribute(self):
        @fold
        def rec(n):
            if n <= 0: return 0
            return rec(n - 1)

        self.assertEqual(rec.__chevron_glyph__, "☾")


class TestWitnessDecorator(unittest.TestCase):
    """𓂀 Witness — observe without modifying."""

    def test_non_mutating_passes(self):
        @witness
        def log_data(data):
            _ = len(data)  # Read-only operation
            return data

        result = log_data([1, 2, 3])
        self.assertEqual(result, [1, 2, 3])

    def test_mutation_detected(self):
        @witness
        def bad_witness(data):
            data.append(99)  # Mutation!
            return data

        with self.assertRaises(ChevronContractError) as ctx:
            bad_witness([1, 2, 3])
        self.assertIn("𓂀", str(ctx.exception))
        self.assertIn("mutated", str(ctx.exception))

    def test_glyph_attribute(self):
        @witness
        def observe(x):
            return x

        self.assertEqual(observe.__chevron_glyph__, "𓂀")


class TestWeaverDecorator(unittest.TestCase):
    """☤ Weaver — merge without data loss."""

    def test_merge_preserves_data(self):
        @weaver
        def merge(streams):
            return [item for stream in streams for item in stream]

        result = merge([[1, 2], [3, 4]])
        self.assertEqual(result, [1, 2, 3, 4])

    def test_data_loss_detected(self):
        @weaver
        def lossy_merge(streams):
            # Only keeps first stream — loses data from second
            return streams[0]

        with self.assertRaises(ChevronContractError) as ctx:
            lossy_merge([[1, 2], [3, 4]])
        self.assertIn("☤", str(ctx.exception))
        self.assertIn("lost data", str(ctx.exception))

    def test_glyph_attribute(self):
        @weaver
        def merge(x):
            return x

        self.assertEqual(merge.__chevron_glyph__, "☤")


if __name__ == "__main__":
    unittest.main(verbosity=2)
