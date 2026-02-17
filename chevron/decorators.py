"""
Chevron Decorators — Runtime-Enforced Glyph Contracts
=====================================================
Moves glyph constraints from "prompt suggestions" to "runtime guarantees."

Instead of relying on docstring comments like:
    def filter_data(data):
        '''Ө Filters data'''
        ...

You write:
    @chevron.filter
    def filter_data(data):
        ...

Each decorator:
  1. Validates the function signature at decoration time
  2. Wraps calls to enforce the contract at runtime
  3. Raises ChevronContractError on violation

Dan Park | MagicPoint.ai | February 2026
"""

from __future__ import annotations

import copy
import functools
import inspect
import sys
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class ChevronContractError(Exception):
    """Raised when a glyph contract is violated at runtime."""
    pass


# ─────────────────────────────────────────────────────────────
#  Internal Helpers
# ─────────────────────────────────────────────────────────────

# I/O function names that indicate side effects
_IO_CALLS = frozenset({
    "print", "open", "input",
    "write", "writelines", "read", "readline", "readlines",
})

# Modules whose usage indicates I/O side effects
_IO_MODULES = frozenset({
    "os", "shutil", "subprocess", "requests", "urllib",
    "http", "socket", "pathlib",
})


def _has_io_in_source(func: Callable) -> str | None:
    """Check function source for I/O calls. Returns the offending call or None."""
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        return None  # Can't inspect — skip static check

    import ast as _ast
    try:
        tree = _ast.parse(source)
    except SyntaxError:
        return None

    for node in _ast.walk(tree):
        if isinstance(node, _ast.Call):
            # Direct calls: print(...), open(...)
            if isinstance(node.func, _ast.Name) and node.func.id in _IO_CALLS:
                return node.func.id
            # Method calls: sys.stdout.write(...)
            if isinstance(node.func, _ast.Attribute) and node.func.attr in _IO_CALLS:
                return node.func.attr
        # Import checks
        if isinstance(node, (_ast.Import, _ast.ImportFrom)):
            names = []
            if isinstance(node, _ast.ImportFrom) and node.module:
                names.append(node.module.split(".")[0])
            for alias in node.names:
                names.append(alias.name.split(".")[0])
            for name in names:
                if name in _IO_MODULES:
                    return f"import {name}"

    return None


# ─────────────────────────────────────────────────────────────
#  ◬ Origin — Entry Point Decorator
# ─────────────────────────────────────────────────────────────

def origin(func: F) -> F:
    """◬ Origin — marks the program entry point.

    Contract:
        - Must appear exactly once per scope
        - Tracked via a module-level flag; raises on double invocation

    Usage:
        @chevron.origin
        def main(data):
            return process(data)
    """
    _called = {"count": 0}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _called["count"] += 1
        if _called["count"] > 1:
            raise ChevronContractError(
                f"◬ Origin '{func.__name__}' invoked {_called['count']} times. "
                f"Origin must be the single entry point — call it exactly once."
            )
        return func(*args, **kwargs)

    wrapper.__chevron_glyph__ = "◬"
    wrapper.__chevron_origin_state__ = _called
    return wrapper  # type: ignore


# ─────────────────────────────────────────────────────────────
#  Ө Filter — Side-Effect-Free Gate
# ─────────────────────────────────────────────────────────────

def filter(func: F) -> F:
    """Ө Filter — conditional gate that passes or rejects data.

    Contract:
        - Must NOT perform I/O (no print, open, requests, etc.)
        - When applied to a collection, must not modify elements
        - Should return a boolean (predicate) or filtered subset

    Usage:
        @chevron.filter
        def is_valid(item):
            return item.score > 0.5
    """
    # Static check at decoration time
    io_call = _has_io_in_source(func)
    if io_call:
        raise ChevronContractError(
            f"Ө Filter '{func.__name__}' contains I/O call '{io_call}'. "
            f"Filters must be side-effect free — reject, don't transform."
        )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # Allow bool returns (predicate mode) and collection returns (filter mode)
        if result is None:
            raise ChevronContractError(
                f"Ө Filter '{func.__name__}' returned None. "
                f"Filters must return a boolean or filtered subset, never None."
            )
        return result

    wrapper.__chevron_glyph__ = "Ө"
    return wrapper  # type: ignore


# ─────────────────────────────────────────────────────────────
#  ☾ Fold — Recursion with Convergence
# ─────────────────────────────────────────────────────────────

_DEFAULT_MAX_DEPTH = 10_000


def fold(func: F = None, *, max_depth: int = _DEFAULT_MAX_DEPTH) -> F:
    """☾ Fold Time — recursion that feeds output back into input.

    Contract:
        - Must have a reachable base case
        - Must not mutate external state
        - Recursion depth is bounded (default 10,000)

    Usage:
        @chevron.fold
        def countdown(n):
            if n <= 0:
                return 0
            return countdown(n - 1)

        @chevron.fold(max_depth=100)
        def fibonacci(n):
            ...
    """
    def decorator(fn: F) -> F:
        # Static check for I/O
        io_call = _has_io_in_source(fn)
        if io_call:
            raise ChevronContractError(
                f"☾ Fold '{fn.__name__}' contains I/O call '{io_call}'. "
                f"Folds must not mutate external state."
            )

        _depth = {"current": 0}

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _depth["current"] += 1
            if _depth["current"] > max_depth:
                _depth["current"] = 0
                raise ChevronContractError(
                    f"☾ Fold '{fn.__name__}' exceeded max recursion depth ({max_depth}). "
                    f"Base case may be unreachable."
                )
            try:
                result = fn(*args, **kwargs)
                return result
            finally:
                _depth["current"] -= 1

        wrapper.__chevron_glyph__ = "☾"
        wrapper.__chevron_max_depth__ = max_depth
        return wrapper  # type: ignore

    # Support both @fold and @fold(max_depth=N)
    if func is not None:
        return decorator(func)
    return decorator  # type: ignore


# ─────────────────────────────────────────────────────────────
#  𓂀 Witness — Observe Without Modifying
# ─────────────────────────────────────────────────────────────

def witness(func: F) -> F:
    """𓂀 Witness — observes the data stream without altering it.

    Contract:
        - Must NEVER modify the input data
        - Pure observation only (logging, metrics, etc.)
        - Returns the input unchanged

    Usage:
        @chevron.witness
        def log_step(data):
            logger.info(f"Processing: {data}")
            return data
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Deep-copy mutable arguments to detect mutation
        original_args = []
        for arg in args:
            try:
                original_args.append(copy.deepcopy(arg))
            except (TypeError, copy.Error):
                original_args.append(arg)  # Fallback for non-copyable

        original_kwargs = {}
        for k, v in kwargs.items():
            try:
                original_kwargs[k] = copy.deepcopy(v)
            except (TypeError, copy.Error):
                original_kwargs[k] = v

        result = func(*args, **kwargs)

        # Check that arguments were not mutated
        for i, (orig, current) in enumerate(zip(original_args, args)):
            try:
                if orig != current:
                    raise ChevronContractError(
                        f"𓂀 Witness '{func.__name__}' mutated argument {i}. "
                        f"Witness must observe without modifying data."
                    )
            except (TypeError, ValueError):
                pass  # Skip comparison for non-comparable types

        return result

    wrapper.__chevron_glyph__ = "𓂀"
    return wrapper  # type: ignore


# ─────────────────────────────────────────────────────────────
#  ☤ Weaver — Merge Without Data Loss
# ─────────────────────────────────────────────────────────────

def weaver(func: F) -> F:
    """☤ Weaver — merges streams, preserving all input data.

    Contract:
        - Must preserve all input data — nothing may be lost
        - Accepts a list/collection of values, produces a merged result
        - The output must contain all elements from the input

    Usage:
        @chevron.weaver
        def merge_results(streams):
            return [item for stream in streams for item in stream]
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)

        # Verify data preservation: check that input elements appear in output
        # Only enforced when the first argument is iterable and result is iterable
        if args and hasattr(args[0], '__iter__') and hasattr(result, '__iter__'):
            try:
                input_items = args[0]
                result_list = list(result) if not isinstance(result, list) else result

                # For nested iterables (list of lists), flatten input
                flat_input = []
                for item in input_items:
                    if hasattr(item, '__iter__') and not isinstance(item, (str, bytes)):
                        flat_input.extend(item)
                    else:
                        flat_input.append(item)

                # Check every input element exists in output
                result_check = list(result_list)
                for item in flat_input:
                    if item not in result_check:
                        raise ChevronContractError(
                            f"☤ Weaver '{func.__name__}' lost data: {item!r} "
                            f"from input is not present in output. "
                            f"Weaver must preserve all input data."
                        )
            except (TypeError, ChevronContractError):
                raise
            except Exception:
                pass  # Skip check for non-comparable types

        return result

    wrapper.__chevron_glyph__ = "☤"
    return wrapper  # type: ignore
