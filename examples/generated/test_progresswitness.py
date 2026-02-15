import pytest
import sys
import json
import inspect
import ast
from unittest.mock import MagicMock, patch, call
from typing import Dict, Any

import progresswitness

# -----------------------------------------------------------------------------
# 1. Structural Tests
# -----------------------------------------------------------------------------

def test_interface_class_exists():
    """Verify ProgressWitness class exists."""
    assert hasattr(progresswitness, "ProgressWitness")
    assert inspect.isclass(progresswitness.ProgressWitness)

def test_interface_method_signatures():
    """Verify method signatures match the SCP contract."""
    witness = progresswitness.ProgressWitness()
    
    # emit_progress(current: int, total: int, label: str) -> None
    sig_progress = inspect.signature(witness.emit_progress)
    params_progress = list(sig_progress.parameters.keys())
    assert params_progress == ['current', 'total', 'label']
    assert sig_progress.return_annotation is None or sig_progress.return_annotation == type(None)

    # emit_result(result: Dict[str, Any]) -> None
    sig_result = inspect.signature(witness.emit_result)
    params_result = list(sig_result.parameters.keys())
    assert params_result == ['result']
    
    # emit_error(error: str, context: str) -> None
    sig_error = inspect.signature(witness.emit_error)
    params_error = list(sig_error.parameters.keys())
    assert params_error == ['error', 'context']

def test_interface_public_api():
    """Verify no unexpected public methods exist."""
    witness = progresswitness.ProgressWitness()
    public_methods = [
        m for m in dir(witness) 
        if not m.startswith('_') and callable(getattr(witness, m))
    ]
    expected_methods = {'emit_progress', 'emit_result', 'emit_error'}
    assert set(public_methods) == expected_methods

# -----------------------------------------------------------------------------
# 2. Constraint Tests
# -----------------------------------------------------------------------------

def test_constraint_no_exceptions_raised():
    """Constraint: Must NEVER raise exceptions that halt the pipeline."""
    witness = progresswitness.ProgressWitness()
    
    # Mock stdout to raise an exception
    with patch('sys.stdout.write', side_effect=IOError("Disk full")):
        try:
            witness.emit_progress(50, 100, "Testing")
            witness.emit_result({"key": "value"})
            witness.emit_error("Something broke", "TestContext")
        except Exception as e:
            pytest.fail(f"Method raised exception {e} instead of suppressing it.")

def test_constraint_line_buffered():
    """Constraint: Output must be line-buffered (flush called)."""
    witness = progresswitness.ProgressWitness()
    
    with patch('sys.stdout') as mock_stdout:
        witness.emit_progress(1, 10, "buffering check")
        mock_stdout.flush.assert_called()
        
        mock_stdout.reset_mock()
        witness.emit_result({"a": 1})
        mock_stdout.flush.assert_called()
        
        mock_stdout.reset_mock()
        witness.emit_error("err", "ctx")
        mock_stdout.flush.assert_called()

def test_constraint_structured_markers():
    """Constraint: Must use structured markers [PROGRESS], [RESULT], [ERROR]."""
    witness = progresswitness.ProgressWitness()
    
    with patch('sys.stdout') as mock_stdout:
        # Test PROGRESS
        witness.emit_progress(10, 100, "loading")
        args, _ = mock_stdout.write.call_args
        assert args[0].startswith("[PROGRESS]")
        
        # Test RESULT
        witness.emit_result({"done": True})
        args, _ = mock_stdout.write.call_args
        assert args[0].startswith("[RESULT]")
        
        # Test ERROR
        witness.emit_error("fail", "ctx")
        args, _ = mock_stdout.write.call_args
        assert args[0].startswith("[ERROR]")

def test_constraint_no_forbidden_imports():
    """Constraint: Verify no forbidden project modules are imported via AST."""
    with open(progresswitness.__file__, "r") as f:
        tree = ast.parse(f.read())
    
    allowed_modules = {'sys', 'json', 'typing'}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name in allowed_modules, f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            assert node.module in allowed_modules, f"Forbidden import from: {node.module}"

# -----------------------------------------------------------------------------
# 3. Behavioral Tests
# -----------------------------------------------------------------------------

def test_behavior_emit_progress_calculation():
    """Verify percentage calculation logic."""
    witness = progresswitness.ProgressWitness()
    
    with patch('sys.stdout') as mock_stdout:
        # 50%
        witness.emit_progress(50, 100, "Halfway")
        mock_stdout.write.assert_called_with("[PROGRESS] 50% | 50/100 | Halfway\n")
        
        # 0 Total (ZeroDivisionError prevention)
        witness.emit_progress(0, 0, "Start")
        mock_stdout.write.assert_called_with("[PROGRESS] 0% | 0/0 | Start\n")
        
        # Clamping > 100%
        witness.emit_progress(150, 100, "Overachiever")
        mock_stdout.write.assert_called_with("[PROGRESS] 100% | 150/100 | Overachiever\n")
        
        # Clamping < 0%
        witness.emit_progress(-10, 100, "Underflow")
        mock_stdout.write.assert_called_with("[PROGRESS] 0% | -10/100 | Underflow\n")

def test_behavior_emit_result_json():
    """Verify result is serialized to JSON."""
    witness = progresswitness.ProgressWitness()
    data = {"id": 123, "status": "ok", "nested": {"a": 1}}
    
    with patch('sys.stdout') as mock_stdout:
        witness.emit_result(data)
        args, _ = mock_stdout.write.call_args
        output = args[0].strip()
        
        assert output.startswith("[RESULT] ")
        json_part = output.replace("[RESULT] ", "")
        parsed = json.loads(json_part)
        assert parsed == data

def test_behavior_emit_result_serialization_failure():
    """Verify fallback when JSON serialization fails."""
    witness = progresswitness.ProgressWitness()
    
    # Create an object that is not JSON serializable
    class Unserializable:
        pass
    
    bad_data = {"obj": Unserializable()}
    
    with patch('sys.stdout') as mock_stdout:
        # Should catch TypeError inside and call emit_error
        # Note: The implementation might call emit_error internally
        witness.emit_result(bad_data)
        
        # Check if [ERROR] was logged eventually
        # We look at the last call or any call
        calls = mock_stdout.write.call_args_list
        error_logged = any("[ERROR]" in str(c) for c in calls)
        assert error_logged, "Should emit error on serialization failure"

def test_behavior_emit_error_sanitization():
    """Verify error messages are sanitized (newlines removed)."""
    witness = progresswitness.ProgressWitness()
    
    error_msg = "Line1\nLine2"
    context = "Module\nName"
    
    with patch('sys.stdout') as mock_stdout:
        witness.emit_error(error_msg, context)
        args, _ = mock_stdout.write.call_args
        output = args[0]
        
        assert "\n" not in output.strip() # Only the final newline allowed
        assert "Line1 Line2" in output
        assert "Module Name" in output

# -----------------------------------------------------------------------------
# 4. Isolation Tests
# -----------------------------------------------------------------------------

def test_isolation_no_global_state_modification():
    """Verify the module does not modify global state or arguments."""
    witness = progresswitness.ProgressWitness()
    data = {"original": "value"}
    data_copy = data.copy()
    
    with patch('sys.stdout'):
        witness.emit_result(data)
    
    assert data == data_copy, "emit_result modified the input dictionary"

def test_isolation_pure_logging():
    """Verify methods return None (pure side-effect logging)."""
    witness = progresswitness.ProgressWitness()
    with patch('sys.stdout'):
        ret_prog = witness.emit_progress(1, 10, "test")
        ret_res = witness.emit_result({})
        ret_err = witness.emit_error("e", "c")
    
    assert ret_prog is None
    assert ret_res is None
    assert ret_err is None