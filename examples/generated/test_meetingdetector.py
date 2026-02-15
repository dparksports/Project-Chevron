import pytest
import sys
import os
import json
import ast
import inspect
import types
from unittest.mock import MagicMock, patch, mock_open, call

import meetingdetector
from meetingdetector import MeetingDetector, ClassificationResult, Classification, DetectionReport

# ---------------------------------------------------------------------------
# 1. Structural Tests
# ---------------------------------------------------------------------------

def test_interface_detect_meetings_signature():
    """Verify detect_meetings signature matches SCP contract."""
    sig = inspect.signature(MeetingDetector.detect_meetings)
    params = list(sig.parameters.keys())
    assert params == ['self', 'directory', 'provider', 'skip_checked']
    assert sig.parameters['directory'].annotation == str
    assert sig.parameters['provider'].annotation == str
    assert sig.parameters['skip_checked'].annotation == bool
    assert sig.return_annotation == DetectionReport

def test_interface_classify_transcript_signature():
    """Verify classify_transcript signature matches SCP contract."""
    sig = inspect.signature(MeetingDetector.classify_transcript)
    params = list(sig.parameters.keys())
    assert params == ['self', 'text', 'provider']
    assert sig.parameters['text'].annotation == str
    assert sig.parameters['provider'].annotation == str
    assert sig.return_annotation == Classification

def test_interface_public_api():
    """Verify only expected public methods exist."""
    public_methods = [
        m for m in dir(MeetingDetector) 
        if callable(getattr(MeetingDetector, m)) and not m.startswith('_')
    ]
    expected = {'detect_meetings', 'classify_transcript'}
    assert set(public_methods) == expected

# ---------------------------------------------------------------------------
# 2. Constraint Tests
# ---------------------------------------------------------------------------

def test_constraint_forbidden_imports():
    """Verify module does not import forbidden dependencies (Whisper, sentence-transformers)."""
    source_file = inspect.getsourcefile(meetingdetector)
    with open(source_file, "r") as f:
        tree = ast.parse(f.read())

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name.lower()
                assert "whisper" not in name, f"Forbidden import found: {alias.name}"
                assert "sentence_transformers" not in name, f"Forbidden import found: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod = node.module.lower()
                assert "whisper" not in mod, f"Forbidden import found: {node.module}"
                assert "sentence_transformers" not in mod, f"Forbidden import found: {node.module}"

def test_constraint_no_transcript_modification():
    """Verify transcripts are never opened in write mode (The Filter constraint)."""
    detector = MeetingDetector()
    transcript_path = '/tmp/test_transcript_v1.txt'
    
    with patch('glob.glob', return_value=[transcript_path]), \
         patch('os.path.exists', return_value=False), \
         patch('builtins.open', mock_open(read_data="some content")) as mocked_file, \
         patch('meetingdetector.LLMProvider') as mock_llm:
        
        # Mock valid LLM response to ensure full flow runs
        mock_llm.generate.return_value = json.dumps({
            "is_meeting": True, "confidence": 0.9, "reasoning": "ok", "category": "MEETING"
        })
        
        detector.detect_meetings('/tmp', 'gpt-4')
        
        # Inspect all open calls
        for call_args in mocked_file.call_args_list:
            args, _ = call_args
            filename = args[0]
            mode = args[1] if len(args) > 1 else 'r'
            
            # If opening the transcript file, ensure it is NOT writable
            if filename == transcript_path:
                assert 'w' not in mode and 'a' not in mode and '+' not in mode, \
                    f"Transcript opened in write mode: {mode}. Violation of The Filter."

# ---------------------------------------------------------------------------
# 3. Behavioral Tests
# ---------------------------------------------------------------------------

def test_behavior_classify_transcript_success():
    """Verify classify_transcript parses valid LLM JSON correctly."""
    detector = MeetingDetector()
    
    mock_response = json.dumps({
        "is_meeting": True,
        "confidence": 0.95,
        "reasoning": "Clear dialogue",
        "category": "MEETING"
    })
    
    with patch('meetingdetector.LLMProvider') as mock_llm:
        mock_llm.generate.return_value = mock_response
        
        result = detector.classify_transcript("Speaker A: Hello.", "gpt-4")
        
        assert isinstance(result, Classification)
        assert result.is_meeting is True
        assert result.confidence == 0.95
        assert result.category == ClassificationResult.MEETING
        assert result.reasoning == "Clear dialogue"
        
        # Verify prompt determinism (contains specific instructions)
        args, _ = mock_llm.generate.call_args
        prompt = args[0]
        assert "You are a quality control system" in prompt
        assert "Respond with valid JSON only" in prompt

def test_behavior_classify_transcript_hallucination():
    """Verify classify_transcript correctly handles hallucination response."""
    detector = MeetingDetector()
    
    mock_response = json.dumps({
        "is_meeting": False,
        "confidence": 0.99,
        "reasoning": "Repetitive loops",
        "category": "HALLUCINATION"
    })
    
    with patch('meetingdetector.LLMProvider') as mock_llm:
        mock_llm.generate.return_value = mock_response
        
        result = detector.classify_transcript("Thank you. Thank you.", "gpt-4")
        
        assert result.is_meeting is False
        assert result.category == ClassificationResult.HALLUCINATION

def test_behavior_classify_transcript_failure():
    """Verify classify_transcript handles LLM failure gracefully (returns UNKNOWN)."""
    detector = MeetingDetector()
    
    with patch('meetingdetector.LLMProvider') as mock_llm:
        mock_llm.generate.side_effect = Exception("API Error")
        
        result = detector.classify_transcript("text", "gpt-4")
        
        assert result.category == ClassificationResult.UNKNOWN
        assert result.is_meeting is False
        assert "API Error" in result.reasoning

def test_behavior_detect_meetings_flow():
    """Verify batch detection flow, sidecar creation, and progress reporting."""
    detector = MeetingDetector()
    
    transcript_files = ['/data/f1_transcript_m.txt', '/data/f2_transcript_m.txt']
    
    # Setup mocks
    with patch('glob.glob', return_value=transcript_files), \
         patch('os.path.exists', side_effect=[False, False]), \
         patch('builtins.open', mock_open(read_data="transcript content")) as mocked_open, \
         patch('meetingdetector.LLMProvider') as mock_llm, \
         patch('sys.stdout', new_callable=MagicMock) as mock_stdout:
             
        # Mock LLM responses for 2 files
        mock_llm.generate.side_effect = [
            json.dumps({"is_meeting": True, "category": "MEETING"}),
            json.dumps({"is_meeting": False, "category": "HALLUCINATION"})
        ]
        
        report = detector.detect_meetings('/data', 'gpt-4', skip_checked=True)
        
        # Verify Report
        assert report.total_files == 2
        assert report.processed_files == 2
        assert report.meetings_detected == 1
        assert report.hallucinations_detected == 1
        
        # Verify Sidecar Writes
        # We expect writes to .classification.json files
        handle = mocked_open()
        write_calls = [c for c in handle.write.call_args_list]
        assert len(write_calls) >= 2 # At least 2 writes (one per file)
        
        # Verify Progress Output
        combined_output = "".join([str(arg) for call_args in mock_stdout.write.call_args_list for arg in call_args[0]])
        assert "[DETECT_PROGRESS]" in combined_output

def test_behavior_detect_meetings_skip_checked():
    """Verify skip_checked=True skips LLM call if sidecar exists."""
    detector = MeetingDetector()
    transcript = '/data/f1_transcript_m.txt'
    
    cached_classification = {
        "is_meeting": True,
        "confidence": 1.0,
        "reasoning": "Cached",
        "category": "MEETING"
    }
    
    with patch('glob.glob', return_value=[transcript]), \
         patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=json.dumps(cached_classification))) as mocked_open, \
         patch('meetingdetector.LLMProvider') as mock_llm:
        
        report = detector.detect_meetings('/data', 'gpt-4', skip_checked=True)
        
        # Should NOT call LLM
        mock_llm.generate.assert_not_called()
        
        # Should count as meeting based on cache
        assert report.meetings_detected == 1
        # Processed count logic: skipped files are not "processed" in the loop counter
        assert report.processed_files == 0

def test_behavior_empty_file_handling():
    """Verify empty files are marked UNKNOWN without LLM call."""
    detector = MeetingDetector()
    transcript = '/data/empty_transcript_m.txt'
    
    with patch('glob.glob', return_value=[transcript]), \
         patch('os.path.exists', return_value=False), \
         patch('builtins.open', mock_open(read_data="")), \
         patch('meetingdetector.LLMProvider') as mock_llm:
        
        report = detector.detect_meetings('/data', 'gpt-4')
        
        # Should not call LLM for empty file
        mock_llm.generate.assert_not_called()
        
        # Result should be UNKNOWN
        res = report.results['empty_transcript_m.txt']
        assert res.category == ClassificationResult.UNKNOWN
        assert "empty" in res.reasoning

# ---------------------------------------------------------------------------
# 4. Isolation Tests
# ---------------------------------------------------------------------------

def test_isolation_no_global_state():
    """Verify module does not rely on global mutable state."""
    module_vars = vars(meetingdetector)
    for name, val in module_vars.items():
        if not name.startswith("__") and not isinstance(val, (type, types.ModuleType, types.FunctionType)):
            # Allow constants (uppercase)
            if name.isupper(): 
                continue
            # Fail if mutable global found
            assert not isinstance(val, (list, dict, set)), f"Found mutable global state: {name}"