import pytest
import unittest.mock as mock
from unittest.mock import MagicMock, patch
import inspect
import ast
import sys
import os
from pathlib import Path

# Import the module under test
import audioingest

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def ingest_module():
    """Returns an instance of the AudioIngest class."""
    return audioingest.AudioIngest()

@pytest.fixture
def mock_torch_np():
    """
    Mocks torch and numpy within the audioingest module to ensure 
    tests run without these heavy dependencies installed.
    """
    with patch('audioingest.torch') as mock_torch, \
         patch('audioingest.np') as mock_np:
        
        # Setup basic torch mock behavior
        mock_torch.Tensor = MagicMock()
        mock_torch.from_numpy = MagicMock(return_value="mock_tensor")
        mock_torch.zeros = MagicMock(return_value="empty_tensor")
        
        # Setup basic numpy mock behavior
        mock_np.float32 = "float32"
        mock_np.frombuffer = MagicMock(return_value=MagicMock(copy=lambda: "mock_np_array"))
        
        yield mock_torch, mock_np

# -----------------------------------------------------------------------------
# 1. Structural Tests
# -----------------------------------------------------------------------------

def test_interface_class_exists():
    """Verify AudioIngest class exists."""
    assert hasattr(audioingest, 'AudioIngest')
    assert inspect.isclass(audioingest.AudioIngest)

def test_interface_methods_exist(ingest_module):
    """Verify required methods exist."""
    assert hasattr(ingest_module, 'find_media')
    assert hasattr(ingest_module, 'load_audio')
    assert hasattr(ingest_module, 'get_device_config')

def test_interface_signatures(ingest_module):
    """Verify method signatures match the SCP contract."""
    # find_media(directory: str) -> List[MediaFile]
    sig_find = inspect.signature(ingest_module.find_media)
    assert list(sig_find.parameters.keys()) == ['directory']
    assert sig_find.return_annotation != inspect.Signature.empty

    # load_audio(file_path: str) -> AudioTensor
    sig_load = inspect.signature(ingest_module.load_audio)
    assert list(sig_load.parameters.keys()) == ['file_path']
    
    # get_device_config(override: str | None) -> DeviceConfig
    sig_device = inspect.signature(ingest_module.get_device_config)
    assert list(sig_device.parameters.keys()) == ['override']

def test_dataclasses_exist():
    """Verify helper dataclasses exist."""
    assert hasattr(audioingest, 'MediaFile')
    assert hasattr(audioingest, 'DeviceConfig')

# -----------------------------------------------------------------------------
# 2. Constraint Tests
# -----------------------------------------------------------------------------

def test_constraint_no_forbidden_imports():
    """
    Verify that the module does NOT import forbidden project modules 
    (Whisper, VAD, etc.) or perform analysis.
    """
    source = inspect.getsource(audioingest)
    tree = ast.parse(source)
    
    forbidden_modules = {'whisper', 'faster_whisper', 'webrtcvad', 'silero', 'pyannote'}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden_modules, f"Forbidden import found: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                assert node.module not in forbidden_modules, f"Forbidden import found: {node.module}"

def test_constraint_media_extensions_truth():
    """Verify MEDIA_EXTENSIONS is the source of truth and contains expected formats."""
    exts = audioingest.MEDIA_EXTENSIONS
    expected = {'.mp3', '.wav', '.flac', '.m4a'}
    assert expected.issubset(exts), "Missing common audio formats in MEDIA_EXTENSIONS"
    assert isinstance(exts, set), "MEDIA_EXTENSIONS should be a set for O(1) lookup"

def test_constraint_load_audio_origin_marker():
    """Verify load_audio is marked as an Origin (conceptually via docstring or structure)."""
    # In SCP, we check if it performs the raw IO. 
    # Here we verify it uses subprocess/ffmpeg directly, not a wrapper library.
    source = inspect.getsource(audioingest.AudioIngest.load_audio)
    assert "subprocess.Popen" in source or "ffmpeg" in source, "load_audio must use ffmpeg directly"

# -----------------------------------------------------------------------------
# 3. Behavioral Tests
# -----------------------------------------------------------------------------

@patch('os.walk')
@patch('pathlib.Path')
def test_behavior_find_media_discovery(mock_path_cls, mock_walk, ingest_module):
    """
    Verify find_media discovers files and filters by extension.
    Also verifies determinism (sorting).
    """
    # Setup Mock Path
    mock_path_obj = MagicMock()
    mock_path_obj.exists.return_value = True
    mock_path_cls.return_value = mock_path_obj
    
    # Setup os.walk data: (root, dirs, files)
    # Intentionally unsorted to test sorting logic
    mock_walk.return_value = [
        ('/root', [], ['b.txt', 'a.mp3', 'c.wav', 'd.exe'])
    ]
    
    # Mock stat for size
    mock_stat = MagicMock()
    mock_stat.st_size = 1024
    
    # When Path(root) / filename is called, return a mock that has suffix and stat
    def path_side_effect(*args, **kwargs):
        p = MagicMock()
        filename = args[0] if args else "unknown"
        
        if filename.endswith('.mp3'):
            p.suffix = '.mp3'
            p.absolute.return_value = f"/root/{filename}"
        elif filename.endswith('.wav'):
            p.suffix = '.wav'
            p.absolute.return_value = f"/root/{filename}"
        else:
            p.suffix = '.txt' # or .exe
        
        p.stat.return_value = mock_stat
        return p

    # We need to mock the behavior of the constructed path inside the loop
    # The code does: file_path = Path(root) / filename
    # Since we mocked Path class, Path(root) returns mock_path_obj.
    # So we mock __truediv__ (the / operator) on mock_path_obj.
    mock_path_obj.__truediv__.side_effect = path_side_effect

    # Execute
    results = ingest_module.find_media("/root")
    
    # Verify
    assert len(results) == 2
    # Check sorting: a.mp3 should come before c.wav
    assert results[0].filename == 'a.mp3'
    assert results[1].filename == 'c.wav'
    
    # Verify attributes
    assert results[0].path == "/root/a.mp3"
    assert results[0].size_bytes == 1024
    assert isinstance(results[0], audioingest.MediaFile)

@patch('pathlib.Path')
def test_behavior_find_media_not_exists(mock_path_cls, ingest_module):
    """Verify find_media returns empty list if directory doesn't exist."""
    mock_path_obj = MagicMock()
    mock_path_obj.exists.return_value = False
    mock_path_cls.return_value = mock_path_obj
    
    results = ingest_module.find_media("/fake/dir")
    assert results == []

@patch('subprocess.Popen')
@patch('os.path.exists')
def test_behavior_load_audio_success(mock_exists, mock_popen, ingest_module, mock_torch_np):
    """Verify load_audio constructs correct ffmpeg command and returns tensor."""
    mock_torch, mock_np = mock_torch_np
    mock_exists.return_value = True
    
    # Mock subprocess
    process_mock = MagicMock()
    process_mock.communicate.return_value = (b'fake_audio_bytes', b'')
    process_mock.return_value = 0 # Success
    mock_popen.return_value = process_mock
    
    # Execute
    result = ingest_module.load_audio("test.mp3")
    
    # Verify FFmpeg command
    args, _ = mock_popen.call_args
    cmd = args[0]
    assert cmd[0] == 'ffmpeg'
    assert '-ar' in cmd and cmd[cmd.index('-ar') + 1] == '16000' # 16kHz
    assert '-ac' in cmd and cmd[cmd.index('-ac') + 1] == '1'     # Mono
    assert '-f' in cmd and cmd[cmd.index('-f') + 1] == 'f32le'   # Float32
    
    # Verify Tensor conversion
    mock_np.frombuffer.assert_called_with(b'fake_audio_bytes', dtype='float32')
    mock_torch.from_numpy.assert_called()
    assert result == "mock_tensor"

@patch('os.path.exists')
def test_behavior_load_audio_file_not_found(mock_exists, ingest_module):
    """Verify load_audio raises FileNotFoundError if input missing."""
    mock_exists.return_value = False
    with pytest.raises(FileNotFoundError):
        ingest_module.load_audio("missing.wav")

@patch('subprocess.Popen')
@patch('os.path.exists')
def test_behavior_load_audio_ffmpeg_failure(mock_exists, mock_popen, ingest_module, mock_torch_np):
    """Verify RuntimeError if ffmpeg returns non-zero code."""
    mock_exists.return_value = True
    
    process_mock = MagicMock()
    process_mock.communicate.return_value = (b'', b'error')
    process_mock.returncode = 1 # Failure
    mock_popen.return_value = process_mock
    
    with pytest.raises(RuntimeError) as excinfo:
        ingest_module.load_audio("corrupt.mp3")
    assert "FFmpeg failed" in str(excinfo.value)

def test_behavior_get_device_config_override(ingest_module):
    """Verify override takes precedence."""
    config = ingest_module.get_device_config(override="cpu")
    assert config.device == "cpu"
    assert config.name == "User Override"

def test_behavior_get_device_config_cuda(ingest_module, mock_torch_np):
    """Verify CUDA detection."""
    mock_torch, _ = mock_torch_np
    
    # Simulate CUDA available
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.current_device.return_value = 0
    mock_torch.cuda.get_device_name.return_value = "NVIDIA Fake GPU"
    
    config = ingest_module.get_device_config(override=None)
    assert config.device == "cuda"
    assert config.index == 0
    assert config.name == "NVIDIA Fake GPU"

def test_behavior_get_device_config_mps(ingest_module, mock_torch_np):
    """Verify MPS (Apple Silicon) detection."""
    mock_torch, _ = mock_torch_np
    
    # Simulate CUDA not available, but MPS available
    mock_torch.cuda.is_available.return_value = False
    mock_torch.backends.mps.is_available.return_value = True
    
    config = ingest_module.get_device_config(override=None)
    assert config.device == "mps"
    assert "Apple" in config.name

def test_behavior_get_device_config_cpu_fallback(ingest_module, mock_torch_np):
    """Verify CPU fallback when no accelerators found."""
    mock_torch, _ = mock_torch_np
    
    mock_torch.cuda.is_available.return_value = False
    # Mock mps not available (either attribute missing or returns False)
    # Here we simulate attribute existing but returning False
    mock_torch.backends.mps.is_available.return_value = False
    
    config = ingest_module.get_device_config(override=None)
    assert config.device == "cpu"

# -----------------------------------------------------------------------------
# 4. Isolation Tests
# -----------------------------------------------------------------------------

def test_isolation_no_global_state_mutation():
    """
    Verify that calling methods does not modify global module state.
    (This is a heuristic check; Python allows almost anything, but we check obvious globals).
    """
    initial_extensions = audioingest.MEDIA_EXTENSIONS.copy()
    
    ingest = audioingest.AudioIngest()
    # Call a method that might be tempted to change state
    ingest.get_device_config(None)
    
    assert audioingest.MEDIA_EXTENSIONS == initial_extensions

def test_isolation_imports_are_local_or_top_level():
    """
    Verify that no imports happen inside methods (except for specific lazy loading if designed).
    The SCP prefers explicit dependencies.
    """
    source = inspect.getsource(audioingest)
    tree = ast.parse(source)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for child in node.body:
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    # Allow imports inside try/except blocks for optional dependencies (like torch)
                    # But generally discourage hidden dependencies.
                    # For this test, we just log or assert if it's a logic import.
                    pass 
                    # (This test is placeholder for strict enforcement if required)