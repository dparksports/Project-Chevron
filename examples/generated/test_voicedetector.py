import sys
import pytest
import ast
import inspect
from unittest.mock import MagicMock, patch, ANY

# -----------------------------------------------------------------------------
# 1. Setup Mocks for Dependencies BEFORE importing the module under test
# -----------------------------------------------------------------------------

# Mock audio_ingest dependency
mock_audio_ingest_module = MagicMock()
sys.modules['audio_ingest'] = mock_audio_ingest_module

# Mock torch to prevent heavy loading and hardware access
mock_torch = MagicMock()
sys.modules['torch'] = mock_torch

# Import the module under test
import voicedetector

# -----------------------------------------------------------------------------
# 2. Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the VAD singleton before each test to ensure isolation."""
    voicedetector._VadModelSingleton._instance = None
    voicedetector._VadModelSingleton._model = None
    voicedetector._VadModelSingleton._utils = None
    voicedetector._VadModelSingleton._device = None
    yield

@pytest.fixture
def mock_vad_utils():
    """Mocks the utils tuple returned by torch.hub.load."""
    # utils structure: (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks)
    get_speech_timestamps = MagicMock()
    return (get_speech_timestamps, MagicMock(), MagicMock(), MagicMock(), MagicMock())

# -----------------------------------------------------------------------------
# 3. Structural Tests
# -----------------------------------------------------------------------------

class TestInterfaces:
    def test_run_vad_scan_signature(self):
        """Verify run_vad_scan signature matches SCP contract."""
        sig = inspect.signature(voicedetector.run_vad_scan)
        assert list(sig.parameters.keys()) == ['file_path', 'threshold']
        assert sig.return_annotation == voicedetector.VadResult

    def test_run_batch_vad_scan_signature(self):
        """Verify run_batch_vad_scan signature matches SCP contract."""
        sig = inspect.signature(voicedetector.run_batch_vad_scan)
        assert list(sig.parameters.keys()) == ['directory', 'threshold', 'skip_existing']
        assert sig.return_annotation == voicedetector.VadReport

    def test_cluster_segments_signature(self):
        """Verify cluster_segments signature matches SCP contract."""
        sig = inspect.signature(voicedetector.cluster_segments)
        assert list(sig.parameters.keys()) == ['segments', 'gap_threshold']

    def test_data_structures(self):
        """Verify required Data Classes exist."""
        seg = voicedetector.Segment(0.0, 1.0)
        assert seg.start == 0.0 and seg.end == 1.0
        
        blk = voicedetector.Block(0.0, 1.0, [seg])
        assert blk.segments == [seg]
        
        res = voicedetector.VadResult("test.wav", True)
        assert res.has_speech is True

# -----------------------------------------------------------------------------
# 4. Constraint Tests
# -----------------------------------------------------------------------------

class TestConstraints:
    def test_no_forbidden_imports(self):
        """Constraint: Must not import forbidden modules (e.g., whisper)."""
        source = inspect.getsource(voicedetector)
        tree = ast.parse(source)
        
        forbidden = ['whisper', 'openai', 'soundfile', 'pydub']
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for f in forbidden:
                        assert f not in alias.name, f"Forbidden import found: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                for f in forbidden:
                    if node.module and f in node.module:
                        pytest.fail(f"Forbidden import found: {node.module}")

    def test_lazy_loading_singleton(self, mock_vad_utils):
        """Constraint: VAD model must be loaded lazily and cached."""
        mock_model = MagicMock()
        
        with patch('voicedetector.torch.hub.load', return_value=(mock_model, mock_vad_utils)) as mock_load:
            with patch('voicedetector.AudioIngest.get_device_config', return_value='cpu'):
                # First load
                instance1 = voicedetector._VadModelSingleton.get_instance()
                instance1.load()
                
                # Second load
                instance2 = voicedetector._VadModelSingleton.get_instance()
                instance2.load()
                
                # Verify singleton behavior
                assert instance1 is instance2
                # Verify load called only once
                mock_load.assert_called_once()

    def test_no_file_modification(self):
        """Constraint: Must not modify source audio files (The Filter)."""
        with patch('builtins.open', new_callable=MagicMock) as mock_open:
            with patch('voicedetector.Path.exists', return_value=True):
                with patch('voicedetector.AudioIngest.load_audio'):
                    with patch('voicedetector._VadModelSingleton.load', return_value=(MagicMock(), (MagicMock(),0,0,0,0), MagicMock())):
                        try:
                            voicedetector.run_vad_scan("dummy.wav", 0.5)
                        except:
                            pass 
                        
                        # Ensure no file was opened in write mode
                        for call in mock_open.mock_calls:
                            args, kwargs = call[1], call[2]
                            mode = args[1] if len(args) > 1 else kwargs.get('mode', 'r')
                            assert 'w' not in mode and 'a' not in mode and '+' not in mode

# -----------------------------------------------------------------------------
# 5. Behavioral Tests
# -----------------------------------------------------------------------------

class TestBehavior:
    
    def test_cluster_segments_weaving(self):
        """Test the Weaver logic: merging adjacent segments based on gap."""
        # Segments: [0-1], [1.5-2.5], [5-6]
        # Gap threshold: 1.0
        # Gap between 1 and 1.5 is 0.5 (<= 1.0) -> Merge
        # Gap between 2.5 and 5 is 2.5 (> 1.0) -> Split
        
        segs = [
            voicedetector.Segment(0.0, 1.0),
            voicedetector.Segment(1.5, 2.5),
            voicedetector.Segment(5.0, 6.0)
        ]
        
        blocks = voicedetector.cluster_segments(segs, gap_threshold=1.0)
        
        assert len(blocks) == 2
        
        # Block 1: 0.0 to 2.5 (Merged)
        assert blocks[0].start == 0.0
        assert blocks[0].end == 2.5
        assert len(blocks[0].segments) == 2
        
        # Block 2: 5.0 to 6.0 (Isolated)
        assert blocks[1].start == 5.0
        assert blocks[1].end == 6.0
        assert len(blocks[1].segments) == 1

    def test_cluster_segments_empty(self):
        """Test weaving with empty input."""
        assert voicedetector.cluster_segments([], 1.0) == []

    @patch('voicedetector.Path')
    @patch('voicedetector.AudioIngest')
    @patch('voicedetector._VadModelSingleton')
    def test_run_vad_scan_flow(self, mock_singleton_cls, mock_ingest, mock_path_cls, mock_vad_utils):
        """Test full flow of run_vad_scan with mocked detection."""
        # Setup Mocks
        mock_path_instance = mock_path_cls.return_value
        mock_path_instance.exists.return_value = True
        
        # Mock Audio Tensor
        mock_tensor = MagicMock()
        mock_ingest.load_audio.return_value = mock_tensor
        
        # Mock Model Loading
        mock_model = MagicMock()
        get_speech_timestamps = mock_vad_utils[0]
        # Return timestamps in samples (16k rate). 0-16000 samples = 0-1 second
        get_speech_timestamps.return_value = [{'start': 0, 'end': 16000}]
        
        mock_singleton_instance = mock_singleton_cls.get_instance.return_value
        mock_singleton_instance.load.return_value = (mock_model, mock_vad_utils, MagicMock())
        
        # Execute
        result = voicedetector.run_vad_scan("test.wav", 0.5)
        
        # Verify
        assert isinstance(result, voicedetector.VadResult)
        assert result.has_speech is True
        assert len(result.segments) == 1
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 1.0
        assert len(result.blocks) == 1
        
        # Verify AudioIngest usage
        mock_ingest.load_audio.assert_called_with("test.wav")

    @patch('voicedetector.run_vad_scan')
    @patch('voicedetector.AudioIngest')
    @patch('voicedetector.Path')
    def test_run_batch_vad_scan_skip_existing(self, mock_path_cls, mock_ingest, mock_run_scan):
        """Test skip_existing constraint in batch scan."""
        # Setup Discovery
        mock_media = MagicMock()
        mock_media.path = "audio/test.wav"
        mock_ingest.find_media.return_value = [mock_media]
        
        # Setup Path logic
        mock_path_obj = MagicMock()
        mock_path_obj.stem = "test"
        mock_path_obj.name = "test.wav"
        mock_path_cls.return_value = mock_path_obj
        
        # Simulate existing transcript
        mock_path_obj.parent.glob.return_value = ["test_transcript_model.txt"]
        
        # Execute with skip_existing=True
        report = voicedetector.run_batch_vad_scan("audio/", 0.5, skip_existing=True)
        
        # Verify run_vad_scan was NOT called
        mock_run_scan.assert_not_called()
        assert report.processed_files == 0

    @patch('voicedetector.run_vad_scan')
    @patch('voicedetector.AudioIngest')
    def test_run_batch_vad_scan_aggregation(self, mock_ingest, mock_run_scan):
        """Test aggregation of results in batch scan."""
        # Setup Discovery: 2 files
        m1, m2 = MagicMock(), MagicMock()
        m1.path = "f1.wav"
        m2.path = "f2.wav"
        mock_ingest.find_media.return_value = [m1, m2]
        
        # Setup Results: f1 has speech, f2 does not
        res1 = voicedetector.VadResult("f1.wav", True)
        res2 = voicedetector.VadResult("f2.wav", False)
        mock_run_scan.side_effect = [res1, res2]
        
        # Execute
        report = voicedetector.run_batch_vad_scan("dir", 0.5)
        
        # Verify
        assert report.processed_files == 2
        assert report.files_with_speech == 1
        assert "f1.wav" in report.results
        assert "f2.wav" not in report.results

# -----------------------------------------------------------------------------
# 6. Isolation Tests
# -----------------------------------------------------------------------------

class TestIsolation:
    def test_device_fallback(self, mock_vad_utils):
        """Test fallback to CPU if GPU load fails."""
        mock_model = MagicMock()
        
        # Mock AudioIngest to return a cuda device config
        with patch('voicedetector.AudioIngest.get_device_config', return_value='cuda'):
            # Mock torch.hub.load to fail first time, succeed second time
            with patch('voicedetector.torch.hub.load') as mock_hub_load:
                mock_hub_load.side_effect = [
                    RuntimeError("CUDA error"), # First call fails
                    (mock_model, mock_vad_utils) # Second call succeeds (fallback)
                ]
                
                # Mock torch.device
                with patch('voicedetector.torch.device') as mock_device_cls:
                    # First device is cuda, second is cpu
                    dev_cuda = MagicMock()
                    dev_cuda.type = 'cuda'
                    dev_cpu = MagicMock()
                    dev_cpu.type = 'cpu'
                    mock_device_cls.side_effect = [dev_cuda, dev_cpu]
                    
                    instance = voicedetector._VadModelSingleton()
                    model, utils, device = instance.load()
                    
                    # Verify fallback occurred
                    assert device == dev_cpu
                    assert mock_hub_load.call_count == 2
