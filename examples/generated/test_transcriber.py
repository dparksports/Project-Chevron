import sys
import os
import pytest
from unittest.mock import MagicMock, patch, mock_open, ANY
import inspect
import ast

# -----------------------------------------------------------------------------
# MOCK EXTERNAL DEPENDENCIES BEFORE IMPORT
# -----------------------------------------------------------------------------
# We mock these modules so that 'import transcriber' does not fail
# and does not try to load real heavy libraries or hardware interfaces.
mock_faster_whisper = MagicMock()
mock_audio_ingest = MagicMock()
mock_voice_detector = MagicMock()
mock_numpy = MagicMock()

sys.modules["faster_whisper"] = mock_faster_whisper
sys.modules["AudioIngest"] = mock_audio_ingest
sys.modules["VoiceDetector"] = mock_voice_detector
sys.modules["numpy"] = mock_numpy

# Now we can import the module under test
import transcriber

# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset the singleton model state in transcriber before each test."""
    transcriber._GLOBAL_MODEL = None
    transcriber._CURRENT_MODEL_NAME = None
    yield
    transcriber._GLOBAL_MODEL = None
    transcriber._CURRENT_MODEL_NAME = None

@pytest.fixture
def mock_whisper_model():
    """Returns a mock for the WhisperModel class constructor."""
    with patch("transcriber.WhisperModel") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        
        # Setup default transcribe return values
        # transcribe returns (segments_generator, info)
        Segment = MagicMock()
        Segment.start = 0.0
        Segment.end = 1.0
        Segment.text = "Test transcript"
        Segment.avg_logprob = -0.5
        
        Info = MagicMock()
        Info.language = "en"
        Info.duration = 1.0
        
        # Generator for segments
        def segment_gen(*args, **kwargs):
            yield Segment

        mock_instance.transcribe.return_value = (segment_gen(), Info)
        
        yield mock_class

# -----------------------------------------------------------------------------
# 1. STRUCTURAL TESTS
# -----------------------------------------------------------------------------

class TestStructural:
    def test_load_model_signature(self):
        sig = inspect.signature(transcriber.load_model)
        assert list(sig.parameters.keys()) == ["model_name", "device", "compute"]
        # Note: The return annotation is the class type, which is mocked, but we check it exists
        assert sig.return_annotation is not inspect.Signature.empty

    def test_transcribe_file_signature(self):
        sig = inspect.signature(transcriber.transcribe_file)
        assert list(sig.parameters.keys()) == ["file_path", "model_name", "beam_size"]
        assert sig.return_annotation == transcriber.Transcript

    def test_batch_transcribe_signature(self):
        sig = inspect.signature(transcriber.batch_transcribe)
        assert list(sig.parameters.keys()) == ["directory", "model_name", "beam_size"]

    def test_transcribe_segment_signature(self):
        sig = inspect.signature(transcriber.transcribe_segment)
        assert list(sig.parameters.keys()) == ["file_path", "start", "end"]
        assert sig.return_annotation == transcriber.Transcript

# -----------------------------------------------------------------------------
# 2. BEHAVIORAL TESTS
# -----------------------------------------------------------------------------

class TestBehavior:
    
    def test_load_model_gpu_success(self, mock_whisper_model):
        """Test that load_model initializes the model with requested config."""
        model = transcriber.load_model("tiny", "cuda", "float16")
        
        mock_whisper_model.assert_called_with("tiny", device="cuda", compute_type="float16")
        assert model == mock_whisper_model.return_value
        assert transcriber._GLOBAL_MODEL == model
        assert transcriber._CURRENT_MODEL_NAME == "tiny"

    def test_load_model_cpu_fallback(self, mock_whisper_model):
        """Test fallback to CPU if GPU loading fails."""
        # Side effect: First call raises Exception, second call succeeds
        # We use side_effect on the constructor mock
        mock_whisper_model.side_effect = [RuntimeError("CUDA error"), MagicMock()]
        
        model = transcriber.load_model("tiny", "cuda", "float16")
        
        # Check calls
        assert mock_whisper_model.call_count == 2
        # First call (GPU)
        mock_whisper_model.assert_any_call("tiny", device="cuda", compute_type="float16")
        # Second call (CPU Fallback)
        mock_whisper_model.assert_any_call("tiny", device="cpu", compute_type="int8")
        
        assert transcriber._GLOBAL_MODEL is not None

    @patch("transcriber.Path")
    @patch("builtins.open", new_callable=mock_open)
    def test_transcribe_file_success(self, mock_file, mock_path, mock_whisper_model):
        """Test full transcription flow including file writing."""
        # Setup Path mocks
        p = MagicMock()
        p.exists.return_value = True
        p.stem = "test_audio"
        p.name = "test_audio.wav"
        p.parent = MagicMock()
        
        # Output path does NOT exist (so we don't skip)
        output_p = MagicMock()
        output_p.exists.return_value = False
        p.parent.__truediv__.return_value = output_p
        
        mock_path.return_value = p
        
        # Setup Model
        transcriber.load_model("tiny", "cpu", "int8")
        
        # Execute
        result = transcriber.transcribe_file("test_audio.wav", "tiny", 5)
        
        # Verify Model Called
        transcriber._GLOBAL_MODEL.transcribe.assert_called_with(
            "test_audio.wav", beam_size=5, vad_filter=True
        )
        
        # Verify File Write
        # Expected filename: test_audio_transcript_tiny.txt
        p.parent.__truediv__.assert_called() 
        mock_file.assert_called_with(output_p, 'w', encoding='utf-8')
        handle = mock_file()
        handle.write.assert_called_with("Test transcript")
        
        # Verify Return
        assert isinstance(result, transcriber.Transcript)
        assert result.text == "Test transcript"

    @patch("transcriber.Path")
    @patch("builtins.open", new_callable=mock_open, read_data="Existing text")
    def test_transcribe_file_skip_existing(self, mock_file, mock_path):
        """Test that existing transcripts cause the function to skip processing."""
        p = MagicMock()
        p.exists.return_value = True
        p.stem = "test_audio"
        
        output_p = MagicMock()
        output_p.exists.return_value = True # Transcript exists
        p.parent.__truediv__.return_value = output_p
        
        mock_path.return_value = p
        
        # Execute
        result = transcriber.transcribe_file("test_audio.wav", "tiny", 5)
        
        # Verify we read the file
        mock_file.assert_called_with(output_p, 'r', encoding='utf-8')
        assert result.text == "Existing text"
        
        # Verify model was NOT loaded/called (Global model is None)
        assert transcriber._GLOBAL_MODEL is None

    @patch("transcriber.AudioIngest")
    @patch("transcriber.transcribe_file")
    def test_batch_transcribe(self, mock_tf, mock_ai):
        """Test batch processing iterates over discovered media."""
        # Setup discovery
        mock_ai.find_media.return_value = ["file1.wav", "file2.mp3"]
        mock_tf.return_value = transcriber.Transcript("path", "model", "text", [])
        
        results = transcriber.batch_transcribe("/data", "tiny", 5)
        
        assert len(results) == 2
        assert mock_tf.call_count == 2
        mock_tf.assert_any_call("file1.wav", "tiny", 5)
        mock_tf.assert_any_call("file2.mp3", "tiny", 5)

    @patch("transcriber.AudioIngest")
    def test_transcribe_segment(self, mock_ai, mock_whisper_model):
        """Test segment transcription logic."""
        # Mock numpy array behavior
        mock_array = MagicMock()
        mock_array.__len__.return_value = 16000
        # Slicing returns another mock
        mock_array.__getitem__.return_value = mock_array
        
        mock_numpy.array.return_value = mock_array
        mock_ai.load_audio.return_value = [0]*16000 
        
        # Execute
        result = transcriber.transcribe_segment("test.wav", 0.0, 0.5)
        
        # Verify slicing logic and transcription call
        assert transcriber._GLOBAL_MODEL is not None
        transcriber._GLOBAL_MODEL.transcribe.assert_called()
        
        # Check return
        assert isinstance(result, transcriber.Transcript)
        assert result.duration == 0.5

# -----------------------------------------------------------------------------
# 3. CONSTRAINT TESTS
# -----------------------------------------------------------------------------

class TestConstraints:
    
    def test_progress_markers(self, capsys, mock_whisper_model):
        """Verify [PROGRESS] markers are emitted to stdout."""
        transcriber.load_model("tiny", "cpu", "int8")
        captured = capsys.readouterr()
        assert "[PROGRESS]" in captured.out
        assert "Loading model" in captured.out

    def test_output_filename_convention(self, mock_whisper_model):
        """Verify the output filename follows the contract."""
        with patch("transcriber.Path") as mock_path:
            p = MagicMock()
            p.exists.return_value = True
            p.stem = "recording"
            p.parent = MagicMock()
            output_p = MagicMock()
            output_p.exists.return_value = False
            p.parent.__truediv__.return_value = output_p
            mock_path.return_value = p
            
            with patch("builtins.open", mock_open()):
                transcriber.transcribe_file("recording.wav", "base", 5)
                
                # Verify the path construction was attempted with correct string
                # p.parent / "recording_transcript_base.txt"
                p.parent.__truediv__.assert_called_with("recording_transcript_base.txt")

    def test_no_forbidden_imports(self):
        """Verify no forbidden modules are imported."""
        source_path = inspect.getfile(transcriber)
        with open(source_path, "r") as f:
            tree = ast.parse(f.read())
            
        forbidden = ["requests", "urllib", "http", "flask", "django"]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split('.')[0] not in forbidden, f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert node.module.split('.')[0] not in forbidden, f"Forbidden import: {node.module}"

# -----------------------------------------------------------------------------
# 4. ISOLATION TESTS
# -----------------------------------------------------------------------------

class TestIsolation:
    
    def test_dependency_isolation(self):
        """Verify only allowed project dependencies are imported."""
        allowed = ["AudioIngest", "VoiceDetector", "faster_whisper"]
        # Standard libs are okay
        
        source_path = inspect.getfile(transcriber)
        with open(source_path, "r") as f:
            tree = ast.parse(f.read())
            
        project_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name.split('.')[0]
                    # Check if it looks like a project module (CamelCase usually)
                    if name in ["AudioIngest", "VoiceDetector", "Analysis", "Search"]:
                        project_imports.append(name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module in ["AudioIngest", "VoiceDetector", "Analysis", "Search"]:
                    project_imports.append(node.module)

        for imp in project_imports:
            assert imp in allowed, f"Unexpected project dependency: {imp}"