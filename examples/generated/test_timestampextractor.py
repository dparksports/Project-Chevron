import pytest
import sys
import os
import ast
import inspect
from unittest.mock import MagicMock, patch, PropertyMock, call
from pathlib import Path
from dataclasses import dataclass

# ==============================================================================
# 🛠️ Fixtures & Mocks
# ==============================================================================

@pytest.fixture(autouse=True)
def mock_dependencies():
    """
    Mock all external dependencies to ensure isolation.
    This runs before importing the module under test.
    """
    with patch.dict(sys.modules, {
        "cv2": MagicMock(),
        "PIL": MagicMock(),
        "PIL.Image": MagicMock(),
        "torch": MagicMock(),
        "transformers": MagicMock(),
        "AudioIngest": MagicMock(),  # Project dependency
    }):
        # Setup specific mock behaviors
        sys.modules["cv2"].CAP_PROP_FRAME_COUNT = 7
        sys.modules["cv2"].CAP_PROP_POS_FRAMES = 1
        
        # Mock torch attributes
        sys.modules["torch"].cuda.is_available.return_value = False
        sys.modules["torch"].float32 = "float32"
        sys.modules["torch"].bfloat16 = "bfloat16"
        
        yield

# Import module after mocking
import timestampextractor

# ==============================================================================
# 🏗️ Structural Tests
# ==============================================================================

class TestStructure:
    """Verifies the module structure and public API contract."""

    def test_class_existence(self):
        assert hasattr(timestampextractor, "TimestampExtractor")
        assert hasattr(timestampextractor, "TimestampResult")
        assert hasattr(timestampextractor, "RenameResult")

    def test_extract_timestamps_signature(self):
        sig = inspect.signature(timestampextractor.TimestampExtractor.extract_timestamps)
        params = list(sig.parameters.keys())
        assert params == ["self", "video_path", "num_frames"]
        assert sig.return_annotation == timestampextractor.TimestampResult

    def test_batch_rename_signature(self):
        sig = inspect.signature(timestampextractor.TimestampExtractor.batch_rename)
        params = list(sig.parameters.keys())
        assert params == ["self", "folder_path", "crop_ratio"]
        # Note: Python < 3.9 might return string representation for list[RenameResult]
        # We just check it exists; strict type check is done by static analysis usually.

# ==============================================================================
# ⚖️ Constraint Tests
# ==============================================================================

class TestConstraints:
    """Verifies SCP constraints and restrictions."""

    def test_forbidden_imports(self):
        """Verify no forbidden modules are imported."""
        source_file = timestampextractor.__file__
        with open(source_file, "r") as f:
            tree = ast.parse(f.read())

        forbidden = ["whisper", "vad", "llm", "openai", "anthropic"]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for bad in forbidden:
                        assert bad not in alias.name.lower(), f"Forbidden import found: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for bad in forbidden:
                        assert bad not in node.module.lower(), f"Forbidden import found: {node.module}"

    def test_singleton_lazy_loading(self):
        """Verify VLM model is loaded lazily and cached."""
        # Reset singleton
        timestampextractor._QwenVLSingleton._instance = None
        
        with patch("timestampextractor.Qwen2_5_VLForConditionalGeneration") as mock_model_cls, \
             patch("timestampextractor.AutoProcessor") as mock_proc_cls:
            
            instance1 = timestampextractor._QwenVLSingleton.get_instance()
            
            # Should not have loaded model yet
            assert instance1._model is None
            
            # Trigger load
            instance1.get_model()
            assert mock_model_cls.from_pretrained.called
            assert mock_proc_cls.from_pretrained.called
            
            # Get instance again
            instance2 = timestampextractor._QwenVLSingleton.get_instance()
            assert instance1 is instance2
            
            # Trigger load again - should not call from_pretrained again
            mock_model_cls.from_pretrained.reset_mock()
            instance2.get_model()
            assert not mock_model_cls.from_pretrained.called

    def test_cleanup_temp_files(self):
        """Verify temporary directories are removed after processing."""
        extractor = timestampextractor.TimestampExtractor()
        
        with patch("os.path.exists", return_value=True), \
             patch("timestampextractor.TimestampExtractor._extract_frames", return_value=[]), \
             patch("shutil.rmtree") as mock_rmtree, \
             patch("pathlib.Path.mkdir"), \
             patch("pathlib.Path.exists", return_value=True):
            
            extractor.extract_timestamps("dummy.mp4")
            
            # Verify rmtree was called on the temp directory
            assert mock_rmtree.called
            args, _ = mock_rmtree.call_args
            assert "temp_frames_dummy.mp4" in str(args[0])

# ==============================================================================
# 🧪 Behavioral Tests
# ==============================================================================

class TestBehavior:
    """Verifies the functional behavior of the module."""

    @patch("timestampextractor._QwenVLSingleton")
    def test_extract_timestamps_consensus(self, mock_singleton_cls):
        """
        Verify majority voting logic.
        Scenario: 5 frames. 3 return '2023-01-01 12:00:00', 2 return '2020-01-01...'.
        Result should be the majority.
        """
        extractor = timestampextractor.TimestampExtractor()
        
        # Mock Model/Processor
        mock_model = MagicMock()
        mock_processor = MagicMock()
        mock_singleton_cls.get_instance.return_value.get_model.return_value = (mock_model, mock_processor, "cpu")
        
        # Mock _extract_frames to return 5 dummy paths
        with patch.object(extractor, '_extract_frames', return_value=[f"f{i}.jpg" for i in range(5)]), \
             patch.object(extractor, '_process_frame') as mock_process:
            
            # Setup voting behavior
            mock_process.side_effect = [
                "2023-01-01 12:00:00",
                "2020-01-01 00:00:00",
                "2023-01-01 12:00:00",
                "2020-01-01 00:00:00",
                "2023-01-01 12:00:00"
            ]
            
            with patch("os.path.exists", return_value=True), \
                 patch("shutil.rmtree"):
                
                result = extractor.extract_timestamps("video.mp4", num_frames=5)
                
                assert result.detected_timestamp == "2023-01-01 12:00:00"
                assert result.consensus_count == 3
                assert result.confidence == 3/5
                assert result.total_frames == 5

    def test_extract_timestamps_no_video(self):
        """Verify behavior when video file does not exist."""
        extractor = timestampextractor.TimestampExtractor()
        with patch("os.path.exists", return_value=False):
            result = extractor.extract_timestamps("ghost.mp4")
            assert result.detected_timestamp is None
            assert result.total_frames == 5  # Requested frames

    @patch("timestampextractor.AudioIngest")
    def test_batch_rename_skip_existing(self, mock_ingest):
        """Verify files already matching the pattern are skipped."""
        extractor = timestampextractor.TimestampExtractor()
        
        # Mock discovery
        mock_media = MagicMock()
        mock_media.path = "/path/to/2023-10-27_12-00-00_video.mp4"
        mock_ingest.find_media.return_value = [mock_media]
        
        results = extractor.batch_rename("/path/to")
        
        assert len(results) == 1
        assert results[0].status == "skipped"
        assert results[0].original_path == mock_media.path

    @patch("timestampextractor.AudioIngest")
    def test_batch_rename_success(self, mock_ingest):
        """Verify successful rename flow."""
        extractor = timestampextractor.TimestampExtractor()
        
        # Mock discovery
        mock_media = MagicMock()
        mock_media.path = "/path/to/video.mp4"
        mock_ingest.find_media.return_value = [mock_media]
        
        # Mock extraction result
        ts_result = timestampextractor.TimestampResult(
            original_path="/path/to/video.mp4",
            detected_timestamp="2023-10-27 12:00:00",
            confidence=1.0,
            consensus_count=5,
            total_frames=5
        )
        
        with patch.object(extractor, 'extract_timestamps', return_value=ts_result), \
             patch("os.rename") as mock_rename:
            
            results = extractor.batch_rename("/path/to")
            
            assert len(results) == 1
            assert results[0].status == "renamed"
            assert results[0].timestamp == "2023-10-27_12-00-00"
            
            # Verify os.rename called correctly
            expected_new_path = "/path/to/2023-10-27_12-00-00_video.mp4"
            mock_rename.assert_called_once_with(mock_media.path, expected_new_path)

    @patch("timestampextractor.AudioIngest")
    def test_batch_rename_low_confidence(self, mock_ingest):
        """Verify rename is aborted if confidence is low."""
        extractor = timestampextractor.TimestampExtractor()
        
        mock_media = MagicMock()
        mock_media.path = "/path/to/video.mp4"
        mock_ingest.find_media.return_value = [mock_media]
        
        # Confidence < 0.5
        ts_result = timestampextractor.TimestampResult(
            original_path="/path/to/video.mp4",
            detected_timestamp="2023-10-27 12:00:00",
            confidence=0.4, 
            consensus_count=2,
            total_frames=5
        )
        
        with patch.object(extractor, 'extract_timestamps', return_value=ts_result), \
             patch("os.rename") as mock_rename:
            
            results = extractor.batch_rename("/path/to")
            
            assert len(results) == 1
            assert results[0].status == "failed"
            mock_rename.assert_not_called()

# ==============================================================================
# 🔒 Isolation Tests
# ==============================================================================

class TestIsolation:
    """Verifies internal isolation logic."""

    def test_extract_frames_cv2_interaction(self):
        """Verify _extract_frames interacts correctly with cv2."""
        extractor = timestampextractor.TimestampExtractor()
        
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.return_value = 100 # 100 frames total
        mock_cap.read.return_value = (True, "dummy_frame_data")
        
        with patch("cv2.VideoCapture", return_value=mock_cap), \
             patch("cv2.imwrite") as mock_imwrite:
            
            output_dir = Path("temp")
            frames = extractor._extract_frames("video.mp4", output_dir, num_frames=2)
            
            assert len(frames) == 2
            assert mock_cap.set.call_count == 2
            assert mock_imwrite.call_count == 2
            mock_cap.release.assert_called_once()

    def test_process_frame_exception_handling(self):
        """Verify _process_frame handles exceptions gracefully."""
        extractor = timestampextractor.TimestampExtractor()
        
        # Mock Image.open to raise exception
        with patch("PIL.Image.open", side_effect=Exception("Corrupt image")):
            result = extractor._process_frame(MagicMock(), MagicMock(), "cpu", "bad.jpg")
            assert result is None

    def test_normalize_timestamp(self):
        """Verify timestamp normalization logic."""
        extractor = timestampextractor.TimestampExtractor()
        
        raw = "2023/01/01 12:30:45"
        normalized = extractor._normalize_timestamp(raw)
        assert normalized == "2023-01-01_12-30-45"
        
        raw_colons = "2023-01-01 12:30:45"
        normalized = extractor._normalize_timestamp(raw_colons)
        assert normalized == "2023-01-01_12-30-45"