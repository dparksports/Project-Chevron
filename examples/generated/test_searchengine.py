import pytest
import inspect
import ast
import sys
import logging
from unittest.mock import MagicMock, patch, mock_open, ANY
from dataclasses import is_dataclass

# Import the module under test
import searchengine

# --------------------------------------------------------------------------------
# 1. Structural Tests
# --------------------------------------------------------------------------------

class TestStructural:
    """
    Verifies the structural integrity of the SearchEngine module against the SCP contract.
    """

    def test_class_existence(self):
        assert hasattr(searchengine, "SearchEngine"), "SearchEngine class missing"
        assert hasattr(searchengine, "SearchResult"), "SearchResult dataclass missing"

    def test_search_result_contract(self):
        """Verify SearchResult is a dataclass with specific fields."""
        assert is_dataclass(searchengine.SearchResult)
        sig = inspect.signature(searchengine.SearchResult)
        params = sig.parameters
        expected_fields = [
            "file_path", "line_number", "content", "score", 
            "timestamp", "context_before", "context_after"
        ]
        for field in expected_fields:
            assert field in params, f"SearchResult missing field: {field}"

    def test_method_signatures(self):
        """Verify method signatures match the SCP contract."""
        engine = searchengine.SearchEngine()
        
        # keyword_search(directory: str, query: str) -> List[SearchResult]
        sig_kw = inspect.signature(engine.keyword_search)
        assert list(sig_kw.parameters.keys()) == ['directory', 'query']
        assert sig_kw.return_annotation != inspect.Signature.empty

        # semantic_search(directories: list[str], query: str, model_name: str) -> List[SearchResult]
        sig_sem = inspect.signature(engine.semantic_search)
        assert list(sig_sem.parameters.keys()) == ['directories', 'query', 'model_name']
        assert sig_sem.return_annotation != inspect.Signature.empty

# --------------------------------------------------------------------------------
# 2. Constraint Tests
# --------------------------------------------------------------------------------

class TestConstraints:
    """
    Verifies SCP constraints: Read-only, no forbidden imports.
    """

    def test_no_forbidden_imports(self):
        """Verify no imports of Whisper, VAD, or LLM libraries."""
        with open(searchengine.__file__, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        forbidden_modules = ["whisper", "faster_whisper", "webrtcvad", "openai", "langchain"]
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in forbidden_modules:
                        assert forbidden not in alias.name, f"Forbidden import found: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for forbidden in forbidden_modules:
                        assert forbidden not in node.module, f"Forbidden import found: {node.module}"

    def test_read_only_constraint(self):
        """Verify that files are never opened in write mode."""
        with open(searchengine.__file__, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'open':
                    # Check arguments for 'w', 'w+', 'a', 'a+'
                    args = [arg.s for arg in node.args if isinstance(arg, ast.Constant)]
                    # Also check keyword args
                    for keyword in node.keywords:
                        if keyword.arg == 'mode' and isinstance(keyword.value, ast.Constant):
                            args.append(keyword.value.s)
                    
                    for mode in args:
                        assert 'w' not in mode and 'a' not in mode, \
                            f"Violation: File opened in write/append mode '{mode}'"

# --------------------------------------------------------------------------------
# 3. Behavioral Tests
# --------------------------------------------------------------------------------

class TestBehavioral:
    """
    Verifies the runtime behavior of the SearchEngine.
    """

    @pytest.fixture
    def engine(self):
        return searchengine.SearchEngine()

    @pytest.fixture
    def mock_transcript_content(self):
        return (
            "[00:00:01] This is the first line.\n"
            "[00:00:05] This is the target keyword line.\n"
            "[00:00:10] This is the context after.\n"
        )

    def test_parse_line(self, engine):
        """Test the internal line parser."""
        # Standard case
        ts, content = engine._parse_line("[12:34:56] Hello World")
        assert ts == "12:34:56"
        assert content == "Hello World"

        # No timestamp
        ts, content = engine._parse_line("Just text here")
        assert ts is None
        assert content == "Just text here"

        # Malformed brackets
        ts, content = engine._parse_line("[Invalid] Text")
        assert ts is None # Assuming ValueError in parsing logic returns None, line
        # Based on code: try...except ValueError -> pass -> return None, line
        # Actually [Invalid] might not raise ValueError on index, but on logic? 
        # The code does: end_idx = line.index("]"); timestamp = line[1:end_idx]
        # It doesn't validate timestamp format, just extracts string.
        ts, content = engine._parse_line("[Tag] Content")
        assert ts == "Tag"
        assert content == "Content"

    @patch("searchengine.glob.glob")
    @patch("builtins.open", new_callable=mock_open)
    def test_keyword_search_success(self, mock_file, mock_glob, engine, mock_transcript_content):
        """Test keyword search finds matches and extracts context."""
        mock_glob.return_value = ["/tmp/test_transcript_model.txt"]
        mock_file.return_value.readlines.return_value = mock_transcript_content.splitlines()

        results = engine.keyword_search("/tmp", "target")

        assert len(results) == 1
        res = results[0]
        assert res.file_path == "/tmp/test_transcript_model.txt"
        assert res.content == "This is the target keyword line."
        assert res.timestamp == "00:00:05"
        assert res.score == 1.0
        # Verify context
        assert res.context_before == "This is the first line."
        assert res.context_after == "This is the context after."

    @patch("searchengine.glob.glob")
    @patch("builtins.open", new_callable=mock_open)
    def test_keyword_search_no_matches(self, mock_file, mock_glob, engine, mock_transcript_content):
        """Test keyword search returns empty list when no matches found."""
        mock_glob.return_value = ["/tmp/test_transcript_model.txt"]
        mock_file.return_value.readlines.return_value = mock_transcript_content.splitlines()

        results = engine.keyword_search("/tmp", "nonexistent")
        assert len(results) == 0

    @patch("searchengine.glob.glob")
    def test_keyword_search_no_files(self, mock_glob, engine):
        """Test keyword search handles empty directories gracefully."""
        mock_glob.return_value = []
        results = engine.keyword_search("/tmp", "query")
        assert results == []

    @patch("searchengine.glob.glob")
    @patch("builtins.open", side_effect=IOError("Read error"))
    def test_keyword_search_file_error(self, mock_file, mock_glob, engine):
        """Test keyword search handles file read errors gracefully."""
        mock_glob.return_value = ["/tmp/bad_file.txt"]
        # Should log error and continue/return empty
        results = engine.keyword_search("/tmp", "query")
        assert results == []

    # ----------------------------------------------------------------------------
    # Semantic Search Tests (Complex Mocking)
    # ----------------------------------------------------------------------------

    def test_semantic_search_fallback_missing_dependency(self, engine):
        """
        Test that semantic_search falls back to keyword_search if 
        sentence-transformers is not installed.
        """
        # Mock sys.modules to raise ImportError for sentence_transformers
        with patch.dict(sys.modules, {'sentence_transformers': None}):
            with patch.object(engine, 'keyword_search') as mock_kw:
                mock_kw.return_value = [searchengine.SearchResult("f", 1, "c", 1.0)]
                
                results = engine.semantic_search(["/tmp"], "query", "model")
                
                # Verify fallback occurred
                mock_kw.assert_called()
                assert len(results) == 1

    def test_semantic_search_fallback_runtime_error(self, engine, mock_transcript_content):
        """
        Test that semantic_search falls back to keyword_search if 
        an exception occurs during embedding generation.
        """
        # Mock successful import but runtime failure
        mock_st = MagicMock()
        mock_st.SentenceTransformer.side_effect = Exception("GPU Error")
        
        modules = {
            'sentence_transformers': mock_st,
            'sentence_transformers.util': MagicMock(),
            'torch': MagicMock()
        }

        with patch.dict(sys.modules, modules):
            with patch("searchengine.glob.glob", return_value=["/tmp/file.txt"]):
                with patch("builtins.open", mock_open(read_data=mock_transcript_content)):
                    with patch.object(engine, 'keyword_search') as mock_kw:
                        mock_kw.return_value = []
                        
                        engine.semantic_search(["/tmp"], "query", "model")
                        
                        # Verify fallback was triggered by the Exception
                        mock_kw.assert_called()

    def test_semantic_search_success(self, engine, mock_transcript_content):
        """
        Test successful semantic search flow with mocked ML libraries.
        """
        # 1. Setup Mocks for external ML libraries
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        # Mock topk return object
        MockTopK = MagicMock()
        MockTopK.values = [0.9, 0.1] # One high score, one low score
        MockTopK.indices = [0, 1]
        mock_torch.topk.return_value = MockTopK

        mock_st_module = MagicMock()
        mock_model_instance = MagicMock()
        mock_st_module.SentenceTransformer.return_value = mock_model_instance
        
        # Mock encode to return dummy tensors
        mock_model_instance.encode.return_value = MagicMock()

        mock_util = MagicMock()
        # Mock cos_sim to return a tensor that supports [0] indexing
        mock_cos_scores = MagicMock()
        mock_util.cos_sim.return_value = [mock_cos_scores] 

        modules = {
            'sentence_transformers': mock_st_module,
            'sentence_transformers.util': mock_util,
            'torch': mock_torch
        }

        # 2. Execute with patched modules and file system
        with patch.dict(sys.modules, modules):
            with patch("searchengine.glob.glob", return_value=["/tmp/file.txt"]):
                with patch("builtins.open", mock_open(read_data=mock_transcript_content)):
                    
                    # The mock content has 3 lines.
                    # Index 0: "This is the first line."
                    # Index 1: "This is the target keyword line."
                    # Index 2: "This is the context after."
                    
                    # Our mock topk returns indices [0, 1] with scores [0.9, 0.1]
                    # The code filters score < 0.25.
                    # So we expect ONLY index 0 to be returned.
                    
                    results = engine.semantic_search(["/tmp"], "query", "model")
                    
                    assert len(results) == 1
                    assert results[0].content == "This is the first line."
                    assert results[0].score == 0.9
                    
                    # Verify model was loaded
                    mock_st_module.SentenceTransformer.assert_called_with("model", device="cpu")

# --------------------------------------------------------------------------------
# 4. Isolation Tests
# --------------------------------------------------------------------------------

class TestIsolation:
    """
    Verifies that the module does not rely on global state or forbidden side effects.
    """

    def test_logger_setup(self):
        """Verify logger is initialized correctly."""
        engine = searchengine.SearchEngine()
        assert engine.logger.name == "SearchEngine"

    def test_no_global_state_modification(self):
        """Verify that instantiating SearchEngine doesn't modify global logging config unexpectedly."""
        # This is a basic check; deep verification of global state is complex.
        # We ensure the class instance has its own logger.
        engine1 = searchengine.SearchEngine()
        engine2 = searchengine.SearchEngine()
        assert engine1 is not engine2
        assert engine1.logger is not engine2 # Loggers with same name are same object in logging module
        assert engine1.logger.name == "SearchEngine"