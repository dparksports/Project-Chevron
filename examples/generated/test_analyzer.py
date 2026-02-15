import pytest
import inspect
import ast
import os
from unittest.mock import MagicMock, patch, mock_open
import analyzer

# -----------------------------------------------------------------------------
# 1. Structural Tests
# -----------------------------------------------------------------------------

class TestAnalyzerStructure:
    """
    Verifies the SCP contract regarding class structure and method signatures.
    """

    def test_class_exists(self):
        assert hasattr(analyzer, "Analyzer"), "Analyzer class must exist"
        assert inspect.isclass(analyzer.Analyzer), "Analyzer must be a class"

    def test_summarize_signature(self):
        """
        Verify summarize(transcript_path: str, provider: str) -> str
        """
        sig = inspect.signature(analyzer.Analyzer.summarize)
        params = list(sig.parameters.keys())
        
        assert "transcript_path" in params, "summarize must accept transcript_path"
        assert "provider" in params, "summarize must accept provider"
        assert sig.return_annotation == str, "summarize must return str"

    def test_outline_signature(self):
        """
        Verify outline(transcript_path: str, provider: str) -> str
        """
        sig = inspect.signature(analyzer.Analyzer.outline)
        params = list(sig.parameters.keys())
        
        assert "transcript_path" in params, "outline must accept transcript_path"
        assert "provider" in params, "outline must accept provider"
        assert sig.return_annotation == str, "outline must return str"

    def test_init_signature(self):
        """
        Verify __init__ accepts llm_provider
        """
        sig = inspect.signature(analyzer.Analyzer.__init__)
        params = list(sig.parameters.keys())
        assert "llm_provider" in params, "Analyzer must accept llm_provider dependency"


# -----------------------------------------------------------------------------
# 2. Constraint Tests
# -----------------------------------------------------------------------------

class TestAnalyzerConstraints:
    """
    Verifies constraints: read-only access, no forbidden imports, truncation.
    """

    def test_no_forbidden_imports(self):
        """
        Constraint: Must not import Whisper, VAD, or search dependencies.
        """
        source_code = inspect.getsource(analyzer)
        tree = ast.parse(source_code)
        
        forbidden_modules = {'whisper', 'vad', 'search', 'torch', 'numpy'}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in forbidden_modules, f"Forbidden import found: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    assert node.module not in forbidden_modules, f"Forbidden import found: {node.module}"

    def test_read_only_file_access(self):
        """
        Constraint: Must not modify transcript files — read-only analysis.
        """
        mock_llm = MagicMock()
        analyzer_instance = analyzer.Analyzer(mock_llm)
        
        file_path = "test_transcript.txt"
        
        with patch("builtins.open", mock_open(read_data="content")) as mock_file:
            with patch("os.path.exists", return_value=True):
                analyzer_instance.summarize(file_path, "test_provider")
                
                # Verify open was called with 'r' mode
                mock_file.assert_called_with(file_path, 'r', encoding='utf-8')
                
                # Verify no write methods were called
                mock_file.return_value.write.assert_not_called()

    def test_truncation_logic(self):
        """
        Constraint: Must truncate transcripts that exceed model context limits (48k chars).
        """
        mock_llm = MagicMock()
        analyzer_instance = analyzer.Analyzer(mock_llm)
        
        # Create content larger than 48000 chars
        large_content = "a" * 50000
        file_path = "large_transcript.txt"
        
        with patch("builtins.open", mock_open(read_data=large_content)):
            with patch("os.path.exists", return_value=True):
                analyzer_instance.summarize(file_path, "test_provider")
                
                # Capture the prompt passed to LLM
                args, _ = mock_llm.generate.call_args
                prompt_sent = args[0] if args else _['prompt']
                
                # Verify truncation marker exists
                assert "[...Transcript Truncated due to Context Limits...]" in prompt_sent
                # Verify length is roughly limited (prompt overhead + 48000)
                # The prompt adds some text, so we check if the transcript part is truncated
                assert len(prompt_sent) < 50000


# -----------------------------------------------------------------------------
# 3. Behavioral Tests
# -----------------------------------------------------------------------------

class TestAnalyzerBehavior:
    """
    Verifies the functional behavior of summarize and outline methods.
    """

    @pytest.fixture
    def mock_llm(self):
        provider = MagicMock()
        provider.generate.return_value = "Mocked LLM Response"
        return provider

    @pytest.fixture
    def analyzer_instance(self, mock_llm):
        return analyzer.Analyzer(mock_llm)

    def test_summarize_success(self, analyzer_instance, mock_llm):
        """
        Glyph: ☤ The Weaver
        Verify summarize weaves transcript into a summary via LLM.
        """
        transcript_content = "This is a meeting about project X."
        file_path = "meeting.txt"
        
        with patch("builtins.open", mock_open(read_data=transcript_content)):
            with patch("os.path.exists", return_value=True):
                result = analyzer_instance.summarize(file_path, "gpt-4")
                
                assert result == "Mocked LLM Response"
                
                # Verify LLM call
                mock_llm.generate.assert_called_once()
                call_kwargs = mock_llm.generate.call_args[1]
                assert call_kwargs['provider'] == "gpt-4"
                assert transcript_content in call_kwargs['prompt']
                assert "SUMMARY:" in call_kwargs['prompt']

    def test_outline_success(self, analyzer_instance, mock_llm):
        """
        Glyph: ☤ The Weaver
        Verify outline weaves transcript into an outline via LLM.
        """
        transcript_content = "First we discussed A. Then we discussed B."
        file_path = "lecture.txt"
        
        with patch("builtins.open", mock_open(read_data=transcript_content)):
            with patch("os.path.exists", return_value=True):
                result = analyzer_instance.outline(file_path, "claude-3")
                
                assert result == "Mocked LLM Response"
                
                # Verify LLM call
                mock_llm.generate.assert_called_once()
                call_kwargs = mock_llm.generate.call_args[1]
                assert call_kwargs['provider'] == "claude-3"
                assert transcript_content in call_kwargs['prompt']
                assert "OUTLINE:" in call_kwargs['prompt']

    def test_file_not_found(self, analyzer_instance):
        """
        Verify FileNotFoundError is raised if transcript path is invalid.
        """
        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                analyzer_instance.summarize("non_existent.txt", "local")

    def test_llm_delegation_contract(self, analyzer_instance, mock_llm):
        """
        Verify that the Analyzer strictly delegates to the LLMProvider interface.
        """
        file_path = "test.txt"
        with patch("builtins.open", mock_open(read_data="content")):
            with patch("os.path.exists", return_value=True):
                analyzer_instance.summarize(file_path, "provider_x")
                
                # Check that api_key is passed as None (as per implementation)
                call_kwargs = mock_llm.generate.call_args[1]
                assert call_kwargs.get('api_key') is None


# -----------------------------------------------------------------------------
# 4. Isolation Tests
# -----------------------------------------------------------------------------

class TestAnalyzerIsolation:
    """
    Verifies dependency isolation and purity.
    """

    def test_pure_dependency_injection(self):
        """
        Verify that Analyzer does not instantiate its own LLM provider but accepts one.
        """
        # This is implicitly tested by the __init__ signature, but we verify
        # that we can pass a mock and it is used, ensuring no hardcoded dependencies.
        mock_llm = MagicMock()
        ana = analyzer.Analyzer(mock_llm)
        assert ana._llm_provider is mock_llm

    def test_no_global_state_mutation(self):
        """
        Verify that module-level variables are not mutated (if any exist).
        """
        # Inspect module globals excluding built-ins and imports
        module_vars = [
            name for name, val in inspect.getmembers(analyzer)
            if not name.startswith("__") and not inspect.ismodule(val) and not inspect.isclass(val)
        ]
        # The module should ideally be stateless at the global level
        # Protocol definitions (LLMProvider) are types, not state.
        # We just ensure no obvious global state containers like lists or dicts are exposed.
        for var_name in module_vars:
            val = getattr(analyzer, var_name)
            assert not isinstance(val, (list, dict, set)), f"Global mutable state found: {var_name}"