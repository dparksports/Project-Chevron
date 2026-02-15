import pytest
import unittest.mock as mock
import sys
import os
import inspect
import ast
import json
import urllib.error
from typing import List, Optional

import llmprovider

class TestLLMProviderStructure:
    """
    Verifies the structural integrity of the LLMProvider module
    against the SCP contract.
    """

    def test_method_signatures(self):
        """Verify public API signatures match the contract."""
        # load_local_model(model_name: str) -> Any
        sig_load = inspect.signature(llmprovider.load_local_model)
        assert list(sig_load.parameters.keys()) == ['model_name']
        
        # generate(prompt: str, provider: str, api_key: Optional[str]) -> str
        sig_gen = inspect.signature(llmprovider.generate)
        assert list(sig_gen.parameters.keys()) == ['prompt', 'provider', 'api_key']
        assert sig_gen.return_annotation == str

        # list_models() -> List[ModelInfo]
        sig_list = inspect.signature(llmprovider.list_models)
        assert list(sig_list.parameters.keys()) == []

    def test_model_info_dataclass(self):
        """Verify ModelInfo dataclass structure."""
        info = llmprovider.ModelInfo(name="test", provider="local", is_local=True)
        assert info.name == "test"
        assert info.provider == "local"
        assert info.is_local is True

class TestLLMProviderConstraints:
    """
    Verifies constraints defined in the SCP contract.
    """

    def setup_method(self):
        # Clear cache before each test to ensure isolation
        llmprovider._LOCAL_MODEL_CACHE.clear()

    def teardown_method(self):
        llmprovider._LOCAL_MODEL_CACHE.clear()

    def test_import_isolation(self):
        """Verify no forbidden project modules are imported."""
        source_file = inspect.getsourcefile(llmprovider)
        with open(source_file, "r") as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.name.split('.')[0]
                    # Allowed: standard lib, typing, dataclasses, enum, llama_cpp (conditional)
                    # Forbidden: anything else specific to the project
                    forbidden = ['turboscribe', 'meeting_analysis', 'core', 'utils']
                    if name in forbidden:
                        pytest.fail(f"Forbidden import detected: {name}")

    @mock.patch('os.path.exists', return_value=True)
    def test_local_model_caching(self, mock_exists):
        """Constraint: Must cache loaded local models — never reload same model."""
        
        # Mock llama_cpp module structure
        mock_llama_module = mock.MagicMock()
        mock_llama_class = mock.MagicMock()
        mock_llama_module.Llama = mock_llama_class
        
        # Inject mock module
        with mock.patch.dict(sys.modules, {'llama_cpp': mock_llama_module}):
            # First load
            model1 = llmprovider.load_local_model("test_model.gguf")
            
            # Second load
            model2 = llmprovider.load_local_model("test_model.gguf")
            
            # Verify constructor called only once
            assert mock_llama_class.call_count == 1
            # Verify same instance returned
            assert model1 is model2

    @mock.patch('os.path.exists', return_value=True)
    def test_gpu_oom_fallback(self, mock_exists):
        """Constraint: Must handle GPU OOM gracefully with CPU fallback."""
        
        mock_llama_module = mock.MagicMock()
        mock_llama_class = mock.MagicMock()
        mock_llama_module.Llama = mock_llama_class
        
        # First call raises Exception (simulating GPU failure), second call succeeds
        mock_llama_class.side_effect = [RuntimeError("CUDA OOM"), mock.MagicMock()]
        
        with mock.patch.dict(sys.modules, {'llama_cpp': mock_llama_module}):
            llmprovider.load_local_model("test_model.gguf")
            
            # Verify two calls were made
            assert mock_llama_class.call_count == 2
            
            # Check arguments of calls
            # First call: n_gpu_layers=-1 (GPU attempt)
            call_args_1 = mock_llama_class.call_args_list[0]
            assert call_args_1.kwargs.get('n_gpu_layers') == -1
            
            # Second call: n_gpu_layers=0 (CPU fallback)
            call_args_2 = mock_llama_class.call_args_list[1]
            assert call_args_2.kwargs.get('n_gpu_layers') == 0

class TestLLMProviderBehavior:
    """
    Verifies behavioral correctness of methods.
    """

    def setup_method(self):
        llmprovider._LOCAL_MODEL_CACHE.clear()

    @mock.patch('os.path.exists', return_value=False)
    def test_load_local_model_file_not_found(self, mock_exists):
        """Verify FileNotFoundError if model path invalid."""
        # Mock llama_cpp to ensure import check passes
        with mock.patch.dict(sys.modules, {'llama_cpp': mock.MagicMock()}):
            with pytest.raises(FileNotFoundError):
                llmprovider.load_local_model("missing.gguf")

    def test_load_local_model_import_error(self):
        """Verify ImportError if llama-cpp-python not installed."""
        # Simulate missing module by setting it to None in sys.modules
        with mock.patch.dict(sys.modules, {'llama_cpp': None}):
            with pytest.raises(ImportError) as excinfo:
                llmprovider.load_local_model("test.gguf")
            assert "llama-cpp-python" in str(excinfo.value)

    def test_generate_local(self):
        """Verify local generation uses cached model."""
        mock_model = mock.MagicMock()
        mock_model.create_chat_completion.return_value = {
            'choices': [{'message': {'content': 'Local response'}}]
        }
        
        # Manually inject into cache to bypass loading logic
        llmprovider._LOCAL_MODEL_CACHE['test.gguf'] = mock_model
        
        response = llmprovider.generate("Hello", "local", None)
        assert response == "Local response"
        mock_model.create_chat_completion.assert_called_once()

    def test_generate_local_no_model(self):
        """Verify error if local generation requested without loading model."""
        with pytest.raises(RuntimeError):
            llmprovider.generate("Hello", "local", None)

    @mock.patch('urllib.request.urlopen')
    def test_generate_openai(self, mock_urlopen):
        """Verify OpenAI generation logic."""
        # Setup mock response
        mock_response = mock.MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "OpenAI response"}}]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        response = llmprovider.generate("Hello", "openai", "sk-test")
        
        assert response == "OpenAI response"
        
        # Verify Request
        args, _ = mock_urlopen.call_args
        req = args[0]
        assert req.full_url == "https://api.openai.com/v1/chat/completions"
        assert req.headers['Authorization'] == "Bearer sk-test"
        
        # Verify Data
        data = json.loads(req.data)
        assert data['model'] == "gpt-4o" # Default
        assert data['messages'][0]['content'] == "Hello"

    @mock.patch('urllib.request.urlopen')
    def test_generate_anthropic(self, mock_urlopen):
        """Verify Anthropic generation logic."""
        mock_response = mock.MagicMock()
        mock_response.read.return_value = json.dumps({
            "content": [{"text": "Claude response"}]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        response = llmprovider.generate("Hello", "anthropic", "sk-ant")
        
        assert response == "Claude response"
        
        args, _ = mock_urlopen.call_args
        req = args[0]
        assert req.full_url == "https://api.anthropic.com/v1/messages"
        assert req.headers['x-api-key'] == "sk-ant"

    @mock.patch('urllib.request.urlopen')
    def test_generate_gemini(self, mock_urlopen):
        """Verify Gemini generation logic."""
        mock_response = mock.MagicMock()
        mock_response.read.return_value = json.dumps({
            "candidates": [{"content": {"parts": [{"text": "Gemini response"}]}}]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        response = llmprovider.generate("Hello", "gemini", "AIzaSy")
        
        assert response == "Gemini response"
        
        args, _ = mock_urlopen.call_args
        req = args[0]
        # Check URL contains key
        assert "key=AIzaSy" in req.full_url
        assert "generativelanguage.googleapis.com" in req.full_url

    @mock.patch('urllib.request.urlopen')
    def test_generate_specific_model(self, mock_urlopen):
        """Verify provider:model syntax overrides default model."""
        mock_response = mock.MagicMock()
        mock_response.read.return_value = json.dumps({
            "choices": [{"message": {"content": "GPT-3.5 response"}}]
        }).encode('utf-8')
        mock_urlopen.return_value.__enter__.return_value = mock_response

        llmprovider.generate("test", "openai:gpt-3.5-turbo", "key")
        
        args, _ = mock_urlopen.call_args
        data = json.loads(args[0].data)
        assert data['model'] == "gpt-3.5-turbo"

    def test_generate_missing_api_key(self):
        """Verify ValueError when API key is missing for cloud providers."""
        with pytest.raises(ValueError):
            llmprovider.generate("test", "openai", None)
        with pytest.raises(ValueError):
            llmprovider.generate("test", "anthropic", "")

    def test_generate_unknown_provider(self):
        """Verify ValueError for unknown provider."""
        with pytest.raises(ValueError):
            llmprovider.generate("test", "unknown_provider", "key")

    @mock.patch('os.listdir')
    def test_list_models(self, mock_listdir):
        """Verify list_models aggregates local and cloud models."""
        mock_listdir.return_value = ["model1.gguf", "readme.txt", "model2.gguf"]
        
        models = llmprovider.list_models()
        
        # Check local models
        local_names = [m.name for m in models if m.is_local]
        assert "model1.gguf" in local_names
        assert "model2.gguf" in local_names
        assert "readme.txt" not in local_names
        
        # Check cloud models exist
        cloud_providers = [m.provider for m in models if not m.is_local]
        assert "openai" in cloud_providers
        assert "anthropic" in cloud_providers
        assert "gemini" in cloud_providers

    @mock.patch('urllib.request.urlopen')
    def test_http_error_handling(self, mock_urlopen):
        """Verify HTTP errors are caught and raised as RuntimeError."""
        # Simulate HTTP Error
        err = urllib.error.HTTPError(
            url="http://test", code=401, msg="Unauthorized", hdrs={}, fp=mock.Mock()
        )
        err.read = mock.Mock(return_value=b"Invalid API Key")
        mock_urlopen.side_effect = err
        
        with pytest.raises(RuntimeError) as excinfo:
            llmprovider.generate("test", "openai", "bad_key")
        
        assert "401" in str(excinfo.value)
        assert "Invalid API Key" in str(excinfo.value)