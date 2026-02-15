import os
import json
import urllib.request
import urllib.error
from typing import List, Optional, Any, Dict, Union
from dataclasses import dataclass
from enum import Enum

# ◬ SCP System Prompt — Module: LLMProvider
# Architecture: TurboScribe — GPU Audio Transcription & Analysis
# Protocol: Spatial Constraint Protocol v1.0

# -----------------------------------------------------------------------------
# Internal Types & Constants
# -----------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """
    Data contract for available models.
    """
    name: str
    provider: str
    is_local: bool

# Global cache for the local model singleton.
# This ensures we never reload the same model into VRAM.
_LOCAL_MODEL_CACHE: Dict[str, Any] = {}

# -----------------------------------------------------------------------------
# Module Implementation
# -----------------------------------------------------------------------------

def load_local_model(model_name: str) -> Any:
    """
    ◬ The Origin
    
    Loads a GGUF model with GPU layers. Acts as the cached singleton origin 
    for local inference.
    
    Contract: Accepts initial data (model path) → Produces a data stream (model instance)
    Constraint: Must appear exactly once per program. Must not be nested.
    
    Args:
        model_name: Path to the .gguf file.
        
    Returns:
        LLMInstance: The loaded llama_cpp.Llama object.
    """
    global _LOCAL_MODEL_CACHE

    # Check cache first
    if model_name in _LOCAL_MODEL_CACHE:
        return _LOCAL_MODEL_CACHE[model_name]

    # Import here to avoid module-level dependency failure if not used
    try:
        from llama_cpp import Llama
    except ImportError:
        raise ImportError("Local provider requires 'llama-cpp-python'. Please install it.")

    if not os.path.exists(model_name):
        raise FileNotFoundError(f"Model file not found: {model_name}")

    # Attempt to load with GPU acceleration
    try:
        # n_gpu_layers=-1 attempts to offload all layers to GPU
        print(f"◬ Loading local model to GPU: {model_name}")
        model = Llama(
            model_path=model_name,
            n_gpu_layers=-1,
            verbose=False,
            n_ctx=4096  # Reasonable default context
        )
    except Exception as e:
        # Fallback to CPU if GPU OOM or error occurs
        print(f"⚠️ GPU Load Failed ({e}). Fallback to CPU.")
        try:
            model = Llama(
                model_path=model_name,
                n_gpu_layers=0,
                verbose=False,
                n_ctx=4096
            )
        except Exception as cpu_e:
            raise RuntimeError(f"Failed to load model on both GPU and CPU: {cpu_e}")

    # Cache the instance
    _LOCAL_MODEL_CACHE[model_name] = model
    return model


def generate(prompt: str, provider: str, api_key: Optional[str]) -> str:
    """
    ☤ The Weaver
    
    Weaves prompt + model into response — braids local/cloud into unified output.
    
    Contract: Accepts list of values → Produces single merged value
    Constraint: Must preserve all input data. Nothing may be lost in the weaving.
    
    Args:
        prompt: The text prompt to process.
        provider: 'local', 'openai', 'anthropic', or 'gemini'. 
                  Can also specify model like 'openai:gpt-4'.
        api_key: API key for cloud providers. Ignored for local.
        
    Returns:
        str: The generated text response.
    """
    
    # Parse provider and specific model if provided (e.g., "openai:gpt-4o")
    if ":" in provider:
        service, specific_model = provider.split(":", 1)
    else:
        service, specific_model = provider, None

    service = service.lower()

    if service == "local":
        return _generate_local(prompt)
    elif service == "openai":
        return _generate_openai(prompt, api_key, specific_model or "gpt-4o")
    elif service == "anthropic" or service == "claude":
        return _generate_anthropic(prompt, api_key, specific_model or "claude-3-5-sonnet-20240620")
    elif service == "gemini" or service == "google":
        return _generate_gemini(prompt, api_key, specific_model or "gemini-1.5-flash")
    else:
        raise ValueError(f"Unknown provider: {provider}")


def list_models() -> List[ModelInfo]:
    """
    𓃀 Witnesses available models without modification.
    
    Returns:
        List[ModelInfo]: List of supported/detected models.
    """
    models = []
    
    # Witness Local Models (scan current directory for .gguf)
    try:
        for file in os.listdir("."):
            if file.endswith(".gguf"):
                models.append(ModelInfo(name=file, provider="local", is_local=True))
    except OSError:
        pass # Ignore if directory access fails

    # Witness Cloud Capabilities (Static list of supported interfaces)
    models.append(ModelInfo(name="gpt-4o", provider="openai", is_local=False))
    models.append(ModelInfo(name="claude-3-5-sonnet", provider="anthropic", is_local=False))
    models.append(ModelInfo(name="gemini-1.5-flash", provider="gemini", is_local=False))
    
    return models


# -----------------------------------------------------------------------------
# Private Helper Methods (Implementation Details)
# -----------------------------------------------------------------------------

def _generate_local(prompt: str) -> str:
    """Internal handler for local generation using cached Llama instance."""
    if not _LOCAL_MODEL_CACHE:
        raise RuntimeError("No local model loaded. Call load_local_model() first (◬ Origin).")
    
    # Retrieve the singleton (last loaded model or specific logic could be added)
    # For this implementation, we take the most recently added or only one.
    model = list(_LOCAL_MODEL_CACHE.values())[-1]
    
    output = model.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2048,
        temperature=0.7
    )
    
    return output['choices'][0]['message']['content']


def _generate_openai(prompt: str, api_key: Optional[str], model: str) -> str:
    """Internal handler for OpenAI API via urllib (No external deps)."""
    if not api_key:
        raise ValueError("API key required for OpenAI provider.")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7
    }
    
    return _http_post(url, headers, data, extract_key=["choices", 0, "message", "content"])


def _generate_anthropic(prompt: str, api_key: Optional[str], model: str) -> str:
    """Internal handler for Anthropic API via urllib."""
    if not api_key:
        raise ValueError("API key required for Anthropic provider.")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    data = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    return _http_post(url, headers, data, extract_key=["content", 0, "text"])


def _generate_gemini(prompt: str, api_key: Optional[str], model: str) -> str:
    """Internal handler for Google Gemini API via urllib."""
    if not api_key:
        raise ValueError("API key required for Gemini provider.")

    # Gemini URL structure requires API key in query param
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    return _http_post(url, headers, data, extract_key=["candidates", 0, "content", "parts", 0, "text"])


def _http_post(url: str, headers: Dict[str, str], data: Dict[str, Any], extract_key: List[Union[str, int]]) -> str:
    """Generic HTTP POST helper to avoid code duplication and external dependencies."""
    try:
        req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers=headers, method="POST")
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            
            # Traverse the response using the extraction path
            value = result
            for key in extract_key:
                if isinstance(value, list) and isinstance(key, int):
                    value = value[key]
                elif isinstance(value, dict) and isinstance(key, str):
                    value = value.get(key)
                else:
                    raise ValueError(f"Unexpected response structure at key {key}")
                
                if value is None:
                    raise ValueError(f"Key {key} not found in response")
            
            return str(value)
            
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        raise RuntimeError(f"Provider API Error {e.code}: {error_body}")
    except Exception as e:
        raise RuntimeError(f"Network/Parsing Error: {str(e)}")
