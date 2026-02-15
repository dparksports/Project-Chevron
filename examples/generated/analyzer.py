import os
from typing import Protocol, Optional, List, Any

# ◬ SCP System Prompt — Module: Analyzer
# Architecture: TurboScribe — GPU Audio Transcription & Analysis
# Protocol: Spatial Constraint Protocol v1.0

# -----------------------------------------------------------------------------
# 🔗 Visible Dependency Interfaces
# -----------------------------------------------------------------------------

class LLMProvider(Protocol):
    """
    Unified interface to local (llama.cpp) and cloud (Gemini, OpenAI, Claude) LLMs.
    """
    def load_local_model(self, model_name: str) -> Any:
        ...

    def generate(self, prompt: str, provider: str, api_key: Optional[str]) -> str:
        ...

    def list_models(self) -> List[Any]:
        ...

# -----------------------------------------------------------------------------
# Module Implementation
# -----------------------------------------------------------------------------

class Analyzer:
    """
    Summarizes or outlines transcripts using LLM providers.
    
    Constraints:
    - ⚠️ Must not modify transcript files — read-only analysis
    - ⚠️ Must delegate all LLM calls to LLMProvider
    - ⚠️ Must truncate transcripts that exceed model context limits
    """

    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize the Analyzer with a handle to the LLMProvider.
        """
        self._llm_provider = llm_provider
        # Heuristic limit for context window (approx 12k tokens ~ 48k chars)
        # to ensure safety across various providers without specific metadata.
        self._max_chars = 48000 

    def _read_and_truncate(self, file_path: str) -> str:
        """
        Reads the transcript file and truncates it to fit context limits.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Transcript file not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if len(content) > self._max_chars:
            # Truncate to limit, keeping the beginning
            return content[:self._max_chars] + "\n\n[...Transcript Truncated due to Context Limits...]"
        
        return content

    # ☤ The Weaver
    def summarize(self, transcript_path: str, provider: str) -> str:
        """
        Weaves transcript + LLM into a concise summary.
        
        Glyph: ☤ The Weaver
        Contract: Accepts list of values → Produces single merged value
        Constraint: Must preserve all input data (read-only).
        """
        transcript_text = self._read_and_truncate(transcript_path)
        
        prompt = (
            "You are an expert analyst. Please provide a concise summary of the "
            "following transcript. Capture the main topics, key decisions, and "
            "action items if present.\n\n"
            "TRANSCRIPT:\n"
            f"{transcript_text}\n\n"
            "SUMMARY:"
        )

        # Delegate to LLMProvider
        # Note: API key handling is assumed to be managed by the provider or environment,
        # passing None here as per interface signature allowing it.
        summary = self._llm_provider.generate(prompt=prompt, provider=provider, api_key=None)
        
        return summary

    # ☤ The Weaver
    def outline(self, transcript_path: str, provider: str) -> str:
        """
        Weaves transcript + LLM into a structured outline.
        
        Glyph: ☤ The Weaver
        Contract: Accepts list of values → Produces single merged value
        Constraint: Must preserve all input data (read-only).
        """
        transcript_text = self._read_and_truncate(transcript_path)

        prompt = (
            "You are an expert analyst. Please generate a structured outline of the "
            "following transcript. Use hierarchical bullet points to represent the "
            "flow of conversation and distinct sections.\n\n"
            "TRANSCRIPT:\n"
            f"{transcript_text}\n\n"
            "OUTLINE:"
        )

        # Delegate to LLMProvider
        outline = self._llm_provider.generate(prompt=prompt, provider=provider, api_key=None)

        return outline
