import os
import json
import glob
from typing import Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# ◬ SCP System Prompt — Module: MeetingDetector
# Protocol: Spatial Constraint Protocol v1.0
# ⚠️ CRITICAL: This module must NOT import Whisper, sentence-transformers, or other project modules.
# ⚠️ CRITICAL: Inter-module communication via declared interfaces only.

# ---------------------------------------------------------------------------
# ◬ VISIBLE DEPENDENCY INTERFACES
# ---------------------------------------------------------------------------
# We assume these are available in the runtime environment as per the SCP contract.
# We use a try/except block to allow static analysis without crashing if dependencies are missing.

try:
    from llm_provider import LLMProvider
except ImportError:
    # Placeholder for static analysis / linting context
    class LLMProvider:
        @staticmethod
        def generate(prompt: str, provider: str, api_key: str | None) -> str:
            raise NotImplementedError("LLMProvider interface not linked.")

# ---------------------------------------------------------------------------
# ◬ DATA STRUCTURES
# ---------------------------------------------------------------------------

class ClassificationResult(str, Enum):
    MEETING = "MEETING"
    HALLUCINATION = "HALLUCINATION"
    UNKNOWN = "UNKNOWN"

@dataclass
class Classification:
    is_meeting: bool
    confidence: float
    reasoning: str
    category: ClassificationResult

@dataclass
class DetectionReport:
    directory: str
    total_files: int
    processed_files: int
    meetings_detected: int
    hallucinations_detected: int
    results: Dict[str, Classification]

# ---------------------------------------------------------------------------
# ◬ MODULE IMPLEMENTATION
# ---------------------------------------------------------------------------

class MeetingDetector:
    """
    ◬ Module: MeetingDetector
    Purpose: Uses LLM to classify transcripts as real meetings vs hallucinated content.
    Governed by SCP v1.0.
    """

    # Ө The Filter
    def detect_meetings(self, directory: str, provider: str, skip_checked: bool = True) -> DetectionReport:
        """
        Glyph: The Filter
        Filters transcripts — real meetings pass, hallucinations rejected.
        
        Contract: Accepts (predicate, data) → Produces filtered data
        Constraint: Must never modify data that passes through. Reject, don't transform.
        
        Args:
            directory: Path to scan for transcripts.
            provider: LLM provider string (e.g., 'gpt-4', 'llama3').
            skip_checked: If True, skips files that already have a classification sidecar.
        """
        # Universal contract: filename_transcript_modelname.txt
        search_pattern = os.path.join(directory, "*_transcript_*.txt")
        transcript_files = glob.glob(search_pattern)
        
        results: Dict[str, Classification] = {}
        meetings_count = 0
        hallucinations_count = 0
        processed_count = 0
        
        print(f"[DETECT_PROGRESS] Starting scan in {directory}. Found {len(transcript_files)} transcripts.")

        for i, file_path in enumerate(transcript_files):
            filename = os.path.basename(file_path)
            
            # Check for existing classification if skip_checked is True
            if skip_checked:
                existing = self._load_classification(file_path)
                if existing:
                    results[filename] = existing
                    if existing.is_meeting:
                        meetings_count += 1
                    else:
                        hallucinations_count += 1
                    print(f"[DETECT_PROGRESS] Skipped {filename} (Cached: {existing.category.value})")
                    continue

            # Read transcript content
            content = self._read_file(file_path)
            if not content:
                # Treat empty/unreadable as UNKNOWN
                cls = Classification(False, 0.0, "File empty or unreadable", ClassificationResult.UNKNOWN)
                results[filename] = cls
                continue

            # Classify
            print(f"[DETECT_PROGRESS] Classifying {filename} with {provider}...")
            classification = self.classify_transcript(content, provider)
            
            # Persist result (Sidecar) - Does not modify original transcript
            self._save_classification(file_path, classification)
            
            results[filename] = classification
            if classification.is_meeting:
                meetings_count += 1
            else:
                hallucinations_count += 1
            
            processed_count += 1
            
            # Progress update
            print(f"[DETECT_PROGRESS] {i+1}/{len(transcript_files)}: {filename} -> {classification.category.value}")

        return DetectionReport(
            directory=directory,
            total_files=len(transcript_files),
            processed_files=processed_count,
            meetings_detected=meetings_count,
            hallucinations_detected=hallucinations_count,
            results=results
        )

    # Ө The Filter
    def classify_transcript(self, text: str, provider: str) -> Classification:
        """
        Glyph: The Filter
        Gates a single transcript — pass/reject with confidence score.
        
        Contract: Accepts (predicate, data) → Produces filtered data
        Constraint: Must never modify data that passes through. Reject, don't transform.
        """
        # Deterministic Prompt Construction
        # We limit text to first 4000 chars to fit context and save cost, 
        # usually enough to detect hallucination loops.
        snippet = text[:4000]
        
        prompt = (
            "You are a quality control system for audio transcripts. "
            "Determine if the text below is a valid human conversation/meeting or "
            "hallucinated garbage (e.g. repetitive loops, infinite 'Thank you', random characters, silence).\n\n"
            "TEXT START:\n"
            f"{snippet}\n"
            "TEXT END\n\n"
            "Respond with valid JSON only:\n"
            "{\n"
            '  "is_meeting": bool,\n'
            '  "confidence": float,\n'
            '  "reasoning": "string",\n'
            '  "category": "MEETING" | "HALLUCINATION" | "UNKNOWN"\n'
            "}"
        )

        try:
            # Call LLMProvider
            # api_key is None as we assume environment config or local model
            response_str = LLMProvider.generate(prompt, provider, None)
            
            # Parse JSON
            # Handle potential markdown wrapping
            clean_json = response_str.replace("