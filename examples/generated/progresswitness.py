import sys
import json
from typing import Dict, Any

class ProgressWitness:
    """
    Module: ProgressWitness
    Purpose: Observes and reports pipeline progress. Pure logging — never modifies data.
    Protocol: Spatial Constraint Protocol v1.0
    Glyph: 𓃀 (Witnesses state)
    """

    def __init__(self) -> None:
        """
        Initialize the witness.
        No external dependencies or state retention.
        """
        pass

    def emit_progress(self, current: int, total: int, label: str) -> None:
        """
        𓃀 Witnesses progress state — logs without altering pipeline.
        
        Output format: [PROGRESS] <percent>% | <current>/<total> | <label>
        
        Args:
            current: Current step or item count.
            total: Total steps or items.
            label: Description of the current operation.
        """
        try:
            if total > 0:
                percent = int((current / total) * 100)
            else:
                percent = 0
            
            # Ensure percent is clamped between 0 and 100 for display sanity
            percent = max(0, min(100, percent))
            
            message = f"[PROGRESS] {percent}% | {current}/{total} | {label}"
            
            sys.stdout.write(message + "\n")
            sys.stdout.flush()
        except Exception:
            # ⚠️ Must NEVER raise exceptions that halt the pipeline
            pass

    def emit_result(self, result: Dict[str, Any]) -> None:
        """
        𓃀 Witnesses a completed result — JSON output for UI consumption.
        
        Output format: [RESULT] <json_string>
        
        Args:
            result: Dictionary containing the result data.
        """
        try:
            # Serialize to JSON to ensure structure is preserved
            json_str = json.dumps(result, default=str)
            message = f"[RESULT] {json_str}"
            
            sys.stdout.write(message + "\n")
            sys.stdout.flush()
        except Exception:
            # ⚠️ Must NEVER raise exceptions that halt the pipeline
            # Fallback error emission if serialization fails
            self.emit_error("Failed to serialize result", "ProgressWitness.emit_result")

    def emit_error(self, error: str, context: str) -> None:
        """
        𓃀 Witnesses an error — logs without attempting recovery.
        
        Output format: [ERROR] [<context>] <error>
        
        Args:
            error: The error message or description.
            context: The location or module where the error occurred.
        """
        try:
            # Sanitize inputs to prevent formatting issues
            safe_context = str(context).replace("\n", " ")
            safe_error = str(error).replace("\n", " ")
            
            message = f"[ERROR] [{safe_context}] {safe_error}"
            
            # Writing to stdout to ensure synchronization with PROGRESS/RESULT markers
            # for single-stream UI parsers.
            sys.stdout.write(message + "\n")
            sys.stdout.flush()
        except Exception:
            # ⚠️ Must NEVER raise exceptions that halt the pipeline
            pass
