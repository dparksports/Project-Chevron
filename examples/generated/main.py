"""
TurboScribe — SCP-Generated Application
=========================================
This entire application was generated module-by-module from SCP contracts.
Each module was independently produced by Gemini, verified by the Weaver,
and assembled here into a working pipeline.

Usage:
    python main.py                          # Show architecture overview
    python main.py transcribe <file>        # Transcribe a single file
    python main.py transcribe-dir <dir>     # Batch transcribe a directory
    python main.py search <dir> <query>     # Search transcripts
    python main.py analyze <transcript>     # Analyze/summarize a transcript
"""

import sys
import os

# ─── Import all generated modules ───
# Each module was independently generated from its SCP contract.
# They communicate only through their declared interfaces.
try:
    from audioingest import *
    from voicedetector import *
    from transcriber import *
    from searchengine import *
    from meetingdetector import *
    from llmprovider import *
    from analyzer import *
    from timestampextractor import *
    from progresswitness import *
except ImportError as e:
    print(f"⚠ Module import error: {e}")
    print("  Some modules may not have been generated successfully.")
    print()


def show_help():
    print("=" * 60)
    print("◬  TurboScribe — SCP-Generated Application")
    print("=" * 60)
    print()
    print("Modules:")
    print("  ◬ 𓃀    AudioIngest            Discovers and loads media files from directories. ")
    print("  Ө ☤    VoiceDetector          Detects speech segments in audio using Silero VAD.")
    print("  ◬ ☾    Transcriber            Transcribes audio to text using faster-whisper mod")
    print("  Ө      SearchEngine           Searches transcripts by keyword or semantic embedd")
    print("  Ө      MeetingDetector        Uses LLM to classify transcripts as real meetings ")
    print("  ◬ ☤ 𓃀  LLMProvider            Unified interface to local (llama.cpp) and cloud (")
    print("  ☤      Analyzer               Summarizes or outlines transcripts using LLM provi")
    print("  ☾      TimestampExtractor     Extracts burned-in timestamps from video frames us")
    print("  𓃀      ProgressWitness        Observes and reports pipeline progress. Pure loggi")
    print()
    print("Commands:")
    print("  transcribe <file>          Transcribe an audio/video file")
    print("  transcribe-dir <dir>       Batch transcribe all files in directory")
    print("  search <dir> <query>       Search transcripts for text")
    print("  analyze <file>             Analyze/summarize a transcript")
    print()
    print("Options:")
    print("  --model <name>             Whisper model (default: base.en)")
    print("  --beam-size <n>            Beam size (default: 5)")
    print()
    print("Examples:")
    print("  python main.py transcribe meeting.wav")
    print("  python main.py transcribe-dir ./recordings --model large-v3")
    print("  python main.py search ./transcripts \"quarterly results\"")
    print()


def cmd_transcribe(args):
    """Transcribe a single file."""
    if not args:
        print("Usage: python main.py transcribe <file> [--model <name>]")
        return

    file_path = args[0]
    model_name = "base.en"
    beam_size = 5

    i = 1
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            i += 1
            model_name = args[i]
        elif args[i] == "--beam-size" and i + 1 < len(args):
            i += 1
            beam_size = int(args[i])
        i += 1

    if not os.path.exists(file_path):
        print(f"⚠ File not found: {file_path}")
        return

    try:
        result = transcribe_file(file_path, model_name, beam_size)
        print(f"\n──── Transcript ────")
        print(result.full_text)
        print(f"\n──── Stats ────")
        print(f"  Model: {result.model_name}")
        print(f"  Segments: {len(result.segments)}")
    except Exception as e:
        print(f"⚠ Error: {e}")


def cmd_transcribe_dir(args):
    """Batch transcribe a directory."""
    if not args:
        print("Usage: python main.py transcribe-dir <directory> [--model <name>]")
        return

    directory = args[0]
    model_name = "base.en"
    beam_size = 5

    i = 1
    while i < len(args):
        if args[i] == "--model" and i + 1 < len(args):
            i += 1
            model_name = args[i]
        elif args[i] == "--beam-size" and i + 1 < len(args):
            i += 1
            beam_size = int(args[i])
        i += 1

    if not os.path.isdir(directory):
        print(f"⚠ Directory not found: {directory}")
        return

    try:
        results = batch_transcribe(directory, model_name, beam_size)
        print(f"\n𓂀 Transcribed {len(results)} files.")
        for r in results:
            status = "✔" if r.full_text else "✘"
            print(f"  {status} {os.path.basename(r.file_path)}")
    except Exception as e:
        print(f"⚠ Error: {e}")


def cmd_search(args):
    """Search transcripts."""
    if len(args) < 2:
        print("Usage: python main.py search <directory> <query>")
        return

    directory = args[0]
    query = " ".join(args[1:])

    try:
        results = search_transcripts(directory, query)
        if not results:
            print(f"No results for: {query}")
        else:
            print(f"\nFound {len(results)} matches for \"{query}\":")
            for r in results:
                print(f"  • {r}")
    except NameError:
        # Fallback: simple grep-like search
        print(f"Searching for \"{query}\" in {directory}...")
        for root, dirs, files in os.walk(directory):
            for f in files:
                if f.endswith("_transcript.txt") or f.endswith("_transcript_base.en.txt"):
                    path = os.path.join(root, f)
                    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
                        text = fh.read()
                    if query.lower() in text.lower():
                        print(f"  ✔ {os.path.relpath(path, directory)}")
    except Exception as e:
        print(f"⚠ Error: {e}")


def cmd_analyze(args):
    """Analyze a transcript."""
    if not args:
        print("Usage: python main.py analyze <transcript_file>")
        return

    file_path = args[0]
    if not os.path.exists(file_path):
        print(f"⚠ File not found: {file_path}")
        return

    try:
        result = analyze_transcript(file_path)
        print(result)
    except NameError:
        # Fallback: show basic stats
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        words = len(text.split())
        lines = text.count("\n") + 1
        print(f"\n──── Transcript Analysis ────")
        print(f"  File: {os.path.basename(file_path)}")
        print(f"  Words: {words:,}")
        print(f"  Lines: {lines:,}")
        print(f"  Characters: {len(text):,}")
    except Exception as e:
        print(f"⚠ Error: {e}")


def main():
    if len(sys.argv) < 2:
        show_help()
        return

    command = sys.argv[1].lower()
    args = sys.argv[2:]

    commands = {
        "transcribe": cmd_transcribe,
        "transcribe-dir": cmd_transcribe_dir,
        "search": cmd_search,
        "analyze": cmd_analyze,
        "help": lambda a: show_help(),
        "--help": lambda a: show_help(),
        "-h": lambda a: show_help(),
    }

    if command in commands:
        commands[command](args)
    else:
        print(f"Unknown command: {command}")
        show_help()


if __name__ == "__main__":
    main()
