"""
nexus/fuzzer.py — Autonomous TT/TF Dataset Generation Engine
=============================================================
This script invokes an SCP-generated architecture CLI with a high
temperature to intentionally induce architectural hallucinations.
The Weaver (code_verifier) intercepts these attempts to build a
Contrastive Divergence (CD) / Forward-Forward (FF) dataset.
"""

import sys
import os
import subprocess
import time
import argparse

def main():
    parser = argparse.ArgumentParser(description="Autonomous SCP Fuzzer for TT/TF Dataset Collection")
    parser.add_argument("scp_cli", help="Path to the generated _scp.py CLI file (e.g., systemmonitor_scp.py)")
    parser.add_argument("--iterations", type=int, default=100, help="Number of times to run the full generation suite")
    parser.add_argument("--temperature", type=float, default=0.8, help="API Temperature (higher = more hallucinations)")
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview", help="Model to use")
    parser.add_argument("--output-dir", type=str, default="./fuzz_out", help="Directory to store code output")
    
    args = parser.parse_args()

    scp_cli = os.path.abspath(args.scp_cli)
    if not os.path.exists(scp_cli):
        print(f"Error: Could not find SCP CLI script at {scp_cli}")
        sys.exit(1)

    print("=" * 70)
    print("🚀 Project Chevron: Energy-Based Fuzzing Engine")
    print("=" * 70)
    print(f"Target:      {scp_cli}")
    print(f"Iterations:  {args.iterations}")
    print(f"Temperature: {args.temperature}")
    print(f"Model:       {args.model}")
    print(f"Dataset:     chevron_ebm_dataset.jsonl (logged by templates)")
    print("=" * 70)

    # Set the environment variable that our modified template looks for
    env = os.environ.copy()
    env["CHEVRON_TEMP"] = str(args.temperature)

    for i in range(1, args.iterations + 1):
        print(f"\n[Fuzzing Iteration {i}/{args.iterations}]")
        print("Starting autonomous generation pipeline...")
        
        # We invoke the generated _scp.py script with --all to generate the whole architecture
        cmd = [
            sys.executable,
            scp_cli,
            "--all",
            "--model", args.model,
            "--output-dir", os.path.abspath(args.output_dir)
        ]
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            elapsed = time.time() - start_time
            
            # Print the summary from the bottom of the output
            if result.stdout:
                lines = result.stdout.strip().split("\n")
                summary = [line for line in lines if "✔" in line or "✘" in line or "PASS" in line or "FAIL" in line]
                for line in summary[-10:]:  # Print last 10 lines containing passes/fails
                    print(f"  {line.strip()}")
                    
            if result.stderr:
                print("Stderr (if any):")
                print(result.stderr[:500])
            
            print(f"Iteration {i} completed in {elapsed:.1f}s. Waiting 2s before next run...")
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\nFuzzing interrupted by user. Stopping.")
            break
        except Exception as e:
            print(f"Error executing iteration {i}: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
