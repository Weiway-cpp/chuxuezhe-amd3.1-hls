"""Run a baseline, an agent, or both against one local task."""
import argparse
import json
import os
from pathlib import Path
import sys

from agent.model import load_config
from agent.runner import load_task, run_session

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--config", default=str(ROOT/"config"/"model.example.json"))
    parser.add_argument("--mode", choices=("baseline", "agent", "compare"), default="compare")
    parser.add_argument("--vitis-run", default=os.environ.get("VITIS_RUN", "vitis-run"))
    parser.add_argument('--tool-version', '--required-tool-version', dest='required_tool_version',
                        default=os.environ.get('HLS_TOOL_VERSION', '2025.2'))
    parser.add_argument("--output-root", default=str(ROOT/"runs"/"agent"))
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--total-seconds", type=float, default=600)
    parser.add_argument("--tool-timeout", type=float, default=180)
    parser.add_argument("--generation-timeout", type=float, default=120)
    parser.add_argument("--reserve-seconds", type=float, default=15)
    parser.add_argument('--generation-only', action='store_true',
                        help='Bare baseline only: emit code without invoking verification tools')
    parser.add_argument('--without-skills', action='store_true')
    args = parser.parse_args()
    try:
        summary = run_session(
            load_task(args.task, require_testbench=not args.generation_only), load_config(args.config), args.vitis_run,
            args.output_root, mode=args.mode, max_attempts=args.max_attempts,
            total_seconds=args.total_seconds, tool_timeout=args.tool_timeout,
            generation_timeout=args.generation_timeout, reserve_seconds=args.reserve_seconds,
            generation_only=args.generation_only, use_skills=not args.without_skills,
            required_tool_version=args.required_tool_version)
    except (ValueError, OSError) as exc:
        print("Configuration or I/O error: " + str(exc), file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    target = "agent" if args.mode == "compare" else args.mode
    return 0 if (summary['policies'][target]['status'] == 'generated' if args.generation_only
                 else summary["policies"][target]["passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
