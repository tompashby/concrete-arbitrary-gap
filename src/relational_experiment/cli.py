from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import analyze
from .genealogy import generate_problems
from .io import read_jsonl, write_jsonl
from .model import Problem, Trial
from .render import render_trials
from .runner import run_trials


def data_dir(root: Path, run_name: str | None) -> Path:
    return root / "data" / run_name if run_name else root / "data"


def results_dir(root: Path, run_name: str | None) -> Path:
    return root / "results" / run_name if run_name else root / "results"


def command_generate(args: argparse.Namespace) -> None:
    root = Path(args.root)
    target = data_dir(root, args.run_name)
    problems = generate_problems(args.count, args.seed, args.distractors)
    trials = render_trials(problems, args.seed + 1)
    write_jsonl(target / "problems.jsonl", (p.to_dict() for p in problems))
    write_jsonl(target / "trials.jsonl", (t.to_dict() for t in trials))
    print(f"Generated {len(problems)} problems and {len(trials)} trials")


def command_run(args: argparse.Namespace) -> None:
    root = Path(args.root)
    target = data_dir(root, args.run_name)
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    trials = [Trial.from_dict(row) for row in read_jsonl(target / "trials.jsonl")]
    if args.one_per_condition:
        selected: dict[str, Trial] = {}
        for trial in trials:
            selected.setdefault(trial.condition, trial)
        trials = list(selected.values())
    run_trials(trials, config, target / "responses.jsonl", args.limit)


def command_analyze(args: argparse.Namespace) -> None:
    root = Path(args.root)
    target = data_dir(root, args.run_name)
    print(analyze(
        target / "responses.jsonl",
        results_dir(root, args.run_name),
        target / "trials.jsonl",
        target / "problems.jsonl",
    ))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen relational reasoning benchmark")
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-name")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--count", type=int, default=25)
    generate.add_argument("--seed", type=int, default=20260824)
    generate.add_argument("--distractors", type=int, default=2)
    generate.set_defaults(func=command_generate)
    run = sub.add_parser("run")
    run.add_argument("--config", default="config.json")
    run.add_argument("--limit", type=int)
    run.add_argument("--one-per-condition", action="store_true")
    run.set_defaults(func=command_run)
    analysis = sub.add_parser("analyze")
    analysis.set_defaults(func=command_analyze)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
