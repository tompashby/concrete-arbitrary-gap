from __future__ import annotations

import csv
import math
import random
from collections import defaultdict
from pathlib import Path

from .io import read_jsonl
from .model import Problem, Trial
from .runner import trial_fingerprint


def _exact_mcnemar_p(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(left_only, right_only) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def _paired_interval(differences: list[int], seed: int = 20260825) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = []
    for _ in range(10_000):
        estimates.append(sum(rng.choice(differences) for _ in differences) / len(differences))
    estimates.sort()
    return estimates[249], estimates[9749]


def analyze(
    response_path: Path,
    output_dir: Path,
    trial_path: Path | None = None,
    problem_path: Path | None = None,
    include_errors_as_incorrect: bool = False,
) -> str:
    rows = [
        row for row in read_jsonl(response_path)
        if include_errors_as_incorrect or not row.get("request_error")
    ]
    # A response file may contain smoke tests from several model configurations.
    # Report only the most recently used configuration to avoid mixing protocols.
    fingerprint = rows[-1].get("config_fingerprint") if rows else None
    if fingerprint:
        rows = [row for row in rows if row.get("config_fingerprint") == fingerprint]
    if trial_path is not None:
        expected = {
            trial.trial_id: trial_fingerprint(trial)
            for trial in (Trial.from_dict(row) for row in read_jsonl(trial_path))
        }
        rows = [
            row for row in rows
            if row.get("trial_fingerprint") == expected.get(row.get("trial_id"))
        ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for condition in sorted(grouped):
        condition_rows = grouped[condition]
        summaries.append({
            "condition": condition,
            "n": len(condition_rows),
            "correct": sum(bool(row.get("correct")) for row in condition_rows),
            "accuracy": sum(bool(row.get("correct")) for row in condition_rows) / len(condition_rows),
            "strict_parse_rate": sum(row.get("parse_mode") == "strict" for row in condition_rows) / len(condition_rows),
            "conditional_accuracy": (
                sum(bool(row.get("correct")) for row in condition_rows if row.get("parsed_letter"))
                / max(1, sum(bool(row.get("parsed_letter")) for row in condition_rows))
            ),
            "mean_latency_seconds": sum(row.get("latency_seconds", 0) for row in condition_rows) / len(condition_rows),
        })

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]) if summaries else ["condition"])
        writer.writeheader()
        writer.writerows(summaries)

    by_problem: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_problem[row["problem_id"]][row["condition"]] = row
    disagreements = []
    for problem_id, conditions in by_problem.items():
        outcomes = {c: bool(r.get("correct")) for c, r in conditions.items()}
        if len(set(outcomes.values())) > 1:
            disagreements.append({"problem_id": problem_id, **outcomes})
    if disagreements:
        fields = sorted({key for row in disagreements for key in row})
        with (output_dir / "disagreements.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(disagreements)

    comparisons = []
    requested_pairs = [
        ("concrete", "arbitrary"), ("concrete", "formal"),
        ("explicit", "arbitrary"), ("explicit", "formal"),
        ("explicit", "concrete"),
    ]
    for left, right in requested_pairs:
        paired = [conditions for conditions in by_problem.values() if left in conditions and right in conditions]
        differences = [int(bool(p[left].get("correct"))) - int(bool(p[right].get("correct"))) for p in paired]
        if not differences:
            continue
        left_only = sum(d == 1 for d in differences)
        right_only = sum(d == -1 for d in differences)
        low, high = _paired_interval(differences)
        comparisons.append({
            "left": left, "right": right, "n": len(paired),
            "accuracy_difference": sum(differences) / len(differences),
            "bootstrap_ci_low": low, "bootstrap_ci_high": high,
            "left_only_correct": left_only, "right_only_correct": right_only,
            "exact_mcnemar_p": _exact_mcnemar_p(left_only, right_only),
        })
    with (output_dir / "paired_comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    relation_summaries = []
    if problem_path is not None:
        relation_by_problem = {
            problem.problem_id: problem.correct_relation
            for problem in (Problem.from_dict(row) for row in read_jsonl(problem_path))
        }
        relation_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in rows:
            relation_groups[(relation_by_problem[row["problem_id"]], row["condition"])].append(row)
        for (relation, condition), group_rows in sorted(relation_groups.items()):
            relation_summaries.append({
                "relation": relation,
                "condition": condition,
                "n": len(group_rows),
                "correct": sum(bool(row.get("correct")) for row in group_rows),
                "accuracy": sum(bool(row.get("correct")) for row in group_rows) / len(group_rows),
                "strict_parse_rate": sum(row.get("parse_mode") == "strict" for row in group_rows) / len(group_rows),
            })
        with (output_dir / "summary_by_relation.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(relation_summaries[0]))
            writer.writeheader()
            writer.writerows(relation_summaries)

    lines = ["# Experiment results", "", f"Configuration fingerprint: `{fingerprint or 'unknown'}`", "", "| Condition | N | Correct | Accuracy | Parseable accuracy | Strict parse | Mean latency |", "|---|---:|---:|---:|---:|---:|---:|"]
    for item in summaries:
        lines.append(
            f"| {item['condition']} | {item['n']} | {item['correct']} | "
            f"{item['accuracy']:.1%} | {item['conditional_accuracy']:.1%} | {item['strict_parse_rate']:.1%} | "
            f"{item['mean_latency_seconds']:.2f}s |"
        )
    lines.extend(["", f"Problems with cross-condition correctness disagreement: {len(disagreements)}", ""])
    lines.extend([
        "## Paired condition differences",
        "",
        "| Left minus right | Difference | 95% bootstrap CI | Discordant pairs | Exact p |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in comparisons:
        lines.append(
            f"| {row['left']} − {row['right']} | {row['accuracy_difference']:.1%} | "
            f"[{row['bootstrap_ci_low']:.1%}, {row['bootstrap_ci_high']:.1%}] | "
            f"{row['left_only_correct']} vs {row['right_only_correct']} | {row['exact_mcnemar_p']:.3g} |"
        )
    lines.append("")
    if relation_summaries:
        lookup = {(row["relation"], row["condition"]): row["accuracy"] for row in relation_summaries}
        conditions = sorted(grouped)
        lines.extend([
            "## Accuracy by target relationship",
            "",
            "| Relationship | " + " | ".join(conditions) + " |",
            "|---|" + "---:|" * len(conditions),
        ])
        for relation in sorted({row["relation"] for row in relation_summaries}):
            values = " | ".join(f"{lookup[(relation, condition)]:.1%}" for condition in conditions)
            lines.append(f"| {relation} | {values} |")
        lines.append("")
    report = "\n".join(lines)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    return report
