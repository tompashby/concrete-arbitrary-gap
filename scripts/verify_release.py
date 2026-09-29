"""Verify immutable inputs and machine-readable results in the public release."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ledger = list(csv.DictReader((RESULTS / "evidence_ledger.csv").open()))
    assert len(ledger) == 46
    for row in ledger:
        path = ROOT / row["source"]
        assert path.is_file(), path
        assert sha256(path) == row["sha256"], path

    results = json.loads((RESULTS / "results.json").read_text())
    assert len(results["headline"]) == 4
    assert len(results["prompt_language"]) == 6
    assert all(row["n"] == 500 for row in results["headline"])
    assert all(row["gap"] > 0 for row in results["headline"])
    assert all(row["mcnemar_p"] < 0.001 for row in results["headline"])

    common = json.loads((RESULTS / "common/manifest.json").read_text())
    assert common["total_responses"] == 40_000
    assert all(0 <= int(row["budget"]) <= 512 for row in common["sources"])
    required = (
        RESULTS / "budget_grid.csv",
        RESULTS / "headline.csv",
        RESULTS / "family.csv",
        RESULTS / "nonfloor.csv",
        RESULTS / "prompt_language.csv",
        RESULTS / "evidence_ledger.csv",
        ROOT / "data/common/problems.jsonl",
        ROOT / "data/common/trials.jsonl",
    )
    assert all(path.is_file() for path in required)
    print(f"Release verification passed: {len(ledger)} ledger rows, 40,000 grid responses")


if __name__ == "__main__":
    main()
