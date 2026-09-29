"""Create a manifest for the public results and reproducibility package."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry(path: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    ledger = list(csv.DictReader((RESULTS / "evidence_ledger.csv").open()))
    data_files = sorted(path for path in (ROOT / "data").rglob("*") if path.is_file())
    analysis_files = sorted(path for path in (ROOT / "scripts").glob("*.py"))
    manifest = {
        "results": [
            entry(path)
            for path in sorted(RESULTS.rglob("*"))
            if path.is_file() and path.name != "artifact_manifest.json"
        ],
        "data": [entry(path) for path in data_files],
        "analysis": [entry(path) for path in analysis_files],
        "evidence_ledger_rows": len(ledger),
        "ledger_sources_verified": all(
            sha256(ROOT / row["source"]) == row["sha256"] for row in ledger
        ),
    }
    (RESULTS / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "data_files": len(data_files),
                "result_files": len(manifest["results"]),
                "ledger_rows": len(ledger),
                "ledger_sources_verified": manifest["ledger_sources_verified"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
