"""Build the gap-focused ACL figures and tables from immutable response records."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from relational_experiment.model import Trial  # noqa: E402
from relational_experiment.runner import (  # noqa: E402
    parse_reasoning_trial_answer,
    parse_trial_answer,
    trial_fingerprint,
)

BASE = ROOT / "results/common"
OUT = ROOT / "results"
FIG = ROOT / "generated/figures/headline"
DATA = ROOT / "data/common"
STRICT_DATA = ROOT / "data/strict"
FOLLOW = ROOT / "data/prompt-language"
BOOT = 10_000
SEED = 20260921
MODELS = [
    ("qwen27", "Qwen3.8-27B"),
    ("gemma26", "Gemma 4 26B"),
    ("gemma31", "Gemma 4 31B"),
    ("qwenmax", "Qwen3.8-Max"),
]
STRICT = {
    "qwen27": STRICT_DATA / "responses-qwen-local-strict-off-v1.jsonl",
    "gemma26": STRICT_DATA / "responses-gemma4-26b-strict-off-repair-v1.jsonl",
}
EXPECTED = {
    "trials.jsonl": "a3c81f351212b85ef3f044654a8be90c5fcae3c8937f2688b37d266b13e9230b",
    "problems.jsonl": "fd0c1ec75182b130b42cd20269680d9324ac9225cffd360173aedb239f608681",
}
FOLLOW_RESPONSE = {
    "qwen27": FOLLOW / "responses-qwen-local-semantic-inventory-full500-v1-seed20260914.jsonl",
    "gemma31": FOLLOW / "responses-gemma4-31b-semantic-inventory-full500-v1-seed20260914.jsonl",
}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval(values: np.ndarray, seed: int = SEED) -> list[float]:
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), (BOOT, len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def paired(c: list[int], a: list[int]) -> dict:
    left, right = np.asarray(c, int), np.asarray(a, int)
    assert len(left) == len(right) == 500
    n10 = int(((left == 1) & (right == 0)).sum())
    n01 = int(((left == 0) & (right == 1)).sum())
    return {
        "n": 500,
        "concrete": float(left.mean()),
        "arbitrary": float(right.mean()),
        "gap": float((left - right).mean()),
        "ci95": interval(left - right),
        "concrete_only": n10,
        "arbitrary_only": n01,
        "mcnemar_p": float(binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1),
    }


def paired_subset(c: list[int], a: list[int]) -> dict:
    left, right = np.asarray(c, int), np.asarray(a, int)
    assert len(left) == len(right) and len(left) > 0
    n10 = int(((left == 1) & (right == 0)).sum())
    n01 = int(((left == 0) & (right == 1)).sum())
    return {
        "n": len(left),
        "concrete": float(left.mean()),
        "arbitrary": float(right.mean()),
        "gap": float((left - right).mean()),
        "ci95": interval(left - right),
        "concrete_only": n10,
        "arbitrary_only": n01,
        "mcnemar_p": float(binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1),
    }


def save_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_tex(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(row + r" \\" for row in rows) + "\n")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for name, digest in EXPECTED.items():
        assert sha(DATA / name) == digest
    source_trials = {r["trial_id"]: Trial.from_dict(r) for r in read(DATA / "trials.jsonl")}
    problems = {r["problem_id"]: r for r in read(DATA / "problems.jsonl")}
    ids = sorted({t.problem_id for t in source_trials.values()})
    assert len(ids) == 500 and len(source_trials) == 1000
    budget = list(csv.DictReader((BASE / "behavior.csv").open()))
    paper_budget = [r for r in budget if int(r["budget"]) <= 512]
    with (OUT / "budget_grid.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paper_budget[0]))
        writer.writeheader()
        writer.writerows(paper_budget)
    source_manifest = json.loads((BASE / "manifest.json").read_text())
    for item in source_manifest["sources"]:
        assert sha(ROOT / item["path"]) == item["sha256"]
    assert source_manifest["dataset"] == EXPECTED

    headline = []
    headline_scores = {}
    ledger = [
        dict(study="original-budget-grid", model=item["model"], dataset="redesigned-v3",
             protocol="final-line; dedicated-channel budget=" + str(item["budget"]),
             source=item["path"], sha256=item["sha256"])
        for item in source_manifest["sources"]
        if int(item["budget"]) <= 512
    ]
    for model, title in MODELS:
        if model in STRICT:
            path = STRICT[model]
            rows = read(path)
            assert len(rows) == 1000 and {r["trial_id"] for r in rows} == set(source_trials)
            score = {}
            for row in rows:
                trial = source_trials[row["trial_id"]]
                assert row["trial_fingerprint"] == trial_fingerprint(trial)
                assert row["correct_letter"] == trial.correct_letter
                assert not row.get("request_error") and not row.get("truncated")
                assert not row.get("reasoning_content")
                content = row.get("visible_content") or ""
                assert len(content.splitlines()) <= 1
                answer, _ = parse_trial_answer(content, trial)
                score[trial.problem_id, trial.condition] = (int(answer == trial.correct_letter), int(answer is not None))
            result = paired([score[g, "concrete"][0] for g in ids], [score[g, "arbitrary"][0] for g in ids])
            headline_scores[model] = {(g, c): score[g, c][0] for g in ids for c in ("concrete", "arbitrary")}
            result.update(
                valid_c=sum(score[g, "concrete"][1] for g in ids),
                valid_a=sum(score[g, "arbitrary"][1] for g in ids),
                protocol="strict whole response",
            )
            ledger.append(dict(study="answer-only", model=model, dataset="redesigned-v3", protocol=result["protocol"], source=str(path.relative_to(ROOT)), sha256=sha(path)))
        else:
            prior = next(
                r for r in budget
                if r["model"] == model and r["budget"] == "0" and r["seed"] == "20260914" and r["subset"] == "full"
            )
            result = {
                "n": 500,
                "concrete": float(prior["concrete"]),
                "arbitrary": float(prior["arbitrary"]),
                "gap": float(prior["gap"]),
                "ci95": [float(prior["gap_low"]), float(prior["gap_high"])],
                "concrete_only": int(prior["concrete_only"]),
                "arbitrary_only": int(prior["arbitrary_only"]),
                "mcnemar_p": float(prior["p"]),
                "valid_c": int(prior["concrete_valid"]),
                "valid_a": int(prior["arbitrary_valid"]),
                "protocol": "observationally answer-only",
            }
            old_scores = list(csv.DictReader((BASE / "scores.csv").open()))
            cell = [r for r in old_scores if r["model"] == model and r["budget"] == "0" and r["seed"] == "20260914"]
            assert len(cell) == 1000
            headline_scores[model] = {(r["problem_id"], r["condition"]): int(r["correct"]) for r in cell}
            source = next(s for s in source_manifest["sources"] if s["model"] == model and s["budget"] == 0 and s["seed"] == 20260914)
            raw = read(ROOT / source["path"])
            assert len(raw) == 1000 and all(not (r.get("reasoning_content") or "").strip() for r in raw)
            assert all(len((r.get("visible_content") or r.get("final_content") or "").splitlines()) <= 1 for r in raw)
            ledger.append(dict(study="answer-only", model=model, dataset="redesigned-v3", protocol=result["protocol"], source=source["path"], sha256=source["sha256"]))
        headline.append(dict(model=model, title=title, **result))

    families = {
        "Grandchild": {"grandson", "granddaughter"},
        "Lateral": {"aunt", "uncle", "niece", "nephew", "cousin"},
        "Great-avuncular": {"great-aunt", "great-uncle"},
        "Marriage-linked": {"mother-in-law", "father-in-law", "daughter-in-law", "son-in-law", "sister-in-law", "brother-in-law"},
    }
    family_rows = []
    for family, labels in families.items():
        subset = [g for g in ids if problems[g]["answer"] in labels]
        for model, _ in MODELS:
            score = headline_scores[model]
            result = paired_subset(
                [score[g, "concrete"] for g in subset],
                [score[g, "arbitrary"] for g in subset],
            )
            family_rows.append(dict(model=model, family=family, **result))
    nonfloor = []
    selected = [g for g in ids if problems[g]["answer"] not in families["Great-avuncular"]]
    for model, _ in MODELS:
        score = headline_scores[model]
        nonfloor.append(dict(model=model, **paired_subset(
            [score[g, "concrete"] for g in selected],
            [score[g, "arbitrary"] for g in selected],
        )))

    follow_trials = {r["trial_id"]: Trial.from_dict(r) for r in read(FOLLOW / "trials.jsonl")}
    follow_ids = sorted({t.problem_id for t in follow_trials.values()})
    assert len(follow_ids) == 500 and len(follow_trials) == 3000
    follow = []
    for model, path in FOLLOW_RESPONSE.items():
        rows = read(path)
        assert len(rows) == 3000 and {r["trial_id"] for r in rows} == set(follow_trials)
        score = {}
        for row in rows:
            trial = follow_trials[row["trial_id"]]
            assert row["trial_fingerprint"] == trial_fingerprint(trial)
            assert row["correct_letter"] == trial.correct_letter
            ans, *_ = parse_reasoning_trial_answer(
                row.get("visible_content") or "", trial, row.get("reasoning_content"), row.get("truncated", False)
            )
            score[trial.problem_id, trial.condition] = int(ans == trial.correct_letter and not row.get("request_error"))
        for fmt in ("free", "inventory", "coded"):
            result = paired(
                [score[g, "concrete-" + fmt] for g in follow_ids],
                [score[g, "arbitrary-" + fmt] for g in follow_ids],
            )
            follow.append(dict(model=model, format=fmt, **result))
        ledger.append(dict(study="prompt-language", model=model, dataset="semantic-inventory-full500-v1", protocol="reasoning off; final-line", source=str(path.relative_to(ROOT)), sha256=sha(path)))

    save_csv(OUT / "headline.csv", headline)
    save_csv(OUT / "family.csv", family_rows)
    save_csv(OUT / "nonfloor.csv", nonfloor)
    save_csv(OUT / "prompt_language.csv", follow)
    save_csv(OUT / "evidence_ledger.csv", ledger)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.7, 2.45), layout="constrained")
    yy = np.arange(4)[::-1]
    for y, r in zip(yy, headline):
        color = "#0072B2" if r["protocol"].startswith("strict") else "#D55E00"
        marker = "D" if r["protocol"].startswith("strict") else "o"
        val = 100 * r["gap"]
        ax.errorbar(val, y, xerr=[[val - 100*r["ci95"][0]], [100*r["ci95"][1] - val]],
                    fmt=marker, color=color, ms=6, capsize=3, lw=1.5)
    ax.set(yticks=yy, yticklabels=[r["title"] for r in headline], xlabel="Concrete minus arbitrary accuracy (percentage points)", xlim=(-2, 43))
    ax.axvline(0, color="0.55", lw=.8)
    ax.grid(axis="x", alpha=.18)
    ax.plot([], [], "D", color="#0072B2", label="Strict whole response")
    ax.plot([], [], "o", color="#D55E00", label="Observed answer-only")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"headline_gaps.{ext}", dpi=300)
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.5), layout="constrained")
    for ax, (model, title) in zip(axs.flat, MODELS):
        values = sorted(
            (r for r in paper_budget if r["model"] == model and r["seed"] == "20260914" and r["subset"] == "full"),
            key=lambda r: int(r["budget"]),
        )
        xx = np.arange(len(values))
        for condition, color, marker in (("concrete", "#0072B2", "o"), ("arbitrary", "#D55E00", "s")):
            y = np.array([100 * float(r[condition]) for r in values])
            ax.plot(xx, y, marker=marker, ms=3, lw=1, color=color, label=condition.capitalize())
            if model in STRICT:
                strict = next(r for r in headline if r["model"] == model)
                ax.plot(-.36, 100 * strict[condition], marker="D", ms=4, linestyle="None", color=color)
        ax.set(title=title, ylim=(0, 105), xlim=(-.7, len(values)-.65),
               xticks=xx, xticklabels=["Off*" if r["budget"] == "0" else r["budget"] for r in values],
               ylabel="Accuracy (%)")
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=.18)
    axs[0, 0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.supxlabel("Nominal reasoning budget; diamonds left of Off* are separate strict-off reruns", fontsize=9)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"budget_accuracy.{ext}", dpi=300)
    plt.close(fig)

    (OUT / "results.json").write_text(json.dumps({
        "headline": headline,
        "families": family_rows,
        "nonfloor": nonfloor,
        "prompt_language": follow,
        "dataset_sha256": {name: sha(DATA / name) for name in EXPECTED},
        "follow_dataset_sha256": {name: sha(FOLLOW / name) for name in ("trials.jsonl", "problems.jsonl")},
        "bootstrap_replicates": BOOT,
        "seed": SEED,
    }, indent=2) + "\n")
    print(json.dumps({"headline": headline, "prompt_language": follow}, indent=2))


if __name__ == "__main__":
    main()
