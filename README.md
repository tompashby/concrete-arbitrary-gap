# Concrete–Arbitrary Kinship Inference: Results and Reproducibility Data

This repository contains the frozen model outputs, derived results, and analysis
code for Tom Pashby's *The Concrete–Arbitrary Gap: Kinship Reasoning in LLMs Is
Not Indifferent to Presentation*. It is a data release, not a manuscript source
repository.

Repository: https://github.com/tompashby/concrete-arbitrary-gap  
Contact: Tom Pashby, Middlebury College, tpashby@middlebury.edu

## Contents

### `data/common/`

The common 500-graph, 1,000-prompt benchmark and 40,000 recorded responses for
the four-model reasoning-budget grid through 512 tokens. Each response file
contains one complete 1,000-response condition.

### `data/strict/`

The strict whole-response, thinking-off reruns for local Qwen3.8-27B and Gemma
4 26B-A4B. These are the two strict cells used in the headline answer-only
comparison.

### `data/prompt-language/`

The augmented 500-graph prompt-language experiment, including the frozen
problems, 3,000 trials per model, and the Gemma 4 31B and local Qwen response
records used for the free, inventory, and coded-output comparison.

### `results/`

Machine-readable results used in the study:

- `headline.csv`: answer-only concrete/arbitrary comparison;
- `budget_grid.csv`: full common-protocol grid through 512 tokens;
- `family.csv` and `nonfloor.csv`: relation-family and non-floor analyses;
- `prompt_language.csv`: free, inventory, and coded-output follow-up;
- `evidence_ledger.csv`: source path and SHA-256 digest for every evidential
  record;
- `results.json`: complete headline and follow-up statistics;
- `common/`: trial-level scores, budget summaries, token statistics, seed
  comparisons, parser discrepancies, and the common-grid source manifest;
- `artifact_manifest.json`: hashes and sizes for the public data package.

### `scripts/` and `src/`

The exact answer parser and analysis programs. Answers are scored only from the
recorded final-answer channel. The analysis never searches a reasoning trace for
an answer.

## Reproduce the results

Python 3.11 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/build_common_grid.py
python scripts/build_results.py
python scripts/verify_release.py
```

The reconstruction verifies dataset hashes and trial fingerprints before
scoring. Confidence intervals use 10,000 graph-level bootstrap resamples. Exact
McNemar tests use paired concrete/arbitrary outcomes. The graph, not the prompt,
is the primary sampling unit.

The analysis programs also place optional regenerated plotting and table
fragments under `generated/`; these are not part of the archived data release.

## Protocol distinctions

- Local Qwen and Gemma 26B headline cells are strict whole-response reruns.
- Gemma 31B and Qwen-Max headline cells are observationally answer-only cells.
- Connected budget-grid cells use the original final-line protocol.
- The prompt-language follow-up uses a separate augmented dataset.
- Invalid responses count as failures in the conservative primary analysis.

These distinctions are encoded in `results/evidence_ledger.csv` and should not
be collapsed when reusing the data.

## Models

- `avyukth/Qwen3.8-27B-AWQ-INT4`, served locally with vLLM 0.29.0;
- `google/gemma-4-26B-A4B-it`, converted to the recorded GGUF and served with
  llama.cpp b10917/Vulkan;
- `google/gemma-4-31B-it`, converted to the recorded GGUF and served with
  llama.cpp b10917/Vulkan;
- Alibaba Cloud Model Studio `qwen3.8-max`.

The repository contains model outputs, not model weights or credentials. Model
weights and hosted services remain subject to their providers' licenses and
terms.

## Integrity

`SHA256SUMS` covers every archived file except itself. Run:

```bash
shasum -a 256 -c SHA256SUMS
```

## Citation and license

Citation metadata is in `CITATION.cff`. No reuse license is asserted by this
release candidate. Before publication, the author should choose licenses for
the code and data/model outputs that are compatible with the applicable model
terms.
