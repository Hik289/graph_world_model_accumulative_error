# Artifact Guide

Operational notes for reproducing `Understanding Rollout Error in Graph World Models` from the public `graph_world_model_accumulative_error` repository.

## Review Path

- `src/`: Core source code and reusable implementations.
- `scripts/`: Command-line entry points for experiments, analysis, or reproduction.
- `tests/`: Local tests or smoke checks for fresh checkouts.
- `figures/`: README and paper-facing figures.

## Environment Files

- `requirements.txt`: Primary Python dependency list.

## Smoke Checks

Run these checks before long jobs:

```bash
python -m compileall -q .
python -m pytest tests -q
```

## Reproduction Entry Points

Main tracked entry points for paper-scale or benchmark-scale runs:

- `python scripts/gen_p2_figures.py`

## Figure Assets

- `figures/fig_main.png`
- `figures/fig_pipeline.png`

## Data And Outputs

- Keep local dataset paths, downloaded corpora, checkpoints, and generated run artifacts outside git unless the README identifies them as small checked-in fixtures.
- Record dataset version, preprocessing command, seed, and hardware/runtime notes for every reproduced table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
