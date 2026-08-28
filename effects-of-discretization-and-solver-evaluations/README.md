# Effects of Discretization and Solver Evaluations

This repository contains three main scripts:
- `example_scenario.py`: single-scenario run plus interval-count parameter study
- `batch_evaluations.py`: Monte Carlo style batch evaluation across sampled scenarios
- `solver_evaluations.py`: MPPI ablation and solver-focused evaluation

## Requirements

- Python 3.9 to 3.12

## Installation

From the repository root:

```bash
pdm install
./.venv/bin/python -m ensurepip --upgrade && ./.venv/bin/python -m pip install --no-deps pyinterval
```

## Run Scripts

Run from the repository root:

```bash
python example_scenario.py
python batch_evaluations.py
python solver_evaluations.py
```
