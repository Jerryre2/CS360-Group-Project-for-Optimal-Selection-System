# Optimal Sample Selection System

This repository contains a complete solving system for covering-design style
optimal sample selection problems of the form `L(n, k, j, s)`.

This submitted repository should be treated as the **final noNN version** of
the project. The production solver path is deterministic and structure-aware:
it does **not** include the earlier experimental neural-guidance component.

The system can:

- construct feasible solutions with lazy greedy + local improvement + simulated annealing
- validate saved answers with two independent validators
- certify small and medium instances with exact backends
- report lower bounds and optimality gaps on larger instances
- expose the workflow through both a Python CLI and an iOS app backed by a
  lightweight local API

## Start Here

If you are opening this repository from a USB submission copy, read
[START_HERE.md](START_HERE.md) first.

For the standalone command-line reference, read
[COMMAND_LINE_GUIDE.md](COMMAND_LINE_GUIDE.md).

For the full user manual, read [docs/USER_GUIDELINE.md](docs/USER_GUIDELINE.md).

## Repository Layout

```text
optimal_samples_system/      Core Python package
OptimalSelectionMobile/      iOS app project for Xcode
tests/                       Regression and validation tests
docs/                        User-facing documentation
scripts/                     Helper launch scripts
FUTURE_WORK.md               Future improvement ideas
prove_n15_optimality.py      One-off exact script for the fixed n=15 case
```

## Main Features

- `solve`: solve one instance and optionally save the result
- `validate-*`: check feasibility with the primary validator
- `audit-*`: run both the primary validator and an independent brute-force validator
- `prove-bound`: try to prove whether a smaller solution exists
- `certify-*`: try to certify a saved result as globally optimal
- `summarize-*`: produce a concise optimality summary
- `list-results` / `show-result` / `delete-result`: manage saved JSON results
- `demo`: run built-in benchmark-style demonstrations

## Quick Start

### CLI

Run a simple solve:

```bash
python -m optimal_samples_system solve --m 45 --n 9 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9 \
  --seed 42 --restarts 5
```

Save the result:

```bash
python -m optimal_samples_system solve --m 45 --n 9 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9 \
  --seed 42 --restarts 5 --save
```

List saved results:

```bash
python -m optimal_samples_system list-results --db-dir results_db_v3
```

Audit a saved result with both validators:

```bash
python -m optimal_samples_system audit-result <filename> --db-dir results_db_v3
```

### iOS App

1. Start the local mobile API:

```bash
./scripts/start_mobile_api.sh
```

2. Open the Xcode project:

```bash
open OptimalSelectionMobile/OptimalSelectionMobile.xcodeproj
```

3. Run the app in the iOS Simulator.

The app currently provides tabs for solving, proof/certification, validation,
and result browsing.

## Optional Exact Solvers

The core system runs with Python's standard library.

Optional backends:

- `scipy` / HiGHS for exact or LP-based routines
- `gurobi` for stronger exact verification
- `scip` for stronger exact verification

If these backends are unavailable, the system still runs in heuristic mode and
still supports full feasibility validation.

## Validation and Correctness

Every final solver result is checked before it is returned or saved.

The validation stack includes:

- a primary validation path based on the incremental coverage tracker
- an independent brute-force validator that does not reuse the tracker logic
- result-file validation, audit, and certification commands for offline checking

This design helps distinguish:

- "the solution is feasible"
- "the solution has been independently audited"
- "the solution has or has not been proven optimal"

## Development Note

This repository is organized as a submission-ready engineering project rather
than a research notebook. The recommended user-facing entry points are:

- [START_HERE.md](START_HERE.md)
- [docs/USER_GUIDELINE.md](docs/USER_GUIDELINE.md)
- `python -m optimal_samples_system --help`

For grading and reproduction, please regard the current branch as the
**submission baseline without neural guidance**.
