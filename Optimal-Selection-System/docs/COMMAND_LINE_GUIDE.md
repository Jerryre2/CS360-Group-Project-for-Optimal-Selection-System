# Command Line Guide

This document is a standalone command-line manual for the
`Optimal Sample Selection System`.

It focuses on three things only:

- how to enter commands
- what each command does
- where important files are located

If you want the broader project overview, read [README.md](README.md).
If you want the complete end-user manual, read
[docs/USER_GUIDELINE.md](docs/USER_GUIDELINE.md).

## 1. Repository Navigation

Open Terminal and enter the repository root first:

```bash
cd /Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System
```

Important paths:

- root overview
  - [README.md](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/README.md)
  - [START_HERE.md](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/START_HERE.md)
  - [COMMAND_LINE_GUIDE.md](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/COMMAND_LINE_GUIDE.md)
- detailed manual
  - [docs/USER_GUIDELINE.md](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/docs/USER_GUIDELINE.md)
- core Python package
  - [optimal_samples_system/__main__.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/optimal_samples_system/__main__.py)
  - [optimal_samples_system/cli.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/optimal_samples_system/cli.py)
  - [optimal_samples_system/solver.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/optimal_samples_system/solver.py)
  - [optimal_samples_system/validation.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/optimal_samples_system/validation.py)
  - [optimal_samples_system/certify.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/optimal_samples_system/certify.py)
  - [optimal_samples_system/exact.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/optimal_samples_system/exact.py)
  - [optimal_samples_system/storage.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/optimal_samples_system/storage.py)
- iOS app
  - [OptimalSelectionMobile/OptimalSelectionMobile.xcodeproj](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/OptimalSelectionMobile/OptimalSelectionMobile.xcodeproj)
  - [OptimalSelectionMobile/OptimalSelectionMobile/OptimalSelectionMobileApp.swift](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/OptimalSelectionMobile/OptimalSelectionMobile/OptimalSelectionMobileApp.swift)
- helper scripts
  - [scripts/start_mobile_api.sh](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/scripts/start_mobile_api.sh)
  - [scripts/open_ios_project.sh](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/scripts/open_ios_project.sh)
  - [scripts/example_solve.sh](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/scripts/example_solve.sh)
- tests
  - [tests/test_solver_known_cases.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/tests/test_solver_known_cases.py)
  - [tests/test_validation.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/tests/test_validation.py)
  - [tests/test_certify.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/tests/test_certify.py)
  - [tests/test_storage.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/tests/test_storage.py)

## 2. Main Entry Commands

### Show global help

```bash
python -m optimal_samples_system --help
```

Purpose:

- list all available CLI commands

### Show mobile API help

```bash
python -m optimal_samples_system.mobile_api --help
```

Purpose:

- show how to start the local HTTP bridge used by the iOS app

## 3. Global Command Format

All core commands start with:

```bash
python -m optimal_samples_system <subcommand> [arguments]
```

Examples:

```bash
python -m optimal_samples_system solve ...
python -m optimal_samples_system list-results ...
python -m optimal_samples_system audit-result ...
```

## 4. Core Solve Command

### Command

```bash
python -m optimal_samples_system solve --m <M> --n <N> --k <K> --j <J> --s <S> [options]
```

### Required arguments

- `--m`
  - total universe size
- `--n`
  - number of chosen samples
- `--k`
  - size of each selected group
- `--j`
  - size of each target subset
- `--s`
  - minimum overlap threshold

### Optional semantic arguments

- `--samples`
  - explicit sample list
  - example: `--samples 1,2,3,4,5,6,7,8,9`
- `--coverage-mode`
  - one of:
    - `at_least_one`
    - `at_least_r`
    - `all_subsets`
- `--aggregation-mode`
  - one of:
    - `distinct_subsets`
    - `single_candidate`
- `--r`
  - required when `--coverage-mode at_least_r`

### Optional solver arguments

- `--seed`
  - fixed random seed for reproducibility
- `--restarts`
  - number of heuristic restarts
- `--local-steps`
  - override local-search step budget
- `--sa-iterations`
  - override simulated-annealing iteration budget
- `--candidate-sample-size`
  - size of sampled candidate pool in local search
- `--disable-adaptive-neighborhoods`
  - turn off adaptive neighborhood scheduling

### Optional exact / polishing arguments

- `--disable-ilp`
  - skip exact ILP verification
- `--exact-backend`
  - choose one of:
    - `auto`
    - `scipy`
    - `gurobi`
    - `scip`
- `--exact-time-limit`
  - time limit in seconds for exact verification
- `--force-exact`
  - force exact verification even beyond default thresholds
- `--disable-reduced-exact-polish`
  - skip restricted-candidate exact polishing
- `--reduced-exact-time-limit`
  - time limit in seconds for reduced-core exact polishing
- `--reduced-exact-core-cap`
  - maximum candidate pool size for reduced-core exact polishing

### Optional result-management arguments

- `--save`
  - save result JSON to the result database
- `--db-dir`
  - choose the result database folder

### Example 1: basic solve

```bash
python -m optimal_samples_system solve --m 45 --n 9 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9 \
  --seed 42 --restarts 5
```

### Example 2: save result

```bash
python -m optimal_samples_system solve --m 45 --n 15 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --seed 42 --restarts 5 --save
```

### Example 3: `at_least_r`

```bash
python -m optimal_samples_system solve --m 45 --n 10 --k 6 --j 6 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9,10 \
  --coverage-mode at_least_r --r 4 \
  --aggregation-mode distinct_subsets \
  --seed 42 --restarts 5
```

## 5. Result Database Commands

Saved result files are stored in a result directory such as `results_db_v3/`.

### List saved results

```bash
python -m optimal_samples_system list-results --db-dir results_db_v3
```

Function:

- show all saved result JSON files
- show size, coverage mode, aggregation mode, validation status, timestamp

### Show one saved result

```bash
python -m optimal_samples_system show-result <filename> --db-dir results_db_v3
```

Function:

- show one result in full JSON form
- also show summary header:
  - file name
  - group count
  - validation summary
  - exact size if available

### Delete one saved result

```bash
python -m optimal_samples_system delete-result <filename> --db-dir results_db_v3
```

Function:

- remove one saved result file from the result database

## 6. Validation Commands

Validation checks whether a solution is feasible under the declared problem
definition.

### Validate one saved result

```bash
python -m optimal_samples_system validate-result <filename> --db-dir results_db_v3
```

Function:

- load one saved result
- run the primary validator
- print a validation report

### Validate one external JSON file

```bash
python -m optimal_samples_system validate-file /path/to/result.json
```

Function:

- validate a JSON file outside the result database

### Validate all saved results

```bash
python -m optimal_samples_system validate-results --db-dir results_db_v3
```

Function:

- validate every saved result JSON in the chosen result database

## 7. Audit Commands

Audit is stronger than ordinary validation because it runs:

- the primary tracker-based validator
- an independent brute-force validator

### Audit one saved result

```bash
python -m optimal_samples_system audit-result <filename> --db-dir results_db_v3
```

Function:

- check the result with both validators
- confirm whether both validators agree

### Audit one external JSON file

```bash
python -m optimal_samples_system audit-file /path/to/result.json
```

Function:

- run both validators on one standalone JSON file

### Audit all saved results

```bash
python -m optimal_samples_system audit-results --db-dir results_db_v3
```

Function:

- run both validators on every result in the database

## 8. Optimality Certification Commands

These commands go beyond feasibility and try to determine whether a result is
globally optimal.

### Certify one saved result

```bash
python -m optimal_samples_system certify-result <filename> \
  --db-dir results_db_v3 \
  --exact-backend auto \
  --exact-time-limit 300
```

Function:

- first audit the incumbent
- then try to prove no smaller feasible solution exists
- print a full certificate

Important note:

- success means the result is mathematically certified optimal
- failure does not necessarily mean the solution is wrong
- large instances may remain unresolved

### Summarize one saved result

```bash
python -m optimal_samples_system summarize-result <filename> \
  --db-dir results_db_v3 \
  --exact-backend auto \
  --exact-time-limit 300
```

Function:

- print a concise summary instead of the full certificate
- useful for presentations and large-instance reporting

### Certify one external JSON file

```bash
python -m optimal_samples_system certify-file /path/to/result.json \
  --exact-backend auto \
  --exact-time-limit 300
```

Function:

- run certification on a standalone result JSON file

### Summarize one external JSON file

```bash
python -m optimal_samples_system summarize-file /path/to/result.json \
  --exact-backend auto \
  --exact-time-limit 300
```

Function:

- print the concise optimality summary for an external JSON file

## 9. Prove-Bound Command

### Command

```bash
python -m optimal_samples_system prove-bound --m <M> --n <N> --k <K> --j <J> --s <S> \
  --target-size <B> [options]
```

### Required arguments

- `--m`
- `--n`
- `--k`
- `--j`
- `--s`
- `--target-size`

### Optional arguments

- `--samples`
- `--coverage-mode`
- `--aggregation-mode`
- `--r`
- `--seed`
- `--exact-backend`
- `--exact-time-limit`

### Example

```bash
python -m optimal_samples_system prove-bound --m 45 --n 15 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --target-size 48 \
  --exact-backend scip \
  --exact-time-limit 600
```

Function:

- test whether any feasible solution of size `<= target-size` exists

Possible interpretations:

- `infeasible`
  - there is no solution that small
- `feasible`
  - a solution that small does exist
- `unknown`
  - the backend could not finish the proof in time

## 10. Demo Command

### Command

```bash
python -m optimal_samples_system demo
```

Optional arguments:

- `--db-dir`
- `--seed`
- `--save`

Example:

```bash
python -m optimal_samples_system demo --save --db-dir results_db_v3
```

Function:

- run built-in assignment-style example instances

## 11. Mobile API Commands

The iOS app uses a local HTTP bridge.

### Start the mobile API directly

```bash
python -m optimal_samples_system.mobile_api --host 127.0.0.1 --port 8000
```

Arguments:

- `--host`
- `--port`
- `--log-level`

### Start the mobile API with the helper script

```bash
./scripts/start_mobile_api.sh
```

Or specify host and port:

```bash
./scripts/start_mobile_api.sh 127.0.0.1 8000
```

Function:

- start the backend service used by the iOS app

## 12. Helper Scripts

### Start backend for iOS app

```bash
./scripts/start_mobile_api.sh
```

### Open the iOS project in Xcode

```bash
./scripts/open_ios_project.sh
```

### Run a sample solve

```bash
./scripts/example_solve.sh
```

## 13. Common Workflows

### Workflow A: Solve, save, and list

```bash
python -m optimal_samples_system solve ... --save
python -m optimal_samples_system list-results --db-dir results_db_v3
```

### Workflow B: Show and audit one result

```bash
python -m optimal_samples_system show-result <filename> --db-dir results_db_v3
python -m optimal_samples_system audit-result <filename> --db-dir results_db_v3
```

### Workflow C: Check whether the result is proven optimal

```bash
python -m optimal_samples_system summarize-result <filename> --db-dir results_db_v3
python -m optimal_samples_system certify-result <filename> --db-dir results_db_v3
```

### Workflow D: Use the iOS app

```bash
./scripts/start_mobile_api.sh
./scripts/open_ios_project.sh
```

## 14. Output Files and Their Meanings

### Result JSON files

Typical location:

- `results_db_v3/`

Typical contents:

- problem parameters
- selected groups
- group count
- elapsed time
- exact information if available
- validation snapshot

### Important status fields

- `validation`
  - stored runtime validation metadata
- `exact_size`
  - exact optimum if exact solve succeeded
- `exact_method`
  - exact backend used

## 15. File Responsibility Map

### CLI and entry

- [optimal_samples_system/__main__.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/optimal_samples_system/__main__.py)
  - `python -m optimal_samples_system` entry
- [optimal_samples_system/cli.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/optimal_samples_system/cli.py)
  - command parsing and dispatch

### Core solving

- [optimal_samples_system/config.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/optimal_samples_system/config.py)
  - enums and shared configuration dataclasses
- [optimal_samples_system/instance.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/optimal_samples_system/instance.py)
  - problem instance construction
- [optimal_samples_system/tracking.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/optimal_samples_system/tracking.py)
  - incremental coverage tracking
- [optimal_samples_system/heuristics.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/optimal_samples_system/heuristics.py)
  - greedy, local search, simulated annealing, adaptive neighborhoods
- [optimal_samples_system/solver.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/optimal_samples_system/solver.py)
  - end-to-end solving pipeline

### Exact and proof tools

- [optimal_samples_system/exact.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/optimal_samples_system/exact.py)
  - exact backends, lower bounds, prove-bound support
- [optimal_samples_system/certify.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/optimal_samples_system/certify.py)
  - result certification and summary generation

### Validation and storage

- [optimal_samples_system/validation.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/optimal_samples_system/validation.py)
  - primary validation, independent validation, audit
- [optimal_samples_system/storage.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/optimal_samples_system/storage.py)
  - result saving, listing, showing, deleting

### iOS bridge

- [optimal_samples_system/mobile_api.py](/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System/optimal_samples_system/mobile_api.py)
  - local backend service for the iOS app

## 16. Exit-Status Notes

Some commands intentionally return a non-zero exit code when the result is not
successful.

Examples:

- `validate-*`
  - returns non-zero if validation fails
- `audit-*`
  - returns non-zero if either validator fails or they disagree
- `certify-*`
  - returns non-zero if the result is not certified optimal
- `summarize-*`
  - returns non-zero when the incumbent is invalid or already proven non-optimal

This behavior is useful for scripts and reproducible checking.

## 17. Fast Reference

Show help:

```bash
python -m optimal_samples_system --help
```

Solve:

```bash
python -m optimal_samples_system solve ...
```

Save and list:

```bash
python -m optimal_samples_system solve ... --save
python -m optimal_samples_system list-results --db-dir results_db_v3
```

Validate:

```bash
python -m optimal_samples_system validate-result <filename> --db-dir results_db_v3
```

Audit:

```bash
python -m optimal_samples_system audit-result <filename> --db-dir results_db_v3
```

Summarize:

```bash
python -m optimal_samples_system summarize-result <filename> --db-dir results_db_v3
```

Certify:

```bash
python -m optimal_samples_system certify-result <filename> --db-dir results_db_v3
```

Prove bound:

```bash
python -m optimal_samples_system prove-bound ...
```

Start iOS backend:

```bash
./scripts/start_mobile_api.sh
```
