# User Guideline

This document describes the **final noNN submission version** of the Optimal
Sample Selection System. All usage instructions below refer to the
deterministic noNN pipeline that is currently shipped in the repository.

## 1. Purpose of the System

This project solves optimal sample selection problems modeled as covering-design
instances.

Given parameters `m, n, k, j, s`, the system:

- chooses `n` samples from a universe of size `m`
- considers all `k`-sized candidate groups from those `n` samples
- searches for the smallest family of candidate groups that satisfies the
  required coverage condition over all `j`-sample targets

The system supports:

- heuristic solving for large instances
- exact verification for small and medium instances
- rigorous feasibility validation
- independent auditing of saved results
- optimality certification or lower-bound reporting
- an iOS front end built in Xcode

The final submitted solver path does **not** rely on a neural network. Instead,
it uses structure-aware search, adaptive neighborhoods, elite-guided restarts,
path relinking, and exact post-processing to improve solution quality while
preserving correctness.

## 2. Submission Contents

Important folders and files:

- `optimal_samples_system/`
  - main Python package for the final noNN solver
  - solver, exact backend interface, validation, storage, CLI, mobile API
- `OptimalSelectionMobile/`
  - iOS app project
- `tests/`
  - regression tests for solver, validation, storage, certification
- `docs/USER_GUIDELINE.md`
  - this document
- `README.md`
  - concise overview and quick start
- `START_HERE.md`
  - entry document for a USB submission copy
- `FUTURE_WORK.md`
  - future improvement directions

## 3. System Requirements

### Core CLI System

- Python 3.11 or later recommended
- no mandatory third-party dependency for the core noNN heuristic + validation path

### Optional Exact Backends

- SciPy / HiGHS
- Gurobi
- SCIP

These are optional. If they are missing, the solver still runs, but exact
proofs may fall back or become unavailable.

### iOS App

- macOS
- Xcode
- iOS Simulator or an iPhone device

The iOS app communicates with a local Python API server, so the Python backend
must be started before the app can use solver features.

## 4. Core Concepts

### Coverage Modes

The system supports three coverage semantics:

- `at_least_one`
  - each `j`-target must be hit by at least one `s`-subset contribution
- `at_least_r`
  - each `j`-target must receive at least `r` distinct `s`-subset contributions
- `all_subsets`
  - each `j`-target must receive all `C(j, s)` subset contributions

### Aggregation Modes

- `distinct_subsets`
  - different selected groups may contribute different required `s`-subsets
- `single_candidate`
  - one selected candidate must satisfy the aggregation condition by itself

### Important Result Statuses

- `validated`
  - saved result already carries a successful runtime validation snapshot
- `invalid`
  - validation failed
- `mismatch`
  - the primary and independent validators disagree
- `unverified`
  - no validation snapshot is stored yet

### Important Optimality Statuses

- `已证明最优`
  - globally optimal
- `已证明非最优`
  - a smaller feasible solution exists
- `未证明`
  - current solution is feasible, but optimality is not yet proven
- `当前解无效`
  - the incumbent failed validation, so proof cannot proceed

## 5. First-Time Quick Start

Open Terminal, move into the repository root, and check the CLI:

```bash
cd /path/to/CS360-Group-Project-for-Optimal-Selection-System
python -m optimal_samples_system --help
```

If the help screen appears, the core system is ready.

## 6. Solving a Problem Instance

### Basic Solve Command

```bash
python -m optimal_samples_system solve --m 45 --n 9 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9 \
  --seed 42 --restarts 5
```

### Meaning of Main Parameters

- `m`
  - size of the full universe
- `n`
  - number of chosen samples
- `k`
  - size of each candidate group
- `j`
  - size of each target subset
- `s`
  - required overlap threshold
- `samples`
  - explicit sample labels used in this run
- `seed`
  - random seed for reproducibility
- `restarts`
  - number of multi-start heuristic attempts

### Choosing a Coverage Mode

`at_least_one`:

```bash
python -m optimal_samples_system solve ... \
  --coverage-mode at_least_one
```

`at_least_r` with `r=4`:

```bash
python -m optimal_samples_system solve ... \
  --coverage-mode at_least_r --r 4
```

`all_subsets`:

```bash
python -m optimal_samples_system solve ... \
  --coverage-mode all_subsets
```

### Choosing an Aggregation Mode

Distinct subset accumulation:

```bash
--aggregation-mode distinct_subsets
```

Single-candidate semantics:

```bash
--aggregation-mode single_candidate
```

### Advanced Solve Settings

Useful optional arguments:

- `--local-steps`
- `--sa-iterations`
- `--candidate-sample-size`
- `--disable-adaptive-neighborhoods`
- `--disable-ilp`
- `--exact-backend auto|scipy|gurobi|scip`
- `--exact-time-limit`
- `--force-exact`
- `--disable-reduced-exact-polish`
- `--reduced-exact-time-limit`
- `--reduced-exact-core-cap`

In the final noNN version, these controls influence deterministic search and
exact post-processing only. There is no neural-guidance toggle in the shipped
system.

## 7. Saving and Managing Results

### Save a Result

```bash
python -m optimal_samples_system solve ... --save
```

Saved results are placed in `results_db_v3/` by default.

### Use a Custom Result Directory

```bash
python -m optimal_samples_system solve ... --save --db-dir my_results
```

### List All Saved Results

```bash
python -m optimal_samples_system list-results --db-dir results_db_v3
```

The listing now includes:

- file name
- solution size
- coverage / aggregation modes
- exact result if available
- validation status
- timestamp

### Show One Saved Result

```bash
python -m optimal_samples_system show-result <filename> --db-dir results_db_v3
```

### Delete One Saved Result

```bash
python -m optimal_samples_system delete-result <filename> --db-dir results_db_v3
```

## 8. Validation and Audit

The project provides two levels of correctness checking.

### 8.1 Primary Validation

Validate a result file from the database:

```bash
python -m optimal_samples_system validate-result <filename> --db-dir results_db_v3
```

Validate a standalone JSON file:

```bash
python -m optimal_samples_system validate-file /path/to/result.json
```

Validate all saved results:

```bash
python -m optimal_samples_system validate-results --db-dir results_db_v3
```

### 8.2 Independent Audit

The audit path runs:

- the primary tracker-based validator
- an independent brute-force validator

Audit one saved result:

```bash
python -m optimal_samples_system audit-result <filename> --db-dir results_db_v3
```

Audit one JSON file:

```bash
python -m optimal_samples_system audit-file /path/to/result.json
```

Audit all saved results:

```bash
python -m optimal_samples_system audit-results --db-dir results_db_v3
```

### Why Audit Matters

Validation answers:

- "Is this result feasible under the encoded problem definition?"

Audit answers:

- "Do both validation mechanisms agree that the result is feasible?"

This distinction is useful in presentations and final reporting.

## 9. Proving Optimality or Near-Optimality

### 9.1 Prove a Bound

Use `prove-bound` to ask:

- does a solution of size `<= B` exist?

Example:

```bash
python -m optimal_samples_system prove-bound --m 45 --n 15 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --target-size 48 --exact-backend auto --exact-time-limit 600
```

Possible results:

- `infeasible`
  - no solution exists at or below that size
- `feasible`
  - such a solution exists
- `unknown`
  - the exact backend could not finish in time

### 9.2 Certify a Saved Result

```bash
python -m optimal_samples_system certify-result <filename> --db-dir results_db_v3
```

This command:

1. audits the incumbent result
2. tries to prove that no smaller feasible solution exists
3. reports one of four statuses

Possible status meanings:

- `certified_optimal`
- `not_optimal`
- `unresolved`
- `invalid_incumbent`

### 9.3 Summarize a Saved Result

```bash
python -m optimal_samples_system summarize-result <filename> --db-dir results_db_v3
```

This is the most presentation-friendly command for large instances because it
shows, in one place:

- whether the result is validated
- whether it is proven optimal
- the certified lower bound
- the current optimality gap upper bound

### 9.4 Large-Instance Interpretation

For large instances, a result may be:

- feasible and audited
- not yet proven globally optimal
- still accompanied by a rigorous lower bound

That means the system can still provide meaningful correctness evidence even
when a full exact proof is too expensive.

## 10. Built-In Demo Command

The repository includes a demo mode:

```bash
python -m optimal_samples_system demo
```

Optionally save demo outputs:

```bash
python -m optimal_samples_system demo --save
```

## 11. iOS App Workflow

### 11.1 Start the Backend API

From the repository root:

```bash
./scripts/start_mobile_api.sh
```

Equivalent direct command:

```bash
python -m optimal_samples_system.mobile_api --host 127.0.0.1 --port 8000
```

### 11.2 Open the Xcode Project

```bash
open OptimalSelectionMobile/OptimalSelectionMobile.xcodeproj
```

### 11.3 Run the App

In Xcode:

1. choose an iPhone Simulator
2. click Run
3. keep the local API server running in Terminal

### 11.4 What the App Can Do

The iOS app currently integrates:

- solve
- proof / certification
- validation
- results browsing

It exposes the same major backend capabilities through a mobile-friendly UI.

## 12. Helper Scripts

### Start the Mobile API

```bash
./scripts/start_mobile_api.sh
```

### Open the iOS Project

```bash
./scripts/open_ios_project.sh
```

### Run an Example Solve

```bash
./scripts/example_solve.sh
```

## 13. Recommended Usage Workflows

### Workflow A: Solve and Save

```bash
python -m optimal_samples_system solve ... --save
python -m optimal_samples_system list-results --db-dir results_db_v3
```

### Workflow B: Validate and Audit a Saved Result

```bash
python -m optimal_samples_system validate-result <filename> --db-dir results_db_v3
python -m optimal_samples_system audit-result <filename> --db-dir results_db_v3
```

### Workflow C: Check Whether a Result Is Proven Optimal

```bash
python -m optimal_samples_system summarize-result <filename> --db-dir results_db_v3
python -m optimal_samples_system certify-result <filename> --db-dir results_db_v3
```

### Workflow D: Use the iOS App

```bash
./scripts/start_mobile_api.sh
open OptimalSelectionMobile/OptimalSelectionMobile.xcodeproj
```

## 14. Troubleshooting

### Problem: `python -m optimal_samples_system --help` fails

Check:

- you are in the repository root
- your Python version is recent enough
- you are using the correct interpreter

### Problem: a saved result becomes `invalid`

Possible causes:

- the JSON file was manually edited
- required semantic fields are missing
- the result does not match the declared parameters

The validator is fail-closed on missing semantic fields, which is intentional.

### Problem: `certify-result` says `unresolved`

This does not mean the solution is wrong.
It means:

- the incumbent is feasible
- the exact backend did not finish proving global optimality within the time limit

In this case, use the reported lower bound and gap information.

### Problem: the iOS app cannot connect

Check:

- the backend API is running
- the app is using the correct API base URL
- if using a real device, the phone and the Mac are on the same network

### Problem: Gurobi academic license setup fails

This is usually a network or academic-domain issue, not a solver-code issue.
Use the system with `scip` or heuristic mode if Gurobi is unavailable.

## 15. Suggested Demonstration Order for a Teacher

If demonstrating the system live, a good order is:

1. run a small `solve`
2. save the result
3. run `list-results`
4. run `audit-result`
5. run `summarize-result`
6. show the same flow in the iOS app

## 16. Final Notes

This repository is designed to make a clear distinction between:

- obtaining a good solution
- validating that the solution is correct
- proving whether the solution is globally optimal

That separation is one of the main engineering strengths of the system.
