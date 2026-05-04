# OptimalSelectionApp

This is a macOS SwiftUI wrapper for the noNN Python solver.

The app does not reimplement the optimization algorithm in Swift. Instead, it calls the existing Python CLI:

```bash
python -m optimal_samples_system ...
```

This keeps the algorithm, validator, Gurobi/SCIP exact verification, and result database behavior consistent with the command-line version.

## How To Open In Xcode

Open this project in Xcode:

```text
/Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System-noNN/OptimalSelectionApp/OptimalSelectionApp.xcodeproj
```

You can also run:

```bash
cd /Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System-noNN/OptimalSelectionApp
open OptimalSelectionApp.xcodeproj
```

Then select the `OptimalSelectionApp` scheme and click Run.

## Default Runtime Settings

The app defaults to:

```text
Python:  /opt/homebrew/anaconda3/bin/python
Project: /Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System-noNN
```

If your Python environment changes, update the Python field in the app.

## Integrated Features

The app integrates:

- Solve one instance.
- Save result JSON files using `--save`.
- List saved results.
- Show one saved result.
- Delete one saved result.
- Validate all saved results.
- Validate a specific JSON result file.
- Run `prove-bound` for exact/near-optimality proof.
- Run assignment demo cases.
- Select exact backend: `auto`, `gurobi`, `scip`, or `scipy`.
- Force exact verification on larger instances.

## Button Mapping

`Solve`

Runs:

```bash
python -m optimal_samples_system solve ...
```

`Prove Bound`

Runs:

```bash
python -m optimal_samples_system prove-bound ... --target-size <value>
```

`Validate All`

Runs:

```bash
python -m optimal_samples_system validate-results --db-dir <db-dir>
```

`List Results`

Runs:

```bash
python -m optimal_samples_system list-results --db-dir <db-dir>
```

`Show Result`

Runs:

```bash
python -m optimal_samples_system show-result <filename> --db-dir <db-dir>
```

`Validate File`

Runs:

```bash
python -m optimal_samples_system validate-file <path>
```

`Delete Result`

Runs:

```bash
python -m optimal_samples_system delete-result <filename> --db-dir <db-dir>
```

`Run Demo`

Runs:

```bash
python -m optimal_samples_system demo
```

## Correctness Notes

The Swift app is only a front-end. Correctness is still guaranteed by the Python solver:

- The solver checks `tracker.is_feasible()` before returning final results.
- The independent validator can verify saved outputs.
- `prove-bound` can use Gurobi/SCIP/SciPy for exact bound verification.

## Build Check

The app was compiled successfully with:

```bash
swift build
```

Observed result:

```text
Build complete!
```
