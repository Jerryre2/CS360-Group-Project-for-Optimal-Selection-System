# Start Here

This file is the fastest way to understand how to use the submitted system.

This USB/repository copy corresponds to the **final noNN submission version**.
The solver and app both use the deterministic noNN pipeline rather than the
earlier experimental neural-guidance branch.

## What This Project Contains

This submission includes:

- a Python noNN solver for optimal sample selection / covering-design instances
- full validation and audit tools
- optimality and near-optimality proof tools
- an iOS app project built with Xcode

## If You Only Want to Run the Core System

Open Terminal in the repository root and run:

```bash
python -m optimal_samples_system --help
```

Typical first command:

```bash
python -m optimal_samples_system solve --m 45 --n 9 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9 --seed 42 --restarts 5
```

## If You Want the iOS App

1. Start the local backend:

```bash
./scripts/start_mobile_api.sh
```

2. Open the Xcode project:

```bash
open OptimalSelectionMobile/OptimalSelectionMobile.xcodeproj
```

3. Run the app in the iOS Simulator.

## Most Important Documents

- [README.md](README.md): project overview and quick start
- [COMMAND_LINE_GUIDE.md](COMMAND_LINE_GUIDE.md): standalone command-line guide
- [docs/USER_GUIDELINE.md](docs/USER_GUIDELINE.md): complete user manual
- [FUTURE_WORK.md](FUTURE_WORK.md): future research/engineering directions

## Most Important Commands

Solve an instance:

```bash
python -m optimal_samples_system solve ...
```

Save a result:

```bash
python -m optimal_samples_system solve ... --save
```

List saved results:

```bash
python -m optimal_samples_system list-results --db-dir results_db_v3
```

Run both validators:

```bash
python -m optimal_samples_system audit-result <filename> --db-dir results_db_v3
```

Check optimality summary:

```bash
python -m optimal_samples_system summarize-result <filename> --db-dir results_db_v3
```

## Recommended Reading Order

1. Read [README.md](README.md)
2. Read [docs/USER_GUIDELINE.md](docs/USER_GUIDELINE.md)
3. Run `python -m optimal_samples_system --help`
4. Try one `solve` command
5. If needed, open the iOS app in Xcode
