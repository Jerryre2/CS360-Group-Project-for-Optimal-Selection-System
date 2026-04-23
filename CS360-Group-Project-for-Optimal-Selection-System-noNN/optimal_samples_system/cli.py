"""Command-line interface."""

from __future__ import annotations

import argparse

from .config import (
    AggregationMode,
    CoverageMode,
    ProblemConfig,
    SolverConfig,
    configure_logging,
    parse_samples_arg,
)
from .solver import OptimalSamplesSolver
from .storage import ResultDatabase
from .exact import ILPSolver
from .validation import (
    format_validation_report,
    validate_database,
    validate_result_data,
    validate_result_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimal Samples Selection System")
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR",
    )

    subparsers = parser.add_subparsers(dest="command")

    solve_parser = subparsers.add_parser("solve", help="Solve one instance")
    solve_parser.add_argument("--m", type=int, required=True)
    solve_parser.add_argument("--n", type=int, required=True)
    solve_parser.add_argument("--k", type=int, required=True)
    solve_parser.add_argument("--j", type=int, required=True)
    solve_parser.add_argument("--s", type=int, required=True)
    solve_parser.add_argument(
        "--samples",
        type=parse_samples_arg,
        default=None,
        help="Manual sample list such as 1,4,7,9,13",
    )
    solve_parser.add_argument(
        "--coverage-mode",
        choices=[mode.value for mode in CoverageMode],
        default=CoverageMode.AT_LEAST_ONE.value,
    )
    solve_parser.add_argument(
        "--aggregation-mode",
        choices=[mode.value for mode in AggregationMode],
        default=AggregationMode.DISTINCT_SUBSETS.value,
    )
    solve_parser.add_argument(
        "--r",
        type=int,
        default=None,
        help="Required distinct s-subset count for at_least_r.",
    )
    solve_parser.add_argument("--seed", type=int, default=None)
    solve_parser.add_argument("--restarts", type=int, default=3)
    solve_parser.add_argument("--local-steps", type=int, default=None)
    solve_parser.add_argument("--sa-iterations", type=int, default=None)
    solve_parser.add_argument("--candidate-sample-size", type=int, default=48)
    solve_parser.add_argument("--db-dir", default="results_db_v3")
    solve_parser.add_argument("--save", action="store_true")
    solve_parser.add_argument("--disable-ilp", action="store_true")
    solve_parser.add_argument(
        "--exact-backend",
        choices=["auto", "scipy", "gurobi", "scip"],
        default="auto",
        help="Exact solver backend for ILP verification.",
    )
    solve_parser.add_argument(
        "--exact-time-limit",
        type=int,
        default=60,
        help="Time limit in seconds for exact verification.",
    )
    solve_parser.add_argument(
        "--force-exact",
        action="store_true",
        help="Run exact verification even when the instance exceeds default limits.",
    )
    solve_parser.add_argument(
        "--disable-neural-guidance",
        action="store_true",
        help="Deprecated no-op kept only for old noNN scripts.",
    )

    list_parser = subparsers.add_parser("list-results", help="List saved results")
    list_parser.add_argument("--db-dir", default="results_db_v3")

    show_parser = subparsers.add_parser("show-result", help="Show one saved result")
    show_parser.add_argument("filename")
    show_parser.add_argument("--db-dir", default="results_db_v3")

    delete_parser = subparsers.add_parser("delete-result", help="Delete one saved result")
    delete_parser.add_argument("filename")
    delete_parser.add_argument("--db-dir", default="results_db_v3")

    validate_result_parser = subparsers.add_parser(
        "validate-result", help="Validate one saved result from the result database"
    )
    validate_result_parser.add_argument("filename")
    validate_result_parser.add_argument("--db-dir", default="results_db_v3")

    validate_file_parser = subparsers.add_parser(
        "validate-file", help="Validate one result JSON file"
    )
    validate_file_parser.add_argument("path")

    validate_results_parser = subparsers.add_parser(
        "validate-results", help="Validate all saved result JSON files"
    )
    validate_results_parser.add_argument("--db-dir", default="results_db_v3")

    prove_parser = subparsers.add_parser(
        "prove-bound",
        help=(
            "Try to prove whether a solution exists at or below a target size. "
            "If infeasible, a known incumbent one larger is proven optimal."
        ),
    )
    prove_parser.add_argument("--m", type=int, required=True)
    prove_parser.add_argument("--n", type=int, required=True)
    prove_parser.add_argument("--k", type=int, required=True)
    prove_parser.add_argument("--j", type=int, required=True)
    prove_parser.add_argument("--s", type=int, required=True)
    prove_parser.add_argument("--samples", type=parse_samples_arg, default=None)
    prove_parser.add_argument(
        "--coverage-mode",
        choices=[mode.value for mode in CoverageMode],
        default=CoverageMode.AT_LEAST_ONE.value,
    )
    prove_parser.add_argument(
        "--aggregation-mode",
        choices=[mode.value for mode in AggregationMode],
        default=AggregationMode.DISTINCT_SUBSETS.value,
    )
    prove_parser.add_argument("--r", type=int, default=None)
    prove_parser.add_argument("--seed", type=int, default=None)
    prove_parser.add_argument("--target-size", type=int, required=True)
    prove_parser.add_argument(
        "--exact-backend",
        choices=["auto", "scipy", "gurobi", "scip"],
        default="auto",
    )
    prove_parser.add_argument("--exact-time-limit", type=int, default=300)

    demo_parser = subparsers.add_parser("demo", help="Run assignment-style demos")
    demo_parser.add_argument("--db-dir", default="results_db_v3")
    demo_parser.add_argument("--seed", type=int, default=20260415)
    demo_parser.add_argument("--save", action="store_true")

    return parser


def run_demo(args: argparse.Namespace) -> None:
    demos = [
        ProblemConfig(
            m=45,
            n=7,
            k=6,
            j=5,
            s=5,
            coverage_mode=CoverageMode.AT_LEAST_ONE,
            aggregation_mode=AggregationMode.DISTINCT_SUBSETS,
            seed=args.seed,
        ),
        ProblemConfig(
            m=45,
            n=9,
            k=6,
            j=5,
            s=4,
            coverage_mode=CoverageMode.AT_LEAST_ONE,
            aggregation_mode=AggregationMode.DISTINCT_SUBSETS,
            seed=args.seed + 1,
        ),
        ProblemConfig(
            m=45,
            n=10,
            k=6,
            j=5,
            s=4,
            coverage_mode=CoverageMode.AT_LEAST_R,
            aggregation_mode=AggregationMode.DISTINCT_SUBSETS,
            required_r=4,
            seed=args.seed + 2,
        ),
    ]

    for config in demos:
        solver = OptimalSamplesSolver(config)
        solver.solve(
            SolverConfig(
                n_restarts=2,
                use_ilp=True,
                save_result=args.save,
                db_dir=args.db_dir,
            )
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    if args.command is None:
        parser.print_help()
        return

    if args.command == "solve":
        config = ProblemConfig(
            m=args.m,
            n=args.n,
            k=args.k,
            j=args.j,
            s=args.s,
            samples=args.samples,
            coverage_mode=CoverageMode(args.coverage_mode),
            aggregation_mode=AggregationMode(args.aggregation_mode),
            required_r=args.r,
            seed=args.seed,
        )
        solver = OptimalSamplesSolver(config)
        solver.solve(
            SolverConfig(
                n_restarts=args.restarts,
                use_ilp=not args.disable_ilp,
                exact_backend=args.exact_backend,
                exact_time_limit=args.exact_time_limit,
                force_exact=args.force_exact,
                max_local_steps=args.local_steps,
                max_sa_iterations=args.sa_iterations,
                candidate_sample_size=args.candidate_sample_size,
                use_neural_guidance=False,
                save_result=args.save,
                db_dir=args.db_dir,
            )
        )
        return

    if args.command == "list-results":
        ResultDatabase(args.db_dir).print_all()
        return

    if args.command == "show-result":
        ResultDatabase(args.db_dir).print_result(args.filename)
        return

    if args.command == "delete-result":
        ResultDatabase(args.db_dir).delete(args.filename)
        return

    if args.command == "validate-result":
        data = ResultDatabase(args.db_dir).load(args.filename)
        report = validate_result_data(data, source=args.filename)
        print(format_validation_report(report))
        if not report.is_valid:
            raise SystemExit(1)
        return

    if args.command == "validate-file":
        report = validate_result_file(args.path)
        print(format_validation_report(report))
        if not report.is_valid:
            raise SystemExit(1)
        return

    if args.command == "validate-results":
        reports = validate_database(args.db_dir)
        if not reports:
            print("No saved result JSON files found.")
            return
        for report in reports:
            print(format_validation_report(report))
        if any(not report.is_valid for report in reports):
            raise SystemExit(1)
        return

    if args.command == "prove-bound":
        config = ProblemConfig(
            m=args.m,
            n=args.n,
            k=args.k,
            j=args.j,
            s=args.s,
            samples=args.samples,
            coverage_mode=CoverageMode(args.coverage_mode),
            aggregation_mode=AggregationMode(args.aggregation_mode),
            required_r=args.r,
            seed=args.seed,
        )
        solver = OptimalSamplesSolver(config)
        result = ILPSolver.prove_no_solution_at_or_below(
            solver.instance,
            cardinality_limit=args.target_size,
            backend=args.exact_backend,
            time_limit=args.exact_time_limit,
        )
        print(f"Proof target: no feasible solution with <= {args.target_size} groups")
        print(f"Status: {result.status}")
        print(f"Method: {result.method}")
        print(f"Message: {result.message}")
        if result.solution is not None:
            print(f"Found feasible solution size: {len(result.solution)}")
        if result.status == "infeasible":
            print(
                "Certificate: infeasible at this target size. "
                "A feasible incumbent of target_size + 1 would be globally optimal."
            )
        elif result.status == "unknown":
            raise SystemExit(1)
        return

    if args.command == "demo":
        run_demo(args)
        return

    parser.error(f"Unknown command: {args.command}")
