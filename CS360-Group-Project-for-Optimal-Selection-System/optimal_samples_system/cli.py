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
    solve_parser.add_argument("--disable-neural-guidance", action="store_true")

    list_parser = subparsers.add_parser("list-results", help="List saved results")
    list_parser.add_argument("--db-dir", default="results_db_v3")

    show_parser = subparsers.add_parser("show-result", help="Show one saved result")
    show_parser.add_argument("filename")
    show_parser.add_argument("--db-dir", default="results_db_v3")

    delete_parser = subparsers.add_parser("delete-result", help="Delete one saved result")
    delete_parser.add_argument("filename")
    delete_parser.add_argument("--db-dir", default="results_db_v3")

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
                max_local_steps=args.local_steps,
                max_sa_iterations=args.sa_iterations,
                candidate_sample_size=args.candidate_sample_size,
                use_neural_guidance=not args.disable_neural_guidance,
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

    if args.command == "demo":
        run_demo(args)
        return

    parser.error(f"Unknown command: {args.command}")
