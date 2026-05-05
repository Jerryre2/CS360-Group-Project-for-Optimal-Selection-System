"""Command-line interface."""

from __future__ import annotations

import argparse

from .certify import (
    certify_result_data,
    format_optimality_certificate,
    format_optimality_summary,
)
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
    audit_database,
    audit_result_data,
    audit_result_file,
    format_validation_audit_report,
    format_validation_report,
    validate_database,
    validate_result_data,
    validate_result_file,
)


def build_parser() -> argparse.ArgumentParser:
    solver_defaults = SolverConfig()
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
    solve_parser.add_argument("--restarts", type=int, default=solver_defaults.n_restarts)
    solve_parser.add_argument(
        "--elite-pool-size",
        type=int,
        default=solver_defaults.elite_pool_size,
    )
    solve_parser.add_argument(
        "--disable-elite-guided-restarts",
        action="store_true",
        help="Disable elite-guided restart seeding.",
    )
    solve_parser.add_argument(
        "--disable-path-relinking",
        action="store_true",
        help="Disable path relinking between restart seeds and elite solutions.",
    )
    solve_parser.add_argument("--local-steps", type=int, default=None)
    solve_parser.add_argument("--sa-iterations", type=int, default=None)
    solve_parser.add_argument(
        "--candidate-sample-size",
        type=int,
        default=solver_defaults.candidate_sample_size,
    )
    solve_parser.add_argument(
        "--large-instance-threshold",
        type=int,
        default=solver_defaults.large_instance_candidate_threshold,
        help="Candidate-count threshold above which the solver scales LS/SA budgets.",
    )
    solve_parser.add_argument(
        "--large-instance-min-local-steps",
        type=int,
        default=solver_defaults.large_instance_min_local_steps,
        help="Minimum LS budget to keep when scaling large instances.",
    )
    solve_parser.add_argument(
        "--large-instance-min-sa-iterations",
        type=int,
        default=solver_defaults.large_instance_min_sa_iterations,
        help="Minimum SA budget to keep when scaling large instances.",
    )
    solve_parser.add_argument(
        "--disable-adaptive-neighborhoods",
        action="store_true",
        help="Use the older fixed local-search / SA neighborhood mix.",
    )
    solve_parser.add_argument("--db-dir", default=solver_defaults.db_dir)
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
        default=solver_defaults.exact_time_limit,
        help="Time limit in seconds for exact verification.",
    )
    solve_parser.add_argument(
        "--force-exact",
        action="store_true",
        help="Run exact verification even when the instance exceeds default limits.",
    )
    solve_parser.add_argument(
        "--disable-mid-size-exact-improvement",
        action="store_true",
        help="Skip the time-limited full-model exact improvement phase for mid-size instances.",
    )
    solve_parser.add_argument(
        "--mid-size-exact-threshold",
        type=int,
        default=solver_defaults.mid_size_exact_candidate_threshold,
        help="Maximum candidate count for the mid-size exact improvement phase.",
    )
    solve_parser.add_argument(
        "--mid-size-exact-time-limit",
        type=int,
        default=solver_defaults.mid_size_exact_time_limit,
        help="Time limit in seconds for the mid-size exact improvement phase.",
    )
    solve_parser.add_argument(
        "--disable-local-branching-improvement",
        action="store_true",
        help="Skip local-branching exact improvement around the incumbent.",
    )
    solve_parser.add_argument(
        "--local-branching-threshold",
        type=int,
        default=solver_defaults.local_branching_candidate_threshold,
        help="Maximum candidate count for local-branching exact improvement.",
    )
    solve_parser.add_argument(
        "--local-branching-time-limit",
        type=int,
        default=solver_defaults.local_branching_time_limit,
        help="Time limit in seconds for each local-branching exact improvement round.",
    )
    solve_parser.add_argument(
        "--local-branching-base-radius",
        type=int,
        default=solver_defaults.local_branching_base_radius,
        help="Initial Hamming-distance radius for local-branching exact improvement.",
    )
    solve_parser.add_argument(
        "--local-branching-rounds",
        type=int,
        default=solver_defaults.local_branching_rounds,
        help="Number of local-branching exact improvement radii to try.",
    )
    solve_parser.add_argument(
        "--disable-lp-guided-exact-polish",
        action="store_true",
        help="Skip the LP-guided restricted exact polishing phase.",
    )
    solve_parser.add_argument(
        "--lp-guided-threshold",
        type=int,
        default=solver_defaults.lp_guided_candidate_threshold,
        help="Maximum candidate count for LP-guided restricted exact polishing.",
    )
    solve_parser.add_argument(
        "--lp-guided-time-limit",
        type=int,
        default=solver_defaults.lp_guided_time_limit,
        help="Time limit in seconds for each LP-guided restricted exact polishing round.",
    )
    solve_parser.add_argument(
        "--lp-guided-core-cap",
        type=int,
        default=solver_defaults.lp_guided_core_cap,
        help="Maximum candidate pool size for LP-guided restricted exact polishing.",
    )
    solve_parser.add_argument(
        "--lp-guided-rounds",
        type=int,
        default=solver_defaults.lp_guided_rounds,
        help="Number of LP-guided restricted exact polishing rounds.",
    )
    solve_parser.add_argument(
        "--disable-cluster-exact-repair",
        action="store_true",
        help="Skip cluster-based exact destroy-and-repair improvement.",
    )
    solve_parser.add_argument(
        "--cluster-exact-threshold",
        type=int,
        default=solver_defaults.cluster_exact_candidate_threshold,
        help="Maximum candidate count for cluster-based exact destroy-and-repair improvement.",
    )
    solve_parser.add_argument(
        "--cluster-exact-time-limit",
        type=int,
        default=solver_defaults.cluster_exact_time_limit,
        help="Time limit in seconds for each cluster-based exact destroy-and-repair round.",
    )
    solve_parser.add_argument(
        "--cluster-exact-core-cap",
        type=int,
        default=solver_defaults.cluster_exact_core_cap,
        help="Maximum candidate pool size for cluster-based exact destroy-and-repair cores.",
    )
    solve_parser.add_argument(
        "--cluster-exact-rounds",
        type=int,
        default=solver_defaults.cluster_exact_rounds,
        help="Number of cluster-based exact destroy-and-repair rounds.",
    )
    solve_parser.add_argument(
        "--cluster-exact-destroy-size",
        type=int,
        default=solver_defaults.cluster_exact_destroy_size,
        help="Target cluster size removed before exact repair.",
    )
    solve_parser.add_argument(
        "--disable-cardinality-descent",
        action="store_true",
        help="Skip fixed-cardinality deficit-descent improvement.",
    )
    solve_parser.add_argument(
        "--cardinality-descent-threshold",
        type=int,
        default=solver_defaults.cardinality_descent_candidate_threshold,
        help="Maximum candidate count for fixed-cardinality deficit descent.",
    )
    solve_parser.add_argument(
        "--cardinality-descent-iterations",
        type=int,
        default=solver_defaults.cardinality_descent_iterations,
        help="Iteration budget for each fixed-cardinality deficit-descent round.",
    )
    solve_parser.add_argument(
        "--cardinality-descent-patience",
        type=int,
        default=solver_defaults.cardinality_descent_patience,
        help="No-improvement patience for each fixed-cardinality deficit-descent round.",
    )
    solve_parser.add_argument(
        "--cardinality-descent-sample-size",
        type=int,
        default=solver_defaults.cardinality_descent_sample_size,
        help="Candidate sampling width for fixed-cardinality deficit descent.",
    )
    solve_parser.add_argument(
        "--cardinality-descent-initial-drop",
        type=int,
        default=solver_defaults.cardinality_descent_initial_drop,
        help="Initial group-count drop to target in fixed-cardinality deficit descent.",
    )
    solve_parser.add_argument(
        "--cardinality-descent-rounds",
        type=int,
        default=solver_defaults.cardinality_descent_max_rounds,
        help="Maximum number of fixed-cardinality deficit-descent rounds.",
    )
    solve_parser.add_argument(
        "--disable-reduced-exact-polish",
        action="store_true",
        help="Skip the restricted-candidate exact post-processing stage.",
    )
    solve_parser.add_argument(
        "--reduced-exact-time-limit",
        type=int,
        default=solver_defaults.reduced_exact_time_limit,
        help="Time limit in seconds for reduced-core exact polishing.",
    )
    solve_parser.add_argument(
        "--reduced-exact-core-cap",
        type=int,
        default=solver_defaults.reduced_exact_core_cap,
        help="Maximum candidate pool size for reduced-core exact polishing.",
    )
    solve_parser.add_argument(
        "--reduced-exact-rounds",
        type=int,
        default=solver_defaults.reduced_exact_rounds,
        help="Number of reduced-core exact polishing rounds.",
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

    audit_result_parser = subparsers.add_parser(
        "audit-result",
        help="Run both validators on one saved result from the result database",
    )
    audit_result_parser.add_argument("filename")
    audit_result_parser.add_argument("--db-dir", default="results_db_v3")

    audit_file_parser = subparsers.add_parser(
        "audit-file", help="Run both validators on one result JSON file"
    )
    audit_file_parser.add_argument("path")

    audit_results_parser = subparsers.add_parser(
        "audit-results", help="Run both validators on all saved result JSON files"
    )
    audit_results_parser.add_argument("--db-dir", default="results_db_v3")

    certify_result_parser = subparsers.add_parser(
        "certify-result",
        help="Try to certify one saved result as globally optimal",
    )
    certify_result_parser.add_argument("filename")
    certify_result_parser.add_argument("--db-dir", default="results_db_v3")
    certify_result_parser.add_argument(
        "--exact-backend",
        choices=["auto", "scipy", "gurobi", "scip"],
        default="auto",
    )
    certify_result_parser.add_argument("--exact-time-limit", type=int, default=300)

    summarize_result_parser = subparsers.add_parser(
        "summarize-result",
        help="Show a concise optimality summary for one saved result",
    )
    summarize_result_parser.add_argument("filename")
    summarize_result_parser.add_argument("--db-dir", default="results_db_v3")
    summarize_result_parser.add_argument(
        "--exact-backend",
        choices=["auto", "scipy", "gurobi", "scip"],
        default="auto",
    )
    summarize_result_parser.add_argument("--exact-time-limit", type=int, default=300)

    certify_file_parser = subparsers.add_parser(
        "certify-file",
        help="Try to certify one result JSON file as globally optimal",
    )
    certify_file_parser.add_argument("path")
    certify_file_parser.add_argument(
        "--exact-backend",
        choices=["auto", "scipy", "gurobi", "scip"],
        default="auto",
    )
    certify_file_parser.add_argument("--exact-time-limit", type=int, default=300)

    summarize_file_parser = subparsers.add_parser(
        "summarize-file",
        help="Show a concise optimality summary for one result JSON file",
    )
    summarize_file_parser.add_argument("path")
    summarize_file_parser.add_argument(
        "--exact-backend",
        choices=["auto", "scipy", "gurobi", "scip"],
        default="auto",
    )
    summarize_file_parser.add_argument("--exact-time-limit", type=int, default=300)

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
        solver_config = SolverConfig(
            n_restarts=args.restarts,
            elite_pool_size=args.elite_pool_size,
            exact_backend=args.exact_backend,
            exact_time_limit=args.exact_time_limit,
            force_exact=args.force_exact,
            mid_size_exact_candidate_threshold=args.mid_size_exact_threshold,
            mid_size_exact_time_limit=args.mid_size_exact_time_limit,
            local_branching_candidate_threshold=args.local_branching_threshold,
            local_branching_time_limit=args.local_branching_time_limit,
            local_branching_base_radius=args.local_branching_base_radius,
            local_branching_rounds=args.local_branching_rounds,
            lp_guided_candidate_threshold=args.lp_guided_threshold,
            lp_guided_time_limit=args.lp_guided_time_limit,
            lp_guided_core_cap=args.lp_guided_core_cap,
            lp_guided_rounds=args.lp_guided_rounds,
            cluster_exact_candidate_threshold=args.cluster_exact_threshold,
            cluster_exact_time_limit=args.cluster_exact_time_limit,
            cluster_exact_core_cap=args.cluster_exact_core_cap,
            cluster_exact_rounds=args.cluster_exact_rounds,
            cluster_exact_destroy_size=args.cluster_exact_destroy_size,
            cardinality_descent_candidate_threshold=args.cardinality_descent_threshold,
            cardinality_descent_iterations=args.cardinality_descent_iterations,
            cardinality_descent_patience=args.cardinality_descent_patience,
            cardinality_descent_sample_size=args.cardinality_descent_sample_size,
            cardinality_descent_initial_drop=args.cardinality_descent_initial_drop,
            cardinality_descent_max_rounds=args.cardinality_descent_rounds,
            max_local_steps=args.local_steps,
            max_sa_iterations=args.sa_iterations,
            candidate_sample_size=args.candidate_sample_size,
            large_instance_candidate_threshold=args.large_instance_threshold,
            large_instance_min_local_steps=args.large_instance_min_local_steps,
            large_instance_min_sa_iterations=args.large_instance_min_sa_iterations,
            reduced_exact_time_limit=args.reduced_exact_time_limit,
            reduced_exact_core_cap=args.reduced_exact_core_cap,
            reduced_exact_rounds=args.reduced_exact_rounds,
            save_result=args.save,
            db_dir=args.db_dir,
        )

        if args.disable_elite_guided_restarts:
            solver_config.elite_guided_restarts = False
        if args.disable_path_relinking:
            solver_config.path_relinking = False
        if args.disable_ilp:
            solver_config.use_ilp = False
        if args.disable_mid_size_exact_improvement:
            solver_config.mid_size_exact_improvement = False
        if args.disable_local_branching_improvement:
            solver_config.local_branching_improvement = False
        if args.disable_lp_guided_exact_polish:
            solver_config.lp_guided_exact_polish = False
        if args.disable_cluster_exact_repair:
            solver_config.cluster_exact_repair = False
        if args.disable_cardinality_descent:
            solver_config.cardinality_descent = False
        if args.disable_adaptive_neighborhoods:
            solver_config.adaptive_neighborhoods = False
        if args.disable_reduced_exact_polish:
            solver_config.reduced_exact_polish = False

        solver.solve(solver_config)
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

    if args.command == "audit-result":
        data = ResultDatabase(args.db_dir).load(args.filename)
        report = audit_result_data(data, source=args.filename)
        print(format_validation_audit_report(report))
        if (
            not report.primary_report.is_valid
            or not report.independent_report.is_valid
            or not report.methods_agree
        ):
            raise SystemExit(1)
        return

    if args.command == "audit-file":
        report = audit_result_file(args.path)
        print(format_validation_audit_report(report))
        if (
            not report.primary_report.is_valid
            or not report.independent_report.is_valid
            or not report.methods_agree
        ):
            raise SystemExit(1)
        return

    if args.command == "audit-results":
        reports = audit_database(args.db_dir)
        if not reports:
            print("No saved result JSON files found.")
            return
        for report in reports:
            print(format_validation_audit_report(report))
        if any(
            (
                not report.primary_report.is_valid
                or not report.independent_report.is_valid
                or not report.methods_agree
            )
            for report in reports
        ):
            raise SystemExit(1)
        return

    if args.command == "certify-result":
        data = ResultDatabase(args.db_dir).load(args.filename)
        certificate = certify_result_data(
            data,
            source=args.filename,
            backend=args.exact_backend,
            time_limit=args.exact_time_limit,
        )
        print(format_optimality_certificate(certificate))
        if not certificate.certified_optimal:
            raise SystemExit(1)
        return

    if args.command == "summarize-result":
        data = ResultDatabase(args.db_dir).load(args.filename)
        certificate = certify_result_data(
            data,
            source=args.filename,
            backend=args.exact_backend,
            time_limit=args.exact_time_limit,
        )
        print(format_optimality_summary(certificate))
        if certificate.status in {"invalid_incumbent", "not_optimal"}:
            raise SystemExit(1)
        return

    if args.command == "certify-file":
        import json
        from pathlib import Path

        result_path = Path(args.path).resolve()
        with result_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Result file must contain a JSON object.")
        certificate = certify_result_data(
            data,
            source=str(result_path),
            backend=args.exact_backend,
            time_limit=args.exact_time_limit,
        )
        print(format_optimality_certificate(certificate))
        if not certificate.certified_optimal:
            raise SystemExit(1)
        return

    if args.command == "summarize-file":
        import json
        from pathlib import Path

        result_path = Path(args.path).resolve()
        with result_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError("Result file must contain a JSON object.")
        certificate = certify_result_data(
            data,
            source=str(result_path),
            backend=args.exact_backend,
            time_limit=args.exact_time_limit,
        )
        print(format_optimality_summary(certificate))
        if certificate.status in {"invalid_incumbent", "not_optimal"}:
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
