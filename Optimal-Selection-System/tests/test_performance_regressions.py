import unittest
from unittest.mock import patch

from optimal_samples_system.config import (
    AggregationMode,
    CoverageMode,
    ProblemConfig,
    SolverConfig,
)
from optimal_samples_system.instance import CoverageInstance
from optimal_samples_system.solver import OptimalSamplesSolver
from optimal_samples_system.tracking import CoverageTracker
from optimal_samples_system.exact import ILPSolver


class PerformanceRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ProblemConfig(
            m=20,
            n=7,
            k=5,
            j=4,
            s=3,
            coverage_mode=CoverageMode.AT_LEAST_R,
            aggregation_mode=AggregationMode.DISTINCT_SUBSETS,
            required_r=2,
            seed=12345,
        )
        self.instance = CoverageInstance(self.config)

    def test_candidate_overlap_matches_set_intersection(self) -> None:
        for left_index in range(len(self.instance.candidates)):
            left = set(self.instance.candidate_subset_ids(left_index))
            for right_index in range(len(self.instance.candidates)):
                right = set(self.instance.candidate_subset_ids(right_index))
                self.assertEqual(
                    self.instance.candidate_overlap_in_s_subsets(
                        left_index, right_index
                    ),
                    len(left & right),
                )

    def test_tracker_metrics_match_naive_distinct_subset_logic(self) -> None:
        tracker = CoverageTracker(self.instance)
        solution = [0, 1, 2, 6]
        tracker.reset(solution)

        for candidate_index in solution:
            losses = {}
            for subset_id in self.instance.candidate_subset_ids(candidate_index):
                if tracker.subset_cover_count[subset_id] != 1:
                    continue
                for target_index in self.instance.subset_to_targets[subset_id]:
                    losses[target_index] = losses.get(target_index, 0) + 1

            naive_can_remove = True
            naive_exclusive_count = 0
            naive_newly_uncovered = set()
            for target_index, loss in losses.items():
                remaining = (
                    tracker.target_covered_count[target_index] - loss
                )
                if remaining < self.instance.required_subset_count:
                    naive_can_remove = False
                    naive_exclusive_count += 1
                    naive_newly_uncovered.add(target_index)

            self.assertEqual(tracker.can_remove(candidate_index), naive_can_remove)
            self.assertEqual(
                tracker.exclusive_count(candidate_index), naive_exclusive_count
            )
            self.assertEqual(
                tracker.get_newly_uncovered(candidate_index), naive_newly_uncovered
            )

        outside_candidates = [
            candidate_index
            for candidate_index in range(len(self.instance.candidates))
            if candidate_index not in tracker.in_solution
        ]
        for candidate_index in outside_candidates[:5]:
            per_target_gain = {}
            naive_gain = 0
            for subset_id in self.instance.candidate_subset_ids(candidate_index):
                if tracker.subset_cover_count[subset_id] > 0:
                    continue
                for target_index in self.instance.subset_to_targets[subset_id]:
                    deficit = (
                        self.instance.required_subset_count
                        - tracker.target_covered_count[target_index]
                    )
                    if deficit <= 0:
                        continue
                    current_gain = per_target_gain.get(target_index, 0)
                    if current_gain < deficit:
                        per_target_gain[target_index] = current_gain + 1
                        naive_gain += 1

            self.assertEqual(tracker.marginal_gain(candidate_index), naive_gain)

    def test_solver_result_remains_feasible(self) -> None:
        solver = OptimalSamplesSolver(self.config)
        result = solver.solve(
            SolverConfig(
                n_restarts=1,
                use_ilp=False,
                max_local_steps=100,
                max_sa_iterations=100,
                candidate_sample_size=16,
            )
        )

        tracker = CoverageTracker(solver.instance)
        tracker.reset(result.solution_indices)
        self.assertTrue(tracker.is_feasible())
        self.assertIsNotNone(result.validation)
        self.assertTrue(result.validation["primary_valid"])
        self.assertTrue(result.validation["independent_valid"])
        self.assertTrue(result.validation["methods_agree"])

    def test_restricted_exact_solver_respects_candidate_core(self) -> None:
        config = ProblemConfig(
            m=10,
            n=7,
            k=6,
            j=5,
            s=5,
            coverage_mode=CoverageMode.AT_LEAST_ONE,
            aggregation_mode=AggregationMode.DISTINCT_SUBSETS,
            samples=tuple(range(1, 8)),
            seed=20260504,
        )
        instance = CoverageInstance(config)
        exact_solution, _ = ILPSolver.solve(
            instance, backend="scipy", time_limit=20
        )
        self.assertIsNotNone(exact_solution)

        restricted = ILPSolver.solve_restricted(
            instance,
            candidate_indices=exact_solution,
            backend="scipy",
            time_limit=20,
            incumbent=exact_solution,
        )
        self.assertIn(restricted.status, {"optimal", "feasible"})
        self.assertIsNotNone(restricted.solution)
        self.assertTrue(set(restricted.solution).issubset(set(exact_solution)))
        self.assertEqual(len(restricted.solution), len(exact_solution))

        restricted_decision = ILPSolver.prove_no_solution_at_or_below_restricted(
            instance,
            candidate_indices=exact_solution,
            cardinality_limit=len(exact_solution),
            backend="scipy",
            time_limit=20,
            incumbent=exact_solution,
        )
        self.assertEqual(restricted_decision.status, "feasible")
        self.assertIsNotNone(restricted_decision.solution)

    def test_reduced_exact_polish_preserves_feasibility(self) -> None:
        solver = OptimalSamplesSolver(self.config)
        baseline = solver.solve(
            SolverConfig(
                n_restarts=1,
                use_ilp=False,
                max_local_steps=80,
                max_sa_iterations=80,
                candidate_sample_size=16,
            )
        )

        polished, _ = solver._reduced_exact_polish(
            baseline.solution_indices,
            [],
            SolverConfig(
                n_restarts=1,
                use_ilp=True,
                exact_backend="scipy",
                reduced_exact_polish=True,
                reduced_exact_time_limit=5,
                reduced_exact_core_cap=64,
                reduced_exact_rounds=2,
            ),
        )
        self.assertLessEqual(len(polished), len(baseline.solution_indices))

        tracker = CoverageTracker(solver.instance)
        tracker.reset(polished)
        self.assertTrue(tracker.is_feasible())

    def test_large_instance_profile_keeps_improvement_passes_enabled(self) -> None:
        config = ProblemConfig(
            m=45,
            n=20,
            k=6,
            j=4,
            s=4,
            samples=tuple(range(1, 21)),
            coverage_mode=CoverageMode.AT_LEAST_ONE,
            aggregation_mode=AggregationMode.DISTINCT_SUBSETS,
            seed=42,
        )
        solver = OptimalSamplesSolver(config)
        local_steps, sa_iterations, profile = solver._resolve_search_budgets(
            SolverConfig(use_ilp=False)
        )

        self.assertEqual(profile, "large-instance-scaled")
        self.assertGreater(local_steps, 0)
        self.assertGreater(sa_iterations, 0)

    def test_path_relink_seed_preserves_feasibility(self) -> None:
        config = ProblemConfig(
            m=9,
            n=7,
            k=5,
            j=4,
            s=3,
            samples=tuple(range(1, 8)),
            coverage_mode=CoverageMode.AT_LEAST_R,
            aggregation_mode=AggregationMode.DISTINCT_SUBSETS,
            required_r=2,
            seed=12345,
        )
        solver = OptimalSamplesSolver(config)
        left = solver.solve(
            SolverConfig(
                n_restarts=1,
                use_ilp=False,
                max_local_steps=80,
                max_sa_iterations=80,
                candidate_sample_size=16,
            )
        ).solution_indices
        right = solver.solve(
            SolverConfig(
                n_restarts=2,
                use_ilp=False,
                max_local_steps=80,
                max_sa_iterations=80,
                candidate_sample_size=16,
                elite_guided_restarts=False,
                path_relinking=False,
            )
        ).solution_indices

        relinked = solver._path_relink(left, right)
        tracker = CoverageTracker(solver.instance)
        tracker.reset(relinked)
        self.assertTrue(tracker.is_feasible())
        self.assertLessEqual(len(relinked), len(set(left) | set(right)))

    def test_mid_size_exact_improvement_preserves_feasibility(self) -> None:
        config = ProblemConfig(
            m=10,
            n=7,
            k=6,
            j=5,
            s=5,
            samples=tuple(range(1, 8)),
            coverage_mode=CoverageMode.AT_LEAST_ONE,
            aggregation_mode=AggregationMode.DISTINCT_SUBSETS,
            seed=20260505,
        )
        solver = OptimalSamplesSolver(config)
        baseline = solver.solve(
            SolverConfig(
                n_restarts=1,
                use_ilp=False,
                max_local_steps=60,
                max_sa_iterations=60,
                candidate_sample_size=12,
            )
        )

        improved, _ = solver._mid_size_exact_improvement(
            baseline.solution_indices,
            SolverConfig(
                use_ilp=True,
                exact_backend="scipy",
                mid_size_exact_improvement=True,
                mid_size_exact_candidate_threshold=1000,
                mid_size_exact_time_limit=20,
            ),
        )

        tracker = CoverageTracker(solver.instance)
        tracker.reset(improved)
        self.assertTrue(tracker.is_feasible())

    def test_lp_guided_exact_polish_preserves_feasibility(self) -> None:
        config = ProblemConfig(
            m=10,
            n=7,
            k=6,
            j=5,
            s=5,
            samples=tuple(range(1, 8)),
            coverage_mode=CoverageMode.AT_LEAST_ONE,
            aggregation_mode=AggregationMode.DISTINCT_SUBSETS,
            seed=20260506,
        )
        solver = OptimalSamplesSolver(config)
        baseline = solver.solve(
            SolverConfig(
                n_restarts=1,
                use_ilp=False,
                max_local_steps=60,
                max_sa_iterations=60,
                candidate_sample_size=12,
            )
        )

        improved, _ = solver._lp_guided_exact_polish(
            baseline.solution_indices,
            [],
            SolverConfig(
                use_ilp=True,
                exact_backend="scipy",
                lp_guided_exact_polish=True,
                lp_guided_candidate_threshold=1000,
                lp_guided_time_limit=10,
                lp_guided_core_cap=48,
                lp_guided_rounds=1,
            ),
        )

        tracker = CoverageTracker(solver.instance)
        tracker.reset(improved)
        self.assertTrue(tracker.is_feasible())

    def test_cluster_exact_repair_preserves_feasibility(self) -> None:
        config = ProblemConfig(
            m=10,
            n=7,
            k=6,
            j=5,
            s=5,
            samples=tuple(range(1, 8)),
            coverage_mode=CoverageMode.AT_LEAST_ONE,
            aggregation_mode=AggregationMode.DISTINCT_SUBSETS,
            seed=20260507,
        )
        solver = OptimalSamplesSolver(config)
        baseline = solver.solve(
            SolverConfig(
                n_restarts=1,
                use_ilp=False,
                max_local_steps=60,
                max_sa_iterations=60,
                candidate_sample_size=12,
            )
        )

        improved, _ = solver._cluster_exact_repair(
            baseline.solution_indices,
            [],
            SolverConfig(
                use_ilp=True,
                exact_backend="scipy",
                cluster_exact_repair=True,
                cluster_exact_candidate_threshold=1000,
                cluster_exact_time_limit=10,
                cluster_exact_core_cap=64,
                cluster_exact_rounds=1,
                cluster_exact_destroy_size=6,
            ),
        )

        tracker = CoverageTracker(solver.instance)
        tracker.reset(improved)
        self.assertTrue(tracker.is_feasible())

    def test_cardinality_descent_preserves_feasibility(self) -> None:
        config = ProblemConfig(
            m=10,
            n=7,
            k=6,
            j=5,
            s=5,
            samples=tuple(range(1, 8)),
            coverage_mode=CoverageMode.AT_LEAST_ONE,
            aggregation_mode=AggregationMode.DISTINCT_SUBSETS,
            seed=20260508,
        )
        solver = OptimalSamplesSolver(config)
        baseline = solver.solve(
            SolverConfig(
                n_restarts=1,
                use_ilp=False,
                max_local_steps=60,
                max_sa_iterations=60,
                candidate_sample_size=12,
            )
        )

        improved = solver._cardinality_descent(
            baseline.solution_indices,
            SolverConfig(
                cardinality_descent=True,
                cardinality_descent_candidate_threshold=1000,
                cardinality_descent_iterations=200,
                cardinality_descent_patience=40,
                cardinality_descent_sample_size=16,
                cardinality_descent_initial_drop=1,
                cardinality_descent_max_rounds=2,
            ),
        )

        tracker = CoverageTracker(solver.instance)
        tracker.reset(improved)
        self.assertTrue(tracker.is_feasible())
        self.assertLessEqual(len(improved), len(baseline.solution_indices))

    def test_auto_exact_falls_back_when_gurobi_backend_raises(self) -> None:
        with patch.object(ILPSolver, "_solve_gurobi", side_effect=RuntimeError("license")):
            solution, method = ILPSolver.solve(self.instance, backend="auto", time_limit=5)

        self.assertIsNotNone(method)
        self.assertIsNotNone(solution)


if __name__ == "__main__":
    unittest.main()
