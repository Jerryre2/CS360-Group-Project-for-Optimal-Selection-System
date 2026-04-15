import unittest

from optimal_samples_system.config import (
    AggregationMode,
    CoverageMode,
    ProblemConfig,
    SolverConfig,
)
from optimal_samples_system.instance import CoverageInstance
from optimal_samples_system.solver import OptimalSamplesSolver
from optimal_samples_system.tracking import CoverageTracker


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


if __name__ == "__main__":
    unittest.main()
