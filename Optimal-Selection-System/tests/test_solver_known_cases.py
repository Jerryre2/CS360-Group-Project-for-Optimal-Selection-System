import unittest

from optimal_samples_system.config import (
    AggregationMode,
    CoverageMode,
    ProblemConfig,
    SolverConfig,
)
from optimal_samples_system.solver import OptimalSamplesSolver
from optimal_samples_system.tracking import CoverageTracker


class SolverKnownCasesTests(unittest.TestCase):
    def test_known_covering_design_cases(self) -> None:
        test_cases = [
            (7, 6, 5, 5, CoverageMode.AT_LEAST_ONE, None, AggregationMode.DISTINCT_SUBSETS, 6, True),
            (8, 6, 5, 5, CoverageMode.AT_LEAST_ONE, None, AggregationMode.DISTINCT_SUBSETS, 12, True),
            (8, 6, 4, 4, CoverageMode.AT_LEAST_ONE, None, AggregationMode.DISTINCT_SUBSETS, 7, True),
            (9, 6, 4, 4, CoverageMode.AT_LEAST_ONE, None, AggregationMode.DISTINCT_SUBSETS, 12, True),
            (8, 6, 6, 5, CoverageMode.AT_LEAST_ONE, None, AggregationMode.DISTINCT_SUBSETS, 4, True),
            (9, 6, 5, 4, CoverageMode.AT_LEAST_ONE, None, AggregationMode.DISTINCT_SUBSETS, 3, True),
            (10, 6, 6, 4, CoverageMode.AT_LEAST_ONE, None, AggregationMode.DISTINCT_SUBSETS, 3, True),
            (12, 6, 6, 4, CoverageMode.AT_LEAST_ONE, None, AggregationMode.DISTINCT_SUBSETS, 6, False),
        ]

        for (
            n,
            k,
            j,
            s,
            coverage_mode,
            required_r,
            aggregation_mode,
            expected_optimal,
            expect_exact_certificate,
        ) in test_cases:
            with self.subTest(
                n=n,
                k=k,
                j=j,
                s=s,
                coverage_mode=coverage_mode.value,
                aggregation_mode=aggregation_mode.value,
                expected_optimal=expected_optimal,
            ):
                config = ProblemConfig(
                    m=n,
                    n=n,
                    k=k,
                    j=j,
                    s=s,
                    samples=tuple(range(1, n + 1)),
                    coverage_mode=coverage_mode,
                    aggregation_mode=aggregation_mode,
                    required_r=required_r,
                    seed=42,
                )
                solver = OptimalSamplesSolver(config)
                result = solver.solve(
                    SolverConfig(
                        n_restarts=5,
                        use_ilp=True,
                        exact_backend="scipy",
                        exact_time_limit=60,
                    )
                )

                tracker = CoverageTracker(solver.instance)
                tracker.reset(result.solution_indices)
                self.assertTrue(tracker.is_feasible())
                self.assertEqual(result.num_groups, expected_optimal)
                if expect_exact_certificate:
                    self.assertEqual(result.exact_size, expected_optimal)


if __name__ == "__main__":
    unittest.main()
