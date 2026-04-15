import unittest

from optimal_samples_system.config import CoverageMode, ProblemConfig, SolverConfig
from optimal_samples_system.solver import OptimalSamplesSolver
from optimal_samples_system.tracking import CoverageTracker


class SolverKnownCasesTests(unittest.TestCase):
    def test_known_covering_design_cases(self) -> None:
        test_cases = [
            (7, 6, 5, 5, 6),
            (8, 6, 5, 5, 12),
            (8, 6, 4, 4, 7),
            (9, 6, 4, 4, 12),
            (8, 6, 6, 5, 4),
            (9, 6, 5, 4, 3),
            (10, 6, 6, 4, 3),
            (12, 6, 6, 4, 6),
        ]

        for n, k, j, s, expected_optimal in test_cases:
            with self.subTest(n=n, k=k, j=j, s=s, expected_optimal=expected_optimal):
                config = ProblemConfig(
                    m=n,
                    n=n,
                    k=k,
                    j=j,
                    s=s,
                    samples=tuple(range(1, n + 1)),
                    coverage_mode=CoverageMode.AT_LEAST_ONE,
                    seed=42,
                )
                solver = OptimalSamplesSolver(config)
                result = solver.solve(
                    SolverConfig(
                        n_restarts=5,
                        use_ilp=False,
                    )
                )

                tracker = CoverageTracker(solver.instance)
                tracker.reset(result.solution_indices)
                self.assertTrue(tracker.is_feasible())
                self.assertLessEqual(result.num_groups, expected_optimal)


if __name__ == "__main__":
    unittest.main()
