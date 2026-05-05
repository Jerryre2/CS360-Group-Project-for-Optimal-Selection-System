import unittest

from optimal_samples_system.certify import certify_result_data
from optimal_samples_system.config import CoverageMode, ProblemConfig


class OptimalityCertificationTests(unittest.TestCase):
    def test_certify_known_optimal_solution(self) -> None:
        config = ProblemConfig(
            m=7,
            n=7,
            k=6,
            j=5,
            s=5,
            samples=tuple(range(1, 8)),
            coverage_mode=CoverageMode.AT_LEAST_ONE,
            seed=42,
        )
        result = {
            "groups": [
                [1, 2, 3, 4, 5, 7],
                [1, 2, 3, 4, 6, 7],
                [1, 2, 3, 5, 6, 7],
                [1, 2, 4, 5, 6, 7],
                [1, 3, 4, 5, 6, 7],
                [2, 3, 4, 5, 6, 7],
            ],
            "num_groups": 6,
            "params": config.to_dict(),
        }

        certificate = certify_result_data(
            result,
            source="known-optimal",
            backend="scipy",
            time_limit=30,
        )

        self.assertTrue(certificate.certified_optimal)
        self.assertEqual(certificate.status, "certified_optimal")

    def test_invalid_solution_is_not_certified(self) -> None:
        config = ProblemConfig(
            m=8,
            n=8,
            k=6,
            j=4,
            s=4,
            samples=tuple(range(1, 9)),
            coverage_mode=CoverageMode.AT_LEAST_ONE,
            seed=42,
        )
        result = {
            "groups": [
                [1, 2, 3, 4, 7, 8],
            ],
            "num_groups": 1,
            "params": config.to_dict(),
        }

        certificate = certify_result_data(
            result,
            source="invalid",
            backend="scipy",
            time_limit=30,
        )

        self.assertFalse(certificate.certified_optimal)
        self.assertEqual(certificate.status, "invalid_incumbent")


if __name__ == "__main__":
    unittest.main()
