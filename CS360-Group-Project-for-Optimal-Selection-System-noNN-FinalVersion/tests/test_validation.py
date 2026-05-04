import unittest

from optimal_samples_system.config import CoverageMode, ProblemConfig
from optimal_samples_system.validation import validate_result_data


class ValidationTests(unittest.TestCase):
    def _base_result(self, groups):
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
        return {
            "groups": [list(group) for group in groups],
            "num_groups": len(groups),
            "params": config.to_dict(),
        }

    def test_valid_known_solution_passes(self) -> None:
        groups = [
            (1, 2, 3, 4, 7, 8),
            (1, 2, 3, 5, 7, 8),
            (1, 2, 3, 6, 7, 8),
            (1, 2, 4, 5, 6, 8),
            (1, 3, 4, 5, 6, 7),
            (2, 3, 4, 5, 6, 7),
            (3, 4, 5, 6, 7, 8),
        ]

        report = validate_result_data(self._base_result(groups))

        self.assertTrue(report.is_valid)
        self.assertEqual(report.unsatisfied_targets, 0)
        self.assertEqual(report.num_groups, 7)

    def test_incomplete_solution_fails(self) -> None:
        groups = [
            (1, 2, 3, 4, 7, 8),
        ]

        report = validate_result_data(self._base_result(groups))

        self.assertFalse(report.is_valid)
        self.assertGreater(report.unsatisfied_targets, 0)

    def test_duplicate_and_invalid_group_fail(self) -> None:
        groups = [
            (1, 2, 3, 4, 7, 8),
            (1, 2, 3, 4, 7, 8),
            (1, 2, 3, 4, 7, 99),
        ]

        report = validate_result_data(self._base_result(groups))

        self.assertFalse(report.is_valid)
        messages = [issue.message for issue in report.issues]
        self.assertTrue(any("duplicated" in message for message in messages))
        self.assertTrue(any("outside the instance" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
