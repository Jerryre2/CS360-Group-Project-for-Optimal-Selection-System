import unittest

from optimal_samples_system.config import CoverageMode, ProblemConfig
from optimal_samples_system.instance import CoverageInstance
from optimal_samples_system.validation import (
    audit_solution_indices,
    audit_result_data,
    validate_result_data,
    validate_solution_indices,
    validate_solution_indices_independently,
    validate_result_data_independently,
)


class ValidationTests(unittest.TestCase):
    def _base_result(self, groups, coverage_mode=CoverageMode.AT_LEAST_ONE, required_r=None):
        config = ProblemConfig(
            m=8,
            n=8,
            k=6,
            j=4,
            s=4,
            samples=tuple(range(1, 9)),
            coverage_mode=coverage_mode,
            required_r=required_r,
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

    def test_independent_validator_valid_known_solution_passes(self) -> None:
        groups = [
            (1, 2, 3, 4, 7, 8),
            (1, 2, 3, 5, 7, 8),
            (1, 2, 3, 6, 7, 8),
            (1, 2, 4, 5, 6, 8),
            (1, 3, 4, 5, 6, 7),
            (2, 3, 4, 5, 6, 7),
            (3, 4, 5, 6, 7, 8),
        ]

        report = validate_result_data_independently(self._base_result(groups))

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

    def test_independent_validator_checks_at_least_r(self) -> None:
        groups = [
            (1, 2, 3, 4, 5, 7),
            (1, 2, 3, 4, 5, 8),
            (1, 2, 3, 4, 6, 7),
            (1, 2, 3, 4, 6, 8),
            (1, 2, 3, 4, 7, 8),
            (1, 2, 3, 5, 6, 7),
            (1, 2, 3, 5, 6, 8),
            (1, 2, 4, 5, 6, 7),
            (1, 2, 4, 5, 6, 8),
            (1, 2, 5, 6, 7, 8),
            (3, 4, 5, 6, 7, 8),
        ]
        config = ProblemConfig(
            m=8,
            n=8,
            k=6,
            j=6,
            s=5,
            samples=tuple(range(1, 9)),
            coverage_mode=CoverageMode.AT_LEAST_R,
            required_r=4,
            seed=42,
        )
        result = {
            "groups": [list(group) for group in groups],
            "num_groups": len(groups),
            "params": config.to_dict(),
        }

        report = validate_result_data_independently(result)

        self.assertTrue(report.is_valid)
        self.assertEqual(report.unsatisfied_targets, 0)

    def test_audit_agrees_on_valid_solution(self) -> None:
        groups = [
            (1, 2, 3, 4, 7, 8),
            (1, 2, 3, 5, 7, 8),
            (1, 2, 3, 6, 7, 8),
            (1, 2, 4, 5, 6, 8),
            (1, 3, 4, 5, 6, 7),
            (2, 3, 4, 5, 6, 7),
            (3, 4, 5, 6, 7, 8),
        ]

        report = audit_result_data(self._base_result(groups))

        self.assertTrue(report.primary_report.is_valid)
        self.assertTrue(report.independent_report.is_valid)
        self.assertTrue(report.methods_agree)

    def test_solution_indices_validators_agree(self) -> None:
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
        instance = CoverageInstance(config)
        candidate_lookup = {
            instance.candidate_label(candidate_index): candidate_index
            for candidate_index in range(len(instance.candidates))
        }
        solution_indices = [
            candidate_lookup[group]
            for group in [
                (1, 2, 3, 4, 7, 8),
                (1, 2, 3, 5, 7, 8),
                (1, 2, 3, 6, 7, 8),
                (1, 2, 4, 5, 6, 8),
                (1, 3, 4, 5, 6, 7),
                (2, 3, 4, 5, 6, 7),
                (3, 4, 5, 6, 7, 8),
            ]
        ]

        primary = validate_solution_indices(instance, solution_indices)
        independent = validate_solution_indices_independently(instance, solution_indices)
        audit = audit_solution_indices(instance, solution_indices)

        self.assertTrue(primary.is_valid)
        self.assertTrue(independent.is_valid)
        self.assertTrue(audit.methods_agree)

    def test_missing_semantics_fields_fail_closed(self) -> None:
        result = self._base_result(
            [
                (1, 2, 3, 4, 7, 8),
                (1, 2, 3, 5, 7, 8),
                (1, 2, 3, 6, 7, 8),
                (1, 2, 4, 5, 6, 8),
                (1, 3, 4, 5, 6, 7),
                (2, 3, 4, 5, 6, 7),
                (3, 4, 5, 6, 7, 8),
            ]
        )
        del result["params"]["coverage_mode"]

        report = validate_result_data(result)

        self.assertFalse(report.is_valid)
        self.assertTrue(
            any("coverage_mode" in issue.message for issue in report.issues)
        )

    def test_missing_samples_fail_closed(self) -> None:
        result = self._base_result(
            [
                (1, 2, 3, 4, 7, 8),
                (1, 2, 3, 5, 7, 8),
                (1, 2, 3, 6, 7, 8),
                (1, 2, 4, 5, 6, 8),
                (1, 3, 4, 5, 6, 7),
                (2, 3, 4, 5, 6, 7),
                (3, 4, 5, 6, 7, 8),
            ]
        )
        del result["params"]["samples"]

        report = validate_result_data(result)

        self.assertFalse(report.is_valid)
        self.assertTrue(any("samples" in issue.message for issue in report.issues))


if __name__ == "__main__":
    unittest.main()
