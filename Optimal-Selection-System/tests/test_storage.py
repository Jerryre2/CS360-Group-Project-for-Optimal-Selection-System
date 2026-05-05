import io
import logging
import tempfile
import unittest
from pathlib import Path

from optimal_samples_system.config import CoverageMode, ProblemConfig
from optimal_samples_system.storage import ResultDatabase


class StorageTests(unittest.TestCase):
    def _result_payload(self, groups, validation=None):
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
            "solution_indices": list(range(len(groups))),
            "groups": [list(group) for group in groups],
            "num_groups": len(groups),
            "exact_size": None,
            "exact_method": None,
            "samples": list(range(1, 9)),
            "params": config.to_dict(),
            "solver": {"n_restarts": 1},
            "elapsed_seconds": 0.1,
            "seed": 42,
            "required_subsets_per_target": 1,
            "num_targets": 70,
            "num_candidates": 28,
            "aggregation_mode": "distinct_subsets",
            "coverage_mode": "at_least_one",
            "validation": validation,
        }

    def test_list_all_exposes_validation_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ResultDatabase(tmpdir)
            db.save(
                self._result_payload(
                    groups=[(1, 2, 3, 4, 7, 8)],
                    validation={
                        "primary_valid": True,
                        "independent_valid": True,
                        "methods_agree": True,
                        "unsatisfied_targets": 0,
                        "deficit_units": 0,
                    },
                )
            )

            items = db.list_all()
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["validation_status"], "validated")

    def test_print_result_shows_validation_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = ResultDatabase(tmpdir)
            path = db.save(
                self._result_payload(
                    groups=[(1, 2, 3, 4, 7, 8)],
                    validation={
                        "primary_valid": True,
                        "independent_valid": True,
                        "methods_agree": True,
                        "unsatisfied_targets": 0,
                        "deficit_units": 0,
                    },
                )
            )

            logger = logging.getLogger("optimal_samples")
            stream = io.StringIO()
            handler = logging.StreamHandler(stream)
            logger.addHandler(handler)
            old_level = logger.level
            logger.setLevel(logging.INFO)
            try:
                db.print_result(Path(path).name)
            finally:
                logger.removeHandler(handler)
                logger.setLevel(old_level)

            output = stream.getvalue()
            self.assertIn("Validation: validation=validated", output)
            self.assertIn(f"File: {Path(path).name}", output)


if __name__ == "__main__":
    unittest.main()
