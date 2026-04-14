"""Exact verification models."""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

from .config import AggregationMode, LOGGER
from .instance import CoverageInstance


class ILPSolver:
    @staticmethod
    def solve(instance: CoverageInstance) -> Tuple[Optional[List[int]], Optional[str]]:
        try:
            import numpy as np
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import lil_matrix
        except ImportError:
            LOGGER.info("  Exact solver skipped: scipy is not available.")
            return None, None

        start = time.time()

        if instance.aggregation_mode == AggregationMode.SINGLE_CANDIDATE:
            if instance.covered_by is None:
                raise RuntimeError("Single-candidate cover relation was not built.")

            num_candidates = len(instance.candidates)
            num_targets = len(instance.targets)
            objective = np.ones(num_candidates)
            matrix = lil_matrix((num_targets, num_candidates), dtype=float)

            for target_index, candidate_indices in enumerate(instance.covered_by):
                for candidate_index in candidate_indices:
                    matrix[target_index, candidate_index] = 1.0

            result = milp(
                objective,
                constraints=LinearConstraint(matrix.tocsc(), lb=1.0),
                bounds=Bounds(lb=0.0, ub=1.0),
                integrality=np.ones(num_candidates),
                options={"time_limit": 60},
            )
        else:
            num_candidates = len(instance.candidates)
            num_subsets = len(instance.s_subsets)
            total_variables = num_candidates + num_subsets

            objective = np.concatenate([np.ones(num_candidates), np.zeros(num_subsets)])

            subset_to_candidates = [list() for _ in range(num_subsets)]
            for candidate_index in range(num_candidates):
                for subset_id in instance.candidate_subset_ids(candidate_index):
                    subset_to_candidates[subset_id].append(candidate_index)

            num_rows = num_subsets + len(instance.targets)
            matrix = lil_matrix((num_rows, total_variables), dtype=float)
            lower_bounds = np.full(num_rows, -np.inf)
            upper_bounds = np.full(num_rows, np.inf)

            row = 0
            for subset_id, candidate_indices in enumerate(subset_to_candidates):
                for candidate_index in candidate_indices:
                    matrix[row, candidate_index] = 1.0
                matrix[row, num_candidates + subset_id] = -1.0
                lower_bounds[row] = 0.0
                row += 1

            for subset_ids in instance.target_subset_ids:
                for subset_id in subset_ids:
                    matrix[row, num_candidates + subset_id] = 1.0
                lower_bounds[row] = float(instance.required_subset_count)
                row += 1

            result = milp(
                objective,
                constraints=LinearConstraint(matrix.tocsc(), lb=lower_bounds, ub=upper_bounds),
                bounds=Bounds(lb=0.0, ub=1.0),
                integrality=np.ones(total_variables),
                options={"time_limit": 60},
            )

        elapsed = time.time() - start
        if result.success:
            chosen = [
                index for index in range(len(instance.candidates)) if result.x[index] > 0.5
            ]
            LOGGER.info(
                f"  Exact solve success: {len(chosen)} groups, time {elapsed:.2f}s"
            )
            return chosen, "scipy-milp"

        LOGGER.info("  Exact solve failed or timed out.")
        return None, None
