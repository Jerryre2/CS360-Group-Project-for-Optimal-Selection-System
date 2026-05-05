"""One-off exact proof attempt for the fixed n=15 benchmark instance.

This script targets the benchmark:
    m=45, n=15, k=6, j=5, s=4, samples=1..15

We currently have a feasible incumbent of size 49. Rather than re-solving the
full minimization model, this script tries to prove optimality by solving the
decision problem:

    "Does there exist a feasible solution with at most 48 groups?"

If the answer is infeasible, then the incumbent size 49 is proven optimal.

For this specific benchmark, coverage_mode=AT_LEAST_ONE, so a target is covered
iff a chosen candidate overlaps it in at least `s` elements. That lets us build
a smaller direct set-cover decision MILP with only candidate variables.
"""

from __future__ import annotations

import argparse
import time
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from optimal_samples_system.config import CoverageMode, ProblemConfig
from optimal_samples_system.instance import CoverageInstance
from optimal_samples_system.tracking import CoverageTracker


FIXED_CONFIG = ProblemConfig(
    m=45,
    n=15,
    k=6,
    j=5,
    s=4,
    samples=tuple(range(1, 16)),
    coverage_mode=CoverageMode.AT_LEAST_ONE,
    seed=42,
)


INCUMBENT_GROUPS: Tuple[Tuple[int, ...], ...] = (
    (1, 2, 3, 5, 8, 14),
    (1, 2, 3, 8, 12, 15),
    (1, 2, 3, 8, 13, 15),
    (1, 2, 4, 5, 10, 12),
    (1, 2, 4, 8, 9, 11),
    (1, 2, 4, 10, 14, 15),
    (1, 2, 5, 6, 9, 12),
    (1, 2, 5, 7, 9, 13),
    (1, 2, 7, 10, 11, 14),
    (1, 3, 4, 6, 11, 12),
    (1, 3, 5, 7, 12, 14),
    (1, 3, 6, 7, 9, 15),
    (1, 3, 9, 10, 13, 15),
    (1, 4, 5, 6, 7, 8),
    (1, 4, 7, 9, 13, 14),
    (1, 5, 7, 10, 12, 13),
    (1, 5, 8, 10, 11, 13),
    (1, 5, 11, 13, 14, 15),
    (1, 6, 7, 11, 13, 15),
    (1, 6, 8, 9, 10, 12),
    (2, 3, 4, 7, 11, 13),
    (2, 3, 5, 6, 10, 15),
    (2, 3, 5, 6, 11, 12),
    (2, 3, 5, 9, 10, 12),
    (2, 3, 6, 8, 13, 14),
    (2, 4, 6, 7, 12, 14),
    (2, 4, 6, 10, 11, 13),
    (2, 4, 9, 10, 12, 15),
    (2, 5, 7, 8, 9, 15),
    (2, 5, 11, 12, 13, 14),
    (2, 6, 9, 10, 13, 15),
    (2, 7, 8, 10, 12, 15),
    (3, 4, 5, 6, 9, 13),
    (3, 4, 5, 9, 11, 15),
    (3, 4, 7, 8, 10, 11),
    (3, 4, 7, 9, 12, 14),
    (3, 6, 10, 12, 13, 14),
    (3, 7, 8, 13, 14, 15),
    (3, 9, 11, 12, 14, 15),
    (4, 5, 8, 9, 10, 14),
    (4, 5, 8, 12, 13, 15),
    (4, 6, 8, 12, 14, 15),
    (4, 7, 10, 11, 12, 15),
    (5, 6, 7, 10, 11, 14),
    (5, 7, 9, 12, 14, 15),
    (6, 7, 8, 9, 11, 12),
    (6, 7, 9, 10, 11, 13),
    (6, 8, 9, 11, 14, 15),
    (8, 9, 11, 12, 13, 14),
)


def group_indices(
    instance: CoverageInstance, groups: Sequence[Tuple[int, ...]]
) -> List[int]:
    label_to_index = {
        instance.candidate_label(candidate_index): candidate_index
        for candidate_index in range(len(instance.candidates))
    }
    return [label_to_index[group] for group in groups]


def verify_incumbent(instance: CoverageInstance, indices: Iterable[int]) -> bool:
    tracker = CoverageTracker(instance)
    tracker.reset(indices)
    return tracker.is_feasible()


def build_direct_decision_model(
    instance: CoverageInstance, cardinality_limit: int
) -> Tuple[np.ndarray, LinearConstraint, Bounds, np.ndarray]:
    num_candidates = len(instance.candidates)
    num_targets = len(instance.targets)

    matrix = lil_matrix((num_targets + 1, num_candidates), dtype=float)
    lower_bounds = np.ones(num_targets + 1)
    upper_bounds = np.full(num_targets + 1, np.inf)

    for target_index, target_mask in enumerate(instance.target_masks):
        for candidate_index, candidate_mask in enumerate(instance.candidate_masks):
            if (candidate_mask & target_mask).bit_count() >= instance.s:
                matrix[target_index, candidate_index] = 1.0

    matrix[num_targets, :] = 1.0
    lower_bounds[num_targets] = -np.inf
    upper_bounds[num_targets] = float(cardinality_limit)

    objective = np.zeros(num_candidates)
    constraints = LinearConstraint(matrix.tocsc(), lb=lower_bounds, ub=upper_bounds)
    bounds = Bounds(lb=0.0, ub=1.0)
    integrality = np.ones(num_candidates)
    return objective, constraints, bounds, integrality


def solve_decision_problem(
    instance: CoverageInstance, cardinality_limit: int, time_limit: int
) -> object:
    objective, constraints, bounds, integrality = build_direct_decision_model(
        instance, cardinality_limit
    )
    return milp(
        c=objective,
        constraints=constraints,
        bounds=bounds,
        integrality=integrality,
        options={"time_limit": time_limit},
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-off exact proof attempt for the fixed n=15 benchmark"
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=300,
        help="HiGHS time limit in seconds for the decision MILP.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=48,
        help="Try to prove there is no feasible solution of this size or smaller.",
    )
    args = parser.parse_args()

    instance = CoverageInstance(FIXED_CONFIG)
    incumbent_indices = group_indices(instance, INCUMBENT_GROUPS)
    incumbent_size = len(incumbent_indices)

    print("Fixed benchmark:")
    print(
        "  m=45, n=15, k=6, j=5, s=4, "
        "samples=1..15, coverage_mode=at_least_one"
    )
    print(f"Incumbent size: {incumbent_size}")
    print(f"Incumbent feasible: {verify_incumbent(instance, incumbent_indices)}")
    print(f"Decision target: prove no solution exists with <= {args.target_size} groups")

    start = time.time()
    result = solve_decision_problem(instance, args.target_size, args.time_limit)
    elapsed = time.time() - start

    print(f"\nElapsed: {elapsed:.2f}s")
    print(f"Success: {result.success}")
    print(f"Status: {getattr(result, 'status', None)}")
    print(f"Message: {getattr(result, 'message', None)}")

    if result.success:
        chosen = [
            index for index in range(len(instance.candidates)) if result.x[index] > 0.5
        ]
        print(f"Found feasible solution with {len(chosen)} groups.")
        if len(chosen) <= args.target_size:
            print(
                "Conclusion: the incumbent is not proven optimal because a "
                "solution at or below the decision target was found."
            )
        return

    print(
        "No certificate was obtained within the time limit. "
        "The incumbent remains a best-known feasible upper bound."
    )


if __name__ == "__main__":
    main()
