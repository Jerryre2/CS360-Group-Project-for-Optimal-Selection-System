"""Exact verification models."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .config import AggregationMode, CoverageMode, LOGGER
from .instance import CoverageInstance
from .validation import audit_solution_indices


@dataclass
class ExactBackendResult:
    status: str
    solution: Optional[List[int]]
    method: Optional[str]
    message: str
    elapsed_seconds: float


@dataclass
class LowerBoundResult:
    status: str
    value: Optional[float]
    integer_bound: Optional[int]
    method: Optional[str]
    message: str
    elapsed_seconds: float


class ILPSolver:
    @staticmethod
    def _validated_backend_solution(
        instance: CoverageInstance,
        solution: Sequence[int],
        status: str,
        method: str,
        message: str,
        elapsed_seconds: float,
    ) -> ExactBackendResult:
        audit = audit_solution_indices(
            instance,
            solution,
            source=f"{method}-solution",
            max_uncovered_examples=5,
        )
        if (
            not audit.primary_report.is_valid
            or not audit.independent_report.is_valid
            or not audit.methods_agree
        ):
            details = "; ".join(audit.agreement_issues) or "solution failed validation"
            return ExactBackendResult(
                status="unknown",
                solution=None,
                method=method,
                message=f"{message}; rejected by validator: {details}",
                elapsed_seconds=elapsed_seconds,
            )

        return ExactBackendResult(
            status=status,
            solution=list(solution),
            method=method,
            message=message,
            elapsed_seconds=elapsed_seconds,
        )

    @staticmethod
    def solve(
        instance: CoverageInstance,
        backend: str = "auto",
        time_limit: int = 60,
        incumbent: Optional[Sequence[int]] = None,
    ) -> Tuple[Optional[List[int]], Optional[str]]:
        result = ILPSolver._solve_with_backends(
            instance=instance,
            backend=backend,
            time_limit=time_limit,
            cardinality_limit=None,
            incumbent=incumbent,
            candidate_indices=None,
            local_branch_radius=None,
        )
        if result.status == "optimal" and result.solution is not None:
            LOGGER.info(
                f"  Exact solve success ({result.method}): "
                f"{len(result.solution)} groups, time {result.elapsed_seconds:.2f}s"
            )
            return result.solution, result.method

        LOGGER.info(f"  Exact solve failed or timed out: {result.message}")
        return None, result.method

    @staticmethod
    def improve(
        instance: CoverageInstance,
        backend: str = "auto",
        time_limit: int = 60,
        incumbent: Optional[Sequence[int]] = None,
    ) -> ExactBackendResult:
        result = ILPSolver._solve_with_backends(
            instance=instance,
            backend=backend,
            time_limit=time_limit,
            cardinality_limit=None,
            incumbent=incumbent,
            candidate_indices=None,
            local_branch_radius=None,
        )
        if result.solution is not None:
            LOGGER.info(
                f"  Exact improvement candidate ({result.method}): "
                f"{len(result.solution)} groups, status={result.status}, "
                f"time {result.elapsed_seconds:.2f}s"
            )
        else:
            LOGGER.info(f"  Exact improvement attempt failed or timed out: {result.message}")
        return result

    @staticmethod
    def prove_no_solution_at_or_below(
        instance: CoverageInstance,
        cardinality_limit: int,
        backend: str = "auto",
        time_limit: int = 300,
        incumbent: Optional[Sequence[int]] = None,
    ) -> ExactBackendResult:
        return ILPSolver._solve_with_backends(
            instance=instance,
            backend=backend,
            time_limit=time_limit,
            cardinality_limit=cardinality_limit,
            incumbent=incumbent,
            candidate_indices=None,
            local_branch_radius=None,
        )

    @staticmethod
    def prove_no_solution_at_or_below_restricted(
        instance: CoverageInstance,
        candidate_indices: Sequence[int],
        cardinality_limit: int,
        backend: str = "auto",
        time_limit: int = 300,
        incumbent: Optional[Sequence[int]] = None,
    ) -> ExactBackendResult:
        return ILPSolver._solve_with_backends(
            instance=instance,
            backend=backend,
            time_limit=time_limit,
            cardinality_limit=cardinality_limit,
            incumbent=incumbent,
            candidate_indices=candidate_indices,
            local_branch_radius=None,
        )

    @staticmethod
    def solve_restricted(
        instance: CoverageInstance,
        candidate_indices: Sequence[int],
        backend: str = "auto",
        time_limit: int = 30,
        incumbent: Optional[Sequence[int]] = None,
    ) -> ExactBackendResult:
        return ILPSolver._solve_with_backends(
            instance=instance,
            backend=backend,
            time_limit=time_limit,
            cardinality_limit=None,
            incumbent=incumbent,
            candidate_indices=candidate_indices,
            local_branch_radius=None,
        )

    @staticmethod
    def improve_locally(
        instance: CoverageInstance,
        incumbent: Sequence[int],
        radius: int,
        backend: str = "auto",
        time_limit: int = 30,
        candidate_indices: Optional[Sequence[int]] = None,
    ) -> ExactBackendResult:
        return ILPSolver._solve_with_backends(
            instance=instance,
            backend=backend,
            time_limit=time_limit,
            cardinality_limit=None,
            incumbent=incumbent,
            candidate_indices=candidate_indices,
            local_branch_radius=max(0, int(radius)),
        )

    @staticmethod
    def lower_bound(
        instance: CoverageInstance,
        time_limit: int = 60,
    ) -> LowerBoundResult:
        try:
            import numpy as np
            from scipy.optimize import linprog
            from scipy.sparse import lil_matrix
        except ImportError:
            return LowerBoundResult(
                status="unavailable",
                value=None,
                integer_bound=None,
                method="scipy-lp",
                message="scipy is not available.",
                elapsed_seconds=0.0,
            )

        start = time.time()

        if instance.coverage_mode == CoverageMode.AT_LEAST_ONE:
            num_candidates = len(instance.candidates)
            num_targets = len(instance.targets)
            matrix = lil_matrix((num_targets, num_candidates), dtype=float)
            lower_bounds = np.ones(num_targets)

            for target_index, candidate_indices in enumerate(
                ILPSolver._target_to_covering_candidates_at_least_one(instance)
            ):
                for candidate_index in candidate_indices:
                    matrix[target_index, candidate_index] = 1.0

            objective = np.ones(num_candidates)
            integrality = np.zeros(num_candidates)
        elif instance.aggregation_mode == AggregationMode.SINGLE_CANDIDATE:
            if instance.covered_by is None:
                raise RuntimeError("Single-candidate cover relation was not built.")

            num_candidates = len(instance.candidates)
            num_targets = len(instance.targets)
            matrix = lil_matrix((num_targets, num_candidates), dtype=float)
            lower_bounds = np.ones(num_targets)

            for target_index, candidate_indices in enumerate(instance.covered_by):
                for candidate_index in candidate_indices:
                    matrix[target_index, candidate_index] = 1.0

            objective = np.ones(num_candidates)
        else:
            num_candidates = len(instance.candidates)
            num_subsets = len(instance.s_subsets)
            total_variables = num_candidates + num_subsets
            subset_to_candidates = ILPSolver._subset_to_candidates(instance)

            num_rows = num_subsets + len(instance.targets)
            matrix = lil_matrix((num_rows, total_variables), dtype=float)
            lower_bounds = np.full(num_rows, -np.inf)

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

            objective = np.concatenate(
                [
                    np.ones(num_candidates),
                    np.zeros(num_subsets),
                ]
            )

        upper_matrix = (-matrix).tocsc()
        upper_rhs = -lower_bounds
        result = linprog(
            c=objective,
            A_ub=upper_matrix,
            b_ub=upper_rhs,
            bounds=(0.0, 1.0),
            method="highs",
            options={"time_limit": float(time_limit)},
        )

        elapsed = time.time() - start
        message = str(getattr(result, "message", ""))
        if result.success:
            value = float(result.fun)
            integer_bound = int(math.ceil(value - 1e-9))
            return LowerBoundResult(
                status="optimal",
                value=value,
                integer_bound=integer_bound,
                method="scipy-lp",
                message=message or "optimal LP relaxation",
                elapsed_seconds=elapsed,
            )

        if getattr(result, "status", None) == 2:
            return LowerBoundResult(
                status="infeasible",
                value=None,
                integer_bound=None,
                method="scipy-lp",
                message=message,
                elapsed_seconds=elapsed,
            )

        return LowerBoundResult(
            status="unknown",
            value=None,
            integer_bound=None,
            method="scipy-lp",
            message=message or "LP relaxation failed or timed out.",
            elapsed_seconds=elapsed,
        )

    @staticmethod
    def fractional_candidate_core(
        instance: CoverageInstance,
        core_cap: int,
        time_limit: int = 20,
        incumbent: Optional[Sequence[int]] = None,
        anchors: Optional[Sequence[Sequence[int]]] = None,
        candidate_indices: Optional[Sequence[int]] = None,
    ) -> tuple[list[int], LowerBoundResult]:
        try:
            import numpy as np
            from scipy.optimize import linprog
            from scipy.sparse import lil_matrix
        except ImportError:
            return [], LowerBoundResult(
                status="unavailable",
                value=None,
                integer_bound=None,
                method="scipy-lp",
                message="scipy is not available.",
                elapsed_seconds=0.0,
            )

        start = time.time()
        candidate_universe = ILPSolver._candidate_universe(instance, candidate_indices)
        if not candidate_universe:
            return [], LowerBoundResult(
                status="unavailable",
                value=None,
                integer_bound=None,
                method="scipy-lp",
                message="Candidate universe is empty.",
                elapsed_seconds=0.0,
            )

        index_map = {
            candidate_index: offset
            for offset, candidate_index in enumerate(candidate_universe)
        }
        num_candidates = len(candidate_universe)

        if instance.coverage_mode == CoverageMode.AT_LEAST_ONE:
            num_targets = len(instance.targets)
            matrix = lil_matrix((num_targets, num_candidates), dtype=float)
            lower_bounds = np.ones(num_targets)

            cover_rows = ILPSolver._target_to_covering_candidates_at_least_one(
                instance,
                candidate_universe,
                index_map,
            )
            for target_index, local_candidate_indices in enumerate(cover_rows):
                for local_candidate_index in local_candidate_indices:
                    matrix[target_index, local_candidate_index] = 1.0

            objective = np.ones(num_candidates)
        elif instance.aggregation_mode == AggregationMode.SINGLE_CANDIDATE:
            num_targets = len(instance.targets)
            matrix = lil_matrix((num_targets, num_candidates), dtype=float)
            lower_bounds = np.ones(num_targets)

            cover_rows = ILPSolver._restricted_cover_rows_single_candidate(
                instance,
                candidate_universe,
                index_map,
            )
            for target_index, local_candidate_indices in enumerate(cover_rows):
                for local_candidate_index in local_candidate_indices:
                    matrix[target_index, local_candidate_index] = 1.0

            objective = np.ones(num_candidates)
        else:
            num_subsets = len(instance.s_subsets)
            total_variables = num_candidates + num_subsets
            subset_to_candidates = ILPSolver._subset_to_candidates(
                instance,
                candidate_universe,
                index_map,
            )

            num_rows = num_subsets + len(instance.targets)
            matrix = lil_matrix((num_rows, total_variables), dtype=float)
            lower_bounds = np.full(num_rows, -np.inf)

            row = 0
            for subset_id, local_candidate_indices in enumerate(subset_to_candidates):
                for local_candidate_index in local_candidate_indices:
                    matrix[row, local_candidate_index] = 1.0
                matrix[row, num_candidates + subset_id] = -1.0
                lower_bounds[row] = 0.0
                row += 1

            for subset_ids in instance.target_subset_ids:
                for subset_id in subset_ids:
                    matrix[row, num_candidates + subset_id] = 1.0
                lower_bounds[row] = float(instance.required_subset_count)
                row += 1

            objective = np.concatenate(
                [
                    np.ones(num_candidates),
                    np.zeros(num_subsets),
                ]
            )
            cover_rows = ILPSolver._target_to_covering_candidates_at_least_one(
                instance,
                candidate_universe,
                index_map,
            )

        upper_matrix = (-matrix).tocsc()
        upper_rhs = -lower_bounds
        result = linprog(
            c=objective,
            A_ub=upper_matrix,
            b_ub=upper_rhs,
            bounds=(0.0, 1.0),
            method="highs",
            options={"time_limit": float(time_limit)},
        )

        elapsed = time.time() - start
        message = str(getattr(result, "message", ""))
        if not result.success:
            return [], LowerBoundResult(
                status="unknown",
                value=None,
                integer_bound=None,
                method="scipy-lp",
                message=message or "LP-guided core extraction failed or timed out.",
                elapsed_seconds=elapsed,
            )

        value = float(result.fun)
        lp_bound = LowerBoundResult(
            status="optimal",
            value=value,
            integer_bound=int(math.ceil(value - 1e-9)),
            method="scipy-lp",
            message=message or "optimal LP relaxation",
            elapsed_seconds=elapsed,
        )

        candidate_values = [
            float(result.x[offset])
            for offset in range(num_candidates)
        ]
        cap = min(len(candidate_universe), max(1, core_cap))
        core: set[int] = set(int(candidate_index) for candidate_index in incumbent or ())
        for anchor in anchors or ():
            core.update(int(candidate_index) for candidate_index in anchor)

        positive = [
            candidate_universe[offset]
            for offset, value_i in enumerate(candidate_values)
            if value_i > 1e-7
        ]
        positive.sort(
            key=lambda candidate_index: candidate_values[index_map[candidate_index]],
            reverse=True,
        )
        core.update(positive[:cap])

        if len(core) < cap:
            target_priority = sorted(
                range(len(cover_rows)),
                key=lambda target_index: len(cover_rows[target_index]),
            )
            for target_index in target_priority:
                ranked_row = sorted(
                    cover_rows[target_index],
                    key=lambda local_candidate_index: candidate_values[local_candidate_index],
                    reverse=True,
                )
                for local_candidate_index in ranked_row[:3]:
                    core.add(candidate_universe[local_candidate_index])
                    if len(core) >= cap:
                        break
                if len(core) >= cap:
                    break

        if len(core) < cap:
            ranked_all = sorted(
                candidate_universe,
                key=lambda candidate_index: candidate_values[index_map[candidate_index]],
                reverse=True,
            )
            for candidate_index in ranked_all:
                core.add(candidate_index)
                if len(core) >= cap:
                    break

        return sorted(core), lp_bound

    @staticmethod
    def combinatorial_lower_bound(instance: CoverageInstance) -> LowerBoundResult:
        bounds: List[Tuple[str, int]] = []

        max_candidate_coverage = max(
            (instance.candidate_span(candidate_index) for candidate_index in range(len(instance.candidates))),
            default=0,
        )
        if max_candidate_coverage > 0:
            counting_bound = math.ceil(len(instance.targets) / max_candidate_coverage)
            bounds.append(("counting", counting_bound))

        if (
            instance.coverage_mode == CoverageMode.AT_LEAST_ONE
            and instance.j == instance.s
        ):
            schoenheim = ILPSolver._schoenheim_bound(instance.n, instance.k, instance.j)
            bounds.append(("schoenheim", schoenheim))

        if not bounds:
            return LowerBoundResult(
                status="unavailable",
                value=None,
                integer_bound=None,
                method="combinatorial",
                message="No combinatorial lower bound is available for this instance.",
                elapsed_seconds=0.0,
            )

        best_method, best_bound = max(bounds, key=lambda item: item[1])
        detail = ", ".join(f"{name}={value}" for name, value in bounds)
        return LowerBoundResult(
            status="optimal",
            value=float(best_bound),
            integer_bound=best_bound,
            method=f"combinatorial:{best_method}",
            message=detail,
            elapsed_seconds=0.0,
        )

    @staticmethod
    def _schoenheim_bound(v: int, k: int, t: int) -> int:
        if t <= 0:
            return 1
        if t == 1:
            return math.ceil(v / k)
        return math.ceil((v / k) * ILPSolver._schoenheim_bound(v - 1, k - 1, t - 1))

    @staticmethod
    def _solve_with_backends(
        instance: CoverageInstance,
        backend: str,
        time_limit: int,
        cardinality_limit: Optional[int],
        incumbent: Optional[Sequence[int]],
        candidate_indices: Optional[Sequence[int]],
        local_branch_radius: Optional[int],
    ) -> ExactBackendResult:
        backend_order = [backend]
        if backend == "auto":
            backend_order = ["gurobi", "scip", "scipy"]

        skipped_messages = []
        for backend_name in backend_order:
            try:
                if backend_name == "gurobi":
                    result = ILPSolver._solve_gurobi(
                        instance,
                        time_limit,
                        cardinality_limit,
                        incumbent,
                        candidate_indices,
                        local_branch_radius,
                    )
                elif backend_name == "scip":
                    result = ILPSolver._solve_scip(
                        instance,
                        time_limit,
                        cardinality_limit,
                        incumbent,
                        candidate_indices,
                        local_branch_radius,
                    )
                elif backend_name == "scipy":
                    result = ILPSolver._solve_scipy(
                        instance,
                        time_limit,
                        cardinality_limit,
                        candidate_indices,
                        incumbent,
                        local_branch_radius,
                    )
                else:
                    raise ValueError(f"Unknown exact backend: {backend_name}")
            except Exception as exc:  # pragma: no cover - defensive fallback
                skipped_messages.append(f"{backend_name}: {exc}")
                continue

            if result.status == "unavailable":
                skipped_messages.append(result.message)
                continue

            return result

        return ExactBackendResult(
            status="unavailable",
            solution=None,
            method=None,
            message="; ".join(skipped_messages) or "No exact backend is available.",
            elapsed_seconds=0.0,
        )

    @staticmethod
    def _candidate_universe(
        instance: CoverageInstance,
        candidate_indices: Optional[Sequence[int]],
    ) -> List[int]:
        if candidate_indices is None:
            return list(range(len(instance.candidates)))

        limit = len(instance.candidates)
        return sorted(
            {
                int(candidate_index)
                for candidate_index in candidate_indices
                if 0 <= int(candidate_index) < limit
            }
        )

    @staticmethod
    def _subset_to_candidates(
        instance: CoverageInstance,
        candidate_universe: Sequence[int],
        index_map: Optional[dict[int, int]] = None,
    ) -> List[List[int]]:
        index_lookup = (
            index_map
            if index_map is not None
            else {candidate_index: offset for offset, candidate_index in enumerate(candidate_universe)}
        )
        subset_to_candidates = [list() for _ in range(len(instance.s_subsets))]
        for candidate_index in candidate_universe:
            for subset_id in instance.candidate_subset_ids(candidate_index):
                subset_to_candidates[subset_id].append(index_lookup[candidate_index])
        return subset_to_candidates

    @staticmethod
    def _target_to_covering_candidates_at_least_one(
        instance: CoverageInstance,
        candidate_universe: Sequence[int],
        index_map: Optional[dict[int, int]] = None,
    ) -> List[List[int]]:
        index_lookup = (
            index_map
            if index_map is not None
            else {candidate_index: offset for offset, candidate_index in enumerate(candidate_universe)}
        )
        allowed = set(candidate_universe)
        if (
            instance.aggregation_mode == AggregationMode.SINGLE_CANDIDATE
            and instance.covered_by is not None
        ):
            return [
                [
                    index_lookup[candidate_index]
                    for candidate_index in candidate_indices
                    if candidate_index in allowed
                ]
                for candidate_indices in instance.covered_by
            ]

        covering_candidates: List[List[int]] = []
        for subset_ids in instance.target_subset_ids:
            seen = set()
            row: List[int] = []
            for subset_id in subset_ids:
                for candidate_index in instance.subset_to_candidates[subset_id]:
                    if candidate_index not in allowed or candidate_index in seen:
                        continue
                    seen.add(candidate_index)
                    row.append(index_lookup[candidate_index])
            covering_candidates.append(row)
        return covering_candidates

    @staticmethod
    def _restricted_cover_rows_single_candidate(
        instance: CoverageInstance,
        candidate_universe: Sequence[int],
        index_map: Optional[dict[int, int]] = None,
    ) -> List[List[int]]:
        if instance.covered_by is None:
            raise RuntimeError("Single-candidate cover relation was not built.")

        index_lookup = (
            index_map
            if index_map is not None
            else {candidate_index: offset for offset, candidate_index in enumerate(candidate_universe)}
        )
        allowed = set(candidate_universe)
        return [
            [
                index_lookup[candidate_index]
                for candidate_index in candidate_indices
                if candidate_index in allowed
            ]
            for candidate_indices in instance.covered_by
        ]

    @staticmethod
    def _solve_scipy(
        instance: CoverageInstance,
        time_limit: int,
        cardinality_limit: Optional[int],
        candidate_indices: Optional[Sequence[int]],
        incumbent: Optional[Sequence[int]],
        local_branch_radius: Optional[int],
    ) -> ExactBackendResult:
        try:
            import numpy as np
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import lil_matrix
        except ImportError:
            return ExactBackendResult(
                status="unavailable",
                solution=None,
                method="scipy-milp",
                message="scipy is not available.",
                elapsed_seconds=0.0,
            )

        start = time.time()
        candidate_universe = ILPSolver._candidate_universe(instance, candidate_indices)
        index_map = {
            candidate_index: offset
            for offset, candidate_index in enumerate(candidate_universe)
        }
        incumbent_local = {
            index_map[candidate_index]
            for candidate_index in set(incumbent or [])
            if candidate_index in index_map
        }
        use_local_branch = (
            local_branch_radius is not None and len(incumbent_local) > 0
        )

        if instance.coverage_mode == CoverageMode.AT_LEAST_ONE:
            num_candidates = len(candidate_universe)
            num_targets = len(instance.targets)
            extra_rows = int(cardinality_limit is not None) + int(use_local_branch)
            matrix = lil_matrix((num_targets + extra_rows, num_candidates), dtype=float)
            lower_bounds = np.ones(num_targets + extra_rows)
            upper_bounds = np.full(num_targets + extra_rows, np.inf)

            for target_index, candidate_indices in enumerate(
                ILPSolver._target_to_covering_candidates_at_least_one(
                    instance, candidate_universe, index_map
                )
            ):
                for candidate_index in candidate_indices:
                    matrix[target_index, candidate_index] = 1.0

            if cardinality_limit is not None:
                matrix[num_targets, :] = 1.0
                lower_bounds[num_targets] = -np.inf
                upper_bounds[num_targets] = float(cardinality_limit)
                next_row = num_targets + 1
            else:
                next_row = num_targets

            if use_local_branch:
                rhs = float(local_branch_radius - len(incumbent_local))
                for candidate_index in range(num_candidates):
                    matrix[next_row, candidate_index] = (
                        -1.0 if candidate_index in incumbent_local else 1.0
                    )
                lower_bounds[next_row] = -np.inf
                upper_bounds[next_row] = rhs

            objective = (
                np.zeros(num_candidates)
                if cardinality_limit is not None
                else np.ones(num_candidates)
            )
            integrality = np.ones(num_candidates)
        elif instance.aggregation_mode == AggregationMode.SINGLE_CANDIDATE:
            num_candidates = len(candidate_universe)
            num_targets = len(instance.targets)
            extra_rows = int(cardinality_limit is not None) + int(use_local_branch)
            matrix = lil_matrix((num_targets + extra_rows, num_candidates), dtype=float)
            lower_bounds = np.ones(num_targets + extra_rows)
            upper_bounds = np.full(num_targets + extra_rows, np.inf)

            for target_index, candidate_indices in enumerate(
                ILPSolver._restricted_cover_rows_single_candidate(
                    instance, candidate_universe, index_map
                )
            ):
                for candidate_index in candidate_indices:
                    matrix[target_index, candidate_index] = 1.0

            if cardinality_limit is not None:
                matrix[num_targets, :] = 1.0
                lower_bounds[num_targets] = -np.inf
                upper_bounds[num_targets] = float(cardinality_limit)
                next_row = num_targets + 1
            else:
                next_row = num_targets

            if use_local_branch:
                rhs = float(local_branch_radius - len(incumbent_local))
                for candidate_index in range(num_candidates):
                    matrix[next_row, candidate_index] = (
                        -1.0 if candidate_index in incumbent_local else 1.0
                    )
                lower_bounds[next_row] = -np.inf
                upper_bounds[next_row] = rhs

            objective = (
                np.zeros(num_candidates)
                if cardinality_limit is not None
                else np.ones(num_candidates)
            )
            integrality = np.ones(num_candidates)
        else:
            num_candidates = len(candidate_universe)
            num_subsets = len(instance.s_subsets)
            total_variables = num_candidates + num_subsets
            subset_to_candidates = ILPSolver._subset_to_candidates(
                instance, candidate_universe, index_map
            )

            extra_rows = int(cardinality_limit is not None) + int(use_local_branch)
            num_rows = num_subsets + len(instance.targets) + extra_rows
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

            if cardinality_limit is not None:
                for candidate_index in range(num_candidates):
                    matrix[row, candidate_index] = 1.0
                upper_bounds[row] = float(cardinality_limit)
                row += 1

            if use_local_branch:
                rhs = float(local_branch_radius - len(incumbent_local))
                for candidate_index in range(num_candidates):
                    matrix[row, candidate_index] = (
                        -1.0 if candidate_index in incumbent_local else 1.0
                    )
                lower_bounds[row] = -np.inf
                upper_bounds[row] = rhs
                row += 1

            objective = np.concatenate(
                [
                    (
                        np.zeros(num_candidates)
                        if cardinality_limit is not None
                        else np.ones(num_candidates)
                    ),
                    np.zeros(num_subsets),
                ]
            )
            integrality = np.ones(total_variables)

        result = milp(
            objective,
            constraints=LinearConstraint(
                matrix.tocsc(), lb=lower_bounds, ub=upper_bounds
            ),
            bounds=Bounds(lb=0.0, ub=1.0),
            integrality=integrality,
            options={"time_limit": time_limit},
        )

        elapsed = time.time() - start
        method = "scipy-milp"
        message = str(getattr(result, "message", ""))
        if result.success:
            chosen = [
                candidate_universe[index]
                for index in range(num_candidates)
                if result.x[index] > 0.5
            ]
            return ILPSolver._validated_backend_solution(
                instance=instance,
                solution=chosen,
                status="feasible" if cardinality_limit is not None else "optimal",
                method=method,
                message=message,
                elapsed_seconds=elapsed,
            )

        if getattr(result, "status", None) == 2 or "infeasible" in message.lower():
            return ExactBackendResult(
                status="infeasible",
                solution=None,
                method=method,
                message=message,
                elapsed_seconds=elapsed,
            )

        return ExactBackendResult(
            status="unknown",
            solution=None,
            method=method,
            message=message or "Exact solve failed or timed out.",
            elapsed_seconds=elapsed,
        )

    @staticmethod
    def _solve_gurobi(
        instance: CoverageInstance,
        time_limit: int,
        cardinality_limit: Optional[int],
        incumbent: Optional[Sequence[int]],
        candidate_indices: Optional[Sequence[int]],
        local_branch_radius: Optional[int],
    ) -> ExactBackendResult:
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            return ExactBackendResult(
                status="unavailable",
                solution=None,
                method="gurobi",
                message="gurobipy is not available.",
                elapsed_seconds=0.0,
            )

        start = time.time()
        model = gp.Model("covering_design")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = float(time_limit)

        candidate_universe = ILPSolver._candidate_universe(instance, candidate_indices)
        index_map = {
            candidate_index: offset
            for offset, candidate_index in enumerate(candidate_universe)
        }
        incumbent_local = {
            index_map[candidate_index]
            for candidate_index in set(incumbent or [])
            if candidate_index in index_map
        }
        num_candidates = len(candidate_universe)
        x_vars = model.addVars(num_candidates, vtype=GRB.BINARY, name="x")
        if cardinality_limit is None:
            model.setObjective(gp.quicksum(x_vars[i] for i in range(num_candidates)), GRB.MINIMIZE)
        else:
            model.setObjective(gp.LinExpr(0.0), GRB.MINIMIZE)
            model.addConstr(
                gp.quicksum(x_vars[i] for i in range(num_candidates))
                <= cardinality_limit
            )

        if incumbent is not None:
            incumbent_set = set(incumbent)
            for local_index, candidate_index in enumerate(candidate_universe):
                x_vars[local_index].Start = (
                    1.0 if candidate_index in incumbent_set else 0.0
                )

        if instance.coverage_mode == CoverageMode.AT_LEAST_ONE:
            for target_index, candidate_indices in enumerate(
                ILPSolver._target_to_covering_candidates_at_least_one(
                    instance, candidate_universe, index_map
                )
            ):
                model.addConstr(
                    gp.quicksum(x_vars[i] for i in candidate_indices) >= 1,
                    name=f"cover_target_{target_index}",
                )
        elif instance.aggregation_mode == AggregationMode.SINGLE_CANDIDATE:
            for target_index, candidate_indices in enumerate(
                ILPSolver._restricted_cover_rows_single_candidate(
                    instance, candidate_universe, index_map
                )
            ):
                model.addConstr(
                    gp.quicksum(x_vars[i] for i in candidate_indices) >= 1,
                    name=f"cover_target_{target_index}",
                )
        else:
            subset_to_candidates = ILPSolver._subset_to_candidates(
                instance, candidate_universe, index_map
            )
            y_vars = model.addVars(len(instance.s_subsets), vtype=GRB.BINARY, name="y")
            for subset_id, candidate_indices in enumerate(subset_to_candidates):
                model.addConstr(
                    y_vars[subset_id]
                    <= gp.quicksum(x_vars[i] for i in candidate_indices),
                    name=f"activate_subset_{subset_id}",
                )
            for target_index, subset_ids in enumerate(instance.target_subset_ids):
                model.addConstr(
                    gp.quicksum(y_vars[subset_id] for subset_id in subset_ids)
                    >= instance.required_subset_count,
                    name=f"cover_target_{target_index}",
                )

        if local_branch_radius is not None and incumbent_local:
            inside = gp.quicksum(1.0 - x_vars[i] for i in incumbent_local)
            outside = gp.quicksum(
                x_vars[i] for i in range(num_candidates) if i not in incumbent_local
            )
            model.addConstr(
                inside + outside <= float(local_branch_radius),
                name="local_branching",
            )

        model.optimize()
        elapsed = time.time() - start

        if model.Status == GRB.OPTIMAL:
            chosen = [
                candidate_universe[index]
                for index in range(num_candidates)
                if x_vars[index].X > 0.5
            ]
            return ILPSolver._validated_backend_solution(
                instance=instance,
                solution=chosen,
                status="feasible" if cardinality_limit is not None else "optimal",
                method="gurobi",
                message="optimal",
                elapsed_seconds=elapsed,
            )

        if model.SolCount > 0:
            chosen = [
                candidate_universe[index]
                for index in range(num_candidates)
                if x_vars[index].X > 0.5
            ]
            return ILPSolver._validated_backend_solution(
                instance=instance,
                solution=chosen,
                status="feasible",
                method="gurobi",
                message=f"gurobi status {model.Status}; feasible incumbent found",
                elapsed_seconds=elapsed,
            )

        if model.Status == GRB.INFEASIBLE:
            return ExactBackendResult(
                status="infeasible",
                solution=None,
                method="gurobi",
                message="infeasible",
                elapsed_seconds=elapsed,
            )

        return ExactBackendResult(
            status="unknown",
            solution=None,
            method="gurobi",
            message=f"gurobi status {model.Status}",
            elapsed_seconds=elapsed,
        )

    @staticmethod
    def _solve_scip(
        instance: CoverageInstance,
        time_limit: int,
        cardinality_limit: Optional[int],
        incumbent: Optional[Sequence[int]],
        candidate_indices: Optional[Sequence[int]],
        local_branch_radius: Optional[int],
    ) -> ExactBackendResult:
        try:
            from pyscipopt import Model, quicksum
        except ImportError:
            return ExactBackendResult(
                status="unavailable",
                solution=None,
                method="scip",
                message="pyscipopt is not available.",
                elapsed_seconds=0.0,
            )

        start = time.time()
        model = Model("covering_design")
        model.hideOutput()
        model.setParam("limits/time", float(time_limit))

        candidate_universe = ILPSolver._candidate_universe(instance, candidate_indices)
        index_map = {
            candidate_index: offset
            for offset, candidate_index in enumerate(candidate_universe)
        }
        incumbent_local = {
            index_map[candidate_index]
            for candidate_index in set(incumbent or [])
            if candidate_index in index_map
        }
        num_candidates = len(candidate_universe)
        x_vars = [
            model.addVar(vtype="B", name=f"x_{candidate_index}")
            for candidate_index in range(num_candidates)
        ]

        if cardinality_limit is None:
            model.setObjective(quicksum(x_vars), "minimize")
        else:
            model.setObjective(quicksum([]), "minimize")
            model.addCons(quicksum(x_vars) <= cardinality_limit)

        active_subset_ids = set()
        y_vars = None

        if instance.coverage_mode == CoverageMode.AT_LEAST_ONE:
            for candidate_indices in ILPSolver._target_to_covering_candidates_at_least_one(
                instance, candidate_universe, index_map
            ):
                model.addCons(quicksum(x_vars[i] for i in candidate_indices) >= 1)
        elif instance.aggregation_mode == AggregationMode.SINGLE_CANDIDATE:
            for candidate_indices in ILPSolver._restricted_cover_rows_single_candidate(
                instance, candidate_universe, index_map
            ):
                model.addCons(quicksum(x_vars[i] for i in candidate_indices) >= 1)
        else:
            subset_to_candidates = ILPSolver._subset_to_candidates(
                instance, candidate_universe, index_map
            )
            y_vars = [
                model.addVar(vtype="B", name=f"y_{subset_id}")
                for subset_id in range(len(instance.s_subsets))
            ]
            for subset_id, candidate_indices in enumerate(subset_to_candidates):
                model.addCons(
                    y_vars[subset_id]
                    <= quicksum(x_vars[i] for i in candidate_indices)
                )
            for subset_ids in instance.target_subset_ids:
                model.addCons(
                    quicksum(y_vars[subset_id] for subset_id in subset_ids)
                    >= instance.required_subset_count
                )

        if local_branch_radius is not None and incumbent_local:
            model.addCons(
                quicksum(1.0 - x_vars[i] for i in incumbent_local)
                + quicksum(
                    x_vars[i] for i in range(num_candidates) if i not in incumbent_local
                )
                <= float(local_branch_radius)
            )

        if incumbent is not None:
            try:
                incumbent_set = set(incumbent)
                allowed_incumbent = incumbent_set & set(candidate_universe)
                if allowed_incumbent:
                    incumbent_solution = model.createSol()
                    for local_index, candidate_index in enumerate(candidate_universe):
                        if candidate_index in allowed_incumbent:
                            model.setSolVal(incumbent_solution, x_vars[local_index], 1.0)

                    if y_vars is not None:
                        for candidate_index in allowed_incumbent:
                            active_subset_ids.update(
                                instance.candidate_subset_ids(candidate_index)
                            )
                        for subset_id in active_subset_ids:
                            model.setSolVal(incumbent_solution, y_vars[subset_id], 1.0)

                    model.addSol(incumbent_solution)
            except Exception:
                pass

        model.optimize()
        elapsed = time.time() - start
        status = model.getStatus()

        if status == "optimal":
            chosen = [
                candidate_universe[index]
                for index in range(num_candidates)
                if model.getVal(x_vars[index]) > 0.5
            ]
            return ILPSolver._validated_backend_solution(
                instance=instance,
                solution=chosen,
                status="feasible" if cardinality_limit is not None else "optimal",
                method="scip",
                message=status,
                elapsed_seconds=elapsed,
            )

        if model.getNSols() > 0:
            chosen = [
                candidate_universe[index]
                for index in range(num_candidates)
                if model.getVal(x_vars[index]) > 0.5
            ]
            return ILPSolver._validated_backend_solution(
                instance=instance,
                solution=chosen,
                status="feasible",
                method="scip",
                message=f"{status}; feasible incumbent found",
                elapsed_seconds=elapsed,
            )

        if status == "infeasible":
            return ExactBackendResult(
                status="infeasible",
                solution=None,
                method="scip",
                message=status,
                elapsed_seconds=elapsed,
            )

        return ExactBackendResult(
            status="unknown",
            solution=None,
            method="scip",
            message=status,
            elapsed_seconds=elapsed,
        )
