"""Exact verification models."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .config import AggregationMode, LOGGER
from .instance import CoverageInstance


@dataclass
class ExactBackendResult:
    status: str
    solution: Optional[List[int]]
    method: Optional[str]
    message: str
    elapsed_seconds: float


class ILPSolver:
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
    def prove_no_solution_at_or_below(
        instance: CoverageInstance,
        cardinality_limit: int,
        backend: str = "auto",
        time_limit: int = 300,
    ) -> ExactBackendResult:
        return ILPSolver._solve_with_backends(
            instance=instance,
            backend=backend,
            time_limit=time_limit,
            cardinality_limit=cardinality_limit,
            incumbent=None,
        )

    @staticmethod
    def _solve_with_backends(
        instance: CoverageInstance,
        backend: str,
        time_limit: int,
        cardinality_limit: Optional[int],
        incumbent: Optional[Sequence[int]],
    ) -> ExactBackendResult:
        backend_order = [backend]
        if backend == "auto":
            backend_order = ["gurobi", "scip", "scipy"]

        skipped_messages = []
        for backend_name in backend_order:
            if backend_name == "gurobi":
                result = ILPSolver._solve_gurobi(
                    instance, time_limit, cardinality_limit, incumbent
                )
            elif backend_name == "scip":
                result = ILPSolver._solve_scip(
                    instance, time_limit, cardinality_limit, incumbent
                )
            elif backend_name == "scipy":
                result = ILPSolver._solve_scipy(
                    instance, time_limit, cardinality_limit
                )
            else:
                raise ValueError(f"Unknown exact backend: {backend_name}")

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
    def _subset_to_candidates(instance: CoverageInstance) -> List[List[int]]:
        subset_to_candidates = [list() for _ in range(len(instance.s_subsets))]
        for candidate_index in range(len(instance.candidates)):
            for subset_id in instance.candidate_subset_ids(candidate_index):
                subset_to_candidates[subset_id].append(candidate_index)
        return subset_to_candidates

    @staticmethod
    def _solve_scipy(
        instance: CoverageInstance,
        time_limit: int,
        cardinality_limit: Optional[int],
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

        if instance.aggregation_mode == AggregationMode.SINGLE_CANDIDATE:
            if instance.covered_by is None:
                raise RuntimeError("Single-candidate cover relation was not built.")

            num_candidates = len(instance.candidates)
            num_targets = len(instance.targets)
            extra_rows = 1 if cardinality_limit is not None else 0
            matrix = lil_matrix((num_targets + extra_rows, num_candidates), dtype=float)
            lower_bounds = np.ones(num_targets + extra_rows)
            upper_bounds = np.full(num_targets + extra_rows, np.inf)

            for target_index, candidate_indices in enumerate(instance.covered_by):
                for candidate_index in candidate_indices:
                    matrix[target_index, candidate_index] = 1.0

            if cardinality_limit is not None:
                matrix[num_targets, :] = 1.0
                lower_bounds[num_targets] = -np.inf
                upper_bounds[num_targets] = float(cardinality_limit)

            objective = (
                np.zeros(num_candidates)
                if cardinality_limit is not None
                else np.ones(num_candidates)
            )
            integrality = np.ones(num_candidates)
        else:
            num_candidates = len(instance.candidates)
            num_subsets = len(instance.s_subsets)
            total_variables = num_candidates + num_subsets
            subset_to_candidates = ILPSolver._subset_to_candidates(instance)

            extra_rows = 1 if cardinality_limit is not None else 0
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
                index for index in range(len(instance.candidates)) if result.x[index] > 0.5
            ]
            return ExactBackendResult(
                status="feasible" if cardinality_limit is not None else "optimal",
                solution=chosen,
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

        num_candidates = len(instance.candidates)
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
            for candidate_index in range(num_candidates):
                x_vars[candidate_index].Start = (
                    1.0 if candidate_index in incumbent_set else 0.0
                )

        if instance.aggregation_mode == AggregationMode.SINGLE_CANDIDATE:
            if instance.covered_by is None:
                raise RuntimeError("Single-candidate cover relation was not built.")
            for target_index, candidate_indices in enumerate(instance.covered_by):
                model.addConstr(
                    gp.quicksum(x_vars[i] for i in candidate_indices) >= 1,
                    name=f"cover_target_{target_index}",
                )
        else:
            subset_to_candidates = ILPSolver._subset_to_candidates(instance)
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

        model.optimize()
        elapsed = time.time() - start

        if model.Status == GRB.OPTIMAL:
            chosen = [
                index
                for index in range(num_candidates)
                if x_vars[index].X > 0.5
            ]
            return ExactBackendResult(
                status="feasible" if cardinality_limit is not None else "optimal",
                solution=chosen,
                method="gurobi",
                message="optimal",
                elapsed_seconds=elapsed,
            )

        if model.SolCount > 0:
            chosen = [
                index
                for index in range(num_candidates)
                if x_vars[index].X > 0.5
            ]
            return ExactBackendResult(
                status="feasible",
                solution=chosen,
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
    ) -> ExactBackendResult:
        _ = incumbent
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

        num_candidates = len(instance.candidates)
        x_vars = [
            model.addVar(vtype="B", name=f"x_{candidate_index}")
            for candidate_index in range(num_candidates)
        ]

        if cardinality_limit is None:
            model.setObjective(quicksum(x_vars), "minimize")
        else:
            model.setObjective(quicksum([]), "minimize")
            model.addCons(quicksum(x_vars) <= cardinality_limit)

        if instance.aggregation_mode == AggregationMode.SINGLE_CANDIDATE:
            if instance.covered_by is None:
                raise RuntimeError("Single-candidate cover relation was not built.")
            for candidate_indices in instance.covered_by:
                model.addCons(quicksum(x_vars[i] for i in candidate_indices) >= 1)
        else:
            subset_to_candidates = ILPSolver._subset_to_candidates(instance)
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

        model.optimize()
        elapsed = time.time() - start
        status = model.getStatus()

        if status == "optimal":
            chosen = [
                index
                for index in range(num_candidates)
                if model.getVal(x_vars[index]) > 0.5
            ]
            return ExactBackendResult(
                status="feasible" if cardinality_limit is not None else "optimal",
                solution=chosen,
                method="scip",
                message=status,
                elapsed_seconds=elapsed,
            )

        if model.getNSols() > 0:
            chosen = [
                index
                for index in range(num_candidates)
                if model.getVal(x_vars[index]) > 0.5
            ]
            return ExactBackendResult(
                status="feasible",
                solution=chosen,
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
