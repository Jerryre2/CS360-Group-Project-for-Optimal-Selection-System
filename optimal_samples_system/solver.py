"""End-to-end solver orchestration."""

from __future__ import annotations

import math
import random
import time
from typing import Optional, Sequence

from .config import (
    AggregationMode,
    CoverageMode,
    LOGGER,
    ProblemConfig,
    SolveResult,
    SolverConfig,
    choose_seed,
)
from .exact import ILPSolver
from .heuristics import (
    GreedySolver,
    ImprovedLocalSearch,
    ImprovedSA,
    RedundancyEliminator,
)
from .instance import CoverageInstance
from .storage import ResultDatabase
from .tracking import CoverageTracker


class OptimalSamplesSolver:
    def __init__(
        self,
        config_or_m: ProblemConfig | int,
        n: Optional[int] = None,
        k: Optional[int] = None,
        j: Optional[int] = None,
        s: Optional[int] = None,
        samples: Optional[Sequence[int]] = None,
        coverage_mode: CoverageMode = CoverageMode.AT_LEAST_ONE,
        aggregation_mode: AggregationMode = AggregationMode.DISTINCT_SUBSETS,
        required_r: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        if isinstance(config_or_m, ProblemConfig):
            self.config = config_or_m
        else:
            if None in (n, k, j, s):
                raise ValueError("m, n, k, j, and s must all be supplied.")
            self.config = ProblemConfig(
                m=config_or_m,
                n=int(n),
                k=int(k),
                j=int(j),
                s=int(s),
                samples=tuple(samples) if samples is not None else None,
                coverage_mode=coverage_mode,
                aggregation_mode=aggregation_mode,
                required_r=required_r,
                seed=seed,
            )

        self.seed = choose_seed(self.config.seed)
        self.config.seed = self.seed
        self.rng = random.Random(self.seed)
        self.instance = CoverageInstance(self.config, rng=self.rng)

    def solve(self, solver_config: Optional[SolverConfig] = None) -> SolveResult:
        solver_config = solver_config or SolverConfig()
        start = time.time()

        exact_size: Optional[int] = None
        exact_method: Optional[str] = None

        should_try_exact = (
            solver_config.use_ilp
            and len(self.instance.candidates) <= 5000
            and len(self.instance.targets) <= 10000
        )
        if should_try_exact:
            LOGGER.info("\n[Exact verification]")
            exact_solution, exact_method = ILPSolver.solve(self.instance)
            if exact_solution is not None:
                exact_size = len(exact_solution)

        max_local_steps = solver_config.max_local_steps
        if max_local_steps is None:
            max_local_steps = min(3000, max(200, len(self.instance.candidates) * 3))

        max_sa_iterations = solver_config.max_sa_iterations
        if max_sa_iterations is None:
            max_sa_iterations = min(5000, max(300, len(self.instance.candidates) * 3))

        if len(self.instance.candidates) > 20000:
            LOGGER.info(
                "Large candidate set detected; skipping local search and simulated annealing."
            )
            max_local_steps = 0
            max_sa_iterations = 0

        best_solution = None
        best_size = math.inf

        for restart in range(solver_config.n_restarts):
            LOGGER.info(f"\n===== Restart {restart + 1}/{solver_config.n_restarts} =====")

            LOGGER.info("[Greedy]")
            greedy = GreedySolver(self.rng)
            greedy_solution = greedy.solve(self.instance, randomized=(restart > 0))
            LOGGER.info(f"  Greedy size: {len(greedy_solution)}")

            tracker = CoverageTracker(self.instance)
            tracker.reset(greedy_solution)
            if not tracker.is_feasible():
                raise RuntimeError("Greedy produced an infeasible solution.")

            current_solution = RedundancyEliminator().eliminate(
                self.instance, greedy_solution
            )

            if max_local_steps > 0:
                LOGGER.info("[Local search]")
                local_search = ImprovedLocalSearch(
                    self.instance,
                    current_solution,
                    rng=self.rng,
                    max_steps=max_local_steps,
                    warmup=min(300, max_local_steps),
                    candidate_sample_size=solver_config.candidate_sample_size,
                    use_neural_guidance=solver_config.use_neural_guidance,
                )
                current_solution = local_search.solve()

            if max_sa_iterations > 0:
                LOGGER.info("[Simulated annealing]")
                sa = ImprovedSA(
                    self.instance,
                    current_solution,
                    rng=self.rng,
                    max_iter=max_sa_iterations,
                )
                current_solution = sa.solve()

            current_solution = RedundancyEliminator().eliminate(
                self.instance, current_solution
            )

            tracker.reset(current_solution)
            if not tracker.is_feasible():
                raise RuntimeError("Post-processing produced an infeasible solution.")

            LOGGER.info(
                f"  Restart result: greedy {len(greedy_solution)} -> final {len(current_solution)}"
            )

            if len(current_solution) < best_size:
                best_solution = current_solution
                best_size = len(current_solution)

        if best_solution is None:
            raise RuntimeError("No feasible solution was found.")

        elapsed = time.time() - start
        groups = [self.instance.candidate_label(index) for index in sorted(best_solution)]

        LOGGER.info("\n" + "=" * 70)
        LOGGER.info("Final result")
        LOGGER.info("=" * 70)
        LOGGER.info(f"Best family size: {best_size}")
        if exact_size is not None:
            LOGGER.info(f"Exact size: {exact_size} (gap {best_size - exact_size})")
        LOGGER.info(f"Elapsed time: {elapsed:.2f}s")
        LOGGER.info("Selected groups:")
        for group_number, group in enumerate(groups, start=1):
            LOGGER.info(f"  Group {group_number}: {group}")

        result = SolveResult(
            solution_indices=sorted(best_solution),
            groups=groups,
            num_groups=int(best_size),
            exact_size=exact_size,
            exact_method=exact_method,
            samples=list(self.instance.samples),
            params=self.config.to_dict(),
            solver=solver_config.to_dict(),
            elapsed_seconds=elapsed,
            seed=self.seed,
            required_subsets_per_target=self.instance.required_subset_count,
            num_targets=len(self.instance.targets),
            num_candidates=len(self.instance.candidates),
            aggregation_mode=self.instance.aggregation_mode.value,
            coverage_mode=self.instance.coverage_mode.value,
        )

        if solver_config.save_result:
            ResultDatabase(solver_config.db_dir).save(result.to_dict())

        return result
