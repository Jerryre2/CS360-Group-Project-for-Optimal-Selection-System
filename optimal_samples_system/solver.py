"""End-to-end solver orchestration."""

from __future__ import annotations

import heapq
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
from .validation import audit_solution_indices


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

    def _build_reduced_exact_core(
        self,
        solution: Sequence[int],
        core_cap: int,
        anchor_solutions: Optional[Sequence[Sequence[int]]] = None,
        pivot_candidates: Optional[Sequence[int]] = None,
    ) -> list[int]:
        tracker = CoverageTracker(self.instance)
        tracker.reset(solution)

        incumbent = list(sorted(tracker.in_solution))
        cap = min(
            len(self.instance.candidates),
            max(len(incumbent) + 24, core_cap),
        )
        core = set(incumbent)
        if len(core) >= cap:
            return sorted(core)

        if pivot_candidates is None:
            soft_pivots = heapq.nsmallest(
                min(12, len(incumbent)),
                incumbent,
                key=lambda candidate_index: (
                    tracker.exclusive_count(candidate_index),
                    tracker.redundancy_score(candidate_index),
                ),
            )
            hard_pivots = heapq.nlargest(
                min(12, len(incumbent)),
                incumbent,
                key=lambda candidate_index: (
                    tracker.exclusive_count(candidate_index),
                    -tracker.redundancy_score(candidate_index),
                ),
            )
            pivots = list(dict.fromkeys(hard_pivots + soft_pivots))
        else:
            pivots = list(
                dict.fromkeys(
                    candidate_index
                    for candidate_index in pivot_candidates
                    if candidate_index in tracker.in_solution
                )
            )
            if not pivots:
                pivots = list(sorted(incumbent[: min(12, len(incumbent))]))

        for anchor_solution in anchor_solutions or ():
            for candidate_index in sorted(anchor_solution, key=self.instance.candidate_span, reverse=True)[:6]:
                if candidate_index not in pivots:
                    pivots.append(candidate_index)

        critical_subsets = {
            subset_id
            for pivot in pivots
            for subset_id in self.instance.candidate_subset_ids(pivot)
            if tracker.subset_cover_count[subset_id] == 1
        }

        candidate_scores: dict[int, float] = {}
        for pivot in pivots:
            for subset_id in self.instance.candidate_subset_ids(pivot):
                weight = 8.0 if subset_id in critical_subsets else 1.0
                for candidate_index in self.instance.subset_to_candidates[subset_id]:
                    if candidate_index in core:
                        continue
                    candidate_scores[candidate_index] = (
                        candidate_scores.get(candidate_index, 0.0) + weight
                    )

        if not candidate_scores:
            fallback = heapq.nlargest(
                min(cap - len(core), len(self.instance.candidates) - len(core)),
                [
                    candidate_index
                    for candidate_index in range(len(self.instance.candidates))
                    if candidate_index not in core
                ],
                key=self.instance.candidate_span,
            )
            core.update(fallback)
            return sorted(core)

        def candidate_score(candidate_index: int) -> float:
            overlap = sum(
                self.instance.candidate_overlap_in_s_subsets(candidate_index, pivot)
                for pivot in pivots
            )
            critical_overlap = sum(
                1
                for subset_id in self.instance.candidate_subset_ids(candidate_index)
                if subset_id in critical_subsets
            )
            return (
                candidate_scores.get(candidate_index, 0.0)
                + 3.0 * critical_overlap
                + 0.1 * overlap
                + 0.001 * self.instance.candidate_span(candidate_index)
            )

        ranked = sorted(candidate_scores, key=candidate_score, reverse=True)
        core.update(ranked[: max(0, cap - len(core))])
        return sorted(core)

    def _targeted_reduced_exact_cores(
        self,
        solution: Sequence[int],
        elite_pool: Sequence[Sequence[int]],
        core_cap: int,
    ) -> list[list[int]]:
        tracker = CoverageTracker(self.instance)
        tracker.reset(solution)
        incumbent = list(sorted(tracker.in_solution))
        anchors = [
            elite
            for elite in elite_pool
            if self._solution_signature(elite) != self._solution_signature(incumbent)
        ][:2]

        weakest = heapq.nsmallest(
            min(18, len(incumbent)),
            incumbent,
            key=lambda candidate_index: (
                tracker.exclusive_count(candidate_index),
                tracker.redundancy_score(candidate_index),
                -self.instance.candidate_span(candidate_index),
            ),
        )
        hardest = heapq.nlargest(
            min(18, len(incumbent)),
            incumbent,
            key=lambda candidate_index: (
                tracker.exclusive_count(candidate_index),
                -tracker.redundancy_score(candidate_index),
                self.instance.candidate_span(candidate_index),
            ),
        )

        pivot_sets: list[tuple[int, ...]] = [
            tuple(),
            tuple(weakest),
            tuple(hardest),
        ]

        for anchor in anchors:
            diff = list((set(incumbent) ^ set(anchor)) & set(incumbent))
            if diff:
                diff.sort(
                    key=lambda candidate_index: (
                        tracker.exclusive_count(candidate_index),
                        -self.instance.candidate_span(candidate_index),
                    ),
                    reverse=True,
                )
                pivot_sets.append(tuple(diff[: min(18, len(diff))]))

        cores: list[list[int]] = []
        seen = set()
        for pivots in pivot_sets:
            core = self._build_reduced_exact_core(
                incumbent,
                core_cap,
                anchor_solutions=anchors,
                pivot_candidates=pivots if pivots else None,
            )
            signature = tuple(core)
            if signature in seen:
                continue
            seen.add(signature)
            cores.append(core)

        return cores

    @staticmethod
    def _solution_signature(solution: Sequence[int]) -> tuple[int, ...]:
        return tuple(sorted(solution))

    def _update_elite_pool(
        self,
        elite_pool: list[list[int]],
        solution: Sequence[int],
        solver_config: SolverConfig,
    ) -> list[list[int]]:
        signature = self._solution_signature(solution)
        if any(self._solution_signature(existing) == signature for existing in elite_pool):
            return sorted(elite_pool, key=lambda item: (len(item), self._solution_signature(item)))

        elite_pool = elite_pool + [list(sorted(solution))]
        elite_pool.sort(key=lambda item: (len(item), self._solution_signature(item)))
        return elite_pool[: max(1, solver_config.elite_pool_size)]

    def _backbone_repair_from_union(
        self,
        left: Sequence[int],
        right: Sequence[int],
    ) -> list[int]:
        common = sorted(set(left) & set(right))
        if not common:
            return []

        union = sorted(set(left) | set(right))
        tracker = CoverageTracker(self.instance)
        tracker.reset(common)

        available = [candidate for candidate in union if candidate not in tracker.in_solution]
        while not tracker.is_feasible():
            if not available:
                return []

            candidate_add = max(
                available,
                key=lambda candidate_index: (
                    tracker.marginal_gain(candidate_index),
                    self.instance.candidate_span(candidate_index),
                ),
            )
            if tracker.marginal_gain(candidate_add) <= 0:
                return []
            tracker.add(candidate_add)
            available.remove(candidate_add)

        repaired = RedundancyEliminator().eliminate(
            self.instance, sorted(tracker.in_solution)
        )
        return repaired

    def _path_relink(
        self,
        left: Sequence[int],
        right: Sequence[int],
    ) -> list[int]:
        left_sorted = list(sorted(left))
        right_sorted = list(sorted(right))
        best = list(left_sorted)

        union_seed = RedundancyEliminator().eliminate(
            self.instance, sorted(set(left_sorted) | set(right_sorted))
        )
        if len(union_seed) < len(best):
            best = union_seed

        backbone_seed = self._backbone_repair_from_union(left_sorted, right_sorted)
        if backbone_seed and len(backbone_seed) < len(best):
            best = backbone_seed

        return best

    def _elite_guided_seed(
        self,
        seed_solution: Sequence[int],
        elite_pool: Sequence[Sequence[int]],
        solver_config: SolverConfig,
    ) -> list[int]:
        base = list(sorted(seed_solution))
        if (
            not solver_config.elite_guided_restarts
            or not solver_config.path_relinking
            or not elite_pool
        ):
            return base

        best = base
        for guide in elite_pool[: max(1, solver_config.elite_pool_size)]:
            candidate = self._path_relink(base, guide)
            if candidate and len(candidate) < len(best):
                best = candidate

        if len(best) < len(base):
            LOGGER.info(
                f"  Elite-guided restart seed: {len(base)} -> {len(best)}"
            )
        return best

    def _elite_intensify(
        self,
        solution: Sequence[int],
        elite_pool: Sequence[Sequence[int]],
        solver_config: SolverConfig,
        max_local_steps: int,
        max_sa_iterations: int,
    ) -> list[int]:
        if (
            not solver_config.path_relinking
            or not elite_pool
            or solver_config.elite_pool_size <= 0
        ):
            return list(sorted(solution))

        best = list(sorted(solution))
        for guide in elite_pool[: max(1, solver_config.elite_pool_size)]:
            candidate = self._path_relink(best, guide)
            if candidate and len(candidate) < len(best):
                best = candidate

        if len(best) >= len(solution):
            return list(sorted(solution))

        LOGGER.info(f"  Elite intensification: {len(solution)} -> {len(best)}")

        local_budget = min(max_local_steps // 4, 400) if max_local_steps > 0 else 0
        if local_budget > 0:
            local_search = ImprovedLocalSearch(
                self.instance,
                best,
                rng=self.rng,
                max_steps=local_budget,
                warmup=min(80, local_budget),
                candidate_sample_size=max(
                    solver_config.candidate_sample_size,
                    solver_config.candidate_sample_size + 16,
                ),
                adaptive_neighborhoods=solver_config.adaptive_neighborhoods,
            )
            best = local_search.solve()

        sa_budget = min(max_sa_iterations // 4, 600) if max_sa_iterations > 0 else 0
        if sa_budget > 0:
            sa = ImprovedSA(
                self.instance,
                best,
                rng=self.rng,
                max_iter=sa_budget,
                adaptive_neighborhoods=solver_config.adaptive_neighborhoods,
            )
            best = sa.solve()

        return RedundancyEliminator().eliminate(self.instance, best)

    def _reduced_exact_polish(
        self,
        solution: Sequence[int],
        elite_pool: Sequence[Sequence[int]],
        solver_config: SolverConfig,
    ) -> tuple[list[int], Optional[str]]:
        if not solver_config.use_ilp or not solver_config.reduced_exact_polish:
            return list(solution), None

        current = list(sorted(solution))
        method: Optional[str] = None
        LOGGER.info("[Reduced-core exact polish]")

        for round_index in range(max(1, solver_config.reduced_exact_rounds)):
            cap_factor = 1.0 + 0.5 * round_index
            time_limit = max(
                1,
                solver_config.reduced_exact_time_limit + 2 * round_index,
            )
            core_cap = max(
                len(current) + 24,
                int(math.ceil(solver_config.reduced_exact_core_cap * cap_factor)),
            )
            improved_in_round = False
            cores = self._targeted_reduced_exact_cores(current, elite_pool, core_cap)
            for core_index, core in enumerate(cores, start=1):
                if len(core) <= len(current):
                    continue

                local_radius = max(
                    12,
                    min(
                        solver_config.local_branching_base_radius * 2,
                        len(current) // 6,
                    ),
                )
                local_result = ILPSolver.improve_locally(
                    self.instance,
                    incumbent=current,
                    radius=local_radius,
                    backend=solver_config.exact_backend,
                    time_limit=max(1, time_limit // 2),
                    candidate_indices=core,
                )
                method = local_result.method or method

                candidate_solution = None
                candidate_method = local_result.method
                if local_result.solution is not None and len(local_result.solution) < len(current):
                    candidate_solution = local_result.solution
                else:
                    result = ILPSolver.solve_restricted(
                        self.instance,
                        candidate_indices=core,
                        backend=solver_config.exact_backend,
                        time_limit=time_limit,
                        incumbent=current,
                    )
                    method = result.method or method
                    candidate_method = result.method
                    if result.solution is not None and len(result.solution) < len(current):
                        candidate_solution = result.solution

                if candidate_solution is None:
                    continue

                polished = RedundancyEliminator().eliminate(
                    self.instance,
                    candidate_solution,
                )
                tracker = CoverageTracker(self.instance)
                tracker.reset(polished)
                if not tracker.is_feasible():
                    raise RuntimeError(
                        "Reduced-core exact polish produced an infeasible solution."
                    )

                LOGGER.info(
                    "  Reduced-core polish round "
                    f"{round_index + 1}, core {core_index}: "
                    f"{len(current)} -> {len(polished)} "
                    f"({candidate_method})"
                )
                current = polished
                improved_in_round = True
                break

            if not improved_in_round:
                LOGGER.info(
                    "  Reduced-core polish round "
                    f"{round_index + 1}: no improvement across targeted cores"
                )

        return current, method

    def _mid_size_exact_improvement(
        self,
        solution: Sequence[int],
        solver_config: SolverConfig,
    ) -> tuple[list[int], Optional[str]]:
        if (
            not solver_config.use_ilp
            or not solver_config.mid_size_exact_improvement
            or len(self.instance.candidates) > solver_config.mid_size_exact_candidate_threshold
        ):
            return list(solution), None

        LOGGER.info("[Mid-size exact improvement]")
        current = list(sorted(solution))
        method: Optional[str] = None
        deadline = time.time() + max(1, solver_config.mid_size_exact_time_limit)
        lower_bound = (
            ILPSolver.combinatorial_lower_bound(self.instance).integer_bound or 0
        )

        while len(current) - 1 >= lower_bound:
            remaining = max(1, int(math.ceil(deadline - time.time())))
            if remaining <= 0:
                break

            target_size = len(current) - 1
            result = ILPSolver.prove_no_solution_at_or_below(
                self.instance,
                cardinality_limit=target_size,
                backend=solver_config.exact_backend,
                time_limit=remaining,
                incumbent=current,
            )
            method = result.method or method

            if result.status == "feasible" and result.solution is not None:
                improved = RedundancyEliminator().eliminate(
                    self.instance, result.solution
                )
                tracker = CoverageTracker(self.instance)
                tracker.reset(improved)
                if not tracker.is_feasible():
                    raise RuntimeError(
                        "Mid-size exact improvement produced an infeasible solution."
                    )
                if len(improved) < len(current):
                    LOGGER.info(
                        "  Mid-size exact improvement: "
                        f"{len(current)} -> {len(improved)} "
                        f"({result.method}, target<={target_size})"
                    )
                    current = improved
                    continue

            if result.status == "infeasible":
                LOGGER.info(
                    "  Mid-size exact improvement: no solution at or below "
                    f"{target_size} ({result.method})"
                )
            else:
                LOGGER.info(
                    "  Mid-size exact improvement: no better incumbent "
                    f"({result.method}: {result.message})"
                )
            break

        return current, method

    def _local_branching_improvement(
        self,
        solution: Sequence[int],
        solver_config: SolverConfig,
    ) -> tuple[list[int], Optional[str]]:
        if (
            not solver_config.use_ilp
            or not solver_config.local_branching_improvement
            or len(self.instance.candidates) > solver_config.local_branching_candidate_threshold
        ):
            return list(solution), None

        LOGGER.info("[Local-branching exact improvement]")
        current = list(sorted(solution))
        method: Optional[str] = None
        base_radius = max(2, solver_config.local_branching_base_radius)
        max_rounds = max(1, solver_config.local_branching_rounds)
        round_index = 0

        while round_index < max_rounds:
            radius = max(base_radius, int(math.ceil(base_radius * (1.8 ** round_index))))
            result = ILPSolver.improve_locally(
                self.instance,
                incumbent=current,
                radius=radius,
                backend=solver_config.exact_backend,
                time_limit=max(1, solver_config.local_branching_time_limit),
            )
            method = result.method or method

            if result.solution is not None:
                improved = RedundancyEliminator().eliminate(
                    self.instance,
                    result.solution,
                )
                tracker = CoverageTracker(self.instance)
                tracker.reset(improved)
                if not tracker.is_feasible():
                    raise RuntimeError(
                        "Local-branching exact improvement produced an infeasible solution."
                    )
                if len(improved) < len(current):
                    LOGGER.info(
                        "  Local-branching improvement: "
                        f"{len(current)} -> {len(improved)} "
                        f"({result.method}, radius={radius})"
                    )
                    current = improved
                    round_index = 0
                    continue

            LOGGER.info(
                "  Local-branching improvement: no better incumbent "
                f"({result.method}: {result.message}, radius={radius})"
            )
            round_index += 1

        return current, method

    def _covering_candidates_for_target(self, target_index: int) -> set[int]:
        if (
            self.instance.aggregation_mode == AggregationMode.SINGLE_CANDIDATE
            and self.instance.covered_by is not None
        ):
            return set(self.instance.covered_by[target_index])

        covering = set()
        for subset_id in self.instance.target_subset_ids[target_index]:
            covering.update(self.instance.subset_to_candidates[subset_id])
        return covering

    def _build_frontier_repair_core(
        self,
        base_solution: Sequence[int],
        focus_targets: Sequence[int],
        core_cap: int,
        anchors: Optional[Sequence[Sequence[int]]] = None,
        preferred_candidates: Optional[Sequence[int]] = None,
        lp_time_limit: int = 0,
    ) -> list[int]:
        base = list(sorted(set(base_solution)))
        core_cap = max(len(base) + 32, core_cap)
        focus_target_set = set(focus_targets)
        if not focus_target_set:
            return base

        base_set = set(base)
        repair_scores: dict[int, tuple[int, int, int]] = {}
        for target_index in focus_target_set:
            for candidate_index in self._covering_candidates_for_target(target_index):
                if candidate_index in base_set:
                    continue
                impacted_targets = self.instance.candidate_impacted_targets(candidate_index)
                focus_hits = sum(
                    1 for impacted_target in impacted_targets if impacted_target in focus_target_set
                )
                repair_scores[candidate_index] = (
                    focus_hits,
                    len(impacted_targets),
                    self.instance.candidate_span(candidate_index),
                )

        if not repair_scores:
            return base

        ranked_repair = sorted(
            repair_scores,
            key=lambda candidate_index: repair_scores[candidate_index],
            reverse=True,
        )

        context_targets = set(focus_target_set)
        for candidate_index in ranked_repair[: min(24, len(ranked_repair))]:
            context_targets.update(self.instance.candidate_impacted_targets(candidate_index))

        expanded_scores: dict[int, tuple[int, int, int]] = {}
        for target_index in context_targets:
            for candidate_index in self._covering_candidates_for_target(target_index):
                if candidate_index in base_set:
                    continue
                impacted_targets = self.instance.candidate_impacted_targets(candidate_index)
                expanded_scores[candidate_index] = (
                    sum(
                        1 for impacted_target in impacted_targets if impacted_target in focus_target_set
                    ),
                    sum(
                        1 for impacted_target in impacted_targets if impacted_target in context_targets
                    ),
                    self.instance.candidate_span(candidate_index),
                )

        ranked_expanded = sorted(
            expanded_scores,
            key=lambda candidate_index: expanded_scores[candidate_index],
            reverse=True,
        )

        candidate_universe = set(base)
        candidate_universe.update(preferred_candidates or ())
        candidate_universe.update(ranked_repair[: core_cap])
        candidate_universe.update(ranked_expanded[: core_cap])

        candidate_universe = sorted(candidate_universe)
        if (
            lp_time_limit > 0
            and len(candidate_universe) > core_cap
            and len(candidate_universe) > len(base)
        ):
            lp_core, _ = ILPSolver.fractional_candidate_core(
                self.instance,
                core_cap=core_cap,
                time_limit=lp_time_limit,
                incumbent=base,
                anchors=anchors,
                candidate_indices=candidate_universe,
            )
            if lp_core:
                return sorted(set(base) | set(lp_core))

        ordered = list(base)
        base_set = set(ordered)
        for candidate_index in ranked_repair:
            if candidate_index not in base_set and candidate_index in candidate_universe:
                ordered.append(candidate_index)
                base_set.add(candidate_index)
            if len(ordered) >= core_cap:
                return ordered
        for candidate_index in ranked_expanded:
            if candidate_index not in base_set and candidate_index in candidate_universe:
                ordered.append(candidate_index)
                base_set.add(candidate_index)
            if len(ordered) >= core_cap:
                return ordered

        return ordered

    def _lp_guided_exact_polish(
        self,
        solution: Sequence[int],
        elite_pool: Sequence[Sequence[int]],
        solver_config: SolverConfig,
    ) -> tuple[list[int], Optional[str]]:
        if (
            not solver_config.use_ilp
            or not solver_config.lp_guided_exact_polish
            or len(self.instance.candidates) > solver_config.lp_guided_candidate_threshold
        ):
            return list(solution), None

        LOGGER.info("[LP-guided exact polish]")
        current = list(sorted(solution))
        method: Optional[str] = None
        anchors = list(elite_pool[: max(1, solver_config.elite_pool_size)])

        for round_index in range(max(1, solver_config.lp_guided_rounds)):
            core_cap = max(
                len(current) + 64,
                int(math.ceil(solver_config.lp_guided_core_cap * (1.0 + 0.35 * round_index))),
            )
            core, lp_bound = ILPSolver.fractional_candidate_core(
                self.instance,
                core_cap=core_cap,
                time_limit=max(1, solver_config.lp_guided_time_limit),
                incumbent=current,
                anchors=anchors,
            )
            method = lp_bound.method or method
            if not core:
                LOGGER.info(
                    "  LP-guided exact polish: unavailable "
                    f"({lp_bound.method}: {lp_bound.message})"
                )
                break

            LOGGER.info(
                "  LP-guided core round "
                f"{round_index + 1}: size={len(core)}, "
                f"lp_bound={lp_bound.integer_bound}"
            )

            local_radius = max(
                18,
                min(
                    max(24, solver_config.local_branching_base_radius * 3),
                    len(current) // 4,
                ),
            )
            local_result = ILPSolver.improve_locally(
                self.instance,
                incumbent=current,
                radius=local_radius,
                backend=solver_config.exact_backend,
                time_limit=max(1, solver_config.lp_guided_time_limit // 2),
                candidate_indices=core,
            )
            method = local_result.method or method

            candidate_solution = None
            if local_result.solution is not None and len(local_result.solution) < len(current):
                candidate_solution = local_result.solution
                candidate_method = local_result.method
            else:
                result = ILPSolver.solve_restricted(
                    self.instance,
                    candidate_indices=core,
                    backend=solver_config.exact_backend,
                    time_limit=max(1, solver_config.lp_guided_time_limit),
                    incumbent=current,
                )
                method = result.method or method
                candidate_method = result.method
                if result.solution is not None and len(result.solution) < len(current):
                    candidate_solution = result.solution

            if candidate_solution is None:
                LOGGER.info(
                    "  LP-guided exact polish: no better incumbent "
                    f"({method})"
                )
                continue

            polished = RedundancyEliminator().eliminate(
                self.instance,
                candidate_solution,
            )
            tracker = CoverageTracker(self.instance)
            tracker.reset(polished)
            if not tracker.is_feasible():
                raise RuntimeError(
                    "LP-guided exact polish produced an infeasible solution."
                )
            if len(polished) < len(current):
                LOGGER.info(
                    "  LP-guided exact polish: "
                    f"{len(current)} -> {len(polished)} "
                    f"({candidate_method})"
                )
                current = polished
                continue

        return current, method

    def _cluster_exact_repair(
        self,
        solution: Sequence[int],
        elite_pool: Sequence[Sequence[int]],
        solver_config: SolverConfig,
    ) -> tuple[list[int], Optional[str]]:
        if (
            not solver_config.use_ilp
            or not solver_config.cluster_exact_repair
            or len(self.instance.candidates) > solver_config.cluster_exact_candidate_threshold
        ):
            return list(solution), None

        LOGGER.info("[Cluster exact repair]")
        current = list(sorted(solution))
        method: Optional[str] = None
        round_index = 0
        while round_index < max(1, solver_config.cluster_exact_rounds):
            tracker = CoverageTracker(self.instance)
            tracker.reset(current)
            incumbent = list(sorted(tracker.in_solution))
            if len(incumbent) <= 4:
                break

            weakest = heapq.nsmallest(
                min(10, len(incumbent)),
                incumbent,
                key=lambda candidate_index: (
                    tracker.exclusive_count(candidate_index),
                    tracker.redundancy_score(candidate_index),
                    -self.instance.candidate_span(candidate_index),
                ),
            )
            hardest = heapq.nlargest(
                min(10, len(incumbent)),
                incumbent,
                key=lambda candidate_index: (
                    tracker.exclusive_count(candidate_index),
                    -tracker.redundancy_score(candidate_index),
                    self.instance.candidate_span(candidate_index),
                ),
            )
            seeds = list(dict.fromkeys(weakest + hardest))
            improved_in_round = False

            for anchor in elite_pool[:2]:
                for candidate_index in sorted(set(current) ^ set(anchor)):
                    if candidate_index in tracker.in_solution and candidate_index not in seeds:
                        seeds.append(candidate_index)

            for seed in seeds[:6]:
                core_cap = int(
                    math.ceil(
                        solver_config.cluster_exact_core_cap * (1.0 + 0.5 * round_index)
                    )
                )
                time_limit = max(
                    1,
                    solver_config.cluster_exact_time_limit + 2 * round_index,
                )
                base_destroy_cap = min(
                    max(4, solver_config.cluster_exact_destroy_size),
                    len(incumbent) - 1,
                )
                destroy_cap = min(len(incumbent) - 1, base_destroy_cap)
                cluster_scores: dict[int, tuple[int, int, float, int]] = {}
                for candidate_index in incumbent:
                    overlap = self.instance.candidate_overlap_in_s_subsets(seed, candidate_index)
                    cluster_scores[candidate_index] = (
                        overlap,
                        -tracker.exclusive_count(candidate_index),
                        -int(round(tracker.redundancy_score(candidate_index) * 1000)),
                        self.instance.candidate_span(candidate_index),
                    )
                cluster = sorted(
                    incumbent,
                    key=lambda candidate_index: cluster_scores[candidate_index],
                    reverse=True,
                )[:destroy_cap]
                residual = [candidate_index for candidate_index in incumbent if candidate_index not in set(cluster)]
                residual_tracker = CoverageTracker(self.instance)
                residual_tracker.reset(residual)

                affected_targets = [
                    target_index
                    for target_index in range(len(self.instance.targets))
                    if residual_tracker.target_deficit(target_index) > 0
                ]
                if not affected_targets:
                    continue

                core = self._build_frontier_repair_core(
                    residual,
                    affected_targets,
                    core_cap=max(len(residual) + 64, core_cap),
                    anchors=elite_pool[:2],
                    preferred_candidates=cluster,
                    lp_time_limit=max(1, time_limit // 2),
                )
                if len(core) <= len(residual):
                    continue

                local_radius = max(
                    len(cluster) + 4,
                    solver_config.local_branching_base_radius * 2,
                )
                local_result = ILPSolver.improve_locally(
                    self.instance,
                    incumbent=current,
                    radius=local_radius,
                    backend=solver_config.exact_backend,
                    time_limit=max(1, time_limit // 2),
                    candidate_indices=core,
                )
                method = local_result.method or method

                candidate_solution = None
                candidate_method = local_result.method
                if local_result.solution is not None and len(local_result.solution) < len(current):
                    candidate_solution = local_result.solution
                else:
                    decision = ILPSolver.prove_no_solution_at_or_below_restricted(
                        self.instance,
                        candidate_indices=core,
                        cardinality_limit=len(current) - 1,
                        backend=solver_config.exact_backend,
                        time_limit=max(1, time_limit),
                        incumbent=current,
                    )
                    method = decision.method or method
                    candidate_method = decision.method
                    if decision.solution is not None and len(decision.solution) < len(current):
                        candidate_solution = decision.solution
                    elif decision.status != "infeasible":
                        result = ILPSolver.solve_restricted(
                            self.instance,
                            candidate_indices=core,
                            backend=solver_config.exact_backend,
                            time_limit=max(1, time_limit),
                            incumbent=current,
                        )
                        method = result.method or method
                        candidate_method = result.method
                        if result.solution is not None and len(result.solution) < len(current):
                            candidate_solution = result.solution

                if candidate_solution is None:
                    continue

                polished = RedundancyEliminator().eliminate(
                    self.instance,
                    candidate_solution,
                )
                verify = CoverageTracker(self.instance)
                verify.reset(polished)
                if not verify.is_feasible():
                    raise RuntimeError(
                        "Cluster exact repair produced an infeasible solution."
                    )

                LOGGER.info(
                    "  Cluster exact repair round "
                    f"{round_index + 1}: {len(current)} -> {len(polished)} "
                    f"(seed={seed}, method={candidate_method}, core={len(core)})"
                )
                current = polished
                improved_in_round = True
                break

            if not improved_in_round:
                LOGGER.info(
                    f"  Cluster exact repair round {round_index + 1}: no improvement"
                )
                round_index += 1
            else:
                round_index = 0

        return current, method

    def _restricted_greedy_completion(
        self,
        base_solution: Sequence[int],
        candidate_core: Sequence[int],
        focus_targets: Sequence[int],
        preferred_candidates: Optional[Sequence[int]] = None,
    ) -> Optional[list[int]]:
        tracker = CoverageTracker(self.instance)
        tracker.reset(base_solution)
        available = set(candidate_core) - tracker.in_solution
        focus_target_set = set(focus_targets)
        preferred = set(preferred_candidates or ())

        while not tracker.is_feasible():
            if not available:
                return None

            best_candidate: Optional[int] = None
            best_key: Optional[tuple[float, int, int, int]] = None
            for candidate_index in available:
                marginal_gain = tracker.marginal_gain(candidate_index)
                if marginal_gain <= 0:
                    continue
                focus_hits = sum(
                    1
                    for impacted_target in self.instance.candidate_impacted_targets(candidate_index)
                    if impacted_target in focus_target_set
                )
                key = (
                    float(marginal_gain),
                    focus_hits,
                    1 if candidate_index in preferred else 0,
                    self.instance.candidate_span(candidate_index),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_candidate = candidate_index

            if best_candidate is None:
                return None

            tracker.add(best_candidate)
            available.discard(best_candidate)

        repaired = RedundancyEliminator().eliminate(
            self.instance,
            sorted(tracker.in_solution),
        )
        verify = CoverageTracker(self.instance)
        verify.reset(repaired)
        if not verify.is_feasible():
            return None
        return repaired

    def _ruin_recreate_exact(
        self,
        solution: Sequence[int],
        elite_pool: Sequence[Sequence[int]],
        solver_config: SolverConfig,
    ) -> tuple[list[int], Optional[str]]:
        if (
            not solver_config.use_ilp
            or not solver_config.ruin_recreate_exact
            or len(self.instance.candidates) > solver_config.ruin_recreate_candidate_threshold
        ):
            return list(solution), None

        LOGGER.info("[Ruin-recreate exact]")
        current = list(sorted(solution))
        method: Optional[str] = None

        for round_index in range(max(1, solver_config.ruin_recreate_rounds)):
            tracker = CoverageTracker(self.instance)
            tracker.reset(current)
            incumbent = list(sorted(tracker.in_solution))
            if len(incumbent) <= 8:
                break

            weakest = heapq.nsmallest(
                min(8, len(incumbent)),
                incumbent,
                key=lambda candidate_index: (
                    tracker.exclusive_count(candidate_index),
                    tracker.redundancy_score(candidate_index),
                    -self.instance.candidate_span(candidate_index),
                ),
            )
            hardest = heapq.nlargest(
                min(8, len(incumbent)),
                incumbent,
                key=lambda candidate_index: (
                    tracker.exclusive_count(candidate_index),
                    -tracker.redundancy_score(candidate_index),
                    self.instance.candidate_span(candidate_index),
                ),
            )
            seeds = list(dict.fromkeys(weakest + hardest))
            improved = False

            for seed in seeds[:6]:
                destroy_cap = min(
                    len(incumbent) - 1,
                    max(8, solver_config.ruin_recreate_destroy_size),
                )
                cluster = sorted(
                    incumbent,
                    key=lambda candidate_index: (
                        self.instance.candidate_overlap_in_s_subsets(seed, candidate_index),
                        -tracker.exclusive_count(candidate_index),
                        -int(round(tracker.redundancy_score(candidate_index) * 1000)),
                        self.instance.candidate_span(candidate_index),
                    ),
                    reverse=True,
                )[:destroy_cap]
                residual = [candidate_index for candidate_index in incumbent if candidate_index not in set(cluster)]
                residual_tracker = CoverageTracker(self.instance)
                residual_tracker.reset(residual)
                focus_targets = [
                    target_index
                    for target_index in range(len(self.instance.targets))
                    if residual_tracker.target_deficit(target_index) > 0
                ]
                if not focus_targets:
                    continue

                core = self._build_frontier_repair_core(
                    residual,
                    focus_targets,
                    core_cap=max(len(residual) + 96, solver_config.ruin_recreate_core_cap),
                    anchors=elite_pool[:2],
                    preferred_candidates=cluster,
                    lp_time_limit=max(2, solver_config.ruin_recreate_time_limit // 2),
                )
                if len(core) <= len(residual):
                    continue

                repaired = self._restricted_greedy_completion(
                    residual,
                    core,
                    focus_targets,
                    preferred_candidates=cluster,
                )
                if repaired is None:
                    continue

                candidate_solution = repaired
                local_result = ILPSolver.improve_locally(
                    self.instance,
                    incumbent=repaired,
                    radius=max(
                        solver_config.local_branching_base_radius * 2,
                        len(cluster) + 4,
                    ),
                    backend=solver_config.exact_backend,
                    time_limit=max(2, solver_config.ruin_recreate_time_limit // 2),
                    candidate_indices=core,
                )
                method = local_result.method or method
                if (
                    local_result.solution is not None
                    and len(local_result.solution) < len(candidate_solution)
                ):
                    candidate_solution = local_result.solution

                optimize_result = ILPSolver.solve_restricted(
                    self.instance,
                    candidate_indices=core,
                    backend=solver_config.exact_backend,
                    time_limit=max(2, solver_config.ruin_recreate_time_limit),
                    incumbent=candidate_solution,
                )
                method = optimize_result.method or method
                if (
                    optimize_result.solution is not None
                    and len(optimize_result.solution) < len(candidate_solution)
                ):
                    candidate_solution = optimize_result.solution

                polished = RedundancyEliminator().eliminate(
                    self.instance,
                    candidate_solution,
                )
                verify = CoverageTracker(self.instance)
                verify.reset(polished)
                if not verify.is_feasible():
                    raise RuntimeError(
                        "Ruin-recreate exact produced an infeasible solution."
                    )
                if len(polished) < len(current):
                    LOGGER.info(
                        "  Ruin-recreate round "
                        f"{round_index + 1}: {len(current)} -> {len(polished)} "
                        f"(seed={seed}, core={len(core)}, method={method})"
                    )
                    current = polished
                    improved = True
                    break

            if not improved:
                LOGGER.info(
                    f"  Ruin-recreate round {round_index + 1}: no improvement"
                )

        return current, method

    def _elite_union_exact_polish(
        self,
        solution: Sequence[int],
        elite_pool: Sequence[Sequence[int]],
        solver_config: SolverConfig,
    ) -> tuple[list[int], Optional[str]]:
        if (
            not solver_config.use_ilp
            or not solver_config.elite_union_exact_polish
            or len(self.instance.candidates) > solver_config.elite_union_candidate_threshold
            or not elite_pool
        ):
            return list(solution), None

        LOGGER.info("[Elite-union exact polish]")
        current = list(sorted(solution))
        method: Optional[str] = None

        tracker = CoverageTracker(self.instance)
        tracker.reset(current)
        incumbent = list(sorted(tracker.in_solution))
        weakest = heapq.nsmallest(
            min(12, len(incumbent)),
            incumbent,
            key=lambda candidate_index: (
                tracker.exclusive_count(candidate_index),
                tracker.redundancy_score(candidate_index),
                -self.instance.candidate_span(candidate_index),
            ),
        )
        focus_targets = set()
        for candidate_index in weakest:
            focus_targets.update(self.instance.candidate_impacted_targets(candidate_index))

        core = set(current)
        for elite_solution in elite_pool[: max(2, solver_config.elite_pool_size)]:
            core.update(elite_solution)
        frontier = self._build_frontier_repair_core(
            sorted(core),
            sorted(focus_targets),
            core_cap=max(len(core) + 128, solver_config.elite_union_core_cap),
            anchors=elite_pool[: max(2, solver_config.elite_pool_size)],
            preferred_candidates=weakest,
            lp_time_limit=max(2, solver_config.elite_union_time_limit // 2),
        )
        core.update(frontier)
        core = sorted(core)

        if len(core) <= len(current):
            LOGGER.info("  Elite-union exact polish: core too small to improve")
            return current, None

        local_result = ILPSolver.improve_locally(
            self.instance,
            incumbent=current,
            radius=max(24, solver_config.local_branching_base_radius * 3),
            backend=solver_config.exact_backend,
            time_limit=max(2, solver_config.elite_union_time_limit // 2),
            candidate_indices=core,
        )
        method = local_result.method or method
        candidate_solution = current
        if local_result.solution is not None and len(local_result.solution) < len(candidate_solution):
            candidate_solution = local_result.solution

        restricted_result = ILPSolver.solve_restricted(
            self.instance,
            candidate_indices=core,
            backend=solver_config.exact_backend,
            time_limit=max(2, solver_config.elite_union_time_limit),
            incumbent=candidate_solution,
        )
        method = restricted_result.method or method
        if (
            restricted_result.solution is not None
            and len(restricted_result.solution) < len(candidate_solution)
        ):
            candidate_solution = restricted_result.solution

        polished = RedundancyEliminator().eliminate(
            self.instance,
            candidate_solution,
        )
        verify = CoverageTracker(self.instance)
        verify.reset(polished)
        if not verify.is_feasible():
            raise RuntimeError("Elite-union exact polish produced an infeasible solution.")

        if len(polished) < len(current):
            LOGGER.info(
                "  Elite-union exact polish: "
                f"{len(current)} -> {len(polished)} (core={len(core)}, method={method})"
            )
            return polished, method

        LOGGER.info(
            "  Elite-union exact polish: no improvement "
            f"(core={len(core)}, method={method})"
        )
        return current, method

    def _cardinality_descent(
        self,
        solution: Sequence[int],
        solver_config: SolverConfig,
    ) -> list[int]:
        if (
            not solver_config.cardinality_descent
            or len(self.instance.candidates) > solver_config.cardinality_descent_candidate_threshold
        ):
            return list(solution)

        LOGGER.info("[Cardinality descent]")
        current = list(sorted(solution))
        lower_bound = ILPSolver.combinatorial_lower_bound(self.instance).integer_bound or 0
        drop = max(1, solver_config.cardinality_descent_initial_drop)

        for round_index in range(max(1, solver_config.cardinality_descent_max_rounds)):
            target_size = len(current) - drop
            if target_size <= lower_bound:
                break

            tracker = CoverageTracker(self.instance)
            tracker.reset(current)
            removable = heapq.nsmallest(
                min(drop, len(current) - 1),
                current,
                key=lambda candidate_index: (
                    tracker.exclusive_count(candidate_index),
                    tracker.redundancy_score(candidate_index),
                    -self.instance.candidate_span(candidate_index),
                ),
            )
            seed_solution = [candidate for candidate in current if candidate not in set(removable)]
            tracker.reset(seed_solution)
            outside = set(range(len(self.instance.candidates))) - tracker.in_solution
            stagnation = 0
            success = False
            best_key = (tracker.unsatisfied_targets, tracker.deficit_units)
            best_snapshot = list(tracker.in_solution)

            LOGGER.info(
                f"  Cardinality descent round {round_index + 1}: target_size={target_size}, "
                f"starting_deficit={tracker.deficit_units}"
            )

            for _ in range(max(1, solver_config.cardinality_descent_iterations)):
                if tracker.is_feasible():
                    current = list(sorted(tracker.in_solution))
                    current = RedundancyEliminator().eliminate(self.instance, current)
                    LOGGER.info(
                        f"  Cardinality descent success: {len(solution)} -> {len(current)}"
                    )
                    success = True
                    break

                deficit_targets = [
                    target_index
                    for target_index in range(len(self.instance.targets))
                    if tracker.target_deficit(target_index) > 0
                ]
                if not deficit_targets:
                    break

                sample_targets = self.rng.sample(
                    deficit_targets,
                    min(6, len(deficit_targets)),
                )
                target_focus = set(sample_targets)
                add_pool = set()
                for target_index in sample_targets:
                    add_pool.update(self._covering_candidates_for_target(target_index))
                add_pool.difference_update(tracker.in_solution)

                if not add_pool:
                    break

                if len(add_pool) > solver_config.cardinality_descent_sample_size * 4:
                    add_pool = set(
                        self.rng.sample(
                            list(add_pool),
                            solver_config.cardinality_descent_sample_size * 4,
                        )
                    )

                ranked_add = sorted(
                    add_pool,
                    key=lambda candidate_index: (
                        sum(
                            1
                            for impacted_target in self.instance.candidate_impacted_targets(
                                candidate_index
                            )
                            if impacted_target in target_focus
                        ),
                        self.instance.candidate_span(candidate_index),
                    ),
                    reverse=True,
                )[: max(8, solver_config.cardinality_descent_sample_size // 4)]
                remove_pool = heapq.nsmallest(
                    min(12, len(tracker.in_solution)),
                    tracker.in_solution,
                    key=lambda candidate_index: (
                        tracker.exclusive_count(candidate_index),
                        tracker.redundancy_score(candidate_index),
                        -self.instance.candidate_span(candidate_index),
                    ),
                )

                snapshot = list(tracker.in_solution)
                candidate_move: Optional[tuple[int, int, tuple[int, int]]] = None
                candidate_key = best_key
                for candidate_add in ranked_add:
                    for candidate_remove in remove_pool:
                        tracker.remove(candidate_remove)
                        tracker.add(candidate_add)
                        move_key = (tracker.unsatisfied_targets, tracker.deficit_units)
                        if move_key < candidate_key:
                            candidate_key = move_key
                            candidate_move = (
                                candidate_remove,
                                candidate_add,
                                move_key,
                            )
                        tracker.reset(snapshot)

                if candidate_move is None:
                    stagnation += 1
                    if stagnation >= solver_config.cardinality_descent_patience:
                        break
                    random_add = self.rng.choice(ranked_add)
                    random_remove = self.rng.choice(remove_pool)
                    tracker.remove(random_remove)
                    tracker.add(random_add)
                    random_key = (tracker.unsatisfied_targets, tracker.deficit_units)
                    if random_key <= best_key:
                        outside.add(random_remove)
                        outside.discard(random_add)
                        best_key = random_key
                    else:
                        tracker.reset(snapshot)
                    continue

                candidate_remove, candidate_add, move_key = candidate_move
                tracker.remove(candidate_remove)
                tracker.add(candidate_add)
                outside.add(candidate_remove)
                outside.discard(candidate_add)
                if move_key < best_key:
                    best_key = move_key
                    best_snapshot = list(tracker.in_solution)
                    stagnation = 0
                else:
                    stagnation += 1
                    if stagnation >= solver_config.cardinality_descent_patience:
                        break

            if (
                not success
                and solver_config.use_ilp
                and best_key[0] <= 12
                and best_key[1] <= 12
            ):
                repair_tracker = CoverageTracker(self.instance)
                repair_tracker.reset(best_snapshot)
                deficit_targets = [
                    target_index
                    for target_index in range(len(self.instance.targets))
                    if repair_tracker.target_deficit(target_index) > 0
                ]
                if deficit_targets:
                    if best_key[1] <= 1:
                        micro_candidate = self._micro_deficit_repair(
                            best_snapshot,
                            target_size=target_size,
                            focus_targets=deficit_targets,
                        )
                        if micro_candidate is not None and len(micro_candidate) < len(current):
                            current = micro_candidate
                            LOGGER.info(
                                "  Cardinality descent micro repair: "
                                f"{len(solution)} -> {len(current)} "
                                f"(deficit={best_key[1]})"
                            )
                            success = True

                if deficit_targets and not success:
                    target_set = set(deficit_targets)
                    repair_scores: dict[int, tuple[int, int]] = {}
                    for target_index in deficit_targets:
                        for candidate_index in self._covering_candidates_for_target(target_index):
                            if candidate_index in repair_tracker.in_solution:
                                continue
                            repair_scores[candidate_index] = (
                                sum(
                                    1
                                    for impacted_target in self.instance.candidate_impacted_targets(
                                        candidate_index
                                    )
                                    if impacted_target in target_set
                                ),
                                self.instance.candidate_span(candidate_index),
                            )

                    if repair_scores:
                        repair_candidates = sorted(
                            repair_scores,
                            key=lambda candidate_index: repair_scores[candidate_index],
                            reverse=True,
                        )

                        core = self._build_frontier_repair_core(
                            best_snapshot,
                            deficit_targets,
                            core_cap=max(
                                len(best_snapshot) + 64,
                                solver_config.cluster_exact_core_cap,
                            ),
                            preferred_candidates=repair_candidates[:64],
                            lp_time_limit=max(2, solver_config.cluster_exact_time_limit),
                        )

                        def try_repair_core(core_candidates: Sequence[int]):
                            return ILPSolver.prove_no_solution_at_or_below_restricted(
                                self.instance,
                                candidate_indices=sorted(core_candidates),
                                cardinality_limit=target_size,
                                backend=solver_config.exact_backend,
                                time_limit=max(4, solver_config.cluster_exact_time_limit * 2),
                                incumbent=current,
                            )

                        result = try_repair_core(core)
                        if result.solution is not None and len(result.solution) < len(current):
                            candidate = RedundancyEliminator().eliminate(
                                self.instance,
                                result.solution,
                            )
                            verify = CoverageTracker(self.instance)
                            verify.reset(candidate)
                            if verify.is_feasible():
                                current = candidate
                                LOGGER.info(
                                    "  Cardinality descent exact repair: "
                                    f"{len(solution)} -> {len(current)} "
                                    f"({result.method}, deficit={best_key[1]})"
                                )
                                success = True
                        elif result.status == "infeasible":
                            context_targets = set(deficit_targets)
                            for candidate_index in repair_candidates[:12]:
                                context_targets.update(
                                    self.instance.candidate_impacted_targets(candidate_index)
                                )

                            expanded_core = self._build_frontier_repair_core(
                                best_snapshot,
                                sorted(context_targets),
                                core_cap=max(
                                    len(best_snapshot) + 96,
                                    solver_config.cluster_exact_core_cap * 2,
                                ),
                                preferred_candidates=repair_candidates[:96],
                                lp_time_limit=max(
                                    2,
                                    solver_config.cluster_exact_time_limit + 2,
                                ),
                            )
                            if len(expanded_core) > len(core):
                                result = try_repair_core(expanded_core)
                                if result.solution is not None and len(result.solution) < len(current):
                                    candidate = RedundancyEliminator().eliminate(
                                        self.instance,
                                        result.solution,
                                    )
                                    verify = CoverageTracker(self.instance)
                                    verify.reset(candidate)
                                    if verify.is_feasible():
                                        current = candidate
                                        LOGGER.info(
                                            "  Cardinality descent expanded exact repair: "
                                            f"{len(solution)} -> {len(current)} "
                                            f"({result.method}, deficit={best_key[1]}, core={len(expanded_core)})"
                                        )
                                        success = True
                                elif result.status == "infeasible":
                                    LOGGER.info(
                                        "  Cardinality descent exact repair: "
                                        f"no solution at size {target_size} inside expanded core"
                                    )
                                    if best_key[1] <= 2:
                                        very_expanded_targets = set(context_targets)
                                        for candidate_index in repair_candidates[:32]:
                                            very_expanded_targets.update(
                                                self.instance.candidate_impacted_targets(
                                                    candidate_index
                                                )
                                            )

                                        very_expanded_core = self._build_frontier_repair_core(
                                            best_snapshot,
                                            sorted(very_expanded_targets),
                                            core_cap=max(
                                                len(best_snapshot) + 192,
                                                solver_config.cluster_exact_core_cap * 4,
                                            ),
                                            preferred_candidates=repair_candidates[:160],
                                            lp_time_limit=max(
                                                3,
                                                solver_config.cluster_exact_time_limit + 4,
                                            ),
                                        )
                                        if len(very_expanded_core) > len(expanded_core):
                                            result = try_repair_core(very_expanded_core)
                                            if (
                                                result.solution is not None
                                                and len(result.solution) < len(current)
                                            ):
                                                candidate = RedundancyEliminator().eliminate(
                                                    self.instance,
                                                    result.solution,
                                                )
                                                verify = CoverageTracker(self.instance)
                                                verify.reset(candidate)
                                                if verify.is_feasible():
                                                    current = candidate
                                                    LOGGER.info(
                                                        "  Cardinality descent very-expanded exact repair: "
                                                        f"{len(solution)} -> {len(current)} "
                                                        f"({result.method}, deficit={best_key[1]}, core={len(very_expanded_core)})"
                                                    )
                                                    success = True
                                            elif result.status == "infeasible":
                                                LOGGER.info(
                                                    "  Cardinality descent exact repair: "
                                                    f"no solution at size {target_size} inside very-expanded core"
                                                )
                                            if not success:
                                                local_result = ILPSolver.improve_locally(
                                                    self.instance,
                                                    incumbent=current,
                                                    radius=max(
                                                        18,
                                                        solver_config.local_branching_base_radius * 2,
                                                    ),
                                                    backend=solver_config.exact_backend,
                                                    time_limit=max(
                                                        4,
                                                        solver_config.cluster_exact_time_limit * 2,
                                                    ),
                                                    candidate_indices=very_expanded_core,
                                                )
                                                if (
                                                    local_result.solution is not None
                                                    and len(local_result.solution) < len(current)
                                                ):
                                                    candidate = RedundancyEliminator().eliminate(
                                                        self.instance,
                                                        local_result.solution,
                                                    )
                                                    verify = CoverageTracker(self.instance)
                                                    verify.reset(candidate)
                                                    if verify.is_feasible():
                                                        current = candidate
                                                        LOGGER.info(
                                                            "  Cardinality descent local-branch repair: "
                                                            f"{len(solution)} -> {len(current)} "
                                                            f"({local_result.method}, core={len(very_expanded_core)})"
                                                        )
                                                        success = True
                                            if not success:
                                                optimize_result = ILPSolver.solve_restricted(
                                                    self.instance,
                                                    candidate_indices=very_expanded_core,
                                                    backend=solver_config.exact_backend,
                                                    time_limit=max(
                                                        6,
                                                        solver_config.cluster_exact_time_limit * 3,
                                                    ),
                                                    incumbent=current,
                                                )
                                                if (
                                                    optimize_result.solution is not None
                                                    and len(optimize_result.solution) < len(current)
                                                ):
                                                    candidate = RedundancyEliminator().eliminate(
                                                        self.instance,
                                                        optimize_result.solution,
                                                    )
                                                    verify = CoverageTracker(self.instance)
                                                    verify.reset(candidate)
                                                    if verify.is_feasible():
                                                        current = candidate
                                                        LOGGER.info(
                                                            "  Cardinality descent restricted optimize repair: "
                                                            f"{len(solution)} -> {len(current)} "
                                                            f"({optimize_result.method}, core={len(very_expanded_core)})"
                                                        )
                                                        success = True
                                            if not success and best_key[1] == 1:
                                                ultra_targets = set(very_expanded_targets)
                                                for target_index in deficit_targets:
                                                    covering_candidates = sorted(
                                                        self._covering_candidates_for_target(
                                                            target_index
                                                        ),
                                                        key=self.instance.candidate_span,
                                                        reverse=True,
                                                    )
                                                    for candidate_index in covering_candidates[:12]:
                                                        ultra_targets.update(
                                                            self.instance.candidate_impacted_targets(
                                                                candidate_index
                                                            )
                                                        )

                                                ultra_core = self._build_frontier_repair_core(
                                                    best_snapshot,
                                                    sorted(ultra_targets),
                                                    core_cap=max(
                                                        len(best_snapshot) + 256,
                                                        solver_config.cluster_exact_core_cap * 6,
                                                    ),
                                                    preferred_candidates=repair_candidates[:240],
                                                    lp_time_limit=0,
                                                )
                                                if len(ultra_core) > len(very_expanded_core):
                                                    ultra_result = ILPSolver.solve_restricted(
                                                        self.instance,
                                                        candidate_indices=ultra_core,
                                                        backend=solver_config.exact_backend,
                                                        time_limit=max(
                                                            12,
                                                            solver_config.cluster_exact_time_limit * 6,
                                                        ),
                                                        incumbent=current,
                                                    )
                                                    if (
                                                        ultra_result.solution is not None
                                                        and len(ultra_result.solution) < len(current)
                                                    ):
                                                        candidate = RedundancyEliminator().eliminate(
                                                            self.instance,
                                                            ultra_result.solution,
                                                        )
                                                        verify = CoverageTracker(self.instance)
                                                        verify.reset(candidate)
                                                        if verify.is_feasible():
                                                            current = candidate
                                                            LOGGER.info(
                                                                "  Cardinality descent ultra-expanded repair: "
                                                                f"{len(solution)} -> {len(current)} "
                                                                f"({ultra_result.method}, core={len(ultra_core)})"
                                                            )
                                                            success = True
                        if success:
                            pass
                        elif result.status == "infeasible":
                            LOGGER.info(
                                "  Cardinality descent exact repair: "
                                f"no solution at size {target_size} inside focused core"
                            )

            if success:
                drop = max(1, drop)
                continue

            LOGGER.info(
                f"  Cardinality descent round {round_index + 1}: no feasible solution at size {target_size}"
            )
            if drop > 1:
                drop = max(1, drop // 2)
            else:
                break

        return current

    def _micro_deficit_repair(
        self,
        seed_solution: Sequence[int],
        target_size: int,
        focus_targets: Sequence[int],
    ) -> Optional[list[int]]:
        tracker = CoverageTracker(self.instance)
        tracker.reset(seed_solution)
        if tracker.solution_size != target_size:
            return None

        focus_target_set = set(focus_targets)
        if not focus_target_set:
            return None

        add_pool = set()
        for target_index in focus_target_set:
            add_pool.update(self._covering_candidates_for_target(target_index))
        add_pool.difference_update(tracker.in_solution)
        if not add_pool:
            return None

        ranked_add = sorted(
            add_pool,
            key=lambda candidate_index: (
                sum(
                    1
                    for impacted_target in self.instance.candidate_impacted_targets(candidate_index)
                    if impacted_target in focus_target_set
                ),
                self.instance.candidate_span(candidate_index),
            ),
            reverse=True,
        )[:128]

        base_snapshot = list(tracker.in_solution)
        for candidate_add in ranked_add:
            tracker.add(candidate_add)
            if not tracker.is_feasible():
                tracker.reset(base_snapshot)
                continue

            remove_candidates = sorted(
                [
                    candidate_index
                    for candidate_index in tracker.in_solution
                    if candidate_index != candidate_add
                ],
                key=lambda candidate_index: (
                    tracker.exclusive_count(candidate_index),
                    tracker.redundancy_score(candidate_index),
                    -self.instance.candidate_span(candidate_index),
                ),
            )

            augmented_snapshot = list(tracker.in_solution)
            for candidate_remove in remove_candidates[:96]:
                if not tracker.can_remove(candidate_remove):
                    continue
                tracker.remove(candidate_remove)
                if tracker.is_feasible() and tracker.solution_size <= target_size:
                    candidate = RedundancyEliminator().eliminate(
                        self.instance,
                        sorted(tracker.in_solution),
                    )
                    verify = CoverageTracker(self.instance)
                    verify.reset(candidate)
                    if verify.is_feasible() and len(candidate) <= target_size:
                        return candidate
                tracker.reset(augmented_snapshot)

            tracker.reset(base_snapshot)

        return None

    def _resolve_search_budgets(
        self, solver_config: SolverConfig
    ) -> tuple[int, int, str]:
        max_local_steps = solver_config.max_local_steps
        if max_local_steps is None:
            max_local_steps = min(1800, max(180, len(self.instance.candidates) * 2))

        max_sa_iterations = solver_config.max_sa_iterations
        if max_sa_iterations is None:
            max_sa_iterations = min(2600, max(240, len(self.instance.candidates) * 2))

        threshold = max(1, solver_config.large_instance_candidate_threshold)
        candidate_count = len(self.instance.candidates)
        if candidate_count <= threshold:
            return max_local_steps, max_sa_iterations, "standard"

        scale = threshold / candidate_count
        if solver_config.max_local_steps is None and max_local_steps > 0:
            max_local_steps = max(
                solver_config.large_instance_min_local_steps,
                int(math.ceil(max_local_steps * scale)),
            )
        if solver_config.max_sa_iterations is None and max_sa_iterations > 0:
            max_sa_iterations = max(
                solver_config.large_instance_min_sa_iterations,
                int(math.ceil(max_sa_iterations * scale)),
            )
        return max_local_steps, max_sa_iterations, "large-instance-scaled"

    def solve(self, solver_config: Optional[SolverConfig] = None) -> SolveResult:
        solver_config = solver_config or SolverConfig()
        start = time.time()

        exact_size: Optional[int] = None
        exact_method: Optional[str] = None

        should_try_exact = (
            solver_config.use_ilp
            and (
                solver_config.force_exact
                or (
                    len(self.instance.candidates) <= 5000
                    and len(self.instance.targets) <= 10000
                )
            )
        )
        if should_try_exact:
            LOGGER.info("\n[Exact verification]")
            exact_solution, exact_method = ILPSolver.solve(
                self.instance,
                backend=solver_config.exact_backend,
                time_limit=solver_config.exact_time_limit,
            )
            if exact_solution is not None:
                exact_size = len(exact_solution)
        elif solver_config.use_ilp:
            LOGGER.info(
                "\n[Exact verification skipped: instance exceeds default exact limits; "
                "use --force-exact to override.]"
            )

        max_local_steps, max_sa_iterations, search_profile = self._resolve_search_budgets(
            solver_config
        )
        if search_profile == "large-instance-scaled":
            LOGGER.info(
                "Large candidate set detected; using scaled improvement budgets "
                f"(local_steps={max_local_steps}, sa_iterations={max_sa_iterations}) "
                "instead of skipping post-processing."
            )

        best_solution = None
        best_size = math.inf
        elite_pool: list[list[int]] = []

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
            current_solution = self._elite_guided_seed(
                current_solution,
                elite_pool,
                solver_config,
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
                    adaptive_neighborhoods=solver_config.adaptive_neighborhoods,
                )
                current_solution = local_search.solve()

            if max_sa_iterations > 0:
                LOGGER.info("[Simulated annealing]")
                sa = ImprovedSA(
                    self.instance,
                    current_solution,
                    rng=self.rng,
                    max_iter=max_sa_iterations,
                    adaptive_neighborhoods=solver_config.adaptive_neighborhoods,
                )
                current_solution = sa.solve()

            current_solution = RedundancyEliminator().eliminate(
                self.instance, current_solution
            )
            current_solution = self._elite_intensify(
                current_solution,
                elite_pool,
                solver_config,
                max_local_steps,
                max_sa_iterations,
            )

            tracker.reset(current_solution)
            if not tracker.is_feasible():
                raise RuntimeError("Post-processing produced an infeasible solution.")

            LOGGER.info(
                f"  Restart result: greedy {len(greedy_solution)} -> final {len(current_solution)}"
            )

            elite_pool = self._update_elite_pool(
                elite_pool,
                current_solution,
                solver_config,
            )

            if len(current_solution) < best_size:
                best_solution = current_solution
                best_size = len(current_solution)

                if exact_size is not None and best_size == exact_size:
                    LOGGER.info("Exact optimum matched; stopping remaining restarts early.")
                    break

        if best_solution is None:
            raise RuntimeError("No feasible solution was found.")

        best_solution, local_branch_method = self._local_branching_improvement(
            best_solution,
            solver_config,
        )
        best_size = len(best_solution)
        if local_branch_method is not None and exact_method is None:
            exact_method = local_branch_method

        best_solution, lp_guided_method = self._lp_guided_exact_polish(
            best_solution,
            elite_pool,
            solver_config,
        )
        best_size = len(best_solution)
        if lp_guided_method is not None and exact_method is None:
            exact_method = lp_guided_method

        best_solution, cluster_method = self._cluster_exact_repair(
            best_solution,
            elite_pool,
            solver_config,
        )
        best_size = len(best_solution)
        if cluster_method is not None and exact_method is None:
            exact_method = cluster_method

        best_solution, ruin_method = self._ruin_recreate_exact(
            best_solution,
            elite_pool,
            solver_config,
        )
        best_size = len(best_solution)
        if ruin_method is not None and exact_method is None:
            exact_method = ruin_method

        best_solution, elite_union_method = self._elite_union_exact_polish(
            best_solution,
            elite_pool,
            solver_config,
        )
        best_size = len(best_solution)
        if elite_union_method is not None and exact_method is None:
            exact_method = elite_union_method

        best_solution = self._cardinality_descent(
            best_solution,
            solver_config,
        )
        best_size = len(best_solution)

        best_solution, exact_improve_method = self._mid_size_exact_improvement(
            best_solution,
            solver_config,
        )
        best_size = len(best_solution)
        if exact_improve_method is not None and exact_method is None:
            exact_method = exact_improve_method

        if exact_size is None:
            best_solution, polish_method = self._reduced_exact_polish(
                best_solution,
                elite_pool,
                solver_config,
            )
            best_size = len(best_solution)
            if polish_method is not None and exact_method is None:
                exact_method = polish_method

        final_audit = audit_solution_indices(
            self.instance,
            best_solution,
            source="solver-final",
            max_uncovered_examples=5,
        )
        if (
            not final_audit.primary_report.is_valid
            or not final_audit.independent_report.is_valid
            or not final_audit.methods_agree
        ):
            raise RuntimeError(
                "Final solution failed internal validation or validator agreement."
            )

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
            validation={
                "primary_valid": final_audit.primary_report.is_valid,
                "independent_valid": final_audit.independent_report.is_valid,
                "methods_agree": final_audit.methods_agree,
                "unsatisfied_targets": final_audit.primary_report.unsatisfied_targets,
                "deficit_units": final_audit.primary_report.deficit_units,
            },
        )

        if solver_config.save_result:
            ResultDatabase(solver_config.db_dir).save(result.to_dict())

        return result
