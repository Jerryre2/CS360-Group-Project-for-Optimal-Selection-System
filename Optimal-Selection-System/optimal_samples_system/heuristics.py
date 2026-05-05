"""Heuristic solvers and local improvement components."""

from __future__ import annotations

import heapq
import math
import random
import time
from collections import defaultdict
from typing import List, Optional, Sequence, Set, Tuple

from .config import AggregationMode, LOGGER
from .instance import CoverageInstance
from .tracking import CoverageTracker


class GreedySolver:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def solve(self, instance: CoverageInstance, randomized: bool = False) -> List[int]:
        tracker = CoverageTracker(instance)
        solution: List[int] = []
        num_candidates = len(instance.candidates)
        candidate_spans = [
            instance.candidate_span(candidate_index)
            for candidate_index in range(num_candidates)
        ]
        span_scale = 1.0 / (max(1, len(instance.targets)) + 1.0)
        noise_scale = span_scale / (max(1, num_candidates) + 1.0)

        heap: List[Tuple[float, int, int]] = []
        gain_cache = [0] * num_candidates
        for candidate_index in range(num_candidates):
            gain = tracker.marginal_gain(candidate_index)
            gain_cache[candidate_index] = gain
            priority = gain + candidate_spans[candidate_index] * span_scale
            if randomized:
                priority += self.rng.random() * noise_scale
            heapq.heappush(heap, (-priority, 0, candidate_index))

        step = 0

        while not tracker.is_feasible():
            step += 1
            best_index: Optional[int] = None
            best_gain = -1

            if randomized:
                contender_window = min(24, num_candidates)
                rcl_size = min(8, contender_window)
                contenders: List[Tuple[float, int, int]] = []

                while heap and len(contenders) < contender_window:
                    neg_priority, timestamp, candidate_index = heapq.heappop(heap)
                    if candidate_index in tracker.in_solution:
                        continue

                    if timestamp != step:
                        gain = tracker.marginal_gain(candidate_index)
                        gain_cache[candidate_index] = gain
                        priority = gain + candidate_spans[candidate_index] * span_scale
                        priority += self.rng.random() * noise_scale
                        heapq.heappush(heap, (-priority, step, candidate_index))
                        continue

                    contenders.append((neg_priority, timestamp, candidate_index))

                if contenders:
                    chosen_offset = self.rng.randrange(min(rcl_size, len(contenders)))
                    chosen_priority, _, best_index = contenders[chosen_offset]
                    best_gain = gain_cache[best_index]
                    for offset, contender in enumerate(contenders):
                        if offset == chosen_offset:
                            continue
                        heapq.heappush(heap, contender)
                else:
                    raise RuntimeError(
                        "Greedy construction failed to build a randomized candidate list."
                    )
            else:
                while heap:
                    _, timestamp, candidate_index = heapq.heappop(heap)
                    if candidate_index in tracker.in_solution:
                        continue

                    if timestamp == step:
                        best_index = candidate_index
                        best_gain = gain_cache[candidate_index]
                        break

                    gain = tracker.marginal_gain(candidate_index)
                    gain_cache[candidate_index] = gain
                    priority = gain + candidate_spans[candidate_index] * span_scale
                    heapq.heappush(heap, (-priority, step, candidate_index))

            if best_index is None or best_gain <= 0:
                raise RuntimeError(
                    "Greedy construction failed to find a candidate with positive gain."
                )

            tracker.add(best_index)
            solution.append(best_index)

        return solution


class RedundancyEliminator:
    def eliminate(self, instance: CoverageInstance, solution: Sequence[int]) -> List[int]:
        tracker = CoverageTracker(instance)
        tracker.reset(solution)

        changed = True
        while changed:
            changed = False
            best_candidate: Optional[int] = None
            best_key: Optional[Tuple[int, float]] = None
            for candidate_index in tracker.in_solution:
                if not tracker.can_remove(candidate_index):
                    continue

                key = (
                    tracker.exclusive_count(candidate_index),
                    tracker.redundancy_score(candidate_index),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_candidate = candidate_index

            if best_candidate is not None:
                tracker.remove(best_candidate)
                changed = True

        return sorted(tracker.in_solution)


class ImprovedLocalSearch:
    def __init__(
        self,
        instance: CoverageInstance,
        initial_solution: Sequence[int],
        rng: random.Random,
        max_steps: int = 2000,
        warmup: int = 200,
        candidate_sample_size: int = 48,
        adaptive_neighborhoods: bool = True,
    ):
        self.instance = instance
        self.initial_solution = list(initial_solution)
        self.rng = rng
        self.max_steps = max_steps
        self.warmup = warmup
        self.candidate_sample_size = candidate_sample_size
        self.adaptive_neighborhoods = adaptive_neighborhoods
        self.stats = defaultdict(int)
        self._active_candidate_sample_size = candidate_sample_size
        self._total_targets = max(1, len(instance.targets))
        self._total_required_units = max(
            1, len(instance.targets) * max(1, instance.required_subset_count)
        )
        self._max_subsets_per_candidate = max(1, math.comb(instance.k, instance.s))

    def _candidate_sample_limit(self, sample_size: Optional[int] = None) -> int:
        base = sample_size if sample_size is not None else self._active_candidate_sample_size
        return max(8, min(len(self.instance.candidates), int(base)))

    def _plateau_stage(self, stagnant_steps: int) -> int:
        if not self.adaptive_neighborhoods:
            return 1

        early_threshold = max(40, self.max_steps // 40)
        late_threshold = max(early_threshold + 1, self.max_steps // 8)
        if stagnant_steps < early_threshold:
            return 0
        if stagnant_steps < late_threshold:
            return 1
        return 2

    def _configure_search_profile(self, stagnant_steps: int) -> Tuple[int, Tuple[float, ...]]:
        stage = self._plateau_stage(stagnant_steps)
        base = max(16, self.candidate_sample_size)
        if stage == 0:
            self._active_candidate_sample_size = max(16, base // 2)
            return stage, (0.52, 0.68, 0.06, 0.18, 0.03, 0.01)
        if stage == 1:
            self._active_candidate_sample_size = base
            return stage, (0.38, 0.56, 0.14, 0.26, 0.10, 0.05)

        self._active_candidate_sample_size = min(
            len(self.instance.candidates),
            max(base + 24, base * 2),
        )
        return stage, (0.20, 0.40, 0.20, 0.34, 0.18, 0.10)

    def solve(self) -> List[int]:
        start = time.time()
        cleaned = RedundancyEliminator().eliminate(self.instance, self.initial_solution)
        LOGGER.info(
            f"  Redundancy elimination: {len(self.initial_solution)} -> {len(cleaned)}"
        )

        tracker = CoverageTracker(self.instance)
        tracker.reset(cleaned)
        best = list(sorted(tracker.in_solution))
        best_size = len(best)
        stagnant_steps = 0

        for step in range(self.max_steps):
            _, move_probabilities = self._configure_search_profile(stagnant_steps)
            improved = False

            if self.rng.random() < move_probabilities[0]:
                improved = self._try_remove(tracker)
            if not improved and self.rng.random() < move_probabilities[1]:
                improved = self._try_replace(tracker, step)
            if not improved and self.rng.random() < move_probabilities[2]:
                improved = self._try_pair_exchange(tracker, step)
            if not improved and self.rng.random() < move_probabilities[3]:
                improved = self._try_remove_repair(tracker, step)
            if not improved and self.rng.random() < move_probabilities[4]:
                improved = self._try_add_then_remove_many(tracker, step)
            if not improved and self.rng.random() < move_probabilities[5]:
                improved = self._try_destroy_repair(tracker, step)

            if improved and tracker.solution_size < best_size:
                best = list(sorted(tracker.in_solution))
                best_size = len(best)
                stagnant_steps = 0
                LOGGER.info(f"  Step {step}: new best {best_size} (search)")
            else:
                stagnant_steps += 1

            if step + 1 >= self.warmup and stagnant_steps >= max(600, self.max_steps // 3):
                LOGGER.info(
                    f"  Early stop at step {step + 1}: no improvement for {stagnant_steps} steps"
                )
                break

            if (step + 1) % 1000 == 0:
                LOGGER.info(f"  Progress {step + 1}/{self.max_steps}, best {best_size}")

        elapsed = time.time() - start
        LOGGER.info(
            f"  Local search: {len(self.initial_solution)} -> {best_size}, "
            f"time {elapsed:.2f}s, improvements {self.stats['improvements']}"
        )
        return best

    def _strip_redundancy(self, tracker: CoverageTracker) -> int:
        removed = 0
        changed = True
        while changed:
            changed = False
            best_candidate: Optional[int] = None
            best_key: Optional[Tuple[int, float]] = None
            for candidate_index in tracker.in_solution:
                if not tracker.can_remove(candidate_index):
                    continue

                key = (
                    tracker.exclusive_count(candidate_index),
                    tracker.redundancy_score(candidate_index),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_candidate = candidate_index

            if best_candidate is not None:
                tracker.remove(best_candidate)
                removed += 1
                changed = True
        return removed

    def _sample_outside_candidates(self, tracker: CoverageTracker) -> List[int]:
        outside = [
            candidate_index
            for candidate_index in range(len(self.instance.candidates))
            if candidate_index not in tracker.in_solution
        ]
        limit = self._candidate_sample_limit()
        if len(outside) <= limit:
            return outside
        return self.rng.sample(outside, limit)

    def _sample_relevant_candidates(
        self, tracker: CoverageTracker, candidate_remove: int
    ) -> List[int]:
        if self.instance.aggregation_mode != AggregationMode.DISTINCT_SUBSETS:
            return self._sample_outside_candidates(tracker)

        target_cover_count = 1 if candidate_remove in tracker.in_solution else 0
        relevant = set()
        for subset_id in self.instance.candidate_subset_ids(candidate_remove):
            if tracker.subset_cover_count[subset_id] != target_cover_count:
                continue
            relevant.update(self.instance.subset_to_candidates[subset_id])

        relevant.difference_update(tracker.in_solution)
        relevant.discard(candidate_remove)
        pool = list(relevant)
        limit = self._candidate_sample_limit()
        if len(pool) <= limit:
            return pool
        return self.rng.sample(pool, limit)

    def _sample_structural_add_candidates(
        self,
        tracker: CoverageTracker,
        pivots: Sequence[int],
        sample_size: Optional[int] = None,
    ) -> List[int]:
        limit = self._candidate_sample_limit(sample_size)
        pool = set()
        for pivot in pivots:
            pool.update(self._sample_relevant_candidates(tracker, pivot))

        if not pool:
            pool.update(self._sample_outside_candidates(tracker))

        candidates = list(pool)
        if len(candidates) > max(limit * 4, limit):
            candidates = self.rng.sample(candidates, max(limit * 4, limit))

        def add_score(candidate_index: int) -> float:
            overlap = sum(
                self.instance.candidate_overlap_in_s_subsets(candidate_index, pivot)
                for pivot in pivots
            )
            return (
                tracker.new_subset_gain(candidate_index)
                + 0.05 * overlap
                + 0.001 * self.instance.candidate_span(candidate_index)
            )

        candidates.sort(key=add_score, reverse=True)
        return candidates[:limit]

    def _try_remove(self, tracker: CoverageTracker) -> bool:
        best_candidate: Optional[int] = None
        best_key: Optional[Tuple[int, float, int]] = None
        for candidate_index in tracker.in_solution:
            key = (
                tracker.exclusive_count(candidate_index),
                tracker.redundancy_score(candidate_index),
                -self.instance.candidate_span(candidate_index),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_candidate = candidate_index

        if best_candidate is None:
            return False

        if tracker.can_remove(best_candidate):
            tracker.remove(best_candidate)
            self.stats["improvements"] += 1
            return True

        return False

    def _try_replace(self, tracker: CoverageTracker, step: int) -> bool:
        if not tracker.in_solution:
            return False

        remove_pool = [
            candidate_index
            for candidate_index in heapq.nsmallest(
                min(6, len(tracker.in_solution)),
                tracker.in_solution,
                key=tracker.exclusive_count,
            )
            if tracker.exclusive_count(candidate_index) > 0
        ]
        if not remove_pool:
            return False

        add_context = {}
        best_pair: Optional[Tuple[int, int]] = None
        best_score = float("-inf")

        for candidate_remove in remove_pool:
            add_pool = self._sample_relevant_candidates(tracker, candidate_remove)
            if not add_pool:
                continue

            newly_uncovered = tracker.get_newly_uncovered(candidate_remove)
            newly_uncovered_count = len(newly_uncovered)
            exclusive_count = tracker.exclusive_count(candidate_remove)

            for candidate_add in add_pool:
                add_info = add_context.get(candidate_add)
                if add_info is None:
                    marginal_gain = tracker.marginal_gain(candidate_add)
                    add_info = (
                        self.instance.candidate_impacted_targets(candidate_add),
                        self.instance.candidate_span(candidate_add),
                        marginal_gain,
                    )
                    add_context[candidate_add] = add_info

                impacted, span, marginal_gain = add_info
                recovered_targets = sum(
                    1 for target_index in impacted if target_index in newly_uncovered
                )
                overlap = self.instance.candidate_overlap_in_s_subsets(
                    candidate_remove, candidate_add
                )
                score = (
                    marginal_gain - exclusive_count
                    + 0.05 * overlap
                    + 0.001 * span
                    + 0.05 * recovered_targets / max(1, newly_uncovered_count)
                    - 0.001 * step
                )
                if score > best_score:
                    best_score = score
                    best_pair = (candidate_remove, candidate_add)

        if best_pair is None:
            return False

        candidate_remove, candidate_add = best_pair
        snapshot = list(tracker.in_solution)
        size_before = tracker.solution_size

        tracker.remove(candidate_remove)
        tracker.add(candidate_add)
        if not tracker.is_feasible():
            tracker.reset(snapshot)
            return False

        self._strip_redundancy(tracker)

        if tracker.solution_size < size_before:
            self.stats["improvements"] += 1
            return True

        tracker.reset(snapshot)
        return False

    def _try_remove_repair(self, tracker: CoverageTracker, step: int) -> bool:
        _ = step
        ordered = heapq.nsmallest(
            min(5, len(tracker.in_solution)),
            tracker.in_solution,
            key=tracker.exclusive_count,
        )
        for candidate_remove in ordered:
            snapshot = list(tracker.in_solution)
            size_before = tracker.solution_size
            tracker.remove(candidate_remove)

            if tracker.is_feasible():
                self.stats["improvements"] += 1
                return True

            add_pool = self._sample_relevant_candidates(tracker, candidate_remove)
            if not add_pool:
                tracker.reset(snapshot)
                continue

            candidate_add = max(add_pool, key=tracker.marginal_gain)
            tracker.add(candidate_add)
            if not tracker.is_feasible():
                tracker.reset(snapshot)
                continue

            self._strip_redundancy(tracker)

            if tracker.solution_size < size_before:
                self.stats["improvements"] += 1
                return True

            tracker.reset(snapshot)

        return False

    def _try_pair_exchange(self, tracker: CoverageTracker, step: int) -> bool:
        _ = step
        if len(tracker.in_solution) < 3:
            return False

        remove_candidates = heapq.nsmallest(
            min(8, len(tracker.in_solution)),
            tracker.in_solution,
            key=lambda candidate_index: (
                tracker.exclusive_count(candidate_index),
                tracker.redundancy_score(candidate_index),
                -self.instance.candidate_span(candidate_index),
            ),
        )
        if len(remove_candidates) < 2:
            return False

        pair_rank = []
        for offset, left in enumerate(remove_candidates):
            left_losses = tracker.get_newly_uncovered(left)
            for right in remove_candidates[offset + 1 :]:
                right_losses = tracker.get_newly_uncovered(right)
                combined_uncovered = len(left_losses | right_losses)
                pair_overlap = self.instance.candidate_overlap_in_s_subsets(left, right)
                pair_rank.append(
                    (
                        combined_uncovered,
                        -pair_overlap,
                        tracker.exclusive_count(left) + tracker.exclusive_count(right),
                        left,
                        right,
                    )
                )

        if not pair_rank:
            return False

        tested_pairs = sorted(pair_rank)[: min(10, len(pair_rank))]
        for _, _, _, left, right in tested_pairs:
            snapshot = list(tracker.in_solution)
            size_before = tracker.solution_size
            tracker.remove(left)
            tracker.remove(right)

            add_pool = set(self._sample_relevant_candidates(tracker, left))
            add_pool.update(self._sample_relevant_candidates(tracker, right))
            add_pool.difference_update(tracker.in_solution)
            if not add_pool:
                tracker.reset(snapshot)
                continue

            ranked_add = sorted(
                add_pool,
                key=lambda candidate_index: (
                    tracker.marginal_gain(candidate_index),
                    self.instance.candidate_overlap_in_s_subsets(candidate_index, left)
                    + self.instance.candidate_overlap_in_s_subsets(candidate_index, right),
                    self.instance.candidate_span(candidate_index),
                ),
                reverse=True,
            )
            single_window = ranked_add[: min(12, len(ranked_add))]
            pair_window = ranked_add[: min(6, len(ranked_add))]

            candidate_add_sets = [[candidate_index] for candidate_index in single_window]
            for add_offset, first in enumerate(pair_window):
                for second in pair_window[add_offset + 1 :]:
                    candidate_add_sets.append([first, second])

            for additions in candidate_add_sets:
                tracker.reset(snapshot)
                tracker.remove(left)
                tracker.remove(right)
                for candidate_add in additions:
                    tracker.add(candidate_add)

                if not tracker.is_feasible():
                    continue

                self._strip_redundancy(tracker)
                if tracker.solution_size < size_before:
                    self.stats["improvements"] += 1
                    return True

            tracker.reset(snapshot)

        return False

    def _try_add_then_remove_many(self, tracker: CoverageTracker, step: int) -> bool:
        _ = step
        if not tracker.in_solution:
            return False

        pivots = heapq.nsmallest(
            min(6, len(tracker.in_solution)),
            tracker.in_solution,
            key=tracker.exclusive_count,
        )
        add_pool = self._sample_structural_add_candidates(
            tracker,
            pivots,
            sample_size=max(self._candidate_sample_limit(), 32),
        )
        if not add_pool:
            return False

        best_solution: Optional[List[int]] = None
        best_size = tracker.solution_size
        snapshot = list(tracker.in_solution)
        trials = min(4, max(1, len(add_pool)))
        max_additions = min(3, len(add_pool))

        for trial in range(trials):
            tracker.reset(snapshot)
            add_count = 1 + (trial % max_additions)
            if len(add_pool) <= add_count:
                additions = add_pool[:]
            else:
                top_window = add_pool[: min(len(add_pool), self._candidate_sample_limit())]
                additions = self.rng.sample(top_window, add_count)

            for candidate_add in additions:
                tracker.add(candidate_add)

            self._strip_redundancy(tracker)
            if tracker.is_feasible() and tracker.solution_size < best_size:
                best_solution = list(tracker.in_solution)
                best_size = tracker.solution_size

        tracker.reset(best_solution if best_solution is not None else snapshot)
        if best_solution is None:
            return False

        self.stats["improvements"] += 1
        return True

    def _try_destroy_repair(self, tracker: CoverageTracker, step: int) -> bool:
        _ = step
        if len(tracker.in_solution) <= 2:
            return False

        snapshot = list(tracker.in_solution)
        size_before = tracker.solution_size
        max_destroy = min(14, max(4, len(snapshot) // 12), len(snapshot) - 1)
        min_destroy = min(3, max_destroy)
        destroy_count = self.rng.randint(min_destroy, max_destroy)
        seed = min(
            snapshot,
            key=lambda candidate_index: (
                tracker.exclusive_count(candidate_index),
                tracker.redundancy_score(candidate_index),
                -self.instance.candidate_span(candidate_index),
            ),
        )
        destroy_pool = sorted(
            snapshot,
            key=lambda candidate_index: (
                -self.instance.candidate_overlap_in_s_subsets(seed, candidate_index),
                tracker.exclusive_count(candidate_index),
                tracker.redundancy_score(candidate_index),
            ),
        )
        removed = destroy_pool[:destroy_count]

        for candidate_remove in removed:
            tracker.remove(candidate_remove)

        repair_pool = set()
        for candidate_remove in removed:
            repair_pool.update(self._sample_relevant_candidates(tracker, candidate_remove))
        if not repair_pool:
            repair_pool.update(self._sample_outside_candidates(tracker))

        max_repairs = destroy_count + 6
        repairs = 0
        while not tracker.is_feasible() and repairs < max_repairs:
            repair_candidates = [
                candidate_index
                for candidate_index in repair_pool
                if candidate_index not in tracker.in_solution
            ]
            if not repair_candidates:
                repair_candidates = self._sample_outside_candidates(tracker)
            if not repair_candidates:
                break

            candidate_add = max(
                repair_candidates,
                key=lambda candidate_index: (
                    tracker.marginal_gain(candidate_index),
                    sum(
                        self.instance.candidate_overlap_in_s_subsets(
                            candidate_index, candidate_remove
                        )
                        for candidate_remove in removed
                    ),
                    self.instance.candidate_span(candidate_index),
                ),
            )
            if tracker.marginal_gain(candidate_add) <= 0:
                break

            tracker.add(candidate_add)
            repair_pool.discard(candidate_add)
            repairs += 1

        if tracker.is_feasible():
            self._strip_redundancy(tracker)

        if tracker.is_feasible() and tracker.solution_size < size_before:
            self.stats["improvements"] += 1
            return True

        tracker.reset(snapshot)
        return False


class ImprovedSA:
    def __init__(
        self,
        instance: CoverageInstance,
        initial_solution: Sequence[int],
        rng: random.Random,
        T_start: float = 5.0,
        T_end: float = 0.001,
        max_iter: int = 4000,
        adaptive_neighborhoods: bool = True,
    ):
        self.instance = instance
        self.initial_solution = list(initial_solution)
        self.rng = rng
        self.T_start = T_start
        self.T_end = T_end
        self.max_iter = max_iter
        self.penalty = max(10, len(initial_solution))
        self.adaptive_neighborhoods = adaptive_neighborhoods

    def _annealing_stage(self, no_improve: int, temperature: float) -> int:
        if not self.adaptive_neighborhoods:
            return 1

        hot_cutoff = max(80, self.max_iter // 40)
        cold_cutoff = max(hot_cutoff + 1, self.max_iter // 12)
        if no_improve < hot_cutoff and temperature > self.T_end * 50.0:
            return 0
        if no_improve < cold_cutoff and temperature > self.T_end * 5.0:
            return 1
        return 2

    def _select_move(
        self,
        tracker: CoverageTracker,
        outside: Set[int],
        stage: int,
    ) -> Optional[Tuple[object, ...]]:
        draw = self.rng.random()
        if stage == 0:
            if draw < 0.30:
                return self._move_remove(tracker, outside)
            if draw < 0.60:
                return self._move_replace(tracker, outside)
            if draw < 0.80:
                return self._move_swap2(tracker, outside)
            if draw < 0.95:
                return self._move_add_remove2(tracker, outside)
            if draw < 0.99:
                return self._move_add_then_remove_many(tracker, outside)
            return self._move_destroy_repair(tracker, outside)

        if stage == 1:
            if draw < 0.22:
                return self._move_remove(tracker, outside)
            if draw < 0.48:
                return self._move_replace(tracker, outside)
            if draw < 0.68:
                return self._move_swap2(tracker, outside)
            if draw < 0.84:
                return self._move_add_remove2(tracker, outside)
            if draw < 0.94:
                return self._move_add_then_remove_many(tracker, outside)
            return self._move_destroy_repair(tracker, outside)

        if draw < 0.12:
            return self._move_remove(tracker, outside)
        if draw < 0.32:
            return self._move_replace(tracker, outside)
        if draw < 0.52:
            return self._move_swap2(tracker, outside)
        if draw < 0.72:
            return self._move_add_remove2(tracker, outside)
        if draw < 0.88:
            return self._move_add_then_remove_many(tracker, outside)
        return self._move_destroy_repair(tracker, outside)

    def _cost(self, tracker: CoverageTracker) -> int:
        return tracker.solution_size + self.penalty * tracker.deficit_units

    def solve(self) -> List[int]:
        start = time.time()
        tracker = CoverageTracker(self.instance)
        tracker.reset(self.initial_solution)
        best = list(sorted(tracker.in_solution))
        current_cost = self._cost(tracker)

        temperature = self.T_start
        cooling = (self.T_end / self.T_start) ** (1.0 / max(1, self.max_iter))
        outside = set(range(len(self.instance.candidates))) - tracker.in_solution
        no_improve = 0

        for iteration in range(self.max_iter):
            temperature *= cooling
            stage = self._annealing_stage(no_improve, temperature)
            move = self._select_move(tracker, outside, stage)

            if move is None:
                continue

            new_cost = self._cost(tracker)
            delta = new_cost - current_cost

            if delta <= 0 or self.rng.random() < math.exp(-delta / max(temperature, 1e-10)):
                current_cost = new_cost
                no_improve = 0
                if tracker.is_feasible() and tracker.solution_size < len(best):
                    best = list(sorted(tracker.in_solution))
                    LOGGER.info(
                        f"  SA iteration {iteration}: new best {len(best)} "
                        f"(T={temperature:.4f})"
                    )
            else:
                self._undo(tracker, move, outside)
                no_improve += 1

            if no_improve > 500:
                temperature = min(temperature * 5.0, self.T_start)
                no_improve = 0

        elapsed = time.time() - start
        LOGGER.info(
            f"  Simulated annealing: {len(self.initial_solution)} -> {len(best)}, "
            f"time {elapsed:.2f}s"
        )
        return best

    def _move_remove(
        self, tracker: CoverageTracker, outside: Set[int]
    ) -> Optional[Tuple[str, int]]:
        solution = list(tracker.in_solution)
        if len(solution) <= 1:
            return None
        candidate_index = self.rng.choice(solution)
        tracker.remove(candidate_index)
        outside.add(candidate_index)
        return ("remove", candidate_index)

    def _sync_outside(self, tracker: CoverageTracker, outside: Set[int]) -> None:
        outside.clear()
        outside.update(set(range(len(self.instance.candidates))) - tracker.in_solution)

    def _strip_redundancy(self, tracker: CoverageTracker) -> None:
        changed = True
        while changed:
            changed = False
            best_candidate: Optional[int] = None
            best_key: Optional[Tuple[int, float]] = None
            for candidate_index in tracker.in_solution:
                if not tracker.can_remove(candidate_index):
                    continue
                key = (
                    tracker.exclusive_count(candidate_index),
                    tracker.redundancy_score(candidate_index),
                )
                if best_key is None or key < best_key:
                    best_key = key
                    best_candidate = candidate_index
            if best_candidate is not None:
                tracker.remove(best_candidate)
                changed = True

    def _relevant_outside_candidates(
        self, outside: Set[int], candidate_index: int
    ) -> List[int]:
        relevant = set()
        for subset_id in self.instance.candidate_subset_ids(candidate_index):
            relevant.update(self.instance.subset_to_candidates[subset_id])

        relevant.intersection_update(outside)
        relevant.discard(candidate_index)
        return list(relevant)

    def _move_replace(
        self, tracker: CoverageTracker, outside: Set[int]
    ) -> Optional[Tuple[str, int, int]]:
        solution = list(tracker.in_solution)
        if not solution or not outside:
            return None

        remove_index = self.rng.choice(solution)
        relevant_add = self._relevant_outside_candidates(outside, remove_index)
        if not relevant_add:
            return None

        add_index = self.rng.choice(relevant_add)
        tracker.remove(remove_index)
        tracker.add(add_index)
        outside.add(remove_index)
        outside.discard(add_index)
        return ("replace", remove_index, add_index)

    def _move_swap2(
        self, tracker: CoverageTracker, outside: Set[int]
    ) -> Optional[Tuple[str, List[int], int]]:
        solution = list(tracker.in_solution)
        non_solution = list(outside)
        if len(solution) < 3 or not non_solution:
            return None

        removed = self.rng.sample(solution, 2)
        added = self.rng.choice(non_solution)
        for candidate_index in removed:
            tracker.remove(candidate_index)
            outside.add(candidate_index)
        tracker.add(added)
        outside.discard(added)
        return ("swap2", removed, added)

    def _move_add_remove2(
        self, tracker: CoverageTracker, outside: Set[int]
    ) -> Optional[Tuple[str, List[int], int]]:
        solution = list(tracker.in_solution)
        if len(solution) < 3 or not outside:
            return None

        pivot = self.rng.choice(solution)
        relevant_add = self._relevant_outside_candidates(outside, pivot)
        if not relevant_add:
            return None

        added = self.rng.choice(relevant_add)
        tracker.add(added)
        outside.discard(added)

        removable = sorted(
            [candidate_index for candidate_index in tracker.in_solution if candidate_index != added],
            key=lambda candidate_index: tracker.exclusive_count(candidate_index),
        )
        removed = removable[:2]
        for candidate_index in removed:
            tracker.remove(candidate_index)
            outside.add(candidate_index)
        return ("add_remove2", removed, added)

    def _move_add_then_remove_many(
        self, tracker: CoverageTracker, outside: Set[int]
    ) -> Optional[Tuple[str, List[int]]]:
        if not outside or not tracker.in_solution:
            return None

        snapshot = list(tracker.in_solution)
        pivots = heapq.nsmallest(
            min(5, len(snapshot)),
            snapshot,
            key=tracker.exclusive_count,
        )
        add_pool = set()
        for pivot in pivots:
            add_pool.update(self._relevant_outside_candidates(outside, pivot))
        if not add_pool:
            add_pool.update(outside)

        candidates = list(add_pool)
        if not candidates:
            return None

        def add_score(candidate_index: int) -> float:
            overlap = sum(
                self.instance.candidate_overlap_in_s_subsets(candidate_index, pivot)
                for pivot in pivots
            )
            return overlap + 0.01 * self.instance.candidate_span(candidate_index)

        candidates.sort(key=add_score, reverse=True)
        window = candidates[: min(len(candidates), 24)]
        add_count = self.rng.randint(1, min(3, len(window)))
        additions = self.rng.sample(window, add_count)
        for candidate_add in additions:
            tracker.add(candidate_add)
            outside.discard(candidate_add)

        self._strip_redundancy(tracker)
        if not tracker.is_feasible() or set(snapshot) == tracker.in_solution:
            tracker.reset(snapshot)
            self._sync_outside(tracker, outside)
            return None

        self._sync_outside(tracker, outside)
        return ("snapshot", snapshot)

    def _move_destroy_repair(
        self, tracker: CoverageTracker, outside: Set[int]
    ) -> Optional[Tuple[str, List[int]]]:
        solution = list(tracker.in_solution)
        if len(solution) <= 2:
            return None

        snapshot = solution[:]
        max_destroy = min(6, max(2, len(solution) // 25), len(solution) - 1)
        destroy_count = self.rng.randint(2, max_destroy)
        destroy_pool = heapq.nsmallest(
            min(max_destroy * 3, len(solution)),
            solution,
            key=lambda candidate_index: (
                tracker.exclusive_count(candidate_index),
                tracker.redundancy_score(candidate_index),
            ),
        )
        removed = self.rng.sample(destroy_pool, destroy_count)
        for candidate_remove in removed:
            tracker.remove(candidate_remove)
            outside.add(candidate_remove)

        repair_pool = set()
        for candidate_remove in removed:
            repair_pool.update(self._relevant_outside_candidates(outside, candidate_remove))
        if not repair_pool:
            repair_pool.update(outside)

        repairs = 0
        while not tracker.is_feasible() and repairs < destroy_count + 3:
            repair_candidates = [
                candidate_index
                for candidate_index in repair_pool
                if candidate_index in outside
            ]
            if not repair_candidates:
                repair_candidates = list(outside)
            if not repair_candidates:
                break

            candidate_add = max(repair_candidates, key=tracker.marginal_gain)
            if tracker.marginal_gain(candidate_add) <= 0:
                break

            tracker.add(candidate_add)
            outside.discard(candidate_add)
            repair_pool.discard(candidate_add)
            repairs += 1

        if tracker.is_feasible():
            self._strip_redundancy(tracker)

        if not tracker.is_feasible() or set(snapshot) == tracker.in_solution:
            tracker.reset(snapshot)
            self._sync_outside(tracker, outside)
            return None

        self._sync_outside(tracker, outside)
        return ("snapshot", snapshot)

    def _undo(
        self,
        tracker: CoverageTracker,
        move: Tuple[object, ...],
        outside: Set[int],
    ) -> None:
        move_type = move[0]
        if move_type == "remove":
            candidate_index = int(move[1])
            tracker.add(candidate_index)
            outside.discard(candidate_index)
            return

        if move_type == "replace":
            remove_index = int(move[1])
            add_index = int(move[2])
            tracker.remove(add_index)
            tracker.add(remove_index)
            outside.discard(remove_index)
            outside.add(add_index)
            return

        if move_type == "swap2":
            removed = list(move[1])
            added = int(move[2])
            tracker.remove(added)
            outside.add(added)
            for candidate_index in removed:
                tracker.add(candidate_index)
                outside.discard(candidate_index)
            return

        if move_type == "add_remove2":
            removed = list(move[1])
            added = int(move[2])
            for candidate_index in removed:
                tracker.add(candidate_index)
                outside.discard(candidate_index)
            tracker.remove(added)
            outside.add(added)
            return

        if move_type == "snapshot":
            snapshot = list(move[1])
            tracker.reset(snapshot)
            self._sync_outside(tracker, outside)
