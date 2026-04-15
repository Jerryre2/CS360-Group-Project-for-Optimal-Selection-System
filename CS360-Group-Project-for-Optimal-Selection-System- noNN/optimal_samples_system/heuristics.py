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
                if randomized:
                    priority += self.rng.random() * noise_scale
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
        use_neural_guidance: bool = True,
    ):
        self.instance = instance
        self.initial_solution = list(initial_solution)
        self.rng = rng
        self.max_steps = max_steps
        self.warmup = warmup
        self.candidate_sample_size = candidate_sample_size
        self.stats = defaultdict(int)

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

        for step in range(self.max_steps):
            improved = False

            if self.rng.random() < 0.45:
                improved = self._try_remove(tracker, step)
            if not improved and self.rng.random() < 0.55:
                improved = self._try_replace(tracker, step)
            if not improved and self.rng.random() < 0.30:
                improved = self._try_remove_repair(tracker, step)

            if improved and tracker.solution_size < best_size:
                best = list(sorted(tracker.in_solution))
                best_size = len(best)
                LOGGER.info(f"  Step {step}: new best {best_size} (search)")

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
        if len(outside) <= self.candidate_sample_size:
            return outside
        return self.rng.sample(outside, self.candidate_sample_size)

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
        if len(pool) <= self.candidate_sample_size:
            return pool
        return self.rng.sample(pool, self.candidate_sample_size)

    def _try_remove(self, tracker: CoverageTracker, step: int) -> bool:
        _ = step
        best_candidate: Optional[int] = None
        best_score = float("-inf")
        for candidate_index in tracker.in_solution:
            exclusive_count = tracker.exclusive_count(candidate_index)
            score = -exclusive_count
            if tracker.can_remove(candidate_index):
                score += 0.5
            score += tracker.redundancy_score(candidate_index) * 1e-3
            if score > best_score:
                best_score = score
                best_candidate = candidate_index

        if best_candidate is None:
            return False

        if tracker.can_remove(best_candidate):
            tracker.remove(best_candidate)
            self.stats["improvements"] += 1
            return True

        return False

    def _try_replace(self, tracker: CoverageTracker, step: int) -> bool:
        _ = step
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

            exclusive_count = tracker.exclusive_count(candidate_remove)

            for candidate_add in add_pool:
                add_info = add_context.get(candidate_add)
                if add_info is None:
                    marginal_gain = tracker.marginal_gain(candidate_add)
                    add_info = (
                        marginal_gain,
                        self.instance.candidate_overlap_in_s_subsets(
                            candidate_remove, candidate_add
                        ),
                    )
                    add_context[candidate_add] = add_info

                marginal_gain, overlap = add_info
                score = marginal_gain - exclusive_count + overlap * 1e-3
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


class ImprovedSA:
    def __init__(
        self,
        instance: CoverageInstance,
        initial_solution: Sequence[int],
        rng: random.Random,
        T_start: float = 5.0,
        T_end: float = 0.001,
        max_iter: int = 4000,
    ):
        self.instance = instance
        self.initial_solution = list(initial_solution)
        self.rng = rng
        self.T_start = T_start
        self.T_end = T_end
        self.max_iter = max_iter
        self.penalty = max(10, len(initial_solution))

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
            draw = self.rng.random()
            if draw < 0.35:
                move = self._move_remove(tracker, outside)
            elif draw < 0.70:
                move = self._move_replace(tracker, outside)
            elif draw < 0.90:
                move = self._move_swap2(tracker, outside)
            else:
                move = self._move_add_remove2(tracker, outside)

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
