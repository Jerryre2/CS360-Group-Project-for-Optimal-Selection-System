"""Incremental feasibility and redundancy tracking."""

from __future__ import annotations

from typing import Dict, Iterable, Set, Tuple

from .config import AggregationMode
from .instance import CoverageInstance


class CoverageTracker:
    def __init__(self, instance: CoverageInstance):
        self.instance = instance
        self.in_solution: Set[int] = set()
        self._state_version = 0
        self._removal_losses_cache: Dict[int, Tuple[int, Dict[int, int]]] = {}

        if instance.aggregation_mode == AggregationMode.DISTINCT_SUBSETS:
            self.subset_cover_count = [0] * len(instance.s_subsets)
            self.target_covered_count = [0] * len(instance.targets)
            self.unsatisfied_targets = len(instance.targets)
            self.deficit_units = len(instance.targets) * instance.required_subset_count
        else:
            self.cover_count = [0] * len(instance.targets)
            self.unsatisfied_targets = len(instance.targets)
            self.deficit_units = len(instance.targets)

    def reset(self, solution_indices: Iterable[int]) -> None:
        self.in_solution = set()
        self._state_version += 1
        if self.instance.aggregation_mode == AggregationMode.DISTINCT_SUBSETS:
            self.subset_cover_count = [0] * len(self.instance.s_subsets)
            self.target_covered_count = [0] * len(self.instance.targets)
            self.unsatisfied_targets = len(self.instance.targets)
            self.deficit_units = (
                len(self.instance.targets) * self.instance.required_subset_count
            )
        else:
            self.cover_count = [0] * len(self.instance.targets)
            self.unsatisfied_targets = len(self.instance.targets)
            self.deficit_units = len(self.instance.targets)

        for candidate_index in solution_indices:
            self.add(candidate_index)

    def add(self, candidate_index: int) -> None:
        if candidate_index in self.in_solution:
            return

        self.in_solution.add(candidate_index)
        self._state_version += 1
        if self.instance.aggregation_mode == AggregationMode.DISTINCT_SUBSETS:
            for subset_id in self.instance.candidate_subset_ids(candidate_index):
                if self.subset_cover_count[subset_id] == 0:
                    for target_index in self.instance.subset_to_targets[subset_id]:
                        before = self.target_covered_count[target_index]
                        if before < self.instance.required_subset_count:
                            self.deficit_units -= 1
                        self.target_covered_count[target_index] = before + 1
                        if before + 1 == self.instance.required_subset_count:
                            self.unsatisfied_targets -= 1
                self.subset_cover_count[subset_id] += 1
        else:
            if self.instance.covers is None:
                raise RuntimeError("Single-candidate cover relation was not built.")
            for target_index in self.instance.covers[candidate_index]:
                if self.cover_count[target_index] == 0:
                    self.cover_count[target_index] = 1
                    self.unsatisfied_targets -= 1
                    self.deficit_units -= 1
                else:
                    self.cover_count[target_index] += 1

    def remove(self, candidate_index: int) -> None:
        if candidate_index not in self.in_solution:
            return

        self.in_solution.remove(candidate_index)
        self._state_version += 1
        if self.instance.aggregation_mode == AggregationMode.DISTINCT_SUBSETS:
            for subset_id in self.instance.candidate_subset_ids(candidate_index):
                self.subset_cover_count[subset_id] -= 1
                if self.subset_cover_count[subset_id] == 0:
                    for target_index in self.instance.subset_to_targets[subset_id]:
                        before = self.target_covered_count[target_index]
                        if before <= self.instance.required_subset_count:
                            self.deficit_units += 1
                        self.target_covered_count[target_index] = before - 1
                        if before == self.instance.required_subset_count:
                            self.unsatisfied_targets += 1
        else:
            if self.instance.covers is None:
                raise RuntimeError("Single-candidate cover relation was not built.")
            for target_index in self.instance.covers[candidate_index]:
                self.cover_count[target_index] -= 1
                if self.cover_count[target_index] == 0:
                    self.unsatisfied_targets += 1
                    self.deficit_units += 1

    def is_feasible(self) -> bool:
        return self.unsatisfied_targets == 0

    def target_deficit(self, target_index: int) -> int:
        if self.instance.aggregation_mode == AggregationMode.DISTINCT_SUBSETS:
            return max(
                0,
                self.instance.required_subset_count - self.target_covered_count[target_index],
            )
        return 0 if self.cover_count[target_index] > 0 else 1

    def marginal_gain(self, candidate_index: int) -> int:
        if candidate_index in self.in_solution:
            return 0

        if self.instance.aggregation_mode == AggregationMode.SINGLE_CANDIDATE:
            if self.instance.covers is None:
                raise RuntimeError("Single-candidate cover relation was not built.")
            return sum(
                1
                for target_index in self.instance.covers[candidate_index]
                if self.cover_count[target_index] == 0
            )

        target_gain: Dict[int, int] = {}
        gain = 0
        subset_cover_count = self.subset_cover_count
        target_covered_count = self.target_covered_count
        required_subset_count = self.instance.required_subset_count
        for subset_id in self.instance.candidate_subset_ids(candidate_index):
            if subset_cover_count[subset_id] > 0:
                continue
            for target_index in self.instance.subset_to_targets[subset_id]:
                deficit = required_subset_count - target_covered_count[target_index]
                if deficit <= 0:
                    continue
                current_gain = target_gain.get(target_index, 0)
                if current_gain < deficit:
                    target_gain[target_index] = current_gain + 1
                    gain += 1
        return gain

    def new_subset_gain(self, candidate_index: int) -> int:
        if self.instance.aggregation_mode == AggregationMode.SINGLE_CANDIDATE:
            return self.marginal_gain(candidate_index)

        return sum(
            1
            for subset_id in self.instance.candidate_subset_ids(candidate_index)
            if self.subset_cover_count[subset_id] == 0
        )

    def _removal_losses(self, candidate_index: int) -> Dict[int, int]:
        cached = self._removal_losses_cache.get(candidate_index)
        if cached is not None and cached[0] == self._state_version:
            return cached[1]

        losses: Dict[int, int] = {}
        if self.instance.aggregation_mode == AggregationMode.SINGLE_CANDIDATE:
            if self.instance.covers is None:
                raise RuntimeError("Single-candidate cover relation was not built.")
            for target_index in self.instance.covers[candidate_index]:
                if self.cover_count[target_index] == 1:
                    losses[target_index] = 1
            self._removal_losses_cache[candidate_index] = (self._state_version, losses)
            return losses

        for subset_id in self.instance.candidate_subset_ids(candidate_index):
            if self.subset_cover_count[subset_id] != 1:
                continue
            for target_index in self.instance.subset_to_targets[subset_id]:
                losses[target_index] = losses.get(target_index, 0) + 1
        self._removal_losses_cache[candidate_index] = (self._state_version, losses)
        return losses

    def can_remove(self, candidate_index: int) -> bool:
        if candidate_index not in self.in_solution:
            return False

        losses = self._removal_losses(candidate_index)
        if not losses:
            return True

        if self.instance.aggregation_mode == AggregationMode.SINGLE_CANDIDATE:
            return False

        for target_index, loss in losses.items():
            if (
                self.target_covered_count[target_index] - loss
                < self.instance.required_subset_count
            ):
                return False
        return True

    def exclusive_count(self, candidate_index: int) -> int:
        if candidate_index not in self.in_solution:
            return 0

        losses = self._removal_losses(candidate_index)
        if self.instance.aggregation_mode == AggregationMode.SINGLE_CANDIDATE:
            return len(losses)

        count = 0
        for target_index, loss in losses.items():
            if (
                self.target_covered_count[target_index] - loss
                < self.instance.required_subset_count
            ):
                count += 1
        return count

    def redundancy_score(self, candidate_index: int) -> float:
        impacted = self.instance.candidate_impacted_targets(candidate_index)
        if not impacted:
            return 0.0

        if self.instance.aggregation_mode == AggregationMode.SINGLE_CANDIDATE:
            return sum(self.cover_count[target_index] for target_index in impacted) / len(
                impacted
            )

        total_surplus = sum(
            self.target_covered_count[target_index] - self.instance.required_subset_count
            for target_index in impacted
        )
        return total_surplus / len(impacted)

    def get_newly_uncovered(self, candidate_index: int) -> Set[int]:
        losses = self._removal_losses(candidate_index)
        if self.instance.aggregation_mode == AggregationMode.SINGLE_CANDIDATE:
            return set(losses)

        return {
            target_index
            for target_index, loss in losses.items()
            if self.target_covered_count[target_index] - loss
            < self.instance.required_subset_count
        }

    @property
    def solution_size(self) -> int:
        return len(self.in_solution)
