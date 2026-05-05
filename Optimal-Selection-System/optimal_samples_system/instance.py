"""Problem instance generation and coverage structure construction."""

from __future__ import annotations

import itertools
import math
import random
from typing import List, Optional, Set, Tuple

from .config import AggregationMode, LOGGER, ProblemConfig, choose_seed


class CoverageInstance:
    def __init__(self, config: ProblemConfig, rng: Optional[random.Random] = None):
        config.validate()

        self.config = config
        self.m = config.m
        self.n = config.n
        self.k = config.k
        self.j = config.j
        self.s = config.s
        self.coverage_mode = config.coverage_mode
        self.aggregation_mode = config.aggregation_mode
        self.required_subset_count = config.required_subset_count()
        self.total_s_subsets_per_target = math.comb(self.j, self.s)
        self.rng = rng or random.Random(choose_seed(config.seed))

        if config.samples is None:
            self.samples = tuple(sorted(self.rng.sample(range(1, self.m + 1), self.n)))
        else:
            self.samples = tuple(sorted(config.samples))

        self._sample_positions = tuple(range(self.n))
        self.position_candidates = list(itertools.combinations(self._sample_positions, self.k))
        self.position_targets = list(itertools.combinations(self._sample_positions, self.j))
        self.position_s_subsets = list(itertools.combinations(self._sample_positions, self.s))

        self.candidates = [self._materialize(group) for group in self.position_candidates]
        self.targets = [self._materialize(group) for group in self.position_targets]
        self.s_subsets = [self._materialize(group) for group in self.position_s_subsets]

        self.candidate_masks = [self._to_mask(group) for group in self.position_candidates]
        self.target_masks = [self._to_mask(group) for group in self.position_targets]

        self.s_subset_index = {
            subset: index for index, subset in enumerate(self.position_s_subsets)
        }
        self.target_subset_ids = [
            tuple(
                self.s_subset_index[subset]
                for subset in itertools.combinations(target, self.s)
            )
            for target in self.position_targets
        ]

        self.subset_to_targets = [list() for _ in range(len(self.position_s_subsets))]
        for target_index, subset_ids in enumerate(self.target_subset_ids):
            for subset_id in subset_ids:
                self.subset_to_targets[subset_id].append(target_index)
        self.subset_to_targets = [tuple(indices) for indices in self.subset_to_targets]

        self._candidate_subset_cache: List[Optional[Tuple[int, ...]]] = [
            None
        ] * len(self.position_candidates)
        self._candidate_subset_mask_cache: List[Optional[int]] = [
            None
        ] * len(self.position_candidates)
        self._candidate_impacted_targets_cache: List[Optional[Tuple[int, ...]]] = [
            None
        ] * len(self.position_candidates)

        self.subset_to_candidates = [list() for _ in range(len(self.position_s_subsets))]
        for candidate_index in range(len(self.position_candidates)):
            for subset_id in self.candidate_subset_ids(candidate_index):
                self.subset_to_candidates[subset_id].append(candidate_index)
        self.subset_to_candidates = [
            tuple(indices) for indices in self.subset_to_candidates
        ]

        self.covers: Optional[List[Set[int]]] = None
        self.covered_by: Optional[List[Set[int]]] = None
        if self.aggregation_mode == AggregationMode.SINGLE_CANDIDATE:
            self._build_single_candidate_cover_relation()

        LOGGER.info(self.summary())

    def _materialize(self, positions: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(self.samples[position] for position in positions)

    @staticmethod
    def _to_mask(positions: Tuple[int, ...]) -> int:
        mask = 0
        for position in positions:
            mask |= 1 << position
        return mask

    def _build_single_candidate_cover_relation(self) -> None:
        pair_count = len(self.position_candidates) * len(self.position_targets)
        max_pair_count = 5_000_000
        if pair_count > max_pair_count:
            raise ValueError(
                "single_candidate aggregation would require precomputing too many "
                f"candidate-target pairs ({pair_count:,}). Use distinct_subsets for "
                "larger instances, or reduce n."
            )

        self.covers = [set() for _ in range(len(self.position_candidates))]
        self.covered_by = [set() for _ in range(len(self.position_targets))]

        for candidate_index, candidate_mask in enumerate(self.candidate_masks):
            for target_index, target_mask in enumerate(self.target_masks):
                overlap = (candidate_mask & target_mask).bit_count()
                covered_subset_count = (
                    math.comb(overlap, self.s) if overlap >= self.s else 0
                )
                if covered_subset_count >= self.required_subset_count:
                    self.covers[candidate_index].add(target_index)
                    self.covered_by[target_index].add(candidate_index)

    def summary(self) -> str:
        return (
            f"Parameters: m={self.m}, n={self.n}, k={self.k}, j={self.j}, s={self.s}\n"
            f"Samples: {list(self.samples)}\n"
            f"Semantics: coverage_mode={self.coverage_mode.value}, "
            f"aggregation_mode={self.aggregation_mode.value}, "
            f"required_subsets_per_target={self.required_subset_count}\n"
            f"Targets: {len(self.targets)}, candidates: {len(self.candidates)}, "
            f"s-subsets: {len(self.s_subsets)}"
        )

    def candidate_subset_ids(self, candidate_index: int) -> Tuple[int, ...]:
        cached = self._candidate_subset_cache[candidate_index]
        if cached is not None:
            return cached

        subset_ids = tuple(
            self.s_subset_index[subset]
            for subset in itertools.combinations(
                self.position_candidates[candidate_index], self.s
            )
        )
        self._candidate_subset_cache[candidate_index] = subset_ids
        return subset_ids

    def candidate_subset_mask(self, candidate_index: int) -> int:
        cached = self._candidate_subset_mask_cache[candidate_index]
        if cached is not None:
            return cached

        mask = 0
        for subset_id in self.candidate_subset_ids(candidate_index):
            mask |= 1 << subset_id
        self._candidate_subset_mask_cache[candidate_index] = mask
        return mask

    def candidate_impacted_targets(self, candidate_index: int) -> Tuple[int, ...]:
        cached = self._candidate_impacted_targets_cache[candidate_index]
        if cached is not None:
            return cached

        impacted = set()
        if self.aggregation_mode == AggregationMode.DISTINCT_SUBSETS:
            for subset_id in self.candidate_subset_ids(candidate_index):
                impacted.update(self.subset_to_targets[subset_id])
        else:
            if self.covers is None:
                raise RuntimeError("Single-candidate cover relation was not built.")
            impacted.update(self.covers[candidate_index])

        ordered = tuple(sorted(impacted))
        self._candidate_impacted_targets_cache[candidate_index] = ordered
        return ordered

    def candidate_span(self, candidate_index: int) -> int:
        return len(self.candidate_impacted_targets(candidate_index))

    def candidate_overlap_in_s_subsets(self, left_index: int, right_index: int) -> int:
        return (
            self.candidate_subset_mask(left_index)
            & self.candidate_subset_mask(right_index)
        ).bit_count()

    def candidate_label(self, candidate_index: int) -> Tuple[int, ...]:
        return self.candidates[candidate_index]
