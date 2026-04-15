"""Configuration and shared data structures."""

from __future__ import annotations

import argparse
import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple


LOGGER = logging.getLogger("optimal_samples")


class CoverageMode(str, Enum):
    AT_LEAST_ONE = "at_least_one"
    AT_LEAST_R = "at_least_r"
    ALL_SUBSETS = "all_subsets"


class AggregationMode(str, Enum):
    DISTINCT_SUBSETS = "distinct_subsets"
    SINGLE_CANDIDATE = "single_candidate"


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(message)s")


def parse_samples_arg(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    if not text:
        return None

    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return None

    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Samples must be a comma-separated list of integers."
        ) from exc


def choose_seed(seed: Optional[int]) -> int:
    return seed if seed is not None else int(time.time() * 1000) % (2**31 - 1)


@dataclass
class ProblemConfig:
    m: int
    n: int
    k: int
    j: int
    s: int
    samples: Optional[Tuple[int, ...]] = None
    coverage_mode: CoverageMode = CoverageMode.AT_LEAST_ONE
    aggregation_mode: AggregationMode = AggregationMode.DISTINCT_SUBSETS
    required_r: Optional[int] = None
    seed: Optional[int] = None

    def validate(self) -> None:
        if self.m <= 0:
            raise ValueError("m must be positive.")
        if self.n <= 0:
            raise ValueError("n must be positive.")
        if self.k <= 0 or self.j <= 0 or self.s <= 0:
            raise ValueError("k, j, and s must be positive.")
        if self.n > self.m:
            raise ValueError("n must satisfy n <= m.")
        if self.k > self.n:
            raise ValueError("k must satisfy k <= n.")
        if self.j > self.k:
            raise ValueError("j must satisfy j <= k.")
        if self.s > self.j:
            raise ValueError("s must satisfy s <= j.")

        if self.samples is not None:
            if len(self.samples) != self.n:
                raise ValueError("The manual sample list length must equal n.")
            if len(set(self.samples)) != self.n:
                raise ValueError("Manual samples must be distinct.")
            if any(sample < 1 or sample > self.m for sample in self.samples):
                raise ValueError("Manual samples must lie in the range [1, m].")

        total_s_subsets = math.comb(self.j, self.s)
        if self.coverage_mode == CoverageMode.AT_LEAST_R:
            if self.required_r is None:
                raise ValueError("required_r must be provided for at_least_r.")
            if not 1 <= self.required_r <= total_s_subsets:
                raise ValueError(
                    f"required_r must be in [1, C(j, s)] = [1, {total_s_subsets}]."
                )
        elif self.required_r is not None and self.required_r <= 0:
            raise ValueError("required_r must be positive when supplied.")

    def required_subset_count(self) -> int:
        total_s_subsets = math.comb(self.j, self.s)
        if self.coverage_mode == CoverageMode.AT_LEAST_ONE:
            return 1
        if self.coverage_mode == CoverageMode.ALL_SUBSETS:
            return total_s_subsets
        if self.required_r is None:
            raise ValueError("required_r must be set for CoverageMode.AT_LEAST_R.")
        return self.required_r

    def normalized_samples(self) -> Optional[Tuple[int, ...]]:
        return tuple(sorted(self.samples)) if self.samples is not None else None

    def to_dict(self) -> Dict[str, object]:
        return {
            "m": self.m,
            "n": self.n,
            "k": self.k,
            "j": self.j,
            "s": self.s,
            "samples": list(self.normalized_samples()) if self.samples else None,
            "coverage_mode": self.coverage_mode.value,
            "aggregation_mode": self.aggregation_mode.value,
            "required_r": self.required_r,
            "seed": self.seed,
        }


@dataclass
class SolverConfig:
    n_restarts: int = 5
    use_ilp: bool = True
    max_local_steps: Optional[int] = None
    max_sa_iterations: Optional[int] = None
    candidate_sample_size: int = 48
    use_neural_guidance: bool = True
    save_result: bool = False
    db_dir: str = "results_db_v3"

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_restarts": self.n_restarts,
            "use_ilp": self.use_ilp,
            "max_local_steps": self.max_local_steps,
            "max_sa_iterations": self.max_sa_iterations,
            "candidate_sample_size": self.candidate_sample_size,
            "use_neural_guidance": self.use_neural_guidance,
            "save_result": self.save_result,
            "db_dir": self.db_dir,
        }


@dataclass
class SolveResult:
    solution_indices: List[int]
    groups: List[Tuple[int, ...]]
    num_groups: int
    exact_size: Optional[int]
    exact_method: Optional[str]
    samples: List[int]
    params: Dict[str, object]
    solver: Dict[str, object]
    elapsed_seconds: float
    seed: int
    required_subsets_per_target: int
    num_targets: int
    num_candidates: int
    aggregation_mode: str
    coverage_mode: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "solution_indices": self.solution_indices,
            "groups": [list(group) for group in self.groups],
            "num_groups": self.num_groups,
            "exact_size": self.exact_size,
            "exact_method": self.exact_method,
            "samples": self.samples,
            "params": self.params,
            "solver": self.solver,
            "elapsed_seconds": self.elapsed_seconds,
            "seed": self.seed,
            "required_subsets_per_target": self.required_subsets_per_target,
            "num_targets": self.num_targets,
            "num_candidates": self.num_candidates,
            "aggregation_mode": self.aggregation_mode,
            "coverage_mode": self.coverage_mode,
        }
