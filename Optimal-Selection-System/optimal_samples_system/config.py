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
    elite_pool_size: int = 4
    elite_guided_restarts: bool = True
    path_relinking: bool = True
    use_ilp: bool = True
    exact_backend: str = "auto"
    exact_time_limit: int = 60
    force_exact: bool = False
    mid_size_exact_improvement: bool = False
    mid_size_exact_candidate_threshold: int = 12000
    mid_size_exact_time_limit: int = 20
    local_branching_improvement: bool = False
    local_branching_candidate_threshold: int = 14000
    local_branching_time_limit: int = 8
    local_branching_base_radius: int = 12
    local_branching_rounds: int = 2
    lp_guided_exact_polish: bool = True
    lp_guided_candidate_threshold: int = 18000
    lp_guided_time_limit: int = 4
    lp_guided_core_cap: int = 900
    lp_guided_rounds: int = 1
    cluster_exact_repair: bool = True
    cluster_exact_candidate_threshold: int = 22000
    cluster_exact_time_limit: int = 4
    cluster_exact_core_cap: int = 1200
    cluster_exact_rounds: int = 1
    cluster_exact_destroy_size: int = 10
    ruin_recreate_exact: bool = False
    ruin_recreate_candidate_threshold: int = 30000
    ruin_recreate_rounds: int = 2
    ruin_recreate_destroy_size: int = 18
    ruin_recreate_core_cap: int = 2400
    ruin_recreate_time_limit: int = 10
    elite_union_exact_polish: bool = False
    elite_union_candidate_threshold: int = 18000
    elite_union_core_cap: int = 2400
    elite_union_time_limit: int = 8
    cardinality_descent: bool = False
    cardinality_descent_candidate_threshold: int = 15000
    cardinality_descent_iterations: int = 900
    cardinality_descent_patience: int = 180
    cardinality_descent_sample_size: int = 48
    cardinality_descent_initial_drop: int = 2
    cardinality_descent_max_rounds: int = 3
    max_local_steps: Optional[int] = None
    max_sa_iterations: Optional[int] = None
    candidate_sample_size: int = 48
    large_instance_candidate_threshold: int = 20000
    large_instance_min_local_steps: int = 160
    large_instance_min_sa_iterations: int = 220
    adaptive_neighborhoods: bool = True
    reduced_exact_polish: bool = True
    reduced_exact_time_limit: int = 4
    reduced_exact_core_cap: int = 192
    reduced_exact_rounds: int = 1
    save_result: bool = False
    db_dir: str = "results_db_v3"

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_restarts": self.n_restarts,
            "elite_pool_size": self.elite_pool_size,
            "elite_guided_restarts": self.elite_guided_restarts,
            "path_relinking": self.path_relinking,
            "use_ilp": self.use_ilp,
            "exact_backend": self.exact_backend,
            "exact_time_limit": self.exact_time_limit,
            "force_exact": self.force_exact,
            "mid_size_exact_improvement": self.mid_size_exact_improvement,
            "mid_size_exact_candidate_threshold": self.mid_size_exact_candidate_threshold,
            "mid_size_exact_time_limit": self.mid_size_exact_time_limit,
            "local_branching_improvement": self.local_branching_improvement,
            "local_branching_candidate_threshold": self.local_branching_candidate_threshold,
            "local_branching_time_limit": self.local_branching_time_limit,
            "local_branching_base_radius": self.local_branching_base_radius,
            "local_branching_rounds": self.local_branching_rounds,
            "lp_guided_exact_polish": self.lp_guided_exact_polish,
            "lp_guided_candidate_threshold": self.lp_guided_candidate_threshold,
            "lp_guided_time_limit": self.lp_guided_time_limit,
            "lp_guided_core_cap": self.lp_guided_core_cap,
            "lp_guided_rounds": self.lp_guided_rounds,
            "cluster_exact_repair": self.cluster_exact_repair,
            "cluster_exact_candidate_threshold": self.cluster_exact_candidate_threshold,
            "cluster_exact_time_limit": self.cluster_exact_time_limit,
            "cluster_exact_core_cap": self.cluster_exact_core_cap,
            "cluster_exact_rounds": self.cluster_exact_rounds,
            "cluster_exact_destroy_size": self.cluster_exact_destroy_size,
            "ruin_recreate_exact": self.ruin_recreate_exact,
            "ruin_recreate_candidate_threshold": self.ruin_recreate_candidate_threshold,
            "ruin_recreate_rounds": self.ruin_recreate_rounds,
            "ruin_recreate_destroy_size": self.ruin_recreate_destroy_size,
            "ruin_recreate_core_cap": self.ruin_recreate_core_cap,
            "ruin_recreate_time_limit": self.ruin_recreate_time_limit,
            "elite_union_exact_polish": self.elite_union_exact_polish,
            "elite_union_candidate_threshold": self.elite_union_candidate_threshold,
            "elite_union_core_cap": self.elite_union_core_cap,
            "elite_union_time_limit": self.elite_union_time_limit,
            "cardinality_descent": self.cardinality_descent,
            "cardinality_descent_candidate_threshold": self.cardinality_descent_candidate_threshold,
            "cardinality_descent_iterations": self.cardinality_descent_iterations,
            "cardinality_descent_patience": self.cardinality_descent_patience,
            "cardinality_descent_sample_size": self.cardinality_descent_sample_size,
            "cardinality_descent_initial_drop": self.cardinality_descent_initial_drop,
            "cardinality_descent_max_rounds": self.cardinality_descent_max_rounds,
            "max_local_steps": self.max_local_steps,
            "max_sa_iterations": self.max_sa_iterations,
            "candidate_sample_size": self.candidate_sample_size,
            "large_instance_candidate_threshold": self.large_instance_candidate_threshold,
            "large_instance_min_local_steps": self.large_instance_min_local_steps,
            "large_instance_min_sa_iterations": self.large_instance_min_sa_iterations,
            "adaptive_neighborhoods": self.adaptive_neighborhoods,
            "reduced_exact_polish": self.reduced_exact_polish,
            "reduced_exact_time_limit": self.reduced_exact_time_limit,
            "reduced_exact_core_cap": self.reduced_exact_core_cap,
            "reduced_exact_rounds": self.reduced_exact_rounds,
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
    validation: Optional[Dict[str, object]] = None

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
            "validation": self.validation,
        }
