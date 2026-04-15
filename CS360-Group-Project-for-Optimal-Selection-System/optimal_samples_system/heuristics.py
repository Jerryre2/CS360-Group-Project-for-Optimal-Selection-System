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


class ImprovedNeuralNet:
    def __init__(
        self,
        rng: random.Random,
        input_dim: int = 12,
        hidden1: int = 32,
        hidden2: int = 16,
        learning_rate: float = 0.005,
    ):
        self.rng = rng
        self.input_dim = input_dim
        self.lr = learning_rate
        self.hidden1_size = hidden1
        self.hidden2_size = hidden2

        self.W1 = self._xavier(input_dim, hidden1)
        self.b1 = [0.0] * hidden1
        self.W2 = self._xavier(hidden1, hidden2)
        self.b2 = [0.0] * hidden2
        self.W3 = self._xavier(hidden2, 1)
        self.b3 = [0.0]

        self.positive_buffer: List[Tuple[List[float], float]] = []
        self.negative_buffer: List[Tuple[List[float], float]] = []
        self.buffer_max = 500
        self.batch_size = 20
        self.train_count = 0

        self.feat_sum = [0.0] * input_dim
        self.feat_sq_sum = [0.0] * input_dim
        self.feat_count = 0
        self.feat_mean = [0.0] * input_dim
        self.feat_inv_std = [1.0] * input_dim

    def _xavier(self, fan_in: int, fan_out: int) -> List[List[float]]:
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        return [
            [self.rng.uniform(-limit, limit) for _ in range(fan_out)]
            for _ in range(fan_in)
        ]

    def _update_stats(self, features: Sequence[float]) -> None:
        for index, value in enumerate(features):
            self.feat_sum[index] += value
            self.feat_sq_sum[index] += value * value
        self.feat_count += 1

        if self.feat_count < 10:
            return

        inv_count = 1.0 / self.feat_count
        for index in range(self.input_dim):
            mean = self.feat_sum[index] * inv_count
            variance = self.feat_sq_sum[index] * inv_count - mean * mean
            self.feat_mean[index] = mean
            self.feat_inv_std[index] = 1.0 / math.sqrt(max(variance, 1e-8))

    def _normalize(self, features: Sequence[float]) -> List[float]:
        if self.feat_count < 10:
            return list(features)

        return [
            (features[index] - self.feat_mean[index]) * self.feat_inv_std[index]
            for index in range(self.input_dim)
        ]

    @staticmethod
    def _lrelu(value: float) -> float:
        return value if value > 0 else 0.01 * value

    @staticmethod
    def _lrelu_d(value: float) -> float:
        return 1.0 if value > 0 else 0.01

    @staticmethod
    def _sigmoid(value: float) -> float:
        clipped = max(-500.0, min(500.0, value))
        return 1.0 / (1.0 + math.exp(-clipped))

    def forward(
        self, raw_features: Sequence[float]
    ) -> Tuple[
        float,
        Tuple[List[float], List[float], List[float], List[float], List[float], float, float],
    ]:
        features = self._normalize(raw_features)
        h1_pre = self.b1[:]
        for feature_index, feature_value in enumerate(features):
            row = self.W1[feature_index]
            for hidden_index in range(self.hidden1_size):
                h1_pre[hidden_index] += feature_value * row[hidden_index]

        h1 = [0.0] * self.hidden1_size
        for hidden_index, value in enumerate(h1_pre):
            h1[hidden_index] = self._lrelu(value)

        h2_pre = self.b2[:]
        for hidden_index, hidden_value in enumerate(h1):
            row = self.W2[hidden_index]
            for output_index in range(self.hidden2_size):
                h2_pre[output_index] += hidden_value * row[output_index]

        h2 = [0.0] * self.hidden2_size
        for output_index, value in enumerate(h2_pre):
            h2[output_index] = self._lrelu(value)

        out_pre = self.b3[0]
        for output_index, hidden_value in enumerate(h2):
            out_pre += hidden_value * self.W3[output_index][0]
        out = self._sigmoid(out_pre)
        cache = (features, h1_pre, h1, h2_pre, h2, out_pre, out)
        return out, cache

    def backward(
        self,
        cache: Tuple[List[float], List[float], List[float], List[float], List[float], float, float],
        label: float,
    ) -> None:
        features, h1_pre, h1, h2_pre, h2, _, out = cache

        delta_out = out - label
        delta_h2 = [0.0] * self.hidden2_size
        for output_index in range(self.hidden2_size):
            delta_h2[output_index] = (
                delta_out
                * self.W3[output_index][0]
                * self._lrelu_d(h2_pre[output_index])
            )

        delta_h1 = [0.0] * self.hidden1_size
        for hidden_index in range(self.hidden1_size):
            total = 0.0
            row = self.W2[hidden_index]
            for output_index in range(self.hidden2_size):
                total += delta_h2[output_index] * row[output_index]
            delta_h1[hidden_index] = total * self._lrelu_d(h1_pre[hidden_index])

        lr = self.lr
        for output_index in range(self.hidden2_size):
            self.W3[output_index][0] -= lr * delta_out * h2[output_index]
        self.b3[0] -= lr * delta_out

        for hidden_index in range(self.hidden1_size):
            row = self.W2[hidden_index]
            hidden_value = h1[hidden_index]
            for output_index in range(self.hidden2_size):
                row[output_index] -= lr * delta_h2[output_index] * hidden_value
        for output_index in range(self.hidden2_size):
            self.b2[output_index] -= lr * delta_h2[output_index]

        for feature_index in range(self.input_dim):
            row = self.W1[feature_index]
            feature_value = features[feature_index]
            for hidden_index in range(self.hidden1_size):
                row[hidden_index] -= lr * delta_h1[hidden_index] * feature_value
        for hidden_index in range(self.hidden1_size):
            self.b1[hidden_index] -= lr * delta_h1[hidden_index]

    def add_sample(self, features: Sequence[float], label: float) -> None:
        feature_list = list(features)
        self._update_stats(feature_list)
        buffer = self.positive_buffer if label > 0.5 else self.negative_buffer
        buffer.append((feature_list, label))
        if len(buffer) > self.buffer_max:
            buffer.pop(0)

        total = len(self.positive_buffer) + len(self.negative_buffer)
        if total >= self.batch_size:
            self._train_batch()

    def _train_batch(self) -> None:
        num_positive = min(len(self.positive_buffer), self.batch_size // 2)
        num_negative = min(len(self.negative_buffer), self.batch_size - num_positive)

        batch: List[Tuple[List[float], float]] = []
        if num_positive > 0:
            batch.extend(self.rng.sample(self.positive_buffer, num_positive))
        if num_negative > 0:
            batch.extend(self.rng.sample(self.negative_buffer, num_negative))

        self.rng.shuffle(batch)
        for features, label in batch:
            _, cache = self.forward(features)
            self.backward(cache, label)
        self.train_count += 1

    def predict(self, features: Sequence[float]) -> float:
        if self.train_count == 0:
            return 0.5
        probability, _ = self.forward(features)
        return probability


class FeatureExtractor:
    def __init__(self, instance: CoverageInstance, greedy_size: int):
        self.instance = instance
        self.greedy_size = greedy_size
        self.total_targets = len(instance.targets)
        self.total_subsets = len(instance.s_subsets)
        self.max_subsets_per_candidate = math.comb(instance.k, instance.s)
        self.total_required_units = (
            len(instance.targets) * max(1, instance.required_subset_count)
        )

    def extract_remove(
        self, tracker: CoverageTracker, candidate_index: int, step: int, total_steps: int
    ) -> List[float]:
        return [
            tracker.exclusive_count(candidate_index) / max(1, self.total_targets),
            1.0 if tracker.can_remove(candidate_index) else 0.0,
            tracker.redundancy_score(candidate_index)
            / max(1.0, float(self.instance.required_subset_count)),
            self.instance.candidate_span(candidate_index) / max(1, self.total_targets),
            len(self.instance.candidate_subset_ids(candidate_index))
            / max(1, self.max_subsets_per_candidate),
            tracker.new_subset_gain(candidate_index) / max(1, self.total_subsets),
            tracker.solution_size / max(1, self.greedy_size),
            step / max(1, total_steps),
            tracker.unsatisfied_targets / max(1, self.total_targets),
            tracker.deficit_units / max(1, self.total_required_units),
            0.0,
            self.instance.required_subset_count
            / max(1, self.instance.total_s_subsets_per_target),
        ]

    def extract_replace(
        self,
        tracker: CoverageTracker,
        candidate_remove: int,
        candidate_add: int,
        step: int,
        total_steps: int,
    ) -> List[float]:
        newly_uncovered = tracker.get_newly_uncovered(candidate_remove)
        impacted = self.instance.candidate_impacted_targets(candidate_add)
        recovered_targets = sum(
            1 for target_index in impacted if target_index in newly_uncovered
        )
        overlap = self.instance.candidate_overlap_in_s_subsets(
            candidate_remove, candidate_add
        )

        return [
            tracker.exclusive_count(candidate_remove) / max(1, self.total_targets),
            recovered_targets / max(1, len(newly_uncovered)),
            tracker.marginal_gain(candidate_add) / max(1, self.total_required_units),
            overlap / max(1, self.max_subsets_per_candidate),
            self.instance.candidate_span(candidate_add) / max(1, self.total_targets),
            tracker.redundancy_score(candidate_remove)
            / max(1.0, float(self.instance.required_subset_count)),
            tracker.solution_size / max(1, self.greedy_size),
            step / max(1, total_steps),
            tracker.unsatisfied_targets / max(1, self.total_targets),
            tracker.deficit_units / max(1, self.total_required_units),
            1.0,
            len(newly_uncovered) / max(1, self.total_targets),
        ]


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
        self.use_neural_guidance = use_neural_guidance
        self.nn = ImprovedNeuralNet(rng=self.rng, input_dim=12)
        self.feat = FeatureExtractor(instance, max(1, len(initial_solution)))
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
            use_nn = (
                self.use_neural_guidance
                and step >= self.warmup
                and self.nn.train_count >= 3
                and self.rng.random() > 0.15
            )
            improved = False

            if self.rng.random() < 0.45:
                improved = self._try_remove(tracker, step, use_nn)
            if not improved and self.rng.random() < 0.55:
                improved = self._try_replace(tracker, step, use_nn)
            if not improved and self.rng.random() < 0.30:
                improved = self._try_remove_repair(tracker, step)

            if improved and tracker.solution_size < best_size:
                best = list(sorted(tracker.in_solution))
                best_size = len(best)
                label = "NN" if use_nn else "search"
                LOGGER.info(f"  Step {step}: new best {best_size} ({label})")

            if (step + 1) % 1000 == 0:
                LOGGER.info(
                    f"  Progress {step + 1}/{self.max_steps}, best {best_size}, "
                    f"NN batches {self.nn.train_count}"
                )

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

    def _try_remove(self, tracker: CoverageTracker, step: int, use_nn: bool) -> bool:
        best_candidate: Optional[int] = None
        best_features: Optional[List[float]] = None
        best_score = float("-inf")
        for candidate_index in tracker.in_solution:
            features = self.feat.extract_remove(
                tracker, candidate_index, step, self.max_steps
            )
            score = (
                self.nn.predict(features)
                if use_nn
                else -tracker.exclusive_count(candidate_index)
            )
            if score > best_score:
                best_score = score
                best_candidate = candidate_index
                best_features = features

        if best_candidate is None or best_features is None:
            return False

        if tracker.can_remove(best_candidate):
            tracker.remove(best_candidate)
            self.nn.add_sample(best_features, 1.0)
            self.stats["improvements"] += 1
            return True

        self.nn.add_sample(best_features, 0.0)
        return False

    def _try_replace(self, tracker: CoverageTracker, step: int, use_nn: bool) -> bool:
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

        total_targets = max(1, self.feat.total_targets)
        total_required_units = max(1, self.feat.total_required_units)
        max_subsets_per_candidate = max(1, self.feat.max_subsets_per_candidate)
        solution_ratio = tracker.solution_size / max(1, self.feat.greedy_size)
        step_ratio = step / max(1, self.max_steps)
        unsatisfied_ratio = tracker.unsatisfied_targets / total_targets
        deficit_ratio = tracker.deficit_units / total_required_units

        add_context = {}
        best_pair: Optional[Tuple[int, int, List[float]]] = None
        best_score = float("-inf")

        for candidate_remove in remove_pool:
            add_pool = self._sample_relevant_candidates(tracker, candidate_remove)
            if not add_pool:
                continue

            newly_uncovered = tracker.get_newly_uncovered(candidate_remove)
            newly_uncovered_count = len(newly_uncovered)
            exclusive_count = tracker.exclusive_count(candidate_remove)
            exclusive_ratio = exclusive_count / total_targets
            redundancy_ratio = tracker.redundancy_score(candidate_remove) / max(
                1.0, float(self.instance.required_subset_count)
            )

            for candidate_add in add_pool:
                add_info = add_context.get(candidate_add)
                if add_info is None:
                    marginal_gain = tracker.marginal_gain(candidate_add)
                    add_info = (
                        self.instance.candidate_impacted_targets(candidate_add),
                        self.instance.candidate_span(candidate_add) / total_targets,
                        marginal_gain,
                        marginal_gain / total_required_units,
                    )
                    add_context[candidate_add] = add_info

                impacted, span_ratio, marginal_gain, marginal_gain_ratio = add_info
                recovered_targets = sum(
                    1 for target_index in impacted if target_index in newly_uncovered
                )
                overlap = self.instance.candidate_overlap_in_s_subsets(
                    candidate_remove, candidate_add
                )
                features = [
                    exclusive_ratio,
                    recovered_targets / max(1, newly_uncovered_count),
                    marginal_gain_ratio,
                    overlap / max_subsets_per_candidate,
                    span_ratio,
                    redundancy_ratio,
                    solution_ratio,
                    step_ratio,
                    unsatisfied_ratio,
                    deficit_ratio,
                    1.0,
                    newly_uncovered_count / total_targets,
                ]
                score = (
                    self.nn.predict(features)
                    if use_nn
                    else marginal_gain - exclusive_count
                )
                if score > best_score:
                    best_score = score
                    best_pair = (candidate_remove, candidate_add, features)

        if best_pair is None:
            return False

        candidate_remove, candidate_add, features = best_pair
        snapshot = list(tracker.in_solution)
        size_before = tracker.solution_size

        tracker.remove(candidate_remove)
        tracker.add(candidate_add)
        if not tracker.is_feasible():
            tracker.reset(snapshot)
            self.nn.add_sample(features, 0.0)
            return False

        self._strip_redundancy(tracker)

        if tracker.solution_size < size_before:
            self.nn.add_sample(features, 1.0)
            self.stats["improvements"] += 1
            return True

        tracker.reset(snapshot)
        self.nn.add_sample(features, 0.0)
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
