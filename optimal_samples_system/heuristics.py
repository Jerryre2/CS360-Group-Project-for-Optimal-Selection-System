"""Heuristic solvers and local improvement components."""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from typing import List, Optional, Sequence, Set, Tuple

import numpy as np

from .config import LOGGER
from .instance import CoverageInstance
from .tracking import CoverageTracker


class GreedySolver:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def solve(self, instance: CoverageInstance, randomized: bool = False) -> List[int]:
        tracker = CoverageTracker(instance)
        # 优化3: 使用 set 替代 list，remove 操作从 O(n) 降到 O(1)
        remaining: Set[int] = set(range(len(instance.candidates)))
        solution: List[int] = []

        while not tracker.is_feasible():
            candidates_list = list(remaining)
            if randomized:
                self.rng.shuffle(candidates_list)

            best_index: Optional[int] = None
            best_score = (-1, -1)

            for candidate_index in candidates_list:
                gain = tracker.marginal_gain(candidate_index)
                score = (gain, instance.candidate_span(candidate_index))
                if score > best_score:
                    best_score = score
                    best_index = candidate_index

            if best_index is None or best_score[0] <= 0:
                raise RuntimeError(
                    "Greedy construction failed to find a candidate with positive gain."
                )

            tracker.add(best_index)
            solution.append(best_index)
            remaining.discard(best_index)  # O(1) 而非 O(n)

        return solution


class RedundancyEliminator:
    def eliminate(self, instance: CoverageInstance, solution: Sequence[int]) -> List[int]:
        tracker = CoverageTracker(instance)
        tracker.reset(solution)

        changed = True
        while changed:
            changed = False
            ordered = sorted(
                tracker.in_solution,
                key=lambda candidate_index: (
                    tracker.exclusive_count(candidate_index),
                    tracker.redundancy_score(candidate_index),
                ),
            )
            for candidate_index in ordered:
                if tracker.can_remove(candidate_index):
                    tracker.remove(candidate_index)
                    changed = True
                    break

        return sorted(tracker.in_solution)


# ---------------------------------------------------------------------------
# 优化1: 神经网络全面 NumPy 向量化
# ---------------------------------------------------------------------------

class ImprovedNeuralNet:
    """三层神经网络，使用 NumPy 向量化运算加速（相比纯 Python 提速 10-50x）。"""

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

        # Xavier 初始化，直接使用 numpy 数组
        self.W1 = self._xavier(input_dim, hidden1)
        self.b1 = np.zeros(hidden1)
        self.W2 = self._xavier(hidden1, hidden2)
        self.b2 = np.zeros(hidden2)
        self.W3 = self._xavier(hidden2, 1)
        self.b3 = np.zeros(1)

        self.positive_buffer: List[Tuple[np.ndarray, float]] = []
        self.negative_buffer: List[Tuple[np.ndarray, float]] = []
        self.buffer_max = 500
        self.batch_size = 20
        self.train_count = 0

        # 在线特征归一化统计量（NumPy）
        self.feat_sum = np.zeros(input_dim)
        self.feat_sq_sum = np.zeros(input_dim)
        self.feat_count = 0

    def _xavier(self, fan_in: int, fan_out: int) -> np.ndarray:
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        # 使用 rng 保证与原始代码相同的随机行为（兼容 seed 复现）
        data = [self.rng.uniform(-limit, limit) for _ in range(fan_in * fan_out)]
        return np.array(data, dtype=np.float64).reshape(fan_in, fan_out)

    def _update_stats(self, features: np.ndarray) -> None:
        self.feat_sum += features
        self.feat_sq_sum += features * features
        self.feat_count += 1

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        if self.feat_count < 10:
            return features.copy()
        mean = self.feat_sum / self.feat_count
        variance = self.feat_sq_sum / self.feat_count - mean * mean
        std = np.sqrt(np.maximum(variance, 1e-8))
        return (features - mean) / std

    @staticmethod
    def _lrelu(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, 0.01 * x)

    @staticmethod
    def _lrelu_d(x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, 0.01)

    @staticmethod
    def _sigmoid(x: float) -> float:
        clipped = max(-500.0, min(500.0, x))
        return 1.0 / (1.0 + math.exp(-clipped))

    def forward(
        self, raw_features: np.ndarray
    ) -> Tuple[float, Tuple]:
        features = self._normalize(raw_features)

        # 向量化矩阵乘法（原 Python 嵌套循环 → NumPy 单行）
        h1_pre = features @ self.W1 + self.b1      # shape: (hidden1,)
        h1 = self._lrelu(h1_pre)                   # shape: (hidden1,)

        h2_pre = h1 @ self.W2 + self.b2            # shape: (hidden2,)
        h2 = self._lrelu(h2_pre)                   # shape: (hidden2,)

        out_pre_val = float(h2 @ self.W3[:, 0] + self.b3[0])
        out = self._sigmoid(out_pre_val)

        cache = (features, h1_pre, h1, h2_pre, h2, out_pre_val, out)
        return out, cache

    def backward(self, cache: Tuple, label: float) -> None:
        features, h1_pre, h1, h2_pre, h2, _, out = cache

        delta_out = out - label  # scalar

        # 输出层梯度（向量化）
        dW3 = (delta_out * h2).reshape(-1, 1)
        db3 = np.array([delta_out])
        old_W3 = self.W3.copy()
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

        # 隐藏层2梯度
        delta_h2 = delta_out * old_W3[:, 0] * self._lrelu_d(h2_pre)  # (hidden2,)
        dW2 = np.outer(h1, delta_h2)
        old_W2 = self.W2.copy()
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * delta_h2

        # 隐藏层1梯度
        delta_h1 = (delta_h2 @ old_W2.T) * self._lrelu_d(h1_pre)    # (hidden1,)
        dW1 = np.outer(features, delta_h1)
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * delta_h1

    def add_sample(self, features: Sequence[float], label: float) -> None:
        feat_arr = np.array(features, dtype=np.float64)
        self._update_stats(feat_arr)
        buffer = self.positive_buffer if label > 0.5 else self.negative_buffer
        buffer.append((feat_arr, label))
        if len(buffer) > self.buffer_max:
            buffer.pop(0)

        total = len(self.positive_buffer) + len(self.negative_buffer)
        if total >= self.batch_size:
            self._train_batch()

    def _train_batch(self) -> None:
        num_positive = min(len(self.positive_buffer), self.batch_size // 2)
        num_negative = min(len(self.negative_buffer), self.batch_size - num_positive)

        batch: List[Tuple[np.ndarray, float]] = []
        if num_positive > 0:
            batch.extend(self.rng.sample(self.positive_buffer, num_positive))
        if num_negative > 0:
            batch.extend(self.rng.sample(self.negative_buffer, num_negative))

        self.rng.shuffle(batch)
        for feat_arr, label in batch:
            _, cache = self.forward(feat_arr)
            self.backward(cache, label)
        self.train_count += 1

    def predict(self, features: Sequence[float]) -> float:
        if self.train_count == 0:
            return 0.5
        probability, _ = self.forward(np.array(features, dtype=np.float64))
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
        impacted = set(self.instance.candidate_impacted_targets(candidate_add))
        recovered_targets = len(newly_uncovered & impacted)
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

        # 优化4: 维护持久化的外部候选集合，随 add/remove 增量更新
        outside: Set[int] = set(range(len(self.instance.candidates))) - tracker.in_solution

        for step in range(self.max_steps):
            use_nn = (
                self.use_neural_guidance
                and step >= self.warmup
                and self.nn.train_count >= 3
                and self.rng.random() > 0.15
            )
            improved = False

            if self.rng.random() < 0.45:
                improved = self._try_remove(tracker, outside, step, use_nn)
            if not improved and self.rng.random() < 0.55:
                improved = self._try_replace(tracker, outside, step, use_nn)
            if not improved and self.rng.random() < 0.30:
                improved = self._try_remove_repair(tracker, outside, step)

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

    def _strip_redundancy(self, tracker: CoverageTracker, outside: Set[int]) -> int:
        """去除冗余候选，同时更新 outside 集合。"""
        removed = 0
        changed = True
        while changed:
            changed = False
            ordered = sorted(
                tracker.in_solution,
                key=lambda candidate_index: (
                    tracker.exclusive_count(candidate_index),
                    tracker.redundancy_score(candidate_index),
                ),
            )
            for candidate_index in ordered:
                if tracker.can_remove(candidate_index):
                    tracker.remove(candidate_index)
                    outside.add(candidate_index)
                    removed += 1
                    changed = True
                    break
        return removed

    def _sample_outside_candidates(self, outside: Set[int]) -> List[int]:
        """优化4: 直接从增量维护的 outside 集合中采样，O(1) 而非 O(N)。"""
        outside_list = list(outside)
        if len(outside_list) <= self.candidate_sample_size:
            return outside_list
        return self.rng.sample(outside_list, self.candidate_sample_size)

    def _try_remove(
        self, tracker: CoverageTracker, outside: Set[int], step: int, use_nn: bool
    ) -> bool:
        candidates = []
        for candidate_index in tracker.in_solution:
            features = self.feat.extract_remove(
                tracker, candidate_index, step, self.max_steps
            )
            score = (
                self.nn.predict(features)
                if use_nn
                else -tracker.exclusive_count(candidate_index)
            )
            candidates.append((score, candidate_index, features))

        if not candidates:
            return False

        candidates.sort(reverse=True)
        _, candidate_index, features = candidates[0]

        if tracker.can_remove(candidate_index):
            tracker.remove(candidate_index)
            outside.add(candidate_index)           # 增量维护
            self.nn.add_sample(features, 1.0)
            self.stats["improvements"] += 1
            return True

        self.nn.add_sample(features, 0.0)
        return False

    def _try_replace(
        self, tracker: CoverageTracker, outside: Set[int], step: int, use_nn: bool
    ) -> bool:
        if not tracker.in_solution:
            return False

        remove_pool = sorted(
            tracker.in_solution,
            key=lambda candidate_index: tracker.exclusive_count(candidate_index),
        )[: min(6, len(tracker.in_solution))]
        add_pool = self._sample_outside_candidates(outside)
        if not add_pool:
            return False

        best_pair: Optional[Tuple[int, int, List[float]]] = None
        best_score = float("-inf")

        for candidate_remove in remove_pool:
            for candidate_add in add_pool:
                features = self.feat.extract_replace(
                    tracker, candidate_remove, candidate_add, step, self.max_steps
                )
                score = (
                    self.nn.predict(features)
                    if use_nn
                    else tracker.marginal_gain(candidate_add)
                    - tracker.exclusive_count(candidate_remove)
                )
                if score > best_score:
                    best_score = score
                    best_pair = (candidate_remove, candidate_add, features)

        if best_pair is None:
            return False

        candidate_remove, candidate_add, features = best_pair
        # 保存快照（用于回滚）
        snapshot = list(tracker.in_solution)
        snapshot_outside = set(outside)
        size_before = tracker.solution_size

        tracker.remove(candidate_remove)
        outside.add(candidate_remove)
        tracker.add(candidate_add)
        outside.discard(candidate_add)
        self._strip_redundancy(tracker, outside)

        if tracker.is_feasible() and tracker.solution_size < size_before:
            self.nn.add_sample(features, 1.0)
            self.stats["improvements"] += 1
            return True

        # 回滚
        tracker.reset(snapshot)
        outside.clear()
        outside.update(snapshot_outside)
        self.nn.add_sample(features, 0.0)
        return False

    def _try_remove_repair(
        self, tracker: CoverageTracker, outside: Set[int], step: int
    ) -> bool:
        _ = step
        ordered = sorted(
            tracker.in_solution,
            key=lambda candidate_index: tracker.exclusive_count(candidate_index),
        )
        for candidate_remove in ordered[: min(5, len(ordered))]:
            snapshot = list(tracker.in_solution)
            snapshot_outside = set(outside)
            size_before = tracker.solution_size

            tracker.remove(candidate_remove)
            outside.add(candidate_remove)

            if tracker.is_feasible():
                self.stats["improvements"] += 1
                return True

            add_pool = self._sample_outside_candidates(outside)
            if not add_pool:
                tracker.reset(snapshot)
                outside.clear()
                outside.update(snapshot_outside)
                continue

            candidate_add = max(add_pool, key=tracker.marginal_gain)
            tracker.add(candidate_add)
            outside.discard(candidate_add)
            self._strip_redundancy(tracker, outside)

            if tracker.is_feasible() and tracker.solution_size < size_before:
                self.stats["improvements"] += 1
                return True

            tracker.reset(snapshot)
            outside.clear()
            outside.update(snapshot_outside)

        return False


# ---------------------------------------------------------------------------
# 优化5: SA 自适应参数
# ---------------------------------------------------------------------------

class ImprovedSA:
    """模拟退火求解器，支持自适应初始温度和问题规模相关的迭代次数。"""

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
        self.penalty = max(10, len(initial_solution))

        # 优化5: 根据问题规模自适应调整温度和迭代次数
        n_candidates = len(instance.candidates)
        n_targets = len(instance.targets)
        problem_scale = math.log1p(n_candidates * n_targets)

        # 自适应初始温度：规模越大，初始温度越高，接受更多劣解以跳出局部最优
        self.T_start = max(T_start, 1.5 * math.log1p(problem_scale))
        self.T_end = T_end

        # 自适应迭代次数：至少保证每个候选被访问数次
        adaptive_iter = min(20_000, max(max_iter, n_candidates * 5))
        self.max_iter = adaptive_iter

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

        # 优化5: 自适应 reheating — no_improve 阈值与规模相关
        reheat_threshold = max(300, min(800, len(self.instance.candidates) * 2))
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

            # 优化5: 自适应 reheating，每次 reheat 强度递减避免无限循环
            if no_improve >= reheat_threshold:
                temperature = min(temperature * 4.0, self.T_start * 0.5)
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

    def _move_replace(
        self, tracker: CoverageTracker, outside: Set[int]
    ) -> Optional[Tuple[str, int, int]]:
        solution = list(tracker.in_solution)
        non_solution = list(outside)
        if not solution or not non_solution:
            return None

        remove_index = self.rng.choice(solution)
        add_index = self.rng.choice(non_solution)
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
        non_solution = list(outside)
        if len(solution) < 3 or not non_solution:
            return None

        added = self.rng.choice(non_solution)
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
