"""
Test suite for optimal_samples_system.

涵盖：
- 正确性验证（与 ILP 精确解对比）
- 固定 seed 下结果复现（回归测试）
- 覆盖模式 / 聚合模式组合测试
- CoverageTracker 增量一致性测试
- 边界条件（最小参数、s==j、k==n）
"""

from __future__ import annotations

import random

import pytest

from optimal_samples_system import OptimalSamplesSolver
from optimal_samples_system.config import (
    AggregationMode,
    CoverageMode,
    ProblemConfig,
    SolverConfig,
)
from optimal_samples_system.instance import CoverageInstance
from optimal_samples_system.tracking import CoverageTracker


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------

def _make_solver_config(use_ilp: bool = True, restarts: int = 1) -> SolverConfig:
    """生成轻量级求解器配置，用于单元测试快速验证。"""
    return SolverConfig(
        n_restarts=restarts,
        use_ilp=use_ilp,
        max_local_steps=300,
        max_sa_iterations=300,
        use_neural_guidance=False,  # 测试时关闭 NN 以保证速度
    )


def _solve(m, n, k, j, s, seed=42, coverage_mode=CoverageMode.AT_LEAST_ONE,
           aggregation_mode=AggregationMode.DISTINCT_SUBSETS, required_r=None,
           use_ilp=True):
    config = ProblemConfig(
        m=m, n=n, k=k, j=j, s=s,
        coverage_mode=coverage_mode,
        aggregation_mode=aggregation_mode,
        required_r=required_r,
        seed=seed,
    )
    solver = OptimalSamplesSolver(config)
    return solver.solve(_make_solver_config(use_ilp=use_ilp))


# ---------------------------------------------------------------------------
# 1. 正确性验证：与 ILP 精确解对比
# ---------------------------------------------------------------------------

class TestCorrectnessVsILP:
    """验证启发式解与精确 ILP 解的差距（gap）不超过合理范围。"""

    def test_small_case_gap_zero(self):
        """案例1（m=45,n=7,k=6,j=5,s=5）：启发式应恰好达到最优解。"""
        result = _solve(m=45, n=7, k=6, j=5, s=5, seed=1000)
        assert result.exact_size is not None, "ILP 应该成功"
        assert result.num_groups == result.exact_size, (
            f"Gap 应为 0，但得到 heuristic={result.num_groups}, exact={result.exact_size}"
        )

    def test_medium_case_gap_small(self):
        """案例2（m=45,n=9,k=6,j=5,s=4）：gap 应 <= 1。"""
        result = _solve(m=45, n=9, k=6, j=5, s=4, seed=1001)
        assert result.exact_size is not None, "ILP 应该成功"
        gap = result.num_groups - result.exact_size
        assert gap <= 1, f"Gap 过大：heuristic={result.num_groups}, exact={result.exact_size}"

    def test_solution_is_feasible(self):
        """验证返回的解满足覆盖约束。"""
        result = _solve(m=45, n=8, k=5, j=4, s=3, seed=9999)
        config = ProblemConfig(m=45, n=8, k=5, j=4, s=3, seed=9999)
        instance = CoverageInstance(config)
        tracker = CoverageTracker(instance)
        tracker.reset(result.solution_indices)
        assert tracker.is_feasible(), "返回的解不满足可行性约束"


# ---------------------------------------------------------------------------
# 2. 回归测试：固定 seed 结果必须可复现
# ---------------------------------------------------------------------------

class TestReproducibility:
    """固定 seed 下，两次独立求解结果应完全相同。"""

    def test_same_seed_same_result(self):
        kwargs = dict(m=45, n=7, k=6, j=5, s=5, seed=12345, use_ilp=False)
        r1 = _solve(**kwargs)
        r2 = _solve(**kwargs)
        assert r1.num_groups == r2.num_groups, "相同 seed 下组数应相同"
        assert sorted(r1.solution_indices) == sorted(r2.solution_indices), (
            "相同 seed 下候选索引应完全一致"
        )

    def test_different_seeds_may_differ(self):
        """不同 seed 可能产生不同解（概率意义上，非强制断言）。"""
        r1 = _solve(m=45, n=9, k=6, j=5, s=4, seed=1, use_ilp=False)
        r2 = _solve(m=45, n=9, k=6, j=5, s=4, seed=999, use_ilp=False)
        # 两者都应可行
        assert r1.num_groups > 0
        assert r2.num_groups > 0


# ---------------------------------------------------------------------------
# 3. 覆盖模式组合测试
# ---------------------------------------------------------------------------

class TestCoverageModes:

    def test_at_least_one(self):
        result = _solve(m=30, n=7, k=5, j=4, s=3, seed=1,
                        coverage_mode=CoverageMode.AT_LEAST_ONE)
        assert result.num_groups >= 1

    def test_all_subsets(self):
        result = _solve(m=30, n=7, k=5, j=4, s=3, seed=1,
                        coverage_mode=CoverageMode.ALL_SUBSETS)
        assert result.num_groups >= 1

    def test_at_least_r(self):
        result = _solve(m=30, n=7, k=5, j=4, s=3, seed=1,
                        coverage_mode=CoverageMode.AT_LEAST_R,
                        required_r=2)
        assert result.num_groups >= 1

    def test_single_candidate_aggregation(self):
        result = _solve(m=20, n=6, k=5, j=4, s=3, seed=1,
                        aggregation_mode=AggregationMode.SINGLE_CANDIDATE)
        assert result.num_groups >= 1


# ---------------------------------------------------------------------------
# 4. CoverageTracker 增量一致性测试
# ---------------------------------------------------------------------------

class TestCoverageTracker:

    @pytest.fixture
    def small_instance(self):
        config = ProblemConfig(m=20, n=6, k=4, j=3, s=2, seed=42)
        return CoverageInstance(config)

    def test_add_remove_symmetric(self, small_instance):
        """add 后 remove 应恢复到初始状态。"""
        tracker = CoverageTracker(small_instance)
        init_unsatisfied = tracker.unsatisfied_targets
        init_deficit = tracker.deficit_units

        tracker.add(0)
        tracker.remove(0)

        assert tracker.unsatisfied_targets == init_unsatisfied
        assert tracker.deficit_units == init_deficit

    def test_feasibility_after_full_solution(self, small_instance):
        """加入所有候选后，tracker 必须可行。"""
        tracker = CoverageTracker(small_instance)
        for i in range(len(small_instance.candidates)):
            tracker.add(i)
        assert tracker.is_feasible()

    def test_reset_restores_state(self, small_instance):
        """reset 后状态应与初始一致。"""
        tracker = CoverageTracker(small_instance)
        # 先添加若干候选
        tracker.add(0)
        tracker.add(1)
        tracker.add(2)
        # reset 为空
        tracker.reset([])
        assert tracker.solution_size == 0
        assert tracker.unsatisfied_targets == len(small_instance.targets)

    def test_can_remove_consistency(self, small_instance):
        """can_remove 为 True 的候选移除后 tracker 依然可行。"""
        tracker = CoverageTracker(small_instance)
        # 加入所有候选
        for i in range(len(small_instance.candidates)):
            tracker.add(i)
        assert tracker.is_feasible()

        for candidate_index in list(tracker.in_solution):
            if tracker.can_remove(candidate_index):
                tracker.remove(candidate_index)
                assert tracker.is_feasible(), (
                    f"移除 {candidate_index} 后 tracker 变为不可行"
                )
                tracker.add(candidate_index)  # 恢复

    def test_marginal_gain_zero_in_solution(self, small_instance):
        """已在解中的候选 marginal_gain 应为 0。"""
        tracker = CoverageTracker(small_instance)
        tracker.add(0)
        assert tracker.marginal_gain(0) == 0


# ---------------------------------------------------------------------------
# 5. 边界条件测试
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_minimum_params(self):
        """最小合法参数（n=k=j=s=1）。"""
        result = _solve(m=5, n=1, k=1, j=1, s=1, seed=0, use_ilp=False)
        assert result.num_groups >= 1

    def test_s_equals_j(self):
        """s==j：每个目标只有 1 个 s-subset，等价于 at_least_one 强模式。"""
        result = _solve(m=20, n=6, k=5, j=4, s=4, seed=0)
        assert result.num_groups >= 1

    def test_k_equals_n(self):
        """k==n：只有 1 个候选（全集），解必须包含它。"""
        result = _solve(m=10, n=4, k=4, j=3, s=2, seed=0)
        assert result.num_groups >= 1

    def test_manual_samples(self):
        """手动指定 samples 时结果应正确。"""
        config = ProblemConfig(
            m=45, n=7, k=6, j=5, s=5,
            samples=(4, 10, 15, 22, 24, 30, 32),
        )
        solver = OptimalSamplesSolver(config)
        result = solver.solve(_make_solver_config())
        assert result.num_groups >= 1
        assert list(result.samples) == [4, 10, 15, 22, 24, 30, 32]

    def test_invalid_config_raises(self):
        """非法参数应抛出 ValueError。"""
        with pytest.raises(ValueError):
            ProblemConfig(m=10, n=5, k=6, j=3, s=2).validate()  # k > n

    def test_result_num_groups_matches_solution_length(self):
        """num_groups 应等于 solution_indices 的长度。"""
        result = _solve(m=30, n=7, k=5, j=4, s=3, seed=777)
        assert result.num_groups == len(result.solution_indices)


# ---------------------------------------------------------------------------
# 6. 神经网络（NumPy 化后）基本前向/反向测试
# ---------------------------------------------------------------------------

class TestNeuralNet:

    def test_predict_before_training_returns_half(self):
        from optimal_samples_system.heuristics import ImprovedNeuralNet
        nn = ImprovedNeuralNet(rng=random.Random(0))
        score = nn.predict([0.1] * 12)
        assert score == pytest.approx(0.5)

    def test_predict_after_training_in_range(self):
        from optimal_samples_system.heuristics import ImprovedNeuralNet
        nn = ImprovedNeuralNet(rng=random.Random(42))
        nn.batch_size = 4  # 调小 batch_size 以便快速触发训练
        rng = random.Random(1)
        # 喂入若干样本触发训练
        for _ in range(30):
            feats = [rng.random() for _ in range(12)]
            label = 1.0 if rng.random() > 0.5 else 0.0
            nn.add_sample(feats, label)
        # 训练后预测值应在 (0, 1) 范围内
        score = nn.predict([0.5] * 12)
        assert 0.0 <= score <= 1.0

    def test_forward_output_shape(self):
        """前向传播返回标量 float，cache 形状正确。"""
        import numpy as np
        from optimal_samples_system.heuristics import ImprovedNeuralNet
        nn = ImprovedNeuralNet(rng=random.Random(0))
        out, cache = nn.forward(np.array([0.0] * 12))
        assert isinstance(out, float)
        assert len(cache) == 7  # (features, h1_pre, h1, h2_pre, h2, out_pre, out)
