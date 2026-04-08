"""
Optimal Samples Selection System v2
=====================================

1. 增量覆盖追踪 - O(1) 评估扰动可行性
2. 更强的特征工程 - 12维特征，更高区分度
3. 改进的局部搜索 - 冗余检测 + remove-repair 策略
4. 有效的模拟退火 - 可行性感知邻域 + 自适应温度
5. ILP 精确求解（小规模）- 验证和基准对比
6. 多次重启取最优
"""

import itertools
import random
import math
import time
import json
import os
from collections import defaultdict


# ============================================================
# 第一部分：增量覆盖追踪器
# ============================================================

class CoverageTracker:
    """
    增量维护每个目标子集的覆盖次数

    核心优势：添加/删除一个分组时，只需更新该分组覆盖的目标，
    不需要重新扫描整个解。判断可行性从 O(|targets|) 降到 O(1)。
    """

    def __init__(self, instance):
        self.instance = instance
        self.num_targets = len(instance.targets)
        self.cover_count = [0] * self.num_targets
        self.uncovered_count = self.num_targets
        self.in_solution = set()

    def reset(self, solution_indices):
        self.cover_count = [0] * self.num_targets
        self.uncovered_count = self.num_targets
        self.in_solution = set(solution_indices)
        for ci in solution_indices:
            for ti in self.instance.covers[ci]:
                if self.cover_count[ti] == 0:
                    self.uncovered_count -= 1
                self.cover_count[ti] += 1

    def add(self, ci):
        self.in_solution.add(ci)
        for ti in self.instance.covers[ci]:
            if self.cover_count[ti] == 0:
                self.uncovered_count -= 1
            self.cover_count[ti] += 1

    def remove(self, ci):
        self.in_solution.discard(ci)
        for ti in self.instance.covers[ci]:
            self.cover_count[ti] -= 1
            if self.cover_count[ti] == 0:
                self.uncovered_count += 1

    def is_feasible(self):
        return self.uncovered_count == 0

    def can_remove(self, ci):
        for ti in self.instance.covers[ci]:
            if self.cover_count[ti] == 1:
                return False
        return True

    def exclusive_count(self, ci):
        count = 0
        for ti in self.instance.covers[ci]:
            if self.cover_count[ti] == 1:
                count += 1
        return count

    def redundancy_score(self, ci):
        if not self.instance.covers[ci]:
            return 0
        total = sum(self.cover_count[ti] for ti in self.instance.covers[ci])
        return total / len(self.instance.covers[ci])

    def get_newly_uncovered(self, ci):
        return {ti for ti in self.instance.covers[ci] if self.cover_count[ti] == 1}

    @property
    def solution_size(self):
        return len(self.in_solution)


# ============================================================
# 第二部分：问题建模
# ============================================================

class CoverageInstance:
    def __init__(self, m, n, k, j, s, samples=None):
        self.m, self.n, self.k, self.j, self.s = m, n, k, j, s

        assert 45 <= m <= 54
        assert 7 <= n <= 25
        assert 4 <= k <= 7
        assert s <= j <= k
        assert 3 <= s <= 7

        if samples is None:
            self.samples = sorted(random.sample(range(1, m + 1), n))
        else:
            assert len(samples) == n
            self.samples = sorted(samples)

        self.targets = list(itertools.combinations(self.samples, j))
        self.candidates = list(itertools.combinations(self.samples, k))
        self.candidate_sets = [frozenset(c) for c in self.candidates]
        self.target_sets = [frozenset(t) for t in self.targets]

        self.covers = defaultdict(set)
        self.covered_by = defaultdict(set)
        for ci, cset in enumerate(self.candidate_sets):
            for ti, tset in enumerate(self.target_sets):
                if len(cset & tset) >= self.s:
                    self.covers[ci].add(ti)
                    self.covered_by[ti].add(ci)

        print(f"参数: m={m}, n={n}, k={k}, j={j}, s={s}")
        print(f"样本: {self.samples}")
        print(f"目标子集: {len(self.targets)}, 候选分组: {len(self.candidates)}")

    def get_overlap(self, ci, cj):
        return len(self.covers[ci] & self.covers[cj])


# ============================================================
# 第三部分：贪心算法（带平局打破 + 可随机化）
# ============================================================

class GreedySolver:
    def solve(self, instance, randomized=False):
        uncovered = set(range(len(instance.targets)))
        solution = []
        used = set()
        order = list(range(len(instance.candidates)))

        while uncovered:
            if randomized:
                random.shuffle(order)
            best_ci, best_count = -1, 0

            for ci in order:
                if ci in used:
                    continue
                count = len(instance.covers[ci] & uncovered)
                if count > best_count:
                    best_count = count
                    best_ci = ci

            if best_ci == -1 or best_count == 0:
                break

            solution.append(best_ci)
            used.add(best_ci)
            uncovered -= instance.covers[best_ci]

        return solution


# ============================================================
# 第四部分：冗余消除
# ============================================================

class RedundancyEliminator:
    def eliminate(self, instance, solution):
        tracker = CoverageTracker(instance)
        tracker.reset(solution)

        improved = True
        while improved:
            improved = False
            scored = [(ci, tracker.exclusive_count(ci)) for ci in list(tracker.in_solution)]
            scored.sort(key=lambda x: x[1])
            for ci, exc in scored:
                if exc == 0:
                    tracker.remove(ci)
                    improved = True

        return list(tracker.in_solution)


# ============================================================
# 第五部分：改进的神经网络
# ============================================================

class ImprovedNeuralNet:
    """
    改进：12维输入、更大隐藏层、Leaky ReLU、
    经验回放（平衡正负样本）、特征在线标准化
    """

    def __init__(self, input_dim=12, hidden1=32, hidden2=16):
        self.input_dim = input_dim
        self.W1 = self._xavier(input_dim, hidden1)
        self.b1 = [0.0] * hidden1
        self.W2 = self._xavier(hidden1, hidden2)
        self.b2 = [0.0] * hidden2
        self.W3 = self._xavier(hidden2, 1)
        self.b3 = [0.0]
        self.lr = 0.005

        self.positive_buffer = []
        self.negative_buffer = []
        self.buffer_max = 500
        self.train_count = 0
        self.batch_size = 20

        self.feat_sum = [0.0] * input_dim
        self.feat_sq_sum = [0.0] * input_dim
        self.feat_count = 0

    def _xavier(self, fi, fo):
        lim = math.sqrt(6.0 / (fi + fo))
        return [[random.uniform(-lim, lim) for _ in range(fo)] for _ in range(fi)]

    def _normalize(self, x):
        if self.feat_count < 10:
            return x
        r = []
        for i in range(self.input_dim):
            mean = self.feat_sum[i] / self.feat_count
            var = self.feat_sq_sum[i] / self.feat_count - mean * mean
            std = math.sqrt(max(var, 1e-8))
            r.append((x[i] - mean) / std)
        return r

    def _update_stats(self, x):
        for i in range(self.input_dim):
            self.feat_sum[i] += x[i]
            self.feat_sq_sum[i] += x[i] * x[i]
        self.feat_count += 1

    @staticmethod
    def _lrelu(x): return x if x > 0 else 0.01 * x
    @staticmethod
    def _lrelu_d(x): return 1.0 if x > 0 else 0.01
    @staticmethod
    def _sigmoid(x):
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, x_raw):
        x = self._normalize(x_raw)
        h1p = [self.b1[j] + sum(x[i]*self.W1[i][j] for i in range(self.input_dim))
               for j in range(len(self.b1))]
        h1 = [self._lrelu(v) for v in h1p]
        h2p = [self.b2[j] + sum(h1[i]*self.W2[i][j] for i in range(len(h1)))
               for j in range(len(self.b2))]
        h2 = [self._lrelu(v) for v in h2p]
        op = self.b3[0] + sum(h2[i]*self.W3[i][0] for i in range(len(h2)))
        out = self._sigmoid(op)
        return out, (x, h1p, h1, h2p, h2, op, out)

    def backward(self, cache, label):
        x, h1p, h1, h2p, h2, op, out = cache
        d = out - label
        for i in range(len(h2)):
            self.W3[i][0] -= self.lr * d * h2[i]
        self.b3[0] -= self.lr * d
        d2 = [d * self.W3[i][0] * self._lrelu_d(h2p[i]) for i in range(len(h2))]
        for i in range(len(h1)):
            for j in range(len(d2)):
                self.W2[i][j] -= self.lr * d2[j] * h1[i]
        for j in range(len(d2)):
            self.b2[j] -= self.lr * d2[j]
        d1 = [sum(d2[j]*self.W2[i][j] for j in range(len(d2))) * self._lrelu_d(h1p[i])
              for i in range(len(h1))]
        for i in range(self.input_dim):
            for j in range(len(d1)):
                self.W1[i][j] -= self.lr * d1[j] * x[i]
        for j in range(len(d1)):
            self.b1[j] -= self.lr * d1[j]

    def add_sample(self, features, label):
        self._update_stats(features)
        buf = self.positive_buffer if label > 0.5 else self.negative_buffer
        buf.append((list(features), label))
        if len(buf) > self.buffer_max:
            buf.pop(0)
        total = len(self.positive_buffer) + len(self.negative_buffer)
        if total >= self.batch_size:
            self._train_batch()

    def _train_batch(self):
        np = min(len(self.positive_buffer), self.batch_size // 2)
        nn = min(len(self.negative_buffer), self.batch_size - np)
        batch = []
        if np > 0: batch += random.sample(self.positive_buffer, np)
        if nn > 0: batch += random.sample(self.negative_buffer, nn)
        random.shuffle(batch)
        for feat, lab in batch:
            _, cache = self.forward(feat)
            self.backward(cache, lab)
        self.train_count += 1

    def predict(self, features):
        p, _ = self.forward(features)
        return p


# ============================================================
# 第六部分：12维特征提取
# ============================================================

class FeatureExtractor:
    def __init__(self, instance, greedy_size):
        self.inst = instance
        self.gs = greedy_size
        self.nt = len(instance.targets)

    def extract_remove(self, tracker, ci, step, total):
        exc = tracker.exclusive_count(ci)
        cov_size = len(self.inst.covers[ci])
        alt = 0
        for ti in self.inst.covers[ci]:
            alt += tracker.cover_count[ti] - 1
        alt /= max(1, cov_size)
        min_rem = min((tracker.cover_count[ti]-1 for ti in self.inst.covers[ci]),
                      default=0)
        max_olap = 0
        for oci in tracker.in_solution:
            if oci != ci:
                o = self.inst.get_overlap(ci, oci) / max(1, cov_size)
                if o > max_olap: max_olap = o

        return [
            exc / max(1, self.nt),
            1.0 if exc == 0 else 0.0,
            tracker.redundancy_score(ci) / max(1, tracker.solution_size),
            alt / max(1, tracker.solution_size),
            cov_size / max(1, self.nt),
            min_rem / 5.0,
            max_olap,
            tracker.solution_size / max(1, self.gs),
            step / max(1, total),
            0.0,
            tracker.uncovered_count / max(1, self.nt),
            cov_size / max(1, len(self.inst.candidates)),
        ]

    def extract_replace(self, tracker, ci_rm, ci_add, step, total):
        exc = tracker.exclusive_count(ci_rm)
        newly = tracker.get_newly_uncovered(ci_rm)
        recover = sum(1 for ti in newly if ti in self.inst.covers[ci_add])
        rr = recover / max(1, len(newly)) if newly else 1.0
        extra = sum(1 for ti in self.inst.covers[ci_add]
                    if tracker.cover_count[ti] <= 1 and ti not in self.inst.covers[ci_rm])
        cr = len(self.inst.covers[ci_rm])
        ca = len(self.inst.covers[ci_add])

        return [
            exc / max(1, self.nt),
            rr,
            extra / max(1, self.nt),
            ca / max(1, cr),
            ca / max(1, self.nt),
            tracker.redundancy_score(ci_rm) / max(1, tracker.solution_size),
            self.inst.get_overlap(ci_rm, ci_add) / max(1, cr),
            tracker.solution_size / max(1, self.gs),
            step / max(1, total),
            0.5,
            tracker.uncovered_count / max(1, self.nt),
            len(newly) / max(1, self.nt),
        ]


# ============================================================
# 第七部分：改进的局部搜索
# ============================================================

class ImprovedLocalSearch:
    def __init__(self, instance, greedy_sol, max_steps=3000, warmup=300):
        self.inst = instance
        self.initial = list(greedy_sol)
        self.max_steps = max_steps
        self.warmup = warmup
        self.nn = ImprovedNeuralNet(input_dim=12)
        self.feat = FeatureExtractor(instance, len(greedy_sol))
        self.stats = defaultdict(int)

    def solve(self):
        start = time.time()
        eliminator = RedundancyEliminator()
        clean = eliminator.eliminate(self.inst, self.initial)
        print(f"  冗余消除: {len(self.initial)} -> {len(clean)}")

        tracker = CoverageTracker(self.inst)
        tracker.reset(clean)
        best = list(tracker.in_solution)
        best_size = len(best)

        for step in range(self.max_steps):
            use_nn = (step >= self.warmup and self.nn.train_count >= 3
                      and random.random() > 0.15)
            improved = False

            if random.random() < 0.4:
                improved = self._try_remove(tracker, step, use_nn)
            if not improved and random.random() < 0.5:
                improved = self._try_replace(tracker, step, use_nn)
            if not improved and random.random() < 0.3:
                improved = self._try_remove_repair(tracker, step)

            if improved and tracker.solution_size < best_size:
                best = list(tracker.in_solution)
                best_size = len(best)
                m = "NN" if use_nn else "探索"
                print(f"  步骤 {step}: 新最优 {best_size} ({m})")

            if (step+1) % 1000 == 0:
                print(f"  进度 {step+1}/{self.max_steps}, 最优: {best_size}, "
                      f"NN训练: {self.nn.train_count}")

        tracker.reset(best)
        elapsed = time.time() - start
        print(f"  局部搜索: {len(self.initial)} -> {best_size}, "
              f"耗时 {elapsed:.2f}s, 改进 {self.stats['improvements']} 次")
        return best

    def _try_remove(self, tracker, step, use_nn):
        cands = [(ci, tracker.exclusive_count(ci),
                   self.feat.extract_remove(tracker, ci, step, self.max_steps))
                  for ci in list(tracker.in_solution)]
        if not cands: return False

        if use_nn:
            scored = sorted(cands, key=lambda x: -self.nn.predict(x[2]))
            ci, exc, feat = scored[0]
        else:
            cands.sort(key=lambda x: x[1])
            ci, exc, feat = cands[0]

        if tracker.can_remove(ci):
            tracker.remove(ci)
            self.nn.add_sample(feat, 1.0)
            self.stats['improvements'] += 1
            return True
        self.nn.add_sample(feat, 0.0)
        return False

    def _try_replace(self, tracker, step, use_nn):
        sol = list(tracker.in_solution)
        nis = [ci for ci in range(len(self.inst.candidates)) if ci not in tracker.in_solution]
        if not nis or not sol: return False

        sr = random.sample(sol, min(8, len(sol)))
        sa = random.sample(nis, min(20, len(nis)))
        best_pair, best_sc = None, -1

        for cr in sr:
            for ca in sa:
                feat = self.feat.extract_replace(tracker, cr, ca, step, self.max_steps)
                sc = self.nn.predict(feat) if use_nn else random.random()
                if sc > best_sc:
                    best_sc = sc
                    best_pair = (cr, ca, feat)

        if not best_pair: return False
        cr, ca, feat = best_pair

        tracker.remove(cr)
        tracker.add(ca)
        if tracker.is_feasible():
            for ci in list(tracker.in_solution):
                if tracker.can_remove(ci):
                    tracker.remove(ci)
                    self.nn.add_sample(feat, 1.0)
                    self.stats['improvements'] += 1
                    return True
        tracker.remove(ca)
        tracker.add(cr)
        self.nn.add_sample(feat, 0.0)
        return False

    def _try_remove_repair(self, tracker, step):
        scored = sorted(
            [(ci, tracker.exclusive_count(ci)) for ci in list(tracker.in_solution)],
            key=lambda x: x[1]
        )
        for ci, exc in scored[:5]:
            if exc == 0: continue
            newly = tracker.get_newly_uncovered(ci)
            if len(newly) > 10: continue
            nis = [idx for idx in range(len(self.inst.candidates))
                   if idx not in tracker.in_solution]
            for ca in nis:
                if newly.issubset(self.inst.covers[ca]):
                    tracker.remove(ci)
                    tracker.add(ca)
                    if tracker.is_feasible():
                        for ci2 in list(tracker.in_solution):
                            if tracker.can_remove(ci2):
                                tracker.remove(ci2)
                                self.stats['improvements'] += 1
                                return True
                    tracker.remove(ca)
                    tracker.add(ci)
                    break
        return False


# ============================================================
# 第八部分：改进的模拟退火
# ============================================================

class ImprovedSA:
    def __init__(self, instance, initial, T_start=5.0, T_end=0.001, max_iter=5000):
        self.inst = instance
        self.initial = initial
        self.T_start = T_start
        self.T_end = T_end
        self.max_iter = max_iter

    def _cost(self, tracker):
        return tracker.solution_size + tracker.uncovered_count * 10

    def solve(self):
        start = time.time()
        tracker = CoverageTracker(self.inst)
        tracker.reset(self.initial)
        best = list(tracker.in_solution)
        best_cost = self._cost(tracker)
        cur_cost = best_cost

        T = self.T_start
        cooling = (self.T_end / self.T_start) ** (1.0 / self.max_iter)
        nis = set(range(len(self.inst.candidates))) - tracker.in_solution
        no_improve = 0

        for it in range(self.max_iter):
            T *= cooling
            r = random.random()
            if r < 0.35:
                mv = self._mv_remove(tracker, nis)
            elif r < 0.7:
                mv = self._mv_replace(tracker, nis)
            elif r < 0.9:
                mv = self._mv_swap2(tracker, nis)
            else:
                mv = self._mv_add_rm2(tracker, nis)

            if mv is None: continue
            new_cost = self._cost(tracker)
            delta = new_cost - cur_cost

            if delta <= 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                cur_cost = new_cost
                no_improve = 0
                if tracker.is_feasible() and tracker.solution_size < len(best):
                    best = list(tracker.in_solution)
                    best_cost = self._cost(tracker)
                    print(f"  SA 迭代 {it}: 新最优 {len(best)} (T={T:.4f})")
            else:
                self._undo(tracker, mv, nis)
                no_improve += 1

            if no_improve > 500:
                T = min(T * 5, self.T_start)
                no_improve = 0

        elapsed = time.time() - start
        print(f"  模拟退火: {len(self.initial)} -> {len(best)}, 耗时 {elapsed:.2f}s")
        return best

    def _mv_remove(self, t, nis):
        sl = list(t.in_solution)
        if len(sl) <= 1: return None
        ci = random.choice(sl)
        t.remove(ci); nis.add(ci)
        return ('rm', ci)

    def _mv_replace(self, t, nis):
        sl, nl = list(t.in_solution), list(nis)
        if not sl or not nl: return None
        cr, ca = random.choice(sl), random.choice(nl)
        t.remove(cr); t.add(ca)
        nis.add(cr); nis.discard(ca)
        return ('rp', cr, ca)

    def _mv_swap2(self, t, nis):
        sl, nl = list(t.in_solution), list(nis)
        if len(sl) < 3 or not nl: return None
        rms = random.sample(sl, 2)
        ca = random.choice(nl)
        for ci in rms: t.remove(ci); nis.add(ci)
        t.add(ca); nis.discard(ca)
        return ('s2', rms, ca)

    def _mv_add_rm2(self, t, nis):
        sl, nl = list(t.in_solution), list(nis)
        if len(sl) < 3 or not nl: return None
        ca = random.choice(nl)
        t.add(ca); nis.discard(ca)
        scored = sorted([(ci, t.exclusive_count(ci)) for ci in list(t.in_solution)
                         if ci != ca], key=lambda x: x[1])
        rms = [ci for ci, _ in scored[:2]]
        for ci in rms: t.remove(ci); nis.add(ci)
        return ('ar2', rms, ca)

    def _undo(self, t, mv, nis):
        if mv[0] == 'rm':
            t.add(mv[1]); nis.discard(mv[1])
        elif mv[0] == 'rp':
            t.remove(mv[2]); t.add(mv[1])
            nis.discard(mv[1]); nis.add(mv[2])
        elif mv[0] == 's2':
            t.remove(mv[2]); nis.add(mv[2])
            for ci in mv[1]: t.add(ci); nis.discard(ci)
        elif mv[0] == 'ar2':
            for ci in mv[1]: t.add(ci); nis.discard(ci)
            t.remove(mv[2]); nis.add(mv[2])


# ============================================================
# 第九部分：ILP 精确求解
# ============================================================

class ILPSolver:
    @staticmethod
    def solve(instance):
        try:
            from scipy.optimize import milp, Bounds, LinearConstraint
            from scipy.sparse import lil_matrix
            import numpy as np
        except ImportError:
            print("  ILP: scipy 不可用，跳过")
            return None

        nc = len(instance.candidates)
        nt = len(instance.targets)
        c = np.ones(nc)
        A = lil_matrix((nt, nc), dtype=float)
        for ti in range(nt):
            for ci in instance.covered_by[ti]:
                A[ti, ci] = 1.0

        start = time.time()
        result = milp(c, constraints=LinearConstraint(A.tocsc(), lb=1.0),
                      bounds=Bounds(lb=0, ub=1),
                      integrality=np.ones(nc),
                      options={"time_limit": 60})
        elapsed = time.time() - start

        if result.success:
            sol = [i for i in range(nc) if result.x[i] > 0.5]
            print(f"  ILP 精确解: {len(sol)} 个分组, 耗时 {elapsed:.2f}s")
            return sol
        print(f"  ILP 求解失败")
        return None


# ============================================================
# 第十部分：主求解器
# ============================================================

class OptimalSamplesSolver:
    def __init__(self, m, n, k, j, s, samples=None):
        self.instance = CoverageInstance(m, n, k, j, s, samples)

    def solve(self, n_restarts=3, use_ilp=True):
        total_start = time.time()
        inst = self.instance

        ilp_size = None
        if use_ilp and len(inst.candidates) <= 5000:
            print("\n[ILP 精确求解]")
            ilp_sol = ILPSolver.solve(inst)
            if ilp_sol is not None:
                ilp_size = len(ilp_sol)

        best, best_size = None, float('inf')

        for r in range(n_restarts):
            print(f"\n===== 运行 {r+1}/{n_restarts} =====")

            print("[贪心]")
            g = GreedySolver()
            gs = g.solve(inst, randomized=(r > 0))
            print(f"  贪心: {len(gs)} 个分组")

            t = CoverageTracker(inst)
            t.reset(gs)
            assert t.is_feasible()

            print("[NN 局部搜索]")
            ls = ImprovedLocalSearch(inst, gs,
                max_steps=min(3000, len(inst.candidates) * 5),
                warmup=min(300, len(inst.candidates)))
            lss = ls.solve()

            print("[模拟退火]")
            sa = ImprovedSA(inst, lss,
                max_iter=min(5000, len(inst.candidates) * 3))
            sas = sa.solve()

            final = RedundancyEliminator().eliminate(inst, sas)
            t.reset(final)
            assert t.is_feasible()

            print(f"  本轮: {len(gs)} -> {len(lss)} -> {len(sas)} -> {len(final)}")
            if len(final) < best_size:
                best, best_size = final, len(final)

        elapsed = time.time() - total_start

        print("\n" + "=" * 60)
        print("最终结果")
        print("=" * 60)
        print(f"最优分组数: {best_size}")
        if ilp_size is not None:
            print(f"ILP 最优解: {ilp_size} (差距: {best_size - ilp_size})")
        print(f"总耗时: {elapsed:.2f}s")
        print(f"\n分组方案:")
        for i, ci in enumerate(sorted(best)):
            print(f"  分组 {i+1}: {inst.candidates[ci]}")

        return {
            'solution_indices': best,
            'groups': [inst.candidates[ci] for ci in sorted(best)],
            'num_groups': best_size,
            'ilp_optimal': ilp_size,
            'samples': inst.samples,
            'params': {'m': inst.m, 'n': inst.n, 'k': inst.k,
                       'j': inst.j, 's': inst.s},
            'time': elapsed,
        }


# ============================================================
# 数据库
# ============================================================

class ResultDatabase:
    def __init__(self, db_dir="results_db"):
        self.db_dir = db_dir
        os.makedirs(db_dir, exist_ok=True)

    def save(self, result):
        p = result['params']
        run = self._next_run(p)
        fn = f"{p['m']}-{p['n']}-{p['k']}-{p['j']}-{p['s']}-{run}-{result['num_groups']}.json"
        data = {**result, 'groups': [list(g) for g in result['groups']],
                'run_number': run, 'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
        with open(os.path.join(self.db_dir, fn), 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"已保存: {fn}")

    def _next_run(self, p):
        pfx = f"{p['m']}-{p['n']}-{p['k']}-{p['j']}-{p['s']}-"
        runs = []
        for f in os.listdir(self.db_dir):
            if f.startswith(pfx):
                parts = f.replace('.json','').split('-')
                if len(parts) >= 6:
                    try: runs.append(int(parts[5]))
                    except: pass
        return max(runs, default=0) + 1

    def list_all(self):
        files = sorted(f for f in os.listdir(self.db_dir) if f.endswith('.json'))
        for f in files:
            with open(os.path.join(self.db_dir, f)) as fp:
                d = json.load(fp)
            ilp = f", ILP={d.get('ilp_optimal','?')}" if d.get('ilp_optimal') else ""
            print(f"  {f} | 分组: {d['num_groups']}{ilp} | {d['timestamp']}")
        return files


# ============================================================
# 主程序
# ============================================================

def main():
    print("=" * 60)
    print("Optimal Samples Selection System v2")
    print("=" * 60)

    db = ResultDatabase("results_db_v2")

    print("\n>>> 示例 1: m=45, n=7, k=6, j=5, s=5")
    r1 = OptimalSamplesSolver(45, 7, 6, 5, 5).solve(n_restarts=1)
    db.save(r1)

    print("\n\n>>> 示例 2: m=45, n=9, k=6, j=5, s=4")
    r2 = OptimalSamplesSolver(45, 9, 6, 5, 4).solve(n_restarts=2)
    db.save(r2)

    print("\n\n>>> 示例 3: m=45, n=12, k=6, j=4, s=4")
    r3 = OptimalSamplesSolver(45, 12, 6, 4, 4).solve(n_restarts=3)
    db.save(r3)

    print("\n\n=== 数据库 ===")
    db.list_all()


if __name__ == '__main__':
    main()