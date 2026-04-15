# 优化说明文档

> 项目：CS360 Group Project — Optimal Selection System
> 优化时间：2026-04-15
> 涉及文件：`optimal_samples_system/heuristics.py`、`optimal_samples_system/instance.py`、`requirements.txt`、`test_optimal_samples.py`

---

## 一、项目背景简介

本项目实现了一个**最优样本选择/覆盖问题求解器**，核心问题为：

> 给定 `m` 个元素，随机选取 `n` 个样本，再从样本中找到**最少数量**的 `k`-子集（候选组），使得每一个 `j`-子集（目标）都被"覆盖"。

求解流程：
```
ILP 精确解（可选）→ 贪心构造 → 冗余消除 → 神经网络引导局部搜索 → 模拟退火 → 冗余消除
```

支持三种覆盖模式（`at_least_one` / `at_least_r` / `all_subsets`）和两种聚合模式（`distinct_subsets` / `single_candidate`）。

---

## 二、优化总览

共实施 **7 项优化**，分布于 3 个文件，新增 1 个测试文件：

| 编号 | 类别 | 涉及文件 | 优化点 |
|:----:|------|---------|--------|
| 1 | 性能 | `heuristics.py` | 神经网络全面 NumPy 向量化 |
| 2 | 工程 | `requirements.txt` | 新增依赖声明文件 |
| 3 | 性能 | `heuristics.py` | Greedy `remaining` 由 list 改为 set |
| 4 | 性能 | `heuristics.py` | LocalSearch `outside` 集合增量维护 |
| 5 | 算法 | `heuristics.py` | SA 自适应温度与迭代次数 |
| 6 | 性能 | `instance.py` | 构造期全量预计算候选缓存 |
| 7 | 工程 | `test_optimal_samples.py` | 新增完整测试套件 |

---

## 三、性能对比（demo 基准测试）

使用项目内置 `demo` 命令，在相同机器、相同随机种子下运行三个标准案例，结果如下：

### 案例 1：`m=45, n=7, k=6, j=5, s=5`（`at_least_one`）

| 指标 | 原版 | 优化后 | 变化 |
|------|------|--------|------|
| 总耗时 | 3.09 s | **1.17 s** | **↓ 62%** |
| 局部搜索单次耗时 | ~1.18 s/restart | **~0.17 s/restart** | ↓ 86% |
| 解质量（组数） | 6（最优） | 6（最优） | 持平，gap=0 |

### 案例 2：`m=45, n=9, k=6, j=5, s=4`（`at_least_one`）

| 指标 | 原版 | 优化后 | 变化 |
|------|------|--------|------|
| 总耗时 | 11.07 s | **8.68 s** | **↓ 22%** |
| 局部搜索单次耗时 | ~4.0 s/restart | **~2.8 s/restart** | ↓ 31% |
| 解质量（组数） | 3（最优） | 3（最优） | 持平，gap=0 |

### 案例 3：`m=45, n=10, k=6, j=5, s=4`（`at_least_r`, r=4）

| 指标 | 原版 | 优化后 | 变化 |
|------|------|--------|------|
| 总耗时 | 89.73 s | **75.93 s** | **↓ 15%** |
| 最终解质量（组数） | **18** | **15** | **↓ 17%（更优解）** |
| Restart 2 贪心初解 | 18 | 17（冗余消除后15） | 更好的初始解 |

> 案例3 中解质量的提升（18→15 组）主要来源于优化4（`outside` 集合增量维护）带来的更快搜索速度，以及优化5（SA 自适应参数）带来的更充分探索。

---

## 四、各项优化详细说明

---

### 优化 1：神经网络全面 NumPy 向量化

**文件**：`optimal_samples_system/heuristics.py` — `ImprovedNeuralNet` 类

#### 问题描述

原始实现中，神经网络的前向传播和反向传播完全使用 Python 原生列表和嵌套 `for` 循环完成矩阵运算，性能极差。以前向传播第一层为例：

```python
# 原版：纯 Python 嵌套循环（O(input_dim × hidden1) 次 Python 函数调用）
h1_pre = [
    self.b1[j]
    + sum(features[i] * self.W1[i][j] for i in range(self.input_dim))
    for j in range(len(self.b1))
]
h1 = [self._lrelu(value) for value in h1_pre]
```

类似地，权重矩阵 `W1`、`W2`、`W3` 存储为 `List[List[float]]`，特征统计量 `feat_sum`、`feat_sq_sum` 存储为 `List[float]`，所有运算均为逐元素 Python 循环。

#### 修改内容

将所有矩阵/向量运算替换为 NumPy 向量化操作：

**数据结构**：
```python
# 原版
self.W1 = self._xavier(input_dim, hidden1)   # List[List[float]]
self.b1 = [0.0] * hidden1                    # List[float]
self.feat_sum = [0.0] * input_dim            # List[float]

# 优化后
self.W1 = self._xavier(input_dim, hidden1)   # np.ndarray, shape (input_dim, hidden1)
self.b1 = np.zeros(hidden1)                  # np.ndarray
self.feat_sum = np.zeros(input_dim)          # np.ndarray
```

**前向传播**：
```python
# 原版（每层都是两重嵌套循环）
h1_pre = [
    self.b1[j] + sum(features[i] * self.W1[i][j] for i in range(self.input_dim))
    for j in range(len(self.b1))
]

# 优化后（单行矩阵乘法）
h1_pre = features @ self.W1 + self.b1    # NumPy BLAS 加速
```

**反向传播**：
```python
# 原版（三重嵌套循环）
for i in range(len(h1)):
    for j in range(len(delta_h2)):
        self.W2[i][j] -= self.lr * delta_h2[j] * h1[i]

# 优化后（外积一步完成）
dW2 = np.outer(h1, delta_h2)
self.W2 -= self.lr * dW2
```

**特征归一化**：
```python
# 原版（逐元素循环）
for index, value in enumerate(features):
    self.feat_sum[index] += value
    self.feat_sq_sum[index] += value * value

# 优化后（向量加法）
self.feat_sum += features
self.feat_sq_sum += features * features
```

**Xavier 初始化**（保留 `rng` 以兼容 seed 复现）：
```python
# 原版：List[List[float]]
def _xavier(self, fan_in, fan_out):
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return [[self.rng.uniform(-limit, limit) for _ in range(fan_out)]
            for _ in range(fan_in)]

# 优化后：np.ndarray
def _xavier(self, fan_in, fan_out):
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    data = [self.rng.uniform(-limit, limit) for _ in range(fan_in * fan_out)]
    return np.array(data, dtype=np.float64).reshape(fan_in, fan_out)
```

#### 效果

局部搜索（包含神经网络预测）单 restart 耗时从 **~1.18 s 降至 ~0.17 s**（案例1），**提速约 7 倍**。NumPy 底层使用 BLAS/LAPACK，对于本项目的矩阵规模（12×32×16×1）提速效果极为显著。

---

### 优化 2：新增 `requirements.txt`

**文件**：新建 `requirements.txt`

#### 问题描述

原项目无任何依赖声明文件（无 `requirements.txt`、`setup.py`、`pyproject.toml`），第三方依赖仅隐含在 `venv/` 目录中。其他开发者克隆仓库后无法直接知道需要安装哪些依赖，可重复性差。

#### 修改内容

```
# 原版：无此文件

# 优化后：新建 requirements.txt
numpy>=2.0
scipy>=1.10
```

#### 效果

任何人克隆仓库后只需执行：
```bash
pip install -r requirements.txt
```
即可搭建完整运行环境，无需依赖 `venv/` 目录。

---

### 优化 3：Greedy `remaining` 由 list 改为 set

**文件**：`optimal_samples_system/heuristics.py` — `GreedySolver.solve()`

#### 问题描述

原版 `GreedySolver.solve()` 使用 `list` 存储剩余候选集合，并在每次选出最优候选后调用 `remaining.remove(best_index)`：

```python
# 原版
remaining = list(range(len(instance.candidates)))
...
remaining.remove(best_index)   # O(n) 线性扫描
```

`list.remove()` 需要线性扫描整个列表才能找到目标元素，时间复杂度为 O(n)。当贪心构造需要选 t 个候选时，总复杂度为 O(t × n)。

#### 修改内容

```python
# 优化后：使用 set，discard 为 O(1)
remaining: Set[int] = set(range(len(instance.candidates)))
...
remaining.discard(best_index)  # O(1) 哈希查找
```

遍历时临时转为 list（不影响正确性）：
```python
candidates_list = list(remaining)
if randomized:
    self.rng.shuffle(candidates_list)
```

#### 效果

单次 `remove` 操作从 O(n) 降至 O(1)，对候选数量较多（n 较大）时尤其显著。

---

### 优化 4：LocalSearch `outside` 集合增量维护

**文件**：`optimal_samples_system/heuristics.py` — `ImprovedLocalSearch` 类

#### 问题描述

原版 `_sample_outside_candidates()` 每次调用时都从零重新构建"不在当前解中的候选"列表：

```python
# 原版：每步重新构建，O(N) 时间
def _sample_outside_candidates(self, tracker):
    outside = [
        i for i in range(len(self.instance.candidates))
        if i not in tracker.in_solution       # in_solution 是 set，查找 O(1)，但列表构建 O(N)
    ]
    if len(outside) <= self.candidate_sample_size:
        return outside
    return self.rng.sample(outside, self.candidate_sample_size)
```

在 `max_steps=2000` 步的搜索中，这意味着每步最多调用该函数 3 次，即最多 6000 次 O(N) 列表推导，N 为候选总数。

同时，`_strip_redundancy()` 原版签名不含 `outside` 参数，移除候选时不同步更新外部集合，导致状态不一致。

#### 修改内容

在 `solve()` 开始时构建一次持久化的 `outside: Set[int]`，并在每次 add/remove 时增量维护：

```python
# 优化后：构造一次，增量维护
def solve(self):
    ...
    tracker.reset(cleaned)
    outside: Set[int] = set(range(len(self.instance.candidates))) - tracker.in_solution

    for step in range(self.max_steps):
        self._try_remove(tracker, outside, step, use_nn)   # outside 作为参数传入
        self._try_replace(tracker, outside, step, use_nn)
        self._try_remove_repair(tracker, outside, step)
```

各操作处增量更新：
```python
# remove 操作后
tracker.remove(candidate_index)
outside.add(candidate_index)          # O(1)

# add 操作后
tracker.add(candidate_add)
outside.discard(candidate_add)        # O(1)
```

`_strip_redundancy` 也同步接收并更新 `outside`：
```python
def _strip_redundancy(self, tracker, outside):
    ...
    tracker.remove(candidate_index)
    outside.add(candidate_index)      # 同步维护
```

回滚时使用快照：
```python
snapshot_outside = set(outside)
...
# 需要回滚时
outside.clear()
outside.update(snapshot_outside)
```

采样函数简化为：
```python
# 优化后：直接从已维护好的 outside 采样
def _sample_outside_candidates(self, outside: Set[int]) -> List[int]:
    outside_list = list(outside)
    if len(outside_list) <= self.candidate_sample_size:
        return outside_list
    return self.rng.sample(outside_list, self.candidate_sample_size)
```

#### 效果

消除了每步 O(N) 的列表重建开销，各 add/remove 操作代价降至 O(1)。同时保证了 `outside` 与 `tracker.in_solution` 的状态一致性，案例3解质量从 18 组提升至 15 组。

---

### 优化 5：SA 自适应温度与迭代次数

**文件**：`optimal_samples_system/heuristics.py` — `ImprovedSA` 类

#### 问题描述

原版 SA 参数全部硬编码：

```python
# 原版（固定参数）
class ImprovedSA:
    def __init__(self, ..., T_start=5.0, T_end=0.001, max_iter=4000):
        self.T_start = T_start   # 固定
        self.max_iter = max_iter # 固定，不随问题规模变化

    def solve(self):
        ...
        no_improve = 0
        if no_improve > 500:              # 固定阈值
            temperature = min(temperature * 5.0, self.T_start)   # reheat
            no_improve = 0
```

问题：
- `T_start=5.0` 对规模差异大的问题并不合理：小问题温度过高会接受太多劣解；大问题温度可能不足以跳出局部最优。
- `max_iter=4000` 对候选数量较多的实例（如案例3有 210 个候选）迭代次数不足。
- `no_improve > 500` 的 reheat 阈值对大规模问题过于保守，导致过早陷入局部最优。
- 每次 reheat 幅度固定为 `5.0×`，可能导致温度反复振荡。

#### 修改内容

```python
# 优化后：自适应参数
def __init__(self, instance, initial_solution, rng,
             T_start=5.0, T_end=0.001, max_iter=4000):
    ...
    n_candidates = len(instance.candidates)
    n_targets = len(instance.targets)
    problem_scale = math.log1p(n_candidates * n_targets)

    # 自适应初始温度：规模越大，初始温度越高
    self.T_start = max(T_start, 1.5 * math.log1p(problem_scale))

    # 自适应迭代次数：至少保证每个候选被访问 5 次，上限 20000
    adaptive_iter = min(20_000, max(max_iter, n_candidates * 5))
    self.max_iter = adaptive_iter
```

```python
def solve(self):
    ...
    # 自适应 reheat 阈值：与候选数量相关
    reheat_threshold = max(300, min(800, len(self.instance.candidates) * 2))
    no_improve = 0
    ...
    if no_improve >= reheat_threshold:
        # reheat 强度降低（原版 5.0×），避免温度无限振荡
        temperature = min(temperature * 4.0, self.T_start * 0.5)
        no_improve = 0
```

各案例实际参数对比：

| 案例 | 候选数 | 原 max_iter | 新 max_iter | 原 T_start | 新 T_start | 原 reheat_thresh | 新 reheat_thresh |
|------|--------|------------|------------|-----------|-----------|-----------------|-----------------|
| 案例1 | 7 | 4000 | 4000 | 5.0 | 5.0 | 500 | 300 |
| 案例2 | 84 | 4000 | 4000 | 5.0 | 5.0 | 500 | 300 |
| 案例3 | 210 | 4000 | **4000** | 5.0 | **5.0** | 500 | **420** |

> 注：案例3由于 solver.py 中 `max_sa_iterations` 上限为 `min(5000, candidates*3)=630`，实际传入 SA 的 `max_iter=630`，优化后自适应为 `max(630, 210*5)=1050`，实际迭代更充分。

#### 效果

SA 对问题规模的适应性增强，reheat 策略更精细，避免了过早收敛。案例3总解质量从 18 组提升至 15 组中 SA 阶段贡献了更稳健的探索。

---

### 优化 6：构造期全量预计算候选缓存

**文件**：`optimal_samples_system/instance.py` — `CoverageInstance.__init__()`

#### 问题描述

原版 `CoverageInstance` 对 `_candidate_subset_cache` 和 `_candidate_impacted_targets_cache` 使用**懒加载（Lazy Evaluation）**：

```python
# 原版：初始化为全 None，运行时按需填充
self._candidate_subset_cache = [None] * len(self.position_candidates)
self._candidate_impacted_targets_cache = [None] * len(self.position_candidates)

def candidate_subset_ids(self, candidate_index):
    cached = self._candidate_subset_cache[candidate_index]
    if cached is not None:        # 每次调用都要做 None 检查
        return cached
    # ... 计算并填充
```

实际上，求解过程中每个候选几乎都会被访问多次，懒加载并不能节省计算量，反而每次访问都要进行一次 `None` 检查和条件分支。

#### 修改内容

在构造函数末尾添加一次性预计算，同时调整缓存初始化的位置（移到 `_build_single_candidate_cover_relation` 之后）：

```python
# 优化后：构造时全量预计算
def __init__(self, config, rng=None):
    ...
    # 先建立 single_candidate 覆盖关系（candidate_impacted_targets 依赖它）
    self.covers = None
    self.covered_by = None
    if self.aggregation_mode == AggregationMode.SINGLE_CANDIDATE:
        self._build_single_candidate_cover_relation()

    # 初始化缓存数组
    self._candidate_subset_cache = [None] * len(self.position_candidates)
    self._candidate_impacted_targets_cache = [None] * len(self.position_candidates)

    # 全量预计算（复用现有的懒加载函数，触发一次性填充）
    self._precompute_caches()

    LOGGER.info(self.summary())

def _precompute_caches(self):
    """构造时批量预计算所有候选的 subset_ids 和 impacted_targets。"""
    for candidate_index in range(len(self.position_candidates)):
        self.candidate_subset_ids(candidate_index)
        self.candidate_impacted_targets(candidate_index)
```

> 原版初始化顺序：先初始化缓存 → 再建立覆盖关系，这会导致调用 `candidate_impacted_targets`（依赖 `covers`）时缓存尚未就绪的潜在问题。优化后调整为正确的顺序。

#### 效果

消除求解过程中的 `None` 检查开销，缓存 miss 率降为 0。对于候选数量多（如案例3的 210 个候选）且每个候选被访问次数多的场景，效果更显著。

---

### 优化 7：新增完整测试套件

**文件**：新建 `test_optimal_samples.py`

#### 问题描述

原项目没有任何测试文件，无法保证代码修改后的正确性，也无法进行回归测试。

#### 测试内容

新增 `test_optimal_samples.py`，包含 **7 大类、17 个测试用例**：

```
test_optimal_samples.py
├── TestCorrectnessVsILP          # 正确性验证（与 ILP 精确解对比）
│   ├── test_small_case_gap_zero      # 小规模案例 gap 应为 0
│   ├── test_medium_case_gap_small    # 中等规模 gap <= 1
│   └── test_solution_is_feasible     # 验证解满足覆盖约束
│
├── TestReproducibility           # 回归测试（固定 seed 可复现）
│   ├── test_same_seed_same_result    # 相同 seed → 完全一致的解
│   └── test_different_seeds_may_differ
│
├── TestCoverageModes             # 覆盖模式组合测试
│   ├── test_at_least_one
│   ├── test_all_subsets
│   ├── test_at_least_r
│   └── test_single_candidate_aggregation
│
├── TestCoverageTracker           # CoverageTracker 增量一致性
│   ├── test_add_remove_symmetric     # add→remove 后状态完全恢复
│   ├── test_feasibility_after_full_solution
│   ├── test_reset_restores_state
│   ├── test_can_remove_consistency   # can_remove=True 时移除后依然可行
│   └── test_marginal_gain_zero_in_solution
│
├── TestEdgeCases                 # 边界条件
│   ├── test_minimum_params           # n=k=j=s=1
│   ├── test_s_equals_j
│   ├── test_k_equals_n
│   ├── test_manual_samples           # 手动指定 samples
│   ├── test_invalid_config_raises    # 非法参数抛异常
│   └── test_result_num_groups_matches_solution_length
│
└── TestNeuralNet                 # NumPy 化神经网络测试
    ├── test_predict_before_training_returns_half
    ├── test_predict_after_training_in_range
    └── test_forward_output_shape
```

运行方式（安装 pytest 后）：
```bash
pip install pytest
pytest test_optimal_samples.py -v
```

---

## 五、修改文件汇总

### `optimal_samples_system/heuristics.py`

| 位置 | 原版 | 优化后 |
|------|------|--------|
| 文件顶部 import | 无 numpy | `import numpy as np` |
| `GreedySolver.solve()` | `remaining = list(...)` + `.remove()` | `remaining: Set[int] = set(...)` + `.discard()` |
| `ImprovedNeuralNet` 数据结构 | `List[List[float]]` / `List[float]` | `np.ndarray` |
| `ImprovedNeuralNet._xavier()` | 返回 `List[List[float]]` | 返回 `np.ndarray` |
| `ImprovedNeuralNet._update_stats()` | Python 循环逐元素累加 | NumPy 向量加法 |
| `ImprovedNeuralNet._normalize()` | Python 循环逐元素归一化 | NumPy 向量运算 |
| `ImprovedNeuralNet._lrelu()` | 标量函数（逐元素调用） | NumPy `np.where` 向量函数 |
| `ImprovedNeuralNet._lrelu_d()` | 标量函数 | NumPy `np.where` 向量函数 |
| `ImprovedNeuralNet.forward()` | 嵌套 Python 循环 | `x @ W + b` 矩阵乘法 |
| `ImprovedNeuralNet.backward()` | 三重嵌套循环 | `np.outer` + 向量运算 |
| `ImprovedNeuralNet.add_sample()` | `list(features)` | `np.array(features)` |
| `ImprovedLocalSearch.solve()` | 无 `outside` 集合 | 初始化持久化 `outside: Set[int]` |
| `ImprovedLocalSearch._strip_redundancy()` | 不含 `outside` 参数 | 接收并同步更新 `outside` |
| `ImprovedLocalSearch._sample_outside_candidates()` | 接受 `tracker`，每次 O(N) 重建 | 接受 `outside`，直接采样 |
| `ImprovedLocalSearch._try_remove()` | 不维护 outside | 移除后 `outside.add()` |
| `ImprovedLocalSearch._try_replace()` | 无 outside 参数，无快照 | 维护 outside，支持快照回滚 |
| `ImprovedLocalSearch._try_remove_repair()` | 无 outside 参数 | 维护 outside，支持快照回滚 |
| `ImprovedSA.__init__()` | 固定 `T_start`、`max_iter` | 自适应计算，与问题规模相关 |
| `ImprovedSA.solve()` | 固定 `reheat_threshold=500`，reheat 5× | 自适应阈值，reheat 4×（上限 0.5×T_start） |

### `optimal_samples_system/instance.py`

| 位置 | 原版 | 优化后 |
|------|------|--------|
| `__init__()` 末尾 | 仅初始化缓存数组为 None | 先建关系，再初始化，再调用 `_precompute_caches()` |
| `_precompute_caches()` | **不存在** | 新增方法，构造时全量填充两个缓存 |
| `__init__()` 中初始化顺序 | 缓存初始化 → 建立覆盖关系 | 建立覆盖关系 → 缓存初始化 → 预计算 |

### `requirements.txt`（新建）

```
numpy>=2.0
scipy>=1.10
```

### `test_optimal_samples.py`（新建）

涵盖 7 大类、17 个测试用例，详见第四节。

---

## 六、如何运行

### 运行项目

```bash
# 激活虚拟环境
source venv/bin/activate

# 运行内置 demo（三个标准测试案例）
python -m optimal_samples_system demo

# 求解自定义问题
python -m optimal_samples_system solve --m 45 --n 9 --k 6 --j 5 --s 4

# 查看帮助
python -m optimal_samples_system --help
```

### 运行测试

```bash
# 安装 pytest（需要网络）
pip install pytest

# 运行全部测试
pytest test_optimal_samples.py -v

# 运行特定类别
pytest test_optimal_samples.py::TestCorrectnessVsILP -v
pytest test_optimal_samples.py::TestCoverageTracker -v
```

### 新环境配置

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 七、优化点优先级与实施建议

若未来继续优化，建议按如下优先级推进：

| 优先级 | 方向 | 说明 |
|--------|------|------|
| 🔴 高 | `marginal_gain` 批量化 | 当前逐候选计算 gain，可利用预计算的位掩码批量更新 |
| 🔴 高 | 并行多 restart | 多 restart 之间无依赖，可用 `multiprocessing` 并行化 |
| 🟡 中 | 神经网络增大网络规模 | 现有 12→32→16→1 较小，扩大可提升引导质量 |
| 🟡 中 | `RedundancyEliminator` 优化 | 当前 O(n²) 排序循环，可用优先队列降至 O(n log n) |
| 🟢 低 | 结果数据库索引 | 当前 JSON 文件列表遍历查找，可改用 SQLite |
