# 2026-04-23 noNN 版本调整总结

本文档总结今天对 `CS360-Group-Project-for-Optimal-Selection-System-noNN` 的所有主要调整、验证结果和使用方式。

## 1. 项目主线切换到 noNN

今天明确将 `CS360-Group-Project-for-Optimal-Selection-System-noNN` 作为后续主线版本。

主要原因：

- 之前消融测试显示，在线 NN 引导没有带来稳定收益，反而显著增加运行时间。
- noNN 版本在 `n=15,k=6,j=5,s=4` benchmark 上曾得到更好的解质量和更短时间。
- 后续优化集中在组合优化本身，包括候选过滤、强邻域、MIP exact verification，而不是继续扩展在线神经网络。

当前 noNN 版本中：

- 没有 `ImprovedNeuralNet`。
- 没有在线训练、反向传播、NN batch、NN prediction。
- `--disable-neural-guidance` 仅保留为兼容旧命令的 no-op 参数。

## 2. 验证器功能已整合

新增独立验证器模块：

```text
optimal_samples_system/validation.py
```

新增 CLI 命令：

```bash
python -m optimal_samples_system validate-result <filename> --db-dir results_db_v3
python -m optimal_samples_system validate-file <path>
python -m optimal_samples_system validate-results --db-dir results_db_v3
```

验证器会检查：

- 每个 group 是否是合法的 `k` 元候选。
- group 内是否有重复样本。
- group 是否包含样本全集之外的元素。
- 是否存在重复 group。
- `groups` 和 `solution_indices` 是否一致。
- `num_groups` 元数据是否和实际解析数量一致。
- 所有 target 是否满足覆盖约束。
- 若不可行，会输出未满足 target 数量、deficit units 和若干 uncovered examples。

验证器的作用是证明：

```text
当前输出解是一个正确可行解。
```

它不证明最优性；最优性或近优性由 `prove-bound` 辅助验证。

## 3. 新增 prove-bound 精确验证功能

新增命令：

```bash
python -m optimal_samples_system prove-bound ...
```

功能含义：

```text
判断是否存在一个可行解，其 group 数量 <= target-size。
```

例如当前有 297 组可行解，要证明它是全局最优，需要证明不存在 296 组或更少的解：

```bash
python -m optimal_samples_system prove-bound --m 45 --n 15 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --target-size 296 \
  --exact-backend auto \
  --exact-time-limit 600
```

结果解释：

- `Status: infeasible`：不存在 `<= target-size` 的解。若已有 `target-size + 1` 的可行解，则该解已证明全局最优。
- `Status: feasible`：存在更小或相同大小的解，当前解不是最优。
- `Status: unknown`：给定时间内没有证明出来，只能说当前解是 best-known feasible solution。

这个功能用于证明：

```text
当前解是否最优，或它距离可证明下界有多远。
```

## 4. Exact Solver 后端增强

`exact.py` 已从单一 SciPy/HiGHS 后端扩展为多后端结构：

```text
auto
scipy
gurobi
scip
```

对应 CLI 参数：

```bash
--exact-backend auto|scipy|gurobi|scip
--exact-time-limit <seconds>
--force-exact
```

默认行为：

- 小规模实例会自动尝试 exact verification。
- 大规模实例默认跳过 exact verification，避免程序卡住。
- 如果要强行验证大规模实例，可以加 `--force-exact`。

示例：

```bash
python -m optimal_samples_system solve --m 45 --n 15 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --seed 42 --restarts 5 \
  --force-exact \
  --exact-backend auto \
  --exact-time-limit 600
```

## 5. 已安装 Gurobi 和 SCIP Python 接口

今天已经在当前 Python/Anaconda 环境中安装：

```text
gurobipy 13.0.1
pyscipopt 6.1.0
```

验证结果：

```text
gurobipy (13, 0, 1)
pyscipopt import OK
```

小规模 proof 测试通过：

```text
--exact-backend scip
Status: infeasible
Method: scip

--exact-backend gurobi
Status: infeasible
Method: gurobi

--exact-backend auto
Status: infeasible
Method: gurobi
```

Gurobi 当前 license 状态：

```text
Restricted license - for non-production use only - expires 2027-11-29
```

这说明 Gurobi 可以运行，但大规模 exact proof 可能仍受 license/model size 限制。如果遇到 size limit，需要申请 academic license。

## 6. 新增强邻域：add-then-remove-many

新增 Local Search 和 SA 中的强邻域：

```text
add-then-remove-many
```

思想：

1. 先挑选若干低贡献 group 作为 pivot。
2. 从与这些 pivot 结构相关的候选池中选择 1 到 3 个 group 加入当前解。
3. 加入后执行 redundancy stripping，尽量删除多个旧 group。
4. 只有当新解仍然可行且 group 数更小时才接受。
5. 否则恢复 snapshot，完全回滚。

正确性保证：

- 不改变覆盖语义。
- 不放松约束。
- 每次接受前都检查 `tracker.is_feasible()`。
- 失败时使用 snapshot 回滚。

该邻域的目标不是单步替换，而是跳出普通 replace 难以突破的局部最优。

## 7. 新增强邻域：destroy-repair

新增 Local Search 和 SA 中的强邻域：

```text
destroy-repair
```

思想：

1. 从当前解中选择一批低贡献或冗余度较高的 group。
2. 暂时删除它们，制造一个不可行或更紧的中间状态。
3. 使用结构相关候选池进行 greedy repair。
4. repair 后执行 redundancy stripping。
5. 只有恢复可行并且 group 数变少时才接受。
6. 否则恢复 snapshot。

正确性保证同上：

- 所有 accepted solution 必须可行。
- 所有 failed move 都会 rollback。
- 最终 solver 仍会进行全局 feasibility check。

## 8. 强邻域速度控制

第一次加入强邻域后，解质量提升明显，但运行时间可能变长。例如某次对比：

```text
旧版：298 groups, 68.77s
新版：288 groups, 76.87s
```

新版更好但略慢。为此今天又做了速度控制：

### Local Search 调整

- 强邻域不再每一步都可能触发。
- 只有进入 plateau 后才尝试强邻域。
- `add-then-remove-many` 降低触发频率。
- `destroy-repair` 降低触发频率。
- 缩小 pivot 数量、候选池规模、trial 次数、destroy/repair 规模。
- 加入 plateau early stop：长时间没有改进时提前停止 Local Search。

### SA 调整

- 默认先使用便宜 move：
  - remove
  - replace
  - swap2
  - add_remove2
- 只有连续一段时间无改进后，才小概率启用：
  - add-then-remove-many
  - destroy-repair

效果：

```text
noNN 回归测试时间从约 65s 降到约 21s。
7 tests OK。
```

这属于安全加速，因为它只减少无效搜索，不改变可行性判定。

## 9. 今日验证结果

### 回归测试

命令：

```bash
python -B -m unittest tests.test_solver_known_cases tests.test_performance_regressions tests.test_validation -v
```

结果：

```text
Ran 7 tests in 21.334s
OK
```

覆盖内容：

- 已知 covering design case。
- tracker 与 naive 逻辑一致性。
- candidate overlap 正确性。
- solver 输出仍然可行。
- validator 正确识别合法/非法解。

### 小规模 solve smoke test

命令：

```bash
python -B -m optimal_samples_system solve --m 8 --n 8 --k 6 --j 4 --s 4 \
  --samples 1,2,3,4,5,6,7,8 \
  --seed 42 --restarts 2 \
  --exact-backend auto --exact-time-limit 30
```

结果：

```text
Best family size: 7
Exact size: 7
gap 0
```

说明求解器仍能找到已证明最优解。

## 10. 九个 assignment examples 验证

今天将用户给出的 9 个例子全部用最新 noNN 版本跑了一遍。

映射方式：

```text
A=1, B=2, C=3, ...
samples = 1..n
```

第 6 个例子按照文字说明设置为：

```text
coverage_mode=at_least_r
r=4
```

结果：

| Example | 参数 | 程序结果 | 期望值 | 可行性 | Exact 证明 |
|---|---:|---:|---:|---|---|
| E.g.1 | n=7,k=6,j=5,s=5 | 6 | 6 | PASS | exact=6, gap=0 |
| E.g.2 | n=8,k=6,j=5,s=5 | 12 | 12 | PASS | exact=12, gap=0 |
| E.g.3 | n=8,k=6,j=4,s=4 | 7 | 7 | PASS | exact=7, gap=0 |
| E.g.4 | n=9,k=6,j=4,s=4 | 12 | 12 | PASS | exact=12, gap=0 |
| E.g.5 | n=8,k=6,j=6,s=5 | 4 | 4 | PASS | exact=4, gap=0 |
| E.g.6 | n=8,k=6,j=6,s=5,r=4 | 10 | 10 | PASS | exact=10, gap=0 |
| E.g.7 | n=9,k=6,j=5,s=4 | 3 | 3 | PASS | exact=3, gap=0 |
| E.g.8 | n=10,k=6,j=6,s=4 | 3 | 3 | PASS | exact=3, gap=0 |
| E.g.9 | n=12,k=6,j=6,s=4 | 6 | 6 | PASS | SciPy 120s unknown |

结论：

- 9 个例子全部可行。
- 程序输出数量均不超过用户给出的 minimum。
- 前 8 个已经 exact proof gap=0。
- 第 9 个找到 6 组可行解，与用户给出的 minimum 一致，但 SciPy 在 120 秒内未证明不存在 5 组解。现在 Gurobi/SCIP 已安装，可以继续对第 9 个做更强 proof。

## 11. 如何验证一个 297 组解是不是近优

假设当前输出：

```text
Best family size: 297
```

第一步，验证可行性：

```bash
python -m optimal_samples_system validate-results --db-dir results_db_v3
```

第二步，证明是否存在更小解：

```bash
python -m optimal_samples_system prove-bound --m 45 --n 15 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --target-size 296 \
  --exact-backend auto \
  --exact-time-limit 600
```

如果结果是：

```text
Status: infeasible
```

则 297 是全局最优。

如果结果是：

```text
Status: unknown
```

则不能证明最优，只能说当前是 best-known feasible solution。

如果想证明近优而不是完全最优，可以尝试较低 target-size，例如：

```bash
--target-size 280
```

若证明 infeasible，则说明：

```text
OPT >= 281
gap <= 297 - 281 = 16
relative gap <= 297 / 281 - 1 ≈ 5.7%
```

## 12. 当前推荐运行命令

### 普通 noNN 求解

```bash
cd /Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System-noNN

python -m optimal_samples_system solve --m 45 --n 15 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --seed 42 --restarts 5 \
  --disable-ilp
```

### 求解并保存结果

```bash
python -m optimal_samples_system solve --m 45 --n 15 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --seed 42 --restarts 5 \
  --save
```

### 验证保存结果

```bash
python -m optimal_samples_system validate-results --db-dir results_db_v3
```

### 用 Gurobi/SCIP 证明边界

```bash
python -m optimal_samples_system prove-bound --m 45 --n 15 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --target-size 296 \
  --exact-backend auto \
  --exact-time-limit 600
```

## 13. 正确性保证总结

今天所有优化都没有改变问题语义：

```text
对于每个 j-subset target，必须被所选 k-subset family 覆盖；
覆盖条件仍然是 intersection >= s；
distinct_subsets / at_least_r / all_subsets 语义保持不变。
```

所有启发式优化只改变搜索策略：

- 先尝试哪些候选。
- 是否触发强邻域。
- 搜索多少步。
- 什么时候 early stop。

不会改变 feasibility 判断。

最终输出仍会经过：

```text
tracker.is_feasible()
```

如果不可行，solver 会抛出错误，不会静默输出错误答案。

## 14. 后续建议

短期建议：

- 用 Gurobi/SCIP 对第 9 个 example 跑 `target-size=5` proof。
- 对 n=15 的 297 组解跑 `target-size=296`，如果太难，先跑 `target-size=280/285/290` 建立近优下界。
- 保存每次重要实验结果，并用 validator 生成可行性证据。

报告中可以强调：

- 本项目同时提供 heuristic solver 和 exact verifier。
- 大规模 case 使用 heuristic 求 best-known solution。
- 小规模和部分中规模 case 使用 exact MIP 证明最优。
- 对无法完全证明的大规模 case，使用 prove-bound 给出可证明 lower bound 和 optimality gap。

