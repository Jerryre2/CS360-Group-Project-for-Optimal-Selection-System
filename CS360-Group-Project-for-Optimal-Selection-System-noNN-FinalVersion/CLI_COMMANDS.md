# CLI Commands Cheat Sheet

本文档整理 `CS360-Group-Project-for-Optimal-Selection-System-noNN` 当前版本支持的所有主要 CLI 指令。

进入项目目录：

```bash
cd /Users/jerryge/CS360-Group-Project-for-Optimal-Selection-System-noNN
```

## 1. 查看帮助

查看全部命令：

```bash
python -m optimal_samples_system --help
```

查看某个子命令：

```bash
python -m optimal_samples_system solve --help
python -m optimal_samples_system prove-bound --help
```

调整日志等级：

```bash
python -m optimal_samples_system --log-level DEBUG solve ...
python -m optimal_samples_system --log-level WARNING solve ...
```

## 2. Solve：求解一个实例

基础格式：

```bash
python -m optimal_samples_system solve --m <m> --n <n> --k <k> --j <j> --s <s>
```

常用例子：

```bash
python -m optimal_samples_system solve --m 45 --n 15 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --seed 42 --restarts 5
```

### Solve 参数

问题参数：

```bash
--m 45
--n 15
--k 6
--j 5
--s 4
--samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

覆盖语义：

```bash
--coverage-mode at_least_one
--coverage-mode at_least_r --r 4
--coverage-mode all_subsets
```

聚合语义：

```bash
--aggregation-mode distinct_subsets
--aggregation-mode single_candidate
```

搜索参数：

```bash
--seed 42
--restarts 5
--local-steps 3000
--sa-iterations 5000
--candidate-sample-size 48
```

Exact verification 参数：

```bash
--disable-ilp
--exact-backend auto
--exact-backend gurobi
--exact-backend scip
--exact-backend scipy
--exact-time-limit 600
--force-exact
```

保存结果：

```bash
--save
--db-dir results_db_v3
```

noNN 兼容参数：

```bash
--disable-neural-guidance
```

说明：该参数只是为了兼容旧脚本，在 noNN 版本中不会改变行为。

## 3. 导出结果

使用 `--save` 会将结果保存到数据库目录。

默认保存目录：

```bash
results_db_v3
```

示例：

```bash
python -m optimal_samples_system solve --m 45 --n 15 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --seed 42 --restarts 5 \
  --save
```

指定保存目录：

```bash
python -m optimal_samples_system solve --m 45 --n 15 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --seed 42 --restarts 5 \
  --save \
  --db-dir my_results
```

## 4. List Results：列出保存结果

```bash
python -m optimal_samples_system list-results --db-dir results_db_v3
```

如果使用自定义目录：

```bash
python -m optimal_samples_system list-results --db-dir my_results
```

## 5. Show Result：查看某个保存结果

```bash
python -m optimal_samples_system show-result <filename> --db-dir results_db_v3
```

示例：

```bash
python -m optimal_samples_system show-result result_20260423_125000.json --db-dir results_db_v3
```

## 6. Delete Result：删除某个保存结果

```bash
python -m optimal_samples_system delete-result <filename> --db-dir results_db_v3
```

示例：

```bash
python -m optimal_samples_system delete-result result_20260423_125000.json --db-dir results_db_v3
```

## 7. Validate：验证答案正确性

验证器用于检查输出解是否真的满足覆盖约束。

它会检查：

- group 是否合法。
- group 内是否有重复样本。
- 是否出现样本全集之外的元素。
- 是否存在重复 group。
- `groups` 和 `solution_indices` 是否一致。
- `num_groups` 是否和实际 group 数一致。
- 所有 target 是否满足覆盖约束。

### 验证数据库中的单个结果

```bash
python -m optimal_samples_system validate-result <filename> --db-dir results_db_v3
```

示例：

```bash
python -m optimal_samples_system validate-result result_20260423_125000.json --db-dir results_db_v3
```

### 验证一个 JSON 文件

```bash
python -m optimal_samples_system validate-file /path/to/result.json
```

### 验证整个结果目录

```bash
python -m optimal_samples_system validate-results --db-dir results_db_v3
```

如果全部正确，会显示 `VALID`。

如果有错误，会显示 `INVALID`，并列出错误原因和未覆盖 target 示例。

## 8. Prove Bound：证明是否存在更小解

`prove-bound` 用于验证近优性或最优性。

它回答的问题是：

```text
是否存在一个可行解，其 group 数量 <= target-size？
```

例如当前有 297 组解，要证明它是否全局最优，需要证明不存在 296 组或更少的解：

```bash
python -m optimal_samples_system prove-bound --m 45 --n 15 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --target-size 296 \
  --exact-backend auto \
  --exact-time-limit 600
```

### Prove-bound 参数

问题参数：

```bash
--m 45
--n 15
--k 6
--j 5
--s 4
--samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

覆盖语义：

```bash
--coverage-mode at_least_one
--coverage-mode at_least_r --r 4
--coverage-mode all_subsets
```

目标大小：

```bash
--target-size 296
```

Exact 后端：

```bash
--exact-backend auto
--exact-backend gurobi
--exact-backend scip
--exact-backend scipy
```

时间限制：

```bash
--exact-time-limit 600
```

### Prove-bound 结果解释

```text
Status: infeasible
```

不存在 `<= target-size` 的可行解。

如果你已经有一个 `target-size + 1` 的可行解，那么它就是全局最优。

```text
Status: feasible
```

存在 `<= target-size` 的可行解。

这说明当前解还可以继续改进。

```text
Status: unknown
```

在给定时间内没有证明出来。

此时不能说当前解已经最优，只能说它是当前 best-known feasible solution。

## 9. Demo：运行预设示例

```bash
python -m optimal_samples_system demo
```

保存 demo 结果：

```bash
python -m optimal_samples_system demo --save --db-dir results_db_v3
```

指定 seed：

```bash
python -m optimal_samples_system demo --seed 20260415
```

## 10. 常用工作流

### 工作流 A：求解并保存

```bash
python -m optimal_samples_system solve --m 45 --n 15 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --seed 42 --restarts 5 \
  --save \
  --db-dir results_db_v3
```

### 工作流 B：验证保存答案

```bash
python -m optimal_samples_system validate-results --db-dir results_db_v3
```

### 工作流 C：尝试证明当前解最优

假设当前解是 297 组：

```bash
python -m optimal_samples_system prove-bound --m 45 --n 15 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --target-size 296 \
  --exact-backend auto \
  --exact-time-limit 600
```

### 工作流 D：证明近优下界

如果 `target-size=296` 太难，可以先证明较弱下界：

```bash
python -m optimal_samples_system prove-bound --m 45 --n 15 --k 6 --j 5 --s 4 \
  --samples 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  --target-size 280 \
  --exact-backend auto \
  --exact-time-limit 600
```

如果返回：

```text
Status: infeasible
```

说明：

```text
OPT >= 281
```

若当前解为 297，则：

```text
absolute gap <= 297 - 281 = 16
relative gap <= 297 / 281 - 1 ≈ 5.7%
```

## 11. 最常用三条命令

求解并保存：

```bash
python -m optimal_samples_system solve ... --save
```

验证答案正确性：

```bash
python -m optimal_samples_system validate-results --db-dir results_db_v3
```

证明是否有更小解：

```bash
python -m optimal_samples_system prove-bound ... --target-size <current_solution_size - 1>
```

