---
title: "Optimal Sample Selection System"
subtitle: "Final noNN Submission: A Detailed Technical Report on Modeling, Algorithms, Implementation, and Certification"
author: "OpenAI Codex"
date: "2026-05-04"
lang: "en-US"
---

# Abstract

This report presents a complete engineering and algorithmic account of the **Optimal Sample Selection System**, a software platform for solving covering-design-style optimal sample selection problems of the form $L(n,k,j,s)$. Given a chosen set of $n$ samples from a universe of size $m$, the system enumerates all $k$-subsets as candidate groups and seeks the smallest family of such groups that satisfies a specified coverage requirement over every $j$-subset target. Unlike a standalone heuristic script, the system is designed as a full-stack optimization workflow with five explicit responsibilities: **constructing high-quality feasible solutions, validating feasibility, certifying optimality when possible, reporting rigorous lower bounds when full proof is infeasible, and exposing the workflow through both a command-line interface and an iOS application**. The final submitted version documented here is the **deterministic noNN baseline** rather than an experimental learning-augmented branch.

The methodology combines several layers of reasoning. At the modeling level, the system supports three coverage semantics (`at_least_one`, `at_least_r`, and `all_subsets`) and two aggregation semantics (`distinct_subsets` and `single_candidate`). At the algorithmic level, it uses a multi-stage hybrid pipeline consisting of **lazy greedy construction, redundancy elimination, structure-aware local search, simulated annealing, elite-guided restart intensification, and reduced-core exact polishing**. At the exact layer, the system supports mixed-integer formulations for all supported semantics and can use SciPy/HiGHS, Gurobi, or SCIP to solve exact or decision versions of the problem. At the correctness layer, the system introduces a **two-validator architecture**: one validator replays the solution through an incremental coverage tracker, while a second independent validator recomputes feasibility directly from the mathematical definition. At the certification layer, the system distinguishes between “feasible”, “audited”, “certified optimal”, and “currently unresolved but lower-bounded”, thereby making the interpretation of optimization results significantly more rigorous.

The central contribution of this project is therefore methodological rather than merely numerical. It demonstrates how heuristic optimization, exact verification, and software engineering discipline can be integrated into a coherent system for a hard combinatorial problem. The resulting platform is not only capable of producing solutions, but also capable of explaining what is known about those solutions and why that knowledge is trustworthy.

# 1. Introduction

Optimal sample selection in this project is formulated as a structured covering problem. From a universe $\mathcal{U}$ of size $m$, we focus on a selected working set of $n$ samples. From these $n$ samples, we enumerate all $k$-subsets as candidate groups. The task is then to choose as few candidate groups as possible so that every $j$-subset target receives a sufficient amount of $s$-level coverage.

This problem sits at the intersection of combinatorial design theory, set covering, and practical search-based optimization. Its challenge is not only combinatorial explosion but also semantic subtlety. The phrase “a target is covered” can mean different things depending on the aggregation rule:

- one may require only that a target share at least one $s$-subset with the chosen family,
- one may require at least $r$ distinct $s$-subset hits,
- or one may require all $s$-subsets of the target to be represented.

Similarly, one may allow these contributions to accumulate across several chosen groups (`distinct_subsets`), or one may insist that the required coverage be achieved by a single group (`single_candidate`).

These semantic distinctions matter. They change not only the optimal value but also the internal data structures, the exact formulation, and the interpretation of correctness. A robust solver must therefore do more than search; it must represent the problem semantics precisely and preserve them consistently across heuristics, exact backends, result storage, validation, and proof routines.

The system described in this report was designed with exactly that objective. In the final submission, these principles are implemented with a noNN design so that the optimization behavior remains reproducible and easier to audit. It is built around four guiding principles:

1. **Model fidelity**: every supported semantic variant should correspond to a precise mathematical definition and a consistent internal representation.
2. **Scalable search**: the system should produce high-quality feasible solutions on instances for which complete exact optimization may be too expensive.
3. **Independent correctness checking**: feasibility should not be trusted merely because the solver claims it; it should be independently reconstructable and auditable.
4. **Tiered optimality evidence**: when full global optimality cannot be proven within a time limit, the system should still report the strongest rigorous statement available, such as a certified lower bound.

The remainder of this report follows these principles. We begin with a formal problem formulation, then describe the system architecture, derive the heuristic and exact methodologies, explain the validation and certification logic, and conclude with limitations and future directions.

# 2. Formal Problem Definition

## 2.1 Basic notation

Let the selected working set of samples be

$$
S = \{s_1, s_2, \dots, s_n\}.
$$

The candidate group universe is

$$
\mathcal{C} = \binom{S}{k},
$$

with cardinality

$$
|\mathcal{C}| = \binom{n}{k}.
$$

The target universe is

$$
\mathcal{T} = \binom{S}{j},
$$

with cardinality

$$
|\mathcal{T}| = \binom{n}{j}.
$$

For a target $T \in \mathcal{T}$, define its family of internal $s$-subsets as

$$
\mathcal{Q}(T) = \binom{T}{s},
$$

and therefore

$$
|\mathcal{Q}(T)| = \binom{j}{s}.
$$

For a candidate group $C \in \mathcal{C}$, define

$$
\mathcal{Q}(C) = \binom{C}{s}.
$$

The objective is to construct a family $\mathcal{F} \subseteq \mathcal{C}$ of minimum size such that every target $T \in \mathcal{T}$ satisfies a prescribed coverage rule.

## 2.2 Aggregation semantics

The system supports two distinct aggregation semantics.

### 2.2.1 Distinct-subset aggregation

Under `distinct_subsets`, coverage accumulates across different chosen groups. For a fixed target $T$, define the accumulated covered $s$-subsets as

$$
\Gamma(T;\mathcal{F})
=
\bigcup_{C \in \mathcal{F}}
(\mathcal{Q}(C)\cap\mathcal{Q}(T)).
$$

The total accumulated coverage count is then

$$
g(T;\mathcal{F}) = |\Gamma(T;\mathcal{F})|.
$$

This is the most expressive semantics because different groups may contribute different $s$-subsets of the same target.

### 2.2.2 Single-candidate aggregation

Under `single_candidate`, the requirement must be met by one chosen candidate alone. Define

$$
h(T;\mathcal{F})
=
\max_{C \in \mathcal{F}} |\mathcal{Q}(C)\cap\mathcal{Q}(T)|.
$$

Since

$$
|\mathcal{Q}(C)\cap\mathcal{Q}(T)|=
\binom{|C\cap T|}{s}
\quad \text{if } |C\cap T|\ge s,
$$

and $0$ otherwise, this semantics is determined by the strongest single overlap between the chosen family and each target.

## 2.3 Coverage modes

Let $R$ denote the required coverage level for each target. The system supports three modes.

### 2.3.1 `at_least_one`

Require

$$
R = 1.
$$

Then the feasibility condition is

$$
g(T;\mathcal{F}) \ge 1
\quad\text{or}\quad
h(T;\mathcal{F}) \ge 1
\qquad \forall T\in\mathcal{T}.
$$

### 2.3.2 `at_least_r`

Require

$$
R = r,
\qquad 1 \le r \le \binom{j}{s}.
$$

Feasibility becomes

$$
g(T;\mathcal{F}) \ge r
\quad\text{or}\quad
h(T;\mathcal{F}) \ge r
\qquad \forall T\in\mathcal{T}.
$$

### 2.3.3 `all_subsets`

Require full $s$-subset coverage, i.e.

$$
R = \binom{j}{s}.
$$

Hence

$$
g(T;\mathcal{F}) \ge \binom{j}{s}
\quad\text{or}\quad
h(T;\mathcal{F}) \ge \binom{j}{s}
\qquad \forall T\in\mathcal{T}.
$$

## 2.4 Optimization problem

Introduce binary decision variables

$$
x_C =
\begin{cases}
1, & C \in \mathcal{F}, \\
0, & C \notin \mathcal{F}.
\end{cases}
$$

The optimization objective is

$$
\min \sum_{C \in \mathcal{C}} x_C
$$

subject to the appropriate target coverage constraints under the selected semantics.

This unified view covers all problem variants implemented by the system.

# 3. Computational Difficulty and Design Objectives

The combinatorial growth is immediate:

$$
|\mathcal{C}| = \binom{n}{k}, \qquad
|\mathcal{T}| = \binom{n}{j}.
$$

Even for moderate $n$, these values become large. For example, with $n=15$ and $k=6$,

$$
\binom{15}{6}=5005.
$$

If $j=5$, then

$$
\binom{15}{5}=3003.
$$

This means that a naive method may need to reason about thousands of candidates, thousands of targets, and, depending on semantics, thousands of intermediate $s$-subset relations. A fully exact mixed-integer solve is not always practical within assignment-level time budgets, yet purely heuristic output is often insufficient because it does not indicate whether the solution is merely feasible, near-optimal, or provably optimal.

The system was therefore designed around the following objectives:

1. **Fast construction of a feasible incumbent**.
2. **Aggressive heuristic improvement under tight runtime budgets**.
3. **Optional exact verification whenever instance size permits**.
4. **Rigorous post-hoc validation and auditing regardless of whether exact solving succeeds**.
5. **Graceful degradation from full optimality proof to certified lower bounds on harder instances**.

# 4. System Architecture

The system is implemented as a layered software pipeline. The principal modules are:

- `config.py`: enums, shared configuration objects, result dataclasses
- `instance.py`: instance generation and structural precomputation
- `tracking.py`: incremental feasibility tracking
- `heuristics.py`: greedy construction, redundancy elimination, local search, and simulated annealing
- `exact.py`: exact solve backends, lower bounds, decision proofs, restricted exact polishing
- `validation.py`: primary validation, independent validation, and audit
- `certify.py`: optimality certification and summary generation
- `storage.py`: persistent JSON result database
- `solver.py`: end-to-end orchestration
- `cli.py`: command-line interface
- `mobile_api.py`: local HTTP bridge for the iOS client

The top-level control flow is implemented in `OptimalSamplesSolver.solve()`. Its pipeline is:

1. build the `CoverageInstance`
2. optionally attempt a full exact solve if the instance falls within configured limits
3. run multi-restart heuristic search:
   - lazy greedy
   - redundancy elimination
   - local search
   - simulated annealing
   - final redundancy elimination
4. if a full exact optimum is unavailable, run reduced-core exact polishing
5. audit the final solution with both validators
6. package and optionally store the result

This is best understood as a cooperation between three distinct layers:

- a **search layer** that produces candidate solutions,
- a **proof layer** that tries to improve them or certify them exactly,
- and a **correctness layer** that verifies whether the produced result is trustworthy.

# 5. Structural Preprocessing and Internal Data Representation

## 5.1 CoverageInstance construction

`CoverageInstance` is the foundational object for a run. Given a validated `ProblemConfig`, it:

1. determines the working sample list `samples`
2. enumerates all position-level $k$-subsets
3. enumerates all position-level $j$-subsets
4. enumerates all position-level $s$-subsets
5. materializes them into sample labels
6. builds efficient lookup structures between candidates, $s$-subsets, and targets

The main precomputed structures are:

- `position_candidates`
- `position_targets`
- `position_s_subsets`
- `target_subset_ids`
- `subset_to_targets`
- `candidate_subset_ids`
- `subset_to_candidates`
- `candidate_masks` and `target_masks`

These structures eliminate repeated combinational recomputation during search.

## 5.2 Bitmask representation

Each candidate and target position set is also represented by an integer bitmask

$$
\mathrm{mask}(P)=\sum_{i\in P} 2^i.
$$

Then for two position sets $P_1$ and $P_2$,

$$
|P_1\cap P_2|
=
\mathrm{bitcount}\big(\mathrm{mask}(P_1)\land \mathrm{mask}(P_2)\big).
$$

This provides a compact and fast overlap primitive, especially useful in `single_candidate` modeling and structural scoring.

## 5.3 Reverse indices

One of the most important engineering optimizations is the precomputation of

$$
\texttt{subset\_to\_candidates}[q]
=
\{C\in\mathcal{C}: q\subseteq C\}.
$$

This reverse index is used repeatedly during local search and reduced-core exact polishing. It allows the solver to focus on structurally relevant candidates instead of sampling replacements uniformly from the entire outside pool.

# 6. Incremental Coverage Tracking

## 6.1 State variables

`CoverageTracker` maintains the current solution $\mathcal{F}$ incrementally. In `distinct_subsets` mode, the key arrays are:

- `subset_cover_count[q]`: how many chosen candidates cover subset $q$
- `target_covered_count[T]`: how many distinct $s$-subsets of $T$ are currently covered
- `unsatisfied_targets`: number of targets that still violate the coverage threshold
- `deficit_units`: total target deficit

Define target deficit as

$$
d(T;\mathcal{F})=
\max\bigl(0,\ R-g(T;\mathcal{F})\bigr).
$$

Then

$$
\texttt{deficit\_units}
=
\sum_{T\in\mathcal{T}} d(T;\mathcal{F}).
$$

The solution is feasible if and only if

$$
\texttt{unsatisfied\_targets}=0.
$$

In `single_candidate` mode the same philosophy applies, except that coverage is tracked at the target level through `cover_count[T]` rather than through distinct subset accumulation.

## 6.2 Incremental add/remove updates

When a candidate $C$ is added, the tracker updates only the $s$-subsets contained in $C$ and the targets influenced by those subsets. In `distinct_subsets`, a target is updated only when a subset is activated from zero coverage to positive coverage. This is crucial: adding a candidate that re-covers already active subsets should not inflate the distinct-subset count.

When a candidate is removed, the inverse update is applied. If a subset’s support count drops from one to zero, every target containing that subset loses one unit of distinct coverage.

This yields an efficient incremental view of feasibility and deficit rather than repeatedly recomputing coverage from scratch.

## 6.3 Marginal quantities

Several local-search metrics are derived from the tracker:

- `marginal_gain(c)`: how many deficit units candidate $c$ can immediately reduce
- `can_remove(c)`: whether $c$ may be removed without breaking feasibility
- `exclusive_count(c)`: how many targets would become unsatisfied after removing $c$
- `redundancy_score(c)`: average surplus coverage over targets impacted by $c$

These quantities provide the algebra used by the greedy constructor, redundancy eliminator, local search, and simulated annealing.

# 7. Greedy Construction via Lazy Marginal Re-evaluation

## 7.1 Coverage objective

For `distinct_subsets`, define the capped coverage objective

$$
f(\mathcal{F})
=
\sum_{T\in\mathcal{T}}
\min\bigl(g(T;\mathcal{F}), R\bigr).
$$

This function is monotone nondecreasing: adding candidates cannot decrease the number of covered subsets. More importantly, it exhibits diminishing returns. If

$$
\mathcal{A}\subseteq\mathcal{B}\subseteq\mathcal{C},
$$

then for any candidate $c$,

$$
\Delta(c\mid \mathcal{A})
=
f(\mathcal{A}\cup\{c\})-f(\mathcal{A})
\ge
f(\mathcal{B}\cup\{c\})-f(\mathcal{B})
=
\Delta(c\mid \mathcal{B}).
$$

Thus $f$ behaves as a monotone submodular set function over candidates.

## 7.2 Lazy greedy rationale

In a standard greedy algorithm, each iteration recomputes the marginal gain of all remaining candidates. With thousands of candidates, this becomes expensive. Because marginal gains under a monotone submodular objective never increase, the system uses a max-heap of upper bounds:

- each heap item stores a priority value and a timestamp,
- if a popped item was not re-evaluated in the current iteration, its gain is recomputed and pushed back,
- only an item whose timestamp matches the current iteration is treated as “fresh” and selected.

This reduces the number of marginal gain evaluations dramatically in practice while preserving correctness of the greedy choice.

## 7.3 Practical priority

The heap priority is not based solely on marginal gain. It adds a small structural tie-break based on candidate span, i.e. the number of impacted targets. In randomized restarts, a tiny random perturbation is also added to diversify the initial construction.

# 8. Redundancy Elimination

Greedy solutions are often feasible but bloated. `RedundancyEliminator` repeatedly searches for safe removals. The elimination rule is based on two statistics:

- `exclusive_count(c)`
- `redundancy_score(c)`

Among candidates that can be removed without violating feasibility, the eliminator prefers those with the lexicographically smallest pair

$$
\bigl(\texttt{exclusive\_count}(c),\ \texttt{redundancy\_score}(c)\bigr).
$$

This phase is applied:

- immediately after greedy,
- again after local search,
- again after simulated annealing,
- and again after reduced-core exact polishing.

The repeated use of redundancy elimination is deliberate. Many successful perturbations do not directly reduce the solution size, but they alter the coverage structure so that additional deletions become possible afterward.

# 9. Structure-Aware Local Search

## 9.1 Motivation

The local search phase attempts to improve the current incumbent without the cost of a global exact solve. The main challenge is that most randomly sampled modifications are not helpful. The system addresses this in two complementary ways:

1. **structural candidate filtering**
2. **deterministic move scoring based on coverage and overlap metrics**

The result is a local search engine that is still exact with respect to correctness, but selective with respect to where it spends evaluation effort.

## 9.2 Move families

`ImprovedLocalSearch` supports five move families:

1. `remove`
2. `replace`
3. `remove_repair`
4. `add_then_remove_many`
5. `destroy_repair`

These moves cover a spectrum from greedy shrinkage to larger structural reconfiguration.

## 9.3 Structure-aware replacement sampling

A major performance idea is that not all outside candidates are equally relevant to a removal. Suppose the current move attempts to remove candidate $c^-$. The solver does not sample replacement candidates uniformly from $\mathcal{C}\setminus\mathcal{F}$. Instead it constructs a relevant pool

$$
\mathcal{P}(c^-)
=
\left(
\bigcup_{q\in\mathcal{Q}(c^-)}
\{c\in\mathcal{C}\setminus\mathcal{F}: q\subseteq c\}
\right)\setminus\{c^-\}.
$$

In words, only candidates sharing structurally relevant $s$-subsets with the removed candidate are considered. This sharply increases the probability that an evaluated replacement can actually repair the damage caused by removal.

## 9.4 Deterministic move scoring and acceptance logic

For replace-style moves, the solver ranks candidate additions with a deterministic score built from quantities that can be computed directly from the current tracker state:

$$
\mathrm{score}(c^-, c^+)
=
g(c^+)
- e(c^-)
+ \alpha \, \mathrm{ov}(c^-, c^+)
+ \beta \, \mathrm{span}(c^+)
+ \gamma \, \mathrm{recover}(c^-, c^+)
- \delta \, \mathrm{progress},
$$

where:

- $g(c^+)$ is the marginal gain of the added candidate,
- $e(c^-)$ is the exclusive coverage burden of the removed candidate,
- $\mathrm{ov}(c^-, c^+)$ measures structural overlap,
- $\mathrm{span}(c^+)$ measures how broadly the added candidate touches the target space,
- $\mathrm{recover}(c^-, c^+)$ estimates how much newly uncovered structure can be repaired,
- and `progress` is a small stage-dependent penalty that gently favors earlier improvements.

The exact coefficients are heuristic tuning constants rather than problem-defining semantics. Their role is to prioritize structurally plausible moves, not to redefine feasibility.

The local search is conservative about correctness:

- a move may only be accepted if the resulting solution is feasible,
- and, for genuine improvement, the post-move redundancy-stripped solution must be strictly smaller than the pre-move solution.

Thus the scoring layer changes only the *order in which moves are tried*, not the definition of validity or improvement.

## 9.5 Plateau-aware adaptive neighborhoods

The local search does not use a static move distribution. Instead, it estimates a plateau stage from the number of consecutive non-improving steps and changes its profile accordingly.

Early stage:

- smaller candidate pools,
- more `remove` and `replace`,
- faster exploitation.

Middle stage:

- balanced mix,
- more `remove_repair`.

Late stage:

- larger candidate pools,
- higher probability of `add_then_remove_many` and `destroy_repair`,
- stronger diversification.

This implements a principled intensification/diversification trade-off without increasing the total search budget arbitrarily.

# 10. Large-Neighborhood Moves

## 10.1 Add-then-remove-many

This move is designed for plateaus where direct deletion no longer works. The logic is:

1. choose structurally weak pivot candidates,
2. sample a small set of promising external additions,
3. temporarily add one to three such candidates,
4. run redundancy elimination,
5. keep the best feasible shrinkage if it strictly improves the solution size.

The philosophy is to *buy flexibility first, then compress more aggressively*. Some local minima cannot be escaped by direct remove/replace moves because they lack short-term feasibility-preserving alternatives. Temporarily increasing redundancy can unlock new deletions.

## 10.2 Destroy-repair

This move intentionally removes a small block of structurally weak candidates, then greedily repairs the damage using structurally relevant outside candidates. It is a targeted diversification operator.

Mathematically, it explores neighborhoods that are difficult to reach via one-step exchanges:

$$
\mathcal{F}
\longrightarrow
(\mathcal{F}\setminus D)\cup A
\longrightarrow
\text{redundancy-strip}((\mathcal{F}\setminus D)\cup A),
$$

where $D$ is a small destroyed set and $A$ is a repair set assembled adaptively from the new deficit structure.

# 11. Simulated Annealing

## 11.1 Cost function

The simulated annealing phase is allowed to move through temporarily infeasible states, but feasibility violations are penalized. The cost used is

$$
\mathrm{cost}(\mathcal{F})
=
|\mathcal{F}| + \lambda \cdot \mathrm{deficit\_units}(\mathcal{F}),
$$

where $\lambda$ is a sufficiently large penalty constant.

This is important. A purely feasible-only search may become trapped in small neighborhoods. The penalty-based formulation permits controlled excursions outside feasibility while strongly preferring quick return.

## 11.2 Acceptance rule

Let

$$
\Delta =
\mathrm{cost}_{new} - \mathrm{cost}_{old}.
$$

The move is accepted with probability

$$
P(\text{accept}) =
\begin{cases}
1, & \Delta \le 0,\\
\exp(-\Delta/T), & \Delta > 0.
\end{cases}
$$

This classical Boltzmann rule makes downhill moves deterministic and uphill moves temperature-dependent.

## 11.3 Cooling and reheating

The system uses geometric cooling:

$$
T_{t+1}=\alpha T_t,
\qquad
\alpha=
\left(\frac{T_{\mathrm{end}}}{T_{\mathrm{start}}}\right)^{1/N}.
$$

If a long period passes with no improvement, the temperature is reheated:

$$
T \leftarrow \min(5T,\ T_{\mathrm{start}}).
$$

This allows the search to regain mobility after stagnation.

## 11.4 Move set

The annealing phase uses a related but not identical move set:

- remove
- replace
- swap2
- add_remove2
- add_then_remove_many
- destroy_repair

The selected move mix again depends on an adaptive stage estimate that considers both recent non-improvement and current temperature.

# 12. Reduced-Core Exact Polishing

## 12.1 Motivation

When a full exact solve is too expensive, the current incumbent still contains valuable structural information. The system exploits this by constructing a reduced candidate core for a second, much smaller exact solve.

## 12.2 Core construction

The core begins with the incumbent itself. Then:

1. identify “soft pivots” and “hard pivots” using exclusive count and redundancy,
2. collect critical $s$-subsets uniquely supported by those pivots,
3. score outside candidates by:
   - how many pivot subsets they share,
   - how many critical subsets they touch,
   - how much structural overlap they have with pivots,
   - their candidate span.

If the incumbent is denoted by $\mathcal{F}$ and the reduced core by $\mathcal{K}$, then

$$
\mathcal{F}\subseteq \mathcal{K}\subseteq \mathcal{C},
\qquad
|\mathcal{K}| \ll |\mathcal{C}|.
$$

The exact solver is then run only over $\mathcal{K}$.

## 12.3 Interpretation

This procedure does **not** guarantee the global optimum over the full candidate universe, because the restricted core may omit a globally essential candidate. However, it often improves the incumbent further at a much lower exact cost than solving the full model.

Thus reduced-core polishing is best viewed as a structured heuristic-exact hybrid.

# 13. Exact Optimization Models

## 13.1 At-least-one direct covering model

Under `at_least_one`, define

$$
\mathcal{N}(T)=\{C\in\mathcal{C}: |\mathcal{Q}(C)\cap \mathcal{Q}(T)|\ge 1\}.
$$

The exact model becomes

$$
\min \sum_{C\in\mathcal{C}} x_C
$$

subject to

$$
\sum_{C\in\mathcal{N}(T)} x_C \ge 1
\qquad \forall T\in\mathcal{T},
$$

$$
x_C\in\{0,1\}.
$$

This is a set-cover-style formulation.

## 13.2 Single-candidate model

Under `single_candidate`, define

$$
\mathcal{N}_R(T)=\{C\in\mathcal{C}: |\mathcal{Q}(C)\cap\mathcal{Q}(T)|\ge R\}.
$$

Then

$$
\sum_{C\in\mathcal{N}_R(T)} x_C \ge 1
\qquad \forall T\in\mathcal{T}.
$$

The objective remains cardinality minimization.

## 13.3 Distinct-subset model with auxiliary variables

For `distinct_subsets`, binary candidate variables alone are not enough, because coverage accumulates through distinct $s$-subset activations. Introduce

$$
y_q\in\{0,1\}
\qquad \forall q\in\binom{S}{s},
$$

where $y_q=1$ means that subset $q$ has been activated by at least one selected candidate.

Let

$$
a_{Cq}=
\begin{cases}
1, & q\subseteq C,\\
0, & \text{otherwise}.
\end{cases}
$$

Then activation is encoded by

$$
y_q \le \sum_{C\in\mathcal{C}} a_{Cq}x_C
\qquad \forall q.
$$

For every target $T$,

$$
\sum_{q\in\mathcal{Q}(T)} y_q \ge R.
$$

The model is therefore

$$
\min \sum_{C\in\mathcal{C}} x_C
$$

subject to

$$
y_q \le \sum_{C\in\mathcal{C}} a_{Cq}x_C \qquad \forall q,
$$

$$
\sum_{q\in\mathcal{Q}(T)} y_q \ge R \qquad \forall T\in\mathcal{T},
$$

$$
x_C\in\{0,1\},\quad y_q\in\{0,1\}.
$$

This is exactly the modeling idea used by the exact backend implementation.

## 13.4 Decision proof

Suppose an incumbent solution has size $U$. To prove that it is globally optimal, it is enough to show that the following decision problem is infeasible:

$$
\exists \mathcal{F}\subseteq\mathcal{C}
\quad\text{s.t.}\quad
|\mathcal{F}| \le U-1
\quad\text{and}\quad
\mathcal{F}\text{ is feasible}.
$$

In MIP form, one adds the constraint

$$
\sum_{C\in\mathcal{C}} x_C \le U-1
$$

and asks whether the model is feasible. This is the backbone of `prove-bound` and `certify-result`.

# 14. Lower Bounds and Near-Optimality Evidence

## 14.1 LP relaxation

When exact integer solving is too expensive, the system relaxes the integrality conditions:

$$
x_C\in[0,1],\qquad y_q\in[0,1].
$$

Let the LP optimum be $z_{LP}$. Then

$$
OPT \ge z_{LP}.
$$

Because the true optimum is integral,

$$
OPT \ge \lceil z_{LP}\rceil.
$$

The system reports this as a rigorous lower bound whenever the LP relaxation is solved successfully.

## 14.2 Counting bound

Let $M$ be the largest number of targets coverable by any single candidate:

$$
M = \max_{C\in\mathcal{C}} \mathrm{span}(C).
$$

Then the solution size must satisfy

$$
|\mathcal{F}| \ge \left\lceil\frac{|\mathcal{T}|}{M}\right\rceil.
$$

This is a simple but valid combinatorial lower bound.

## 14.3 Schönheim bound

When `coverage_mode = at_least_one` and $j=s$, the instance aligns with a classical covering-design setting, and the system applies the Schönheim bound:

$$
L(v,k,t)
=
\left\lceil \frac{v}{k}L(v-1,k-1,t-1)\right\rceil,
$$

with base case

$$
L(v,k,1)=\left\lceil \frac{v}{k}\right\rceil.
$$

This often gives a stronger structural lower bound than plain counting.

## 14.4 Certified gap

If the incumbent has size $U$ and the strongest currently certified lower bound is $L$, then the system reports the current upper gap as

$$
\mathrm{gap}_{upper}=U-L.
$$

This is valuable because it makes unresolved large-instance output much more informative than a bare “timed out” message.

# 15. Validation, Audit, and Fail-Closed Correctness

## 15.1 Why one validator is not enough

If the solver and validator share the same internal logic, then a single implementation error can corrupt both simultaneously. The system therefore deliberately separates:

- the **primary validator**, which uses the same incremental semantics as the tracker,
- and the **independent validator**, which recomputes feasibility directly from the formal definition.

## 15.2 Primary validation

The primary validator rebuilds the `CoverageInstance`, parses the result into candidate indices, replays the solution through a fresh `CoverageTracker`, and checks:

$$
\texttt{unsatisfied\_targets}=0
$$

and

$$
\texttt{deficit\_units}=0.
$$

## 15.3 Independent brute-force validation

The independent validator ignores the tracker logic. For each target $T$, it directly computes either

$$
g(T;\mathcal{F})
=
\left|
\bigcup_{C\in\mathcal{F}}
(\mathcal{Q}(C)\cap \mathcal{Q}(T))
\right|
$$

or

$$
h(T;\mathcal{F})
=
\max_{C\in\mathcal{F}}
|\mathcal{Q}(C)\cap\mathcal{Q}(T)|.
$$

The target deficit is then

$$
d(T;\mathcal{F})
=
\max(0,R-g(T;\mathcal{F}))
$$

or

$$
d(T;\mathcal{F})
=
\max(0,R-h(T;\mathcal{F})).
$$

The solution is valid exactly when the aggregate deficit is zero.

## 15.4 Audit layer

The audit layer compares:

- parsed group count,
- feasibility decision,
- number of unsatisfied targets

between the primary and independent validators. If the two disagree, the result is flagged as a mismatch.

## 15.5 Fail-closed result semantics

The result validation code is intentionally fail-closed with respect to missing problem semantics. If semantic fields such as `samples`, `coverage_mode`, or `aggregation_mode` are missing from a stored result, the validator does **not** silently infer defaults and pass the file. This is an important engineering choice because hidden fallback semantics can easily lead to incorrect acceptance of malformed or legacy result files.

## 15.6 Validation gate for exact backend solutions

The validation mechanism is not applied only to human-facing saved results. It is also applied to candidate solutions produced by exact backends. Whenever SciPy, Gurobi, or SCIP returns a feasible incumbent, the system does **not** accept that solution immediately on the basis of solver status alone. Instead, the returned candidate index set is passed through the same dual-validator gate used elsewhere in the system:

1. rebuild the instance under the declared semantics,
2. run the primary incremental validator,
3. run the independent brute-force validator,
4. compare the two outputs for agreement.

If either validator rejects the solution, or if both validators disagree, the backend solution is discarded and reclassified as unresolved rather than accepted. This design is particularly important in a project setting because it prevents exact-solver integration bugs, index mapping mistakes, or restricted-core reconstruction mistakes from silently contaminating the final result stream.

In other words, a feasible status reported by an external optimizer is treated as **necessary but not sufficient** evidence of correctness within the software system. The software only accepts exact solutions after they survive its own semantic audit layer.

## 15.7 Final self-audit before solver return

The same principle is enforced at the top-level solver boundary. After all heuristic and optional exact-polishing stages have completed, the final incumbent is audited once more before being returned to the caller or written to disk. If the final incumbent fails either validator, or if the two validators disagree, the solver raises an internal error instead of returning the solution.

This creates an explicit acceptance invariant for the software:

> **Acceptance invariant.** No solution is returned by the end-to-end solver, and no exact-backend incumbent is accepted into the final output path, unless it passes both validators under the declared problem semantics.

This invariant is stronger than a simple post-hoc report. It actively constrains the control flow of the solver.

## 15.8 Validation snapshot as persistent metadata

After the final incumbent passes self-audit, the solver stores a validation snapshot in the serialized result object. This snapshot records:

- whether the primary validator accepted the solution,
- whether the independent validator accepted the solution,
- whether both methods agreed,
- the number of unsatisfied targets seen by the primary validator,
- the final deficit count.

The purpose of this metadata is twofold. First, it makes saved results self-describing: a result file carries not only the selected groups, but also the status of the correctness checks that were performed at runtime. Second, it enables result-database commands such as `list-results` and `show-result` to surface validation state immediately, without requiring the user to re-run a full validation cycle just to inspect the file.

## 15.9 Verification workflow summary

The full verification workflow can be summarized procedurally as follows.

```text
Input: declared problem semantics + candidate solution
1. Rebuild the CoverageInstance from declared parameters.
2. Normalize and validate candidate indices/groups.
3. Run primary validation via CoverageTracker replay.
4. Run independent validation via direct target-by-target definition checking.
5. Compare the two validator outputs.
6. Accept the solution only if:
   - primary validator says feasible,
   - independent validator says feasible,
   - parsed group count agrees,
   - unsatisfied target count agrees.
Otherwise reject or mark unresolved.
```

This workflow is the core reason the project can make stronger claims than a typical heuristic implementation. It does not merely *search* for solutions; it maintains an explicit contract for what counts as an acceptable result.

# 16. Optimality Certification Workflow

`certify-result` performs a three-stage reasoning process:

1. **Audit the incumbent**  
   If the incumbent is invalid or the validators disagree, certification stops immediately.

2. **Run a decision proof**  
   Try to prove that no feasible solution exists with cardinality at most $U-1$, where $U$ is the incumbent size.

3. **If exact proof times out, compute lower bounds**  
   Use LP relaxation and combinatorial bounds to provide the strongest rigorous fallback statement available.

This can be written as the following operational certification routine:

```text
Input: incumbent result of size U
1. Audit the incumbent with both validators.
2. If the incumbent is invalid, stop with status invalid_incumbent.
3. Ask the exact backend whether a feasible solution exists with size <= U - 1.
4. If infeasible, return certified_optimal.
5. If feasible, return not_optimal.
6. If unresolved, compute rigorous lower bounds L.
7. Return unresolved together with:
   - incumbent feasibility status,
   - certified lower bound L when available,
   - current gap upper bound U - L.
```

The important methodological point is that optimality certification is built **on top of** correctness auditing rather than in parallel with it. The system never tries to prove optimality for a result whose semantic feasibility has not already been independently established.

This produces one of four statuses:

- `certified_optimal`
- `not_optimal`
- `unresolved`
- `invalid_incumbent`

This tiered logic is one of the most important methodological aspects of the system. It separates the notion of correctness from the notion of global optimality and provides an explicit evidence ladder for both.

# 17. Software Interfaces and Result Management

## 17.1 Command-line interface

The CLI exposes all major workflow actions:

- `solve`
- `list-results`
- `show-result`
- `delete-result`
- `validate-*`
- `audit-*`
- `prove-bound`
- `certify-*`
- `summarize-*`
- `demo`

This makes the system suitable both for interactive experimentation and for scripted evaluation.

## 17.2 Result database

Results are stored as JSON files whose names encode:

- the problem parameters,
- the coverage mode,
- the aggregation mode,
- the run number,
- and the final solution size.

Each saved result includes:

- the selected groups,
- candidate indices,
- elapsed time,
- exact metadata,
- and a runtime validation snapshot.

## 17.3 iOS application

The iOS app is intentionally implemented as a thin client over a local Python HTTP bridge. This avoids duplicating solver logic in Swift and keeps the solver, validators, and exact proof layer centralized in one trusted backend.

From a software engineering standpoint, this is a strong design choice: one algorithmic core, multiple presentation layers.

# 18. Experimental Results

## 18.1 Experimental protocol

To provide a consistent empirical snapshot of the current solver version, we evaluated a representative set of instances under a fixed configuration:

- samples: `samples = (1,2,\dots,n)`
- random seed: `42`
- number of restarts: `5`
- coverage mode: `at_least_one`
- aggregation mode: `distinct_subsets`
- exact backend option inside `solve`: `auto`
- exact time limit inside `solve`: `60` seconds

The reported objective value is the final family size returned by the complete pipeline after greedy construction, redundancy elimination, local search, simulated annealing, final redundancy elimination, optional reduced-core exact polishing, and final dual-validator self-audit.

Because these instances are already of moderate scale, the values in Table 1 should be interpreted as **current best audited solver outputs produced by the present configuration**, not automatically as globally certified optima.

## 18.2 Experimental table

Table 1 summarizes ten instances that were run with the current solver configuration.

| Case $(m,n,k,j,s)$ | Best family size returned | Exact optimum obtained during `solve` | Result interpretation |
|---|---:|---:|---|
| $(45,25,6,6,4)$ | 243 | No | Audited heuristic incumbent |
| $(45,15,6,4,4)$ | 142 | No | Audited heuristic incumbent |
| $(45,13,6,5,5)$ | 287 | No | Audited heuristic incumbent |
| $(45,22,6,6,4)$ | 138 | No | Audited heuristic incumbent |
| $(45,16,6,4,4)$ | 188 | No | Audited heuristic incumbent |
| $(45,16,6,6,4)$ | 29 | No | Audited heuristic incumbent |
| $(45,16,6,6,5)$ | 271 | No | Audited heuristic incumbent |
| $(45,18,6,5,4)$ | 110 | No | Audited heuristic incumbent |
| $(45,18,6,4,4)$ | 304 | No | Audited heuristic incumbent |
| $(45,20,6,6,4)$ | 90 | No | Audited heuristic incumbent |

## 18.3 Interpretation of the table

Several observations follow from Table 1.

First, the system remains effective on instances whose combinatorial candidate universe is large enough that ordinary exact verification is often impractical within the default `solve` budget. This shows that the heuristic layers are not merely decorative additions to an exact core; they are the main mechanism by which feasible high-quality incumbents are obtained at scale.

Second, the table illustrates why the project separates **solving**, **auditing**, and **certification** into distinct commands. The absence of an exact optimum during `solve` does not mean that the output lacks correctness evidence. It means only that the ordinary solve-time exact phase did not finish with a global optimum certificate inside the configured budget.

Third, the phrase “audited heuristic incumbent” is deliberate. For these runs, the strongest immediate claim made by the end-to-end solver is:

- the solution is semantically feasible,
- the solution passed the internal dual-validator gate,
- and the recorded family size is therefore a trustworthy incumbent for further certification.

Fourth, these results emphasize the methodological value of the system’s evidence hierarchy. For larger instances, the workflow should be:

1. generate a feasible incumbent with `solve`,
2. confirm or inspect its validation status,
3. run `summarize-result` or `certify-result`,
4. report whether the incumbent is:
   - certified optimal,
   - certified non-optimal,
   - or unresolved but accompanied by a rigorous lower bound.

This prevents a common mistake in heuristic optimization reports, namely overstating a good feasible solution as though it had already been mathematically proven optimal.

## 18.4 Reporting guidance

When presenting experimental outcomes in a report or demonstration, the recommended terminology is:

- **best family size found**: when only heuristic search plus audit has completed,
- **certified optimal**: when decision infeasibility for all smaller sizes has been established,
- **unresolved with certified lower bound**: when the incumbent is feasible and audited, but full optimality proof did not finish.

This language matches the architecture of the software itself and keeps the interpretation of empirical results mathematically honest.

# 19. Methodological Strengths of the System

The methodology of this system is strong in at least six respects:

1. **Semantic precision**  
   It formalizes multiple coverage and aggregation variants cleanly.

2. **Algorithmic layering**  
   It does not rely on a single search strategy but composes greedy construction, local improvement, global perturbation, and exact polishing.

3. **Structure-aware search**  
   It avoids wasting search effort on replacement candidates with no meaningful structural relationship to the damaged portion of the incumbent.

4. **Controlled move-ranking component**  
   Search-ordering heuristics influence which moves are tried first, never the definition of feasibility or correctness.

5. **Independent correctness verification**  
   The system explicitly defends against shared-logic validation failure.

6. **Graceful exactness hierarchy**  
   It cleanly degrades from full proof to lower-bound certificates rather than conflating “unproven” with “incorrect”.

# 20. Limitations

Despite its breadth, the system retains several limitations.

First, no NP-hard covering problem of nontrivial size can be guaranteed to yield a full optimality proof within bounded time on all instances. Large instances may remain unresolved.

Second, the structure-aware scoring rules and reduced-core exact polishing remain heuristic intensification devices rather than globally exhaustive search. They can substantially improve incumbents, but they do not by themselves guarantee convergence to the true optimum on every large instance.

Third, reduced-core exact polishing is structurally informed but not globally exhaustive. Its success depends on whether the constructed candidate core happens to retain the combinatorial ingredients needed for improvement.

Fourth, while the iOS app makes the system more accessible, the true computational core still runs in Python, which means the mobile interface depends on a local backend process.

# 21. Future Directions

Several directions could strengthen the current methodology further:

- elite-guided restarts or path relinking,
- more powerful destroy-repair policies,
- larger or offline-trained learned move policies,
- stronger lower bounds for difficult large instances,
- stronger MIP backends and more sophisticated warm starts,
- richer benchmark suites and automated report generation.

These directions are consistent with the existing software architecture and can be added incrementally without undermining the current correctness guarantees.

# 22. Conclusion

This report has presented the Optimal Sample Selection System as a full methodological pipeline rather than a single optimization heuristic. At the mathematical level, the system formalizes multiple coverage semantics under a common optimization framework. At the algorithmic level, it combines lazy greedy construction, redundancy elimination, structure-aware local search, simulated annealing, and reduced-core exact polishing. At the proof level, it supports exact certification, decision proofs, LP relaxations, and combinatorial lower bounds. At the correctness level, it uses two independent validators and fail-closed result semantics. At the software level, it integrates these capabilities into a command-line platform and an iOS application.

The resulting system does not merely compute solutions; it organizes what can be *claimed* about those solutions. In combinatorial optimization, this distinction is fundamental. A solver that returns a small family is useful. A solver that can explain whether that family is feasible, audited, globally optimal, or only partially certified is substantially more valuable. That is the central methodological contribution of this project.
