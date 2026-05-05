# Future Work

The current system is a hybrid heuristic solver for the covering design problem. It can efficiently produce feasible high-quality solutions and can certify global optimality on smaller instances through the ILP verifier. For larger instances, however, exact optimality is often computationally expensive to prove. The following directions are natural extensions for improving both solution quality and optimality verification.

## 1. Add-Then-Remove-Many Neighborhood

The current local search primarily uses small neighborhoods, such as removing one group, replacing one group, or removing one group and repairing it. These moves are safe and efficient, but they can become too local once the solution is already tight. In that situation, most selected groups have some exclusive coverage, so deleting a single group usually breaks feasibility.

A stronger future neighborhood is an add-then-remove-many move:

```text
1. Temporarily add 1 to 3 promising candidate groups.
2. Recompute coverage redundancy.
3. Greedily remove as many now-redundant groups as possible.
4. Accept the move only if the final solution is feasible and smaller.
```

This allows the solver to first create extra coverage redundancy and then remove several old groups at once. Such a move can discover improvements that a simple one-for-one replacement cannot find.

Correctness would still be preserved because the algorithm would only accept the final solution after checking feasibility with the coverage tracker.

## 2. Destroy-Repair Search

Another promising direction is destroy-repair. Instead of preserving feasibility at every intermediate step, the solver can deliberately remove a batch of weak or low-contribution groups, temporarily producing an infeasible partial solution. It then repairs the solution by greedily adding candidates that cover the largest remaining deficits.

A typical destroy-repair cycle would be:

```text
1. Select several groups to remove, e.g. low-redundancy or low-contribution groups.
2. Remove them and identify uncovered or under-covered targets.
3. Add repair candidates until feasibility is restored.
4. Run redundancy elimination again.
5. Accept the move only if the final solution is feasible and smaller.
```

This type of search is useful because many high-quality covering designs require a structural rearrangement rather than a sequence of single deletions. Destroy-repair can jump between more distant regions of the search space and may escape local optima that local search and simulated annealing cannot easily leave.

The main challenge is controlling the destroy size and repair cost. If the destroyed region is too large, repair becomes expensive and unstable. A practical implementation should start with small destroy sizes and trigger larger destroy-repair moves only after the search has reached a plateau.

## 3. Stronger MIP Solver

The current exact verifier uses `scipy.optimize.milp`, which is convenient and has no heavy project-specific dependency. However, for larger covering design instances, proving optimality can be much harder than simply finding a good feasible solution. The current solver may time out before it can certify that no smaller solution exists.

A useful future improvement is to support stronger mixed-integer programming solvers, such as:

- Gurobi
- SCIP
- CPLEX

These solvers generally provide stronger presolve, cutting planes, branching strategies, incumbent handling, and diagnostic information. They could significantly improve the chance of proving optimality for medium-size instances.

This would not replace the heuristic solver. Instead, the heuristic solution can be used as a high-quality initial upper bound for the exact solver, making the exact proof process more effective.

## 4. Gurobi / SCIP Exact Verification

For final optimality certification, a dedicated Gurobi or SCIP verification module could be added. The goal would not be to solve every large instance from scratch, but to verify whether the best heuristic solution is globally optimal.

For example, if the heuristic finds a feasible solution with 49 groups, the verifier can solve the decision problem:

```text
Does there exist a feasible solution with 48 groups or fewer?
```

If the solver proves this decision problem infeasible, then the 49-group solution is globally optimal.

This proof-oriented formulation is often more useful than simply minimizing from scratch, because it directly targets the optimality certificate needed for reporting results.

## Expected Impact

These extensions address two different limitations of the current system:

- Add-then-remove-many and destroy-repair are designed to improve heuristic solution quality on large instances.
- Stronger MIP solvers and Gurobi / SCIP verification are designed to improve optimality certification on instances where exact proof is currently difficult.

Together, these future directions would make the system stronger both as a practical heuristic solver and as an experimental platform for studying exact versus approximate covering design methods.
