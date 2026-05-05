"""Validation utilities for saved covering design solutions."""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .config import AggregationMode, CoverageMode, ProblemConfig
from .instance import CoverageInstance
from .tracking import CoverageTracker


@dataclass
class ValidationIssue:
    severity: str
    message: str


@dataclass
class ValidationReport:
    source: str
    is_valid: bool
    num_groups: int
    declared_num_groups: Optional[int]
    unsatisfied_targets: int
    deficit_units: int
    issues: List[ValidationIssue]
    uncovered_examples: List[str]


@dataclass
class ValidationAuditReport:
    source: str
    primary_report: ValidationReport
    independent_report: ValidationReport
    methods_agree: bool
    agreement_issues: List[str]


def _issue(issues: List[ValidationIssue], severity: str, message: str) -> None:
    issues.append(ValidationIssue(severity=severity, message=message))


def _problem_config_from_result(data: Dict[str, object]) -> ProblemConfig:
    params = data.get("params")
    if not isinstance(params, dict):
        raise ValueError("Result JSON must contain a 'params' object.")

    if "samples" not in params or not isinstance(params.get("samples"), list):
        raise ValueError(
            "Result JSON params must explicitly contain a 'samples' list for validation."
        )
    if "coverage_mode" not in params:
        raise ValueError(
            "Result JSON params must explicitly contain 'coverage_mode' for validation."
        )
    if "aggregation_mode" not in params:
        raise ValueError(
            "Result JSON params must explicitly contain 'aggregation_mode' for validation."
        )

    samples = params["samples"]
    normalized_samples = tuple(int(sample) for sample in samples)

    return ProblemConfig(
        m=int(params["m"]),
        n=int(params["n"]),
        k=int(params["k"]),
        j=int(params["j"]),
        s=int(params["s"]),
        samples=normalized_samples,
        coverage_mode=CoverageMode(str(params["coverage_mode"])),
        aggregation_mode=AggregationMode(str(params["aggregation_mode"])),
        required_r=(
            int(params["required_r"])
            if params.get("required_r") is not None
            else None
        ),
        seed=(int(params["seed"]) if params.get("seed") is not None else None),
    )


def _parse_declared_size(data: Dict[str, object], issues: List[ValidationIssue]) -> Optional[int]:
    value = data.get("num_groups")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        _issue(issues, "error", f"num_groups is not an integer: {value!r}")
        return None


def _canonical_group(raw_group: object) -> Optional[Tuple[int, ...]]:
    if not isinstance(raw_group, (list, tuple)):
        return None
    try:
        return tuple(sorted(int(item) for item in raw_group))
    except (TypeError, ValueError):
        return None


def _indices_from_groups(
    instance: CoverageInstance,
    groups: Sequence[object],
    issues: List[ValidationIssue],
) -> List[int]:
    candidate_lookup = {
        instance.candidate_label(candidate_index): candidate_index
        for candidate_index in range(len(instance.candidates))
    }
    sample_set = set(instance.samples)
    seen_groups = set()
    indices: List[int] = []

    for group_number, raw_group in enumerate(groups, start=1):
        group = _canonical_group(raw_group)
        if group is None:
            _issue(issues, "error", f"Group {group_number} is not a valid integer list.")
            continue

        if len(group) != instance.k:
            _issue(
                issues,
                "error",
                f"Group {group_number} has size {len(group)}, expected {instance.k}: {group}",
            )
            continue

        if len(set(group)) != len(group):
            _issue(issues, "error", f"Group {group_number} contains duplicate samples: {group}")
            continue

        invalid_samples = [sample for sample in group if sample not in sample_set]
        if invalid_samples:
            _issue(
                issues,
                "error",
                f"Group {group_number} contains samples outside the instance: {invalid_samples}",
            )
            continue

        candidate_index = candidate_lookup.get(group)
        if candidate_index is None:
            _issue(issues, "error", f"Group {group_number} is not a valid candidate: {group}")
            continue

        if group in seen_groups:
            _issue(issues, "error", f"Group {group_number} is duplicated: {group}")
            continue

        seen_groups.add(group)
        indices.append(candidate_index)

    return indices


def _indices_from_solution_indices(
    instance: CoverageInstance,
    raw_indices: object,
    issues: List[ValidationIssue],
) -> List[int]:
    if not isinstance(raw_indices, list):
        _issue(issues, "error", "solution_indices must be a list when groups are absent.")
        return []

    seen_indices = set()
    indices: List[int] = []
    for position, raw_index in enumerate(raw_indices, start=1):
        try:
            candidate_index = int(raw_index)
        except (TypeError, ValueError):
            _issue(issues, "error", f"solution_indices[{position}] is not an integer.")
            continue

        if not 0 <= candidate_index < len(instance.candidates):
            _issue(
                issues,
                "error",
                f"solution_indices[{position}]={candidate_index} is out of range.",
            )
            continue

        if candidate_index in seen_indices:
            _issue(issues, "error", f"Duplicate candidate index: {candidate_index}")
            continue

        seen_indices.add(candidate_index)
        indices.append(candidate_index)

    return indices


def _extract_solution_indices(
    instance: CoverageInstance,
    data: Dict[str, object],
    issues: List[ValidationIssue],
) -> List[int]:
    groups = data.get("groups")
    if isinstance(groups, list):
        group_indices = _indices_from_groups(instance, groups, issues)
        raw_solution_indices = data.get("solution_indices")
        if isinstance(raw_solution_indices, list):
            declared_indices = _indices_from_solution_indices(
                instance, raw_solution_indices, issues
            )
            if set(group_indices) != set(declared_indices):
                _issue(
                    issues,
                    "error",
                    "groups and solution_indices describe different candidate sets.",
                )
        return group_indices

    raw_solution_indices = data.get("solution_indices")
    if raw_solution_indices is not None:
        return _indices_from_solution_indices(instance, raw_solution_indices, issues)

    _issue(issues, "error", "Result JSON must contain either 'groups' or 'solution_indices'.")
    return []


def _extract_solution_groups(
    instance: CoverageInstance,
    data: Dict[str, object],
    issues: List[ValidationIssue],
) -> List[Tuple[int, ...]]:
    return [
        instance.candidate_label(candidate_index)
        for candidate_index in _extract_solution_indices(instance, data, issues)
    ]


def _uncovered_examples(
    instance: CoverageInstance,
    tracker: CoverageTracker,
    max_examples: int,
) -> List[str]:
    examples = []
    for target_index, target in enumerate(instance.targets):
        deficit = tracker.target_deficit(target_index)
        if deficit <= 0:
            continue
        examples.append(f"{target} deficit={deficit}")
        if len(examples) >= max_examples:
            break
    return examples


def _coerce_solution_indices(
    instance: CoverageInstance,
    solution_indices: Sequence[int],
    issues: List[ValidationIssue],
) -> List[int]:
    seen_indices = set()
    normalized: List[int] = []

    for position, raw_index in enumerate(solution_indices, start=1):
        try:
            candidate_index = int(raw_index)
        except (TypeError, ValueError):
            _issue(issues, "error", f"solution_indices[{position}] is not an integer.")
            continue

        if not 0 <= candidate_index < len(instance.candidates):
            _issue(
                issues,
                "error",
                f"solution_indices[{position}]={candidate_index} is out of range.",
            )
            continue

        if candidate_index in seen_indices:
            _issue(issues, "error", f"Duplicate candidate index: {candidate_index}")
            continue

        seen_indices.add(candidate_index)
        normalized.append(candidate_index)

    return normalized


def validate_solution_indices(
    instance: CoverageInstance,
    solution_indices: Sequence[int],
    source: str = "<memory>",
    max_uncovered_examples: int = 10,
) -> ValidationReport:
    issues: List[ValidationIssue] = []
    normalized = _coerce_solution_indices(instance, solution_indices, issues)

    tracker = CoverageTracker(instance)
    tracker.reset(normalized)

    if not tracker.is_feasible():
        _issue(
            issues,
            "error",
            (
                f"Solution is infeasible: {tracker.unsatisfied_targets} targets remain "
                f"unsatisfied, deficit_units={tracker.deficit_units}."
            ),
        )

    has_errors = any(issue.severity == "error" for issue in issues)
    return ValidationReport(
        source=source,
        is_valid=not has_errors and tracker.is_feasible(),
        num_groups=len(normalized),
        declared_num_groups=None,
        unsatisfied_targets=tracker.unsatisfied_targets,
        deficit_units=tracker.deficit_units,
        issues=issues,
        uncovered_examples=_uncovered_examples(
            instance, tracker, max_uncovered_examples
        ),
    )


def validate_result_data(
    data: Dict[str, object],
    source: str = "<memory>",
    max_uncovered_examples: int = 10,
) -> ValidationReport:
    issues: List[ValidationIssue] = []
    declared_num_groups = _parse_declared_size(data, issues)

    try:
        config = _problem_config_from_result(data)
        instance = CoverageInstance(config)
    except Exception as exc:
        _issue(issues, "error", f"Could not rebuild problem instance: {exc}")
        return ValidationReport(
            source=source,
            is_valid=False,
            num_groups=0,
            declared_num_groups=declared_num_groups,
            unsatisfied_targets=-1,
            deficit_units=-1,
            issues=issues,
            uncovered_examples=[],
        )

    solution_indices = _extract_solution_indices(instance, data, issues)
    if declared_num_groups is not None and declared_num_groups != len(solution_indices):
        _issue(
            issues,
            "error",
            f"num_groups={declared_num_groups} but parsed {len(solution_indices)} groups.",
        )

    direct_report = validate_solution_indices(
        instance,
        solution_indices,
        source=source,
        max_uncovered_examples=max_uncovered_examples,
    )
    issues.extend(direct_report.issues)

    exact_size = data.get("exact_size")
    if exact_size is not None:
        try:
            exact_size_int = int(exact_size)
            if len(solution_indices) < exact_size_int:
                _issue(
                    issues,
                    "warning",
                    (
                        f"Parsed solution size {len(solution_indices)} is smaller than "
                        f"declared exact_size={exact_size_int}; check the saved metadata."
                    ),
                )
        except (TypeError, ValueError):
            _issue(issues, "warning", f"exact_size is not an integer: {exact_size!r}")

    has_errors = any(issue.severity == "error" for issue in issues)
    return ValidationReport(
        source=source,
        is_valid=not has_errors and direct_report.is_valid,
        num_groups=len(solution_indices),
        declared_num_groups=declared_num_groups,
        unsatisfied_targets=direct_report.unsatisfied_targets,
        deficit_units=direct_report.deficit_units,
        issues=issues,
        uncovered_examples=direct_report.uncovered_examples,
    )


def _independent_target_deficit(
    target: Tuple[int, ...],
    group_sets: Sequence[frozenset[int]],
    s: int,
    required_subset_count: int,
    aggregation_mode: AggregationMode,
) -> int:
    target_set = frozenset(target)

    if aggregation_mode == AggregationMode.SINGLE_CANDIDATE:
        best_subset_count = 0
        for group_set in group_sets:
            overlap = len(group_set & target_set)
            if overlap < s:
                continue
            best_subset_count = max(best_subset_count, math.comb(overlap, s))
            if best_subset_count >= required_subset_count:
                return 0
        return max(0, required_subset_count - best_subset_count)

    covered_subsets = set()
    for group_set in group_sets:
        overlap = sorted(group_set & target_set)
        if len(overlap) < s:
            continue
        covered_subsets.update(itertools.combinations(overlap, s))
        if len(covered_subsets) >= required_subset_count:
            return 0
    return max(0, required_subset_count - len(covered_subsets))


def validate_result_data_independently(
    data: Dict[str, object],
    source: str = "<memory>",
    max_uncovered_examples: int = 10,
) -> ValidationReport:
    issues: List[ValidationIssue] = []
    declared_num_groups = _parse_declared_size(data, issues)

    try:
        config = _problem_config_from_result(data)
        instance = CoverageInstance(config)
    except Exception as exc:
        _issue(issues, "error", f"Could not rebuild problem instance: {exc}")
        return ValidationReport(
            source=source,
            is_valid=False,
            num_groups=0,
            declared_num_groups=declared_num_groups,
            unsatisfied_targets=-1,
            deficit_units=-1,
            issues=issues,
            uncovered_examples=[],
        )

    solution_indices = _extract_solution_indices(instance, data, issues)
    if declared_num_groups is not None and declared_num_groups != len(solution_indices):
        _issue(
            issues,
            "error",
            f"num_groups={declared_num_groups} but parsed {len(solution_indices)} groups.",
        )

    direct_report = validate_solution_indices_independently(
        instance,
        solution_indices,
        source=source,
        max_uncovered_examples=max_uncovered_examples,
    )
    issues.extend(direct_report.issues)

    exact_size = data.get("exact_size")
    if exact_size is not None:
        try:
            exact_size_int = int(exact_size)
            if len(solution_indices) < exact_size_int:
                _issue(
                    issues,
                    "warning",
                    (
                        f"Parsed solution size {len(solution_indices)} is smaller than "
                        f"declared exact_size={exact_size_int}; check the saved metadata."
                    ),
                )
        except (TypeError, ValueError):
            _issue(issues, "warning", f"exact_size is not an integer: {exact_size!r}")

    has_errors = any(issue.severity == "error" for issue in issues)
    return ValidationReport(
        source=source,
        is_valid=not has_errors and direct_report.is_valid,
        num_groups=len(solution_indices),
        declared_num_groups=declared_num_groups,
        unsatisfied_targets=direct_report.unsatisfied_targets,
        deficit_units=direct_report.deficit_units,
        issues=issues,
        uncovered_examples=direct_report.uncovered_examples,
    )


def validate_solution_indices_independently(
    instance: CoverageInstance,
    solution_indices: Sequence[int],
    source: str = "<memory>",
    max_uncovered_examples: int = 10,
) -> ValidationReport:
    issues: List[ValidationIssue] = []
    normalized = _coerce_solution_indices(instance, solution_indices, issues)

    group_sets = [
        frozenset(instance.candidate_label(candidate_index))
        for candidate_index in normalized
    ]
    unsatisfied_targets = 0
    deficit_units = 0
    uncovered_examples: List[str] = []

    for target in instance.targets:
        deficit = _independent_target_deficit(
            target=target,
            group_sets=group_sets,
            s=instance.s,
            required_subset_count=instance.required_subset_count,
            aggregation_mode=instance.aggregation_mode,
        )
        if deficit <= 0:
            continue
        unsatisfied_targets += 1
        deficit_units += deficit
        if len(uncovered_examples) < max_uncovered_examples:
            uncovered_examples.append(f"{target} deficit={deficit}")

    if unsatisfied_targets > 0:
        _issue(
            issues,
            "error",
            (
                "Independent brute-force check found an infeasible solution: "
                f"{unsatisfied_targets} targets remain unsatisfied, "
                f"deficit_units={deficit_units}."
            ),
        )

    has_errors = any(issue.severity == "error" for issue in issues)
    return ValidationReport(
        source=source,
        is_valid=not has_errors and unsatisfied_targets == 0,
        num_groups=len(normalized),
        declared_num_groups=None,
        unsatisfied_targets=unsatisfied_targets,
        deficit_units=deficit_units,
        issues=issues,
        uncovered_examples=uncovered_examples,
    )


def audit_solution_indices(
    instance: CoverageInstance,
    solution_indices: Sequence[int],
    source: str = "<memory>",
    max_uncovered_examples: int = 10,
) -> ValidationAuditReport:
    primary_report = validate_solution_indices(
        instance,
        solution_indices,
        source=source,
        max_uncovered_examples=max_uncovered_examples,
    )
    independent_report = validate_solution_indices_independently(
        instance,
        solution_indices,
        source=source,
        max_uncovered_examples=max_uncovered_examples,
    )

    agreement_issues: List[str] = []
    if primary_report.num_groups != independent_report.num_groups:
        agreement_issues.append(
            "Parsed group counts differ between the primary and independent validators."
        )
    if primary_report.is_valid != independent_report.is_valid:
        agreement_issues.append(
            "Primary validator and independent validator disagree on feasibility."
        )
    if primary_report.unsatisfied_targets != independent_report.unsatisfied_targets:
        agreement_issues.append(
            "Primary validator and independent validator disagree on the number of unsatisfied targets."
        )

    return ValidationAuditReport(
        source=source,
        primary_report=primary_report,
        independent_report=independent_report,
        methods_agree=not agreement_issues,
        agreement_issues=agreement_issues,
    )


def validate_result_file(path: str | Path) -> ValidationReport:
    result_path = Path(path).resolve()
    with result_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Result file must contain a JSON object.")
    return validate_result_data(data, source=str(result_path))


def validate_result_file_independently(path: str | Path) -> ValidationReport:
    result_path = Path(path).resolve()
    with result_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Result file must contain a JSON object.")
    return validate_result_data_independently(data, source=str(result_path))


def validate_database(db_dir: str | Path) -> List[ValidationReport]:
    directory = Path(db_dir).resolve()
    reports = []
    for path in sorted(directory.glob("*.json")):
        reports.append(validate_result_file(path))
    return reports


def audit_result_data(
    data: Dict[str, object],
    source: str = "<memory>",
    max_uncovered_examples: int = 10,
) -> ValidationAuditReport:
    primary_report = validate_result_data(
        data,
        source=source,
        max_uncovered_examples=max_uncovered_examples,
    )
    independent_report = validate_result_data_independently(
        data,
        source=source,
        max_uncovered_examples=max_uncovered_examples,
    )
    return ValidationAuditReport(
        source=source,
        primary_report=primary_report,
        independent_report=independent_report,
        methods_agree=(
            primary_report.num_groups == independent_report.num_groups
            and primary_report.is_valid == independent_report.is_valid
            and (
                primary_report.unsatisfied_targets
                == independent_report.unsatisfied_targets
            )
        ),
        agreement_issues=[
            issue
            for issue in (
                "Parsed group counts differ between the primary and independent validators."
                if primary_report.num_groups != independent_report.num_groups
                else None,
                "Primary validator and independent validator disagree on feasibility."
                if primary_report.is_valid != independent_report.is_valid
                else None,
                "Primary validator and independent validator disagree on the number of unsatisfied targets."
                if (
                    primary_report.unsatisfied_targets
                    != independent_report.unsatisfied_targets
                )
                else None,
            )
            if issue is not None
        ],
    )


def audit_result_file(path: str | Path) -> ValidationAuditReport:
    result_path = Path(path).resolve()
    with result_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Result file must contain a JSON object.")
    return audit_result_data(data, source=str(result_path))


def audit_database(db_dir: str | Path) -> List[ValidationAuditReport]:
    directory = Path(db_dir).resolve()
    reports = []
    for path in sorted(directory.glob("*.json")):
        reports.append(audit_result_file(path))
    return reports


def format_validation_report(report: ValidationReport) -> str:
    status = "VALID" if report.is_valid else "INVALID"
    lines = [
        f"{report.source}: {status}",
        (
            f"  groups={report.num_groups}"
            + (
                f", declared_num_groups={report.declared_num_groups}"
                if report.declared_num_groups is not None
                else ""
            )
        ),
        (
            f"  unsatisfied_targets={report.unsatisfied_targets}, "
            f"deficit_units={report.deficit_units}"
        ),
    ]

    if report.issues:
        lines.append("  issues:")
        for issue in report.issues:
            lines.append(f"    [{issue.severity}] {issue.message}")

    if report.uncovered_examples:
        lines.append("  uncovered examples:")
        for example in report.uncovered_examples:
            lines.append(f"    {example}")

    return "\n".join(lines)


def format_validation_audit_report(report: ValidationAuditReport) -> str:
    lines = [
        f"Audit source: {report.source}",
        "[Primary validator]",
        format_validation_report(report.primary_report),
        "[Independent validator]",
        format_validation_report(report.independent_report),
        (
            "Agreement: OK"
            if report.methods_agree
            else "Agreement: MISMATCH"
        ),
    ]
    if report.agreement_issues:
        lines.append("  agreement issues:")
        for issue in report.agreement_issues:
            lines.append(f"    {issue}")
    return "\n".join(lines)
