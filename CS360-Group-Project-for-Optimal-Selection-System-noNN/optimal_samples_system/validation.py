"""Validation utilities for saved covering design solutions."""

from __future__ import annotations

import json
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


def _issue(issues: List[ValidationIssue], severity: str, message: str) -> None:
    issues.append(ValidationIssue(severity=severity, message=message))


def _problem_config_from_result(data: Dict[str, object]) -> ProblemConfig:
    params = data.get("params")
    if not isinstance(params, dict):
        raise ValueError("Result JSON must contain a 'params' object.")

    samples = params.get("samples")
    normalized_samples = (
        tuple(int(sample) for sample in samples)
        if isinstance(samples, list)
        else None
    )

    return ProblemConfig(
        m=int(params["m"]),
        n=int(params["n"]),
        k=int(params["k"]),
        j=int(params["j"]),
        s=int(params["s"]),
        samples=normalized_samples,
        coverage_mode=CoverageMode(str(params.get("coverage_mode", CoverageMode.AT_LEAST_ONE.value))),
        aggregation_mode=AggregationMode(
            str(params.get("aggregation_mode", AggregationMode.DISTINCT_SUBSETS.value))
        ),
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

    tracker = CoverageTracker(instance)
    tracker.reset(solution_indices)

    if not tracker.is_feasible():
        _issue(
            issues,
            "error",
            (
                f"Solution is infeasible: {tracker.unsatisfied_targets} targets remain "
                f"unsatisfied, deficit_units={tracker.deficit_units}."
            ),
        )

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
        is_valid=not has_errors and tracker.is_feasible(),
        num_groups=len(solution_indices),
        declared_num_groups=declared_num_groups,
        unsatisfied_targets=tracker.unsatisfied_targets,
        deficit_units=tracker.deficit_units,
        issues=issues,
        uncovered_examples=_uncovered_examples(
            instance, tracker, max_uncovered_examples
        ),
    )


def validate_result_file(path: str | Path) -> ValidationReport:
    result_path = Path(path).resolve()
    with result_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Result file must contain a JSON object.")
    return validate_result_data(data, source=str(result_path))


def validate_database(db_dir: str | Path) -> List[ValidationReport]:
    directory = Path(db_dir).resolve()
    reports = []
    for path in sorted(directory.glob("*.json")):
        reports.append(validate_result_file(path))
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
