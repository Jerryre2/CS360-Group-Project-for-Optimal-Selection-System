"""Optimality certification helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .exact import ILPSolver
from .instance import CoverageInstance
from .validation import (
    ValidationAuditReport,
    _extract_solution_indices,
    _problem_config_from_result,
    audit_result_data,
)


@dataclass
class OptimalityCertificate:
    source: str
    incumbent_size: int
    proof_target: int
    status: str
    method: Optional[str]
    message: str
    audit: ValidationAuditReport
    better_solution_size: Optional[int]
    certified_optimal: bool
    certified_lower_bound: Optional[int]
    relaxation_value: Optional[float]


@dataclass
class OptimalitySummary:
    source: str
    status_code: str
    status_label: str
    incumbent_size: int
    feasibility_label: str
    certified_optimal: bool
    certified_lower_bound: Optional[int]
    gap_upper_bound: Optional[int]
    better_solution_size: Optional[int]
    method: Optional[str]


def certify_result_data(
    data: Dict[str, object],
    source: str = "<memory>",
    backend: str = "auto",
    time_limit: int = 300,
) -> OptimalityCertificate:
    audit = audit_result_data(data, source=source)
    incumbent_size = audit.primary_report.num_groups
    proof_target = incumbent_size - 1

    if (
        not audit.primary_report.is_valid
        or not audit.independent_report.is_valid
        or not audit.methods_agree
    ):
        return OptimalityCertificate(
            source=source,
            incumbent_size=incumbent_size,
            proof_target=proof_target,
            status="invalid_incumbent",
            method=None,
            message=(
                "The incumbent solution failed validation or the validators disagree, "
                "so optimality cannot be certified."
            ),
            audit=audit,
            better_solution_size=None,
            certified_optimal=False,
            certified_lower_bound=None,
            relaxation_value=None,
        )

    config = _problem_config_from_result(data)
    instance = CoverageInstance(config)
    issues = []
    incumbent = _extract_solution_indices(instance, data, issues)
    if issues:
        return OptimalityCertificate(
            source=source,
            incumbent_size=incumbent_size,
            proof_target=proof_target,
            status="invalid_incumbent",
            method=None,
            message="Could not reconstruct incumbent indices for exact certification.",
            audit=audit,
            better_solution_size=None,
            certified_optimal=False,
            certified_lower_bound=None,
            relaxation_value=None,
        )

    if proof_target < 0:
        return OptimalityCertificate(
            source=source,
            incumbent_size=incumbent_size,
            proof_target=proof_target,
            status="certified_optimal",
            method="trivial",
            message="The incumbent has size 0, so no smaller feasible solution exists.",
            audit=audit,
            better_solution_size=None,
            certified_optimal=True,
            certified_lower_bound=incumbent_size,
            relaxation_value=float(incumbent_size),
        )

    result = ILPSolver.prove_no_solution_at_or_below(
        instance=instance,
        cardinality_limit=proof_target,
        backend=backend,
        time_limit=time_limit,
        incumbent=incumbent,
    )

    if result.status == "infeasible":
        return OptimalityCertificate(
            source=source,
            incumbent_size=incumbent_size,
            proof_target=proof_target,
            status="certified_optimal",
            method=result.method,
            message=(
                f"No feasible solution exists with <= {proof_target} groups. "
                f"The incumbent size {incumbent_size} is globally optimal."
            ),
            audit=audit,
            better_solution_size=None,
            certified_optimal=True,
            certified_lower_bound=incumbent_size,
            relaxation_value=float(incumbent_size),
        )

    if result.status == "feasible":
        better_solution_size = (
            len(result.solution) if result.solution is not None else None
        )
        if better_solution_size is not None and better_solution_size < incumbent_size:
            message = (
                f"Found a strictly better feasible solution of size {better_solution_size}; "
                f"the incumbent size {incumbent_size} is not optimal."
            )
        else:
            message = (
                f"A feasible solution exists at or below {proof_target}. "
                f"The incumbent size {incumbent_size} is not certified optimal."
            )
        return OptimalityCertificate(
            source=source,
            incumbent_size=incumbent_size,
            proof_target=proof_target,
            status="not_optimal",
            method=result.method,
            message=message,
            audit=audit,
            better_solution_size=better_solution_size,
            certified_optimal=False,
            certified_lower_bound=None,
            relaxation_value=None,
        )

    lower_bound = ILPSolver.lower_bound(instance, time_limit=max(30, min(time_limit, 300)))
    combinatorial_bound = ILPSolver.combinatorial_lower_bound(instance)
    certified_lower_bound = lower_bound.integer_bound
    lower_bound_method = lower_bound.method
    lower_bound_message = lower_bound.message
    relaxation_value = lower_bound.value

    if (
        combinatorial_bound.integer_bound is not None
        and (
            certified_lower_bound is None
            or combinatorial_bound.integer_bound > certified_lower_bound
        )
    ):
        certified_lower_bound = combinatorial_bound.integer_bound
        lower_bound_method = combinatorial_bound.method
        lower_bound_message = combinatorial_bound.message
        if lower_bound.method != combinatorial_bound.method:
            if lower_bound.status == "optimal" and lower_bound.integer_bound is not None:
                lower_bound_message = (
                    f"{combinatorial_bound.message}; "
                    f"scipy-lp={lower_bound.integer_bound}"
                )
            elif lower_bound.status not in {"unavailable"}:
                lower_bound_message = (
                    f"{combinatorial_bound.message}; "
                    f"scipy-lp status={lower_bound.status}"
                )

    if certified_lower_bound is not None:
        message = (
            f"Exact certification did not finish: {result.message}. "
            "Current incumbent is feasible, and a rigorous lower bound certifies "
            f"OPT >= {certified_lower_bound} ({lower_bound_method}: {lower_bound_message})."
        )
    else:
        message = (
            f"Exact certification did not finish: {result.message}. "
            "The incumbent remains feasible, but global optimality is unproven."
        )

    return OptimalityCertificate(
        source=source,
        incumbent_size=incumbent_size,
        proof_target=proof_target,
        status="unresolved",
        method=result.method,
        message=message,
        audit=audit,
        better_solution_size=None,
        certified_optimal=False,
        certified_lower_bound=certified_lower_bound,
        relaxation_value=relaxation_value,
    )


def summarize_optimality_certificate(
    certificate: OptimalityCertificate,
) -> OptimalitySummary:
    if certificate.status == "certified_optimal":
        status_label = "已证明最优"
    elif certificate.status == "not_optimal":
        status_label = "已证明非最优"
    elif certificate.status == "invalid_incumbent":
        status_label = "当前解无效"
    else:
        status_label = "未证明"

    if (
        certificate.audit.primary_report.is_valid
        and certificate.audit.independent_report.is_valid
        and certificate.audit.methods_agree
    ):
        feasibility_label = "双重验证通过"
    else:
        feasibility_label = "验证未通过"

    gap_upper_bound = None
    if certificate.certified_lower_bound is not None:
        gap_upper_bound = certificate.incumbent_size - certificate.certified_lower_bound

    return OptimalitySummary(
        source=certificate.source,
        status_code=certificate.status,
        status_label=status_label,
        incumbent_size=certificate.incumbent_size,
        feasibility_label=feasibility_label,
        certified_optimal=certificate.certified_optimal,
        certified_lower_bound=certificate.certified_lower_bound,
        gap_upper_bound=gap_upper_bound,
        better_solution_size=certificate.better_solution_size,
        method=certificate.method,
    )


def format_optimality_summary(certificate: OptimalityCertificate) -> str:
    summary = summarize_optimality_certificate(certificate)
    lines = [
        f"Optimality status: {summary.status_label}",
        f"Status code: {summary.status_code}",
        f"Incumbent size: {summary.incumbent_size}",
        f"Feasibility: {summary.feasibility_label}",
        f"Method: {summary.method}",
    ]
    if summary.better_solution_size is not None:
        lines.append(f"Better solution size: {summary.better_solution_size}")
    if summary.certified_lower_bound is not None:
        lines.append(f"Certified lower bound: {summary.certified_lower_bound}")
    if summary.gap_upper_bound is not None:
        lines.append(f"Certified gap upper bound: {summary.gap_upper_bound}")
    return "\n".join(lines)


def format_optimality_certificate(certificate: OptimalityCertificate) -> str:
    lines = [
        format_optimality_summary(certificate),
        "",
        f"Optimality certification source: {certificate.source}",
        f"Incumbent size: {certificate.incumbent_size}",
        f"Proof target: <= {certificate.proof_target}",
        f"Status: {certificate.status}",
        f"Method: {certificate.method}",
        f"Message: {certificate.message}",
    ]
    if certificate.better_solution_size is not None:
        lines.append(
            f"Better solution size found: {certificate.better_solution_size}"
        )
    if certificate.relaxation_value is not None:
        lines.append(f"LP relaxation value: {certificate.relaxation_value:.4f}")
    if certificate.certified_lower_bound is not None:
        lines.append(
            f"Certified lower bound on OPT: {certificate.certified_lower_bound}"
        )
        lines.append(
            f"Certified gap upper bound: {certificate.incumbent_size - certificate.certified_lower_bound}"
        )
    lines.append(
        "Certified optimal: yes" if certificate.certified_optimal else "Certified optimal: no"
    )
    lines.append("")
    lines.append("[Audit summary]")
    lines.append(
        "Primary validator: VALID"
        if certificate.audit.primary_report.is_valid
        else "Primary validator: INVALID"
    )
    lines.append(
        "Independent validator: VALID"
        if certificate.audit.independent_report.is_valid
        else "Independent validator: INVALID"
    )
    lines.append(
        "Validators agree: yes"
        if certificate.audit.methods_agree
        else "Validators agree: no"
    )
    if certificate.audit.agreement_issues:
        lines.append("Agreement issues:")
        for issue in certificate.audit.agreement_issues:
            lines.append(f"  - {issue}")
    return "\n".join(lines)
