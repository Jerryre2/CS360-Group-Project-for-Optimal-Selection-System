"""Lightweight HTTP bridge for the mobile app."""

from __future__ import annotations

import argparse
import io
import json
import logging
import threading
from dataclasses import asdict, is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, unquote, urlparse

from .config import (
    AggregationMode,
    CoverageMode,
    ProblemConfig,
    SolverConfig,
    configure_logging,
    parse_samples_arg,
)
from .certify import (
    certify_result_data,
    format_optimality_certificate,
    format_optimality_summary,
    summarize_optimality_certificate,
)
from .exact import ILPSolver
from .solver import OptimalSamplesSolver
from .storage import ResultDatabase
from .validation import (
    audit_database,
    audit_result_data,
    audit_result_file,
    format_validation_audit_report,
    ValidationIssue,
    ValidationReport,
    format_validation_report,
    validate_database,
    validate_result_data,
    validate_result_file,
)


def _enum_value(value: str, enum_type: type[CoverageMode] | type[AggregationMode]):
    return enum_type(value)


def _problem_config_from_payload(payload: Dict[str, Any]) -> ProblemConfig:
    samples_value = payload.get("samples")
    if isinstance(samples_value, list):
        samples = tuple(int(item) for item in samples_value)
    elif isinstance(samples_value, str):
        samples = parse_samples_arg(samples_value)
    else:
        samples = None

    return ProblemConfig(
        m=int(payload["m"]),
        n=int(payload["n"]),
        k=int(payload["k"]),
        j=int(payload["j"]),
        s=int(payload["s"]),
        samples=samples,
        coverage_mode=_enum_value(
            str(payload.get("coverage_mode", CoverageMode.AT_LEAST_ONE.value)),
            CoverageMode,
        ),
        aggregation_mode=_enum_value(
            str(
                payload.get(
                    "aggregation_mode", AggregationMode.DISTINCT_SUBSETS.value
                )
            ),
            AggregationMode,
        ),
        required_r=(
            int(payload["required_r"])
            if payload.get("required_r") is not None
            else None
        ),
        seed=int(payload["seed"]) if payload.get("seed") is not None else None,
    )


def _solver_config_from_payload(payload: Dict[str, Any]) -> SolverConfig:
    return SolverConfig(
        n_restarts=int(payload.get("n_restarts", 5)),
        use_ilp=bool(payload.get("use_ilp", True)),
        exact_backend=str(payload.get("exact_backend", "auto")),
        exact_time_limit=int(payload.get("exact_time_limit", 60)),
        force_exact=bool(payload.get("force_exact", False)),
        max_local_steps=(
            int(payload["max_local_steps"])
            if payload.get("max_local_steps") is not None
            else None
        ),
        max_sa_iterations=(
            int(payload["max_sa_iterations"])
            if payload.get("max_sa_iterations") is not None
            else None
        ),
        candidate_sample_size=int(payload.get("candidate_sample_size", 48)),
        adaptive_neighborhoods=bool(payload.get("adaptive_neighborhoods", True)),
        reduced_exact_polish=bool(payload.get("reduced_exact_polish", True)),
        reduced_exact_time_limit=int(payload.get("reduced_exact_time_limit", 8)),
        reduced_exact_core_cap=int(payload.get("reduced_exact_core_cap", 256)),
        save_result=bool(payload.get("save_result", False)),
        db_dir=str(payload.get("db_dir", "results_db_v3")),
    )


class _LogCapture:
    def __init__(self) -> None:
        self.buffer = io.StringIO()
        self.handler = logging.StreamHandler(self.buffer)
        self.handler.setFormatter(logging.Formatter("%(message)s"))

    def __enter__(self) -> io.StringIO:
        root = logging.getLogger("optimal_samples")
        root.addHandler(self.handler)
        root.setLevel(logging.INFO)
        return self.buffer

    def __exit__(self, exc_type, exc, tb) -> None:
        logging.getLogger("optimal_samples").removeHandler(self.handler)


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, ValidationIssue):
        return {"severity": value.severity, "message": value.message}
    if isinstance(value, ValidationReport):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class MobileAPIHandler(BaseHTTPRequestHandler):
    server_version = "OptimalSamplesMobileAPI/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        try:
            if parsed.path == "/health":
                self._send_json(
                    {
                        "status": "ok",
                        "service": "optimal-samples-mobile-api",
                    }
                )
                return

            if parsed.path == "/results":
                db_dir = query.get("db_dir", ["results_db_v3"])[0]
                items = ResultDatabase(db_dir).list_all()
                self._send_json({"items": items})
                return

            if parsed.path.startswith("/results/"):
                filename = unquote(parsed.path.split("/", 2)[2])
                db_dir = query.get("db_dir", ["results_db_v3"])[0]
                data = ResultDatabase(db_dir).load(filename)
                self._send_json({"result": data})
                return

            self._send_error_json(HTTPStatus.NOT_FOUND, "Endpoint not found.")
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self._read_json_body()
            if self.path == "/solve":
                self._handle_solve(payload)
                return
            if self.path == "/prove-bound":
                self._handle_prove_bound(payload)
                return
            if self.path == "/validate-result-data":
                report = validate_result_data(
                    payload["result"], source=payload.get("source", "<mobile>")
                )
                self._send_json(
                    {
                        "report": asdict(report),
                        "formatted": format_validation_report(report),
                    }
                )
                return
            if self.path == "/audit-result-data":
                report = audit_result_data(
                    payload["result"], source=payload.get("source", "<mobile>")
                )
                self._send_json(
                    {
                        "report": asdict(report),
                        "formatted": format_validation_audit_report(report),
                    }
                )
                return
            if self.path == "/validate-file":
                report = validate_result_file(str(payload["path"]))
                self._send_json(
                    {
                        "report": asdict(report),
                        "formatted": format_validation_report(report),
                    }
                )
                return
            if self.path == "/audit-file":
                report = audit_result_file(str(payload["path"]))
                self._send_json(
                    {
                        "report": asdict(report),
                        "formatted": format_validation_audit_report(report),
                    }
                )
                return
            if self.path == "/validate-results":
                db_dir = str(payload.get("db_dir", "results_db_v3"))
                reports = validate_database(db_dir)
                self._send_json(
                    {
                        "reports": [asdict(report) for report in reports],
                        "formatted": [format_validation_report(report) for report in reports],
                    }
                )
                return
            if self.path == "/audit-results":
                db_dir = str(payload.get("db_dir", "results_db_v3"))
                reports = audit_database(db_dir)
                self._send_json(
                    {
                        "reports": [asdict(report) for report in reports],
                        "formatted": [format_validation_audit_report(report) for report in reports],
                    }
                )
                return
            if self.path == "/audit-result":
                db_dir = str(payload.get("db_dir", "results_db_v3"))
                filename = str(payload["filename"])
                data = ResultDatabase(db_dir).load(filename)
                report = audit_result_data(data, source=filename)
                self._send_json(
                    {
                        "report": asdict(report),
                        "formatted": format_validation_audit_report(report),
                    }
                )
                return
            if self.path == "/certify-result-data":
                report = certify_result_data(
                    payload["result"],
                    source=payload.get("source", "<mobile>"),
                    backend=str(payload.get("exact_backend", "auto")),
                    time_limit=int(payload.get("exact_time_limit", 300)),
                )
                self._send_json(
                    {
                        "report": asdict(report),
                        "summary": asdict(summarize_optimality_certificate(report)),
                        "summary_text": format_optimality_summary(report),
                        "formatted": format_optimality_certificate(report),
                    }
                )
                return
            if self.path == "/certify-file":
                result_path = Path(str(payload["path"])).resolve()
                with result_path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if not isinstance(data, dict):
                    raise ValueError("Result file must contain a JSON object.")
                report = certify_result_data(
                    data,
                    source=str(result_path),
                    backend=str(payload.get("exact_backend", "auto")),
                    time_limit=int(payload.get("exact_time_limit", 300)),
                )
                self._send_json(
                    {
                        "report": asdict(report),
                        "summary": asdict(summarize_optimality_certificate(report)),
                        "summary_text": format_optimality_summary(report),
                        "formatted": format_optimality_certificate(report),
                    }
                )
                return
            if self.path == "/certify-result":
                db_dir = str(payload.get("db_dir", "results_db_v3"))
                filename = str(payload["filename"])
                data = ResultDatabase(db_dir).load(filename)
                report = certify_result_data(
                    data,
                    source=filename,
                    backend=str(payload.get("exact_backend", "auto")),
                    time_limit=int(payload.get("exact_time_limit", 300)),
                )
                self._send_json(
                    {
                        "report": asdict(report),
                        "summary": asdict(summarize_optimality_certificate(report)),
                        "summary_text": format_optimality_summary(report),
                        "formatted": format_optimality_certificate(report),
                    }
                )
                return
            if self.path == "/demo":
                examples = demo_payloads()
                self._send_json({"examples": examples})
                return

            self._send_error_json(HTTPStatus.NOT_FOUND, "Endpoint not found.")
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        try:
            if parsed.path.startswith("/results/"):
                filename = unquote(parsed.path.split("/", 2)[2])
                db_dir = query.get("db_dir", ["results_db_v3"])[0]
                ResultDatabase(db_dir).delete(filename)
                self._send_json({"deleted": filename})
                return

            self._send_error_json(HTTPStatus.NOT_FOUND, "Endpoint not found.")
        except Exception as exc:  # noqa: BLE001
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _read_json_body(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw_body = self.rfile.read(content_length)
        decoded = json.loads(raw_body.decode("utf-8"))
        if not isinstance(decoded, dict):
            raise ValueError("Request body must be a JSON object.")
        return decoded

    def _handle_solve(self, payload: Dict[str, Any]) -> None:
        problem = _problem_config_from_payload(payload["problem"])
        solver_config = _solver_config_from_payload(payload.get("solver", {}))

        with _LogCapture() as log_buffer:
            solver = OptimalSamplesSolver(problem)
            result = solver.solve(solver_config)

        self._send_json(
            {
                "result": result.to_dict(),
                "logs": log_buffer.getvalue(),
            }
        )

    def _handle_prove_bound(self, payload: Dict[str, Any]) -> None:
        problem = _problem_config_from_payload(payload["problem"])
        solver = OptimalSamplesSolver(problem)
        exact_backend = str(payload.get("exact_backend", "auto"))
        exact_time_limit = int(payload.get("exact_time_limit", 300))
        target_size = int(payload["target_size"])

        with _LogCapture() as log_buffer:
            proof = ILPSolver.prove_no_solution_at_or_below(
                solver.instance,
                cardinality_limit=target_size,
                backend=exact_backend,
                time_limit=exact_time_limit,
            )

        self._send_json(
            {
                "proof": asdict(proof),
                "logs": log_buffer.getvalue(),
            }
        )

    def _send_json(self, payload: Dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, default=_json_default).encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(encoded)

    def _send_error_json(self, status: HTTPStatus, message: str) -> None:
        self._send_json({"error": message}, status=status)


def demo_payloads() -> list[dict[str, Any]]:
    return [
        {
            "title": "Example 5",
            "problem": {
                "m": 45,
                "n": 8,
                "k": 6,
                "j": 6,
                "s": 5,
                "samples": list(range(1, 9)),
                "coverage_mode": "at_least_one",
                "aggregation_mode": "distinct_subsets",
                "seed": 42,
            },
        },
        {
            "title": "n=15 Benchmark",
            "problem": {
                "m": 45,
                "n": 15,
                "k": 6,
                "j": 5,
                "s": 4,
                "samples": list(range(1, 16)),
                "coverage_mode": "at_least_one",
                "aggregation_mode": "distinct_subsets",
                "seed": 42,
            },
        },
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optimal Samples mobile API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level: DEBUG, INFO, WARNING, ERROR",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    server = ThreadingHTTPServer((args.host, args.port), MobileAPIHandler)
    stop_event = threading.Event()

    try:
        logging.getLogger("optimal_samples").info(
            f"Mobile API listening on http://{args.host}:{args.port}"
        )
        while not stop_event.is_set():
            server.handle_request()
    except KeyboardInterrupt:
        logging.getLogger("optimal_samples").info("Mobile API server stopped.")


if __name__ == "__main__":
    main()
