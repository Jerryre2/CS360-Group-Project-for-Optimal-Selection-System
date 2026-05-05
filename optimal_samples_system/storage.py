"""Persistence for solver results."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import AggregationMode, CoverageMode, LOGGER


class ResultDatabase:
    def __init__(self, db_dir: str = "results_db_v3"):
        self.db_dir = Path(db_dir).resolve()
        self.db_dir.mkdir(parents=True, exist_ok=True)

    def _mode_token(self, params: Dict[str, object]) -> str:
        coverage_mode = str(params["coverage_mode"])
        required_r = params.get("required_r")
        if coverage_mode == CoverageMode.AT_LEAST_ONE.value:
            return "one"
        if coverage_mode == CoverageMode.ALL_SUBSETS.value:
            return "all"
        return f"r{required_r}"

    def _aggregation_token(self, params: Dict[str, object]) -> str:
        aggregation_mode = str(params["aggregation_mode"])
        if aggregation_mode == AggregationMode.DISTINCT_SUBSETS.value:
            return "cum"
        return "single"

    def _prefix(self, params: Dict[str, object]) -> str:
        return (
            f"{params['m']}-{params['n']}-{params['k']}-{params['j']}-{params['s']}-"
            f"{self._mode_token(params)}-{self._aggregation_token(params)}"
        )

    def _next_run(self, prefix: str) -> int:
        pattern = re.compile(r"run(\d+)")
        runs = []
        for path in self.db_dir.glob(f"{prefix}-run*-size*.json"):
            match = pattern.search(path.stem)
            if match:
                runs.append(int(match.group(1)))
        return max(runs, default=0) + 1

    def _resolve_filename(self, filename: str) -> Path:
        path = (self.db_dir / filename).resolve()
        if self.db_dir not in path.parents and path != self.db_dir:
            raise ValueError("Invalid database filename.")
        return path

    def _validation_snapshot(self, data: Dict[str, object]) -> Dict[str, object]:
        validation = data.get("validation")
        if not isinstance(validation, dict):
            return {
                "status": "unverified",
                "summary": "validation=unverified",
            }

        primary_valid = bool(validation.get("primary_valid"))
        independent_valid = bool(validation.get("independent_valid"))
        methods_agree = bool(validation.get("methods_agree"))
        unsatisfied_targets = validation.get("unsatisfied_targets")
        deficit_units = validation.get("deficit_units")

        if not methods_agree:
            status = "mismatch"
        elif primary_valid and independent_valid:
            status = "validated"
        else:
            status = "invalid"

        tail_parts = []
        if unsatisfied_targets is not None:
            tail_parts.append(f"unsat={unsatisfied_targets}")
        if deficit_units is not None:
            tail_parts.append(f"deficit={deficit_units}")
        tail = f" ({', '.join(tail_parts)})" if tail_parts else ""

        return {
            "status": status,
            "primary_valid": primary_valid,
            "independent_valid": independent_valid,
            "methods_agree": methods_agree,
            "unsatisfied_targets": unsatisfied_targets,
            "deficit_units": deficit_units,
            "summary": (
                f"validation={status}, primary={primary_valid}, "
                f"independent={independent_valid}, agree={methods_agree}{tail}"
            ),
        }

    def _result_header_lines(self, filename: str, data: Dict[str, object]) -> List[str]:
        validation = self._validation_snapshot(data)
        exact_size = data.get("exact_size")
        lines = [
            f"File: {filename}",
            (
                f"Size: {data.get('num_groups')} | "
                f"{data.get('coverage_mode') or data.get('params', {}).get('coverage_mode')} / "
                f"{data.get('aggregation_mode') or data.get('params', {}).get('aggregation_mode')}"
                + (f" | exact={exact_size}" if exact_size is not None else "")
            ),
            f"Validation: {validation['summary']}",
        ]
        timestamp = data.get("timestamp")
        if timestamp is not None:
            lines.append(f"Timestamp: {timestamp}")
        return lines

    def save(self, result: Dict[str, object]) -> Path:
        params = dict(result["params"])
        prefix = self._prefix(params)
        run_number = self._next_run(prefix)
        filename = f"{prefix}-run{run_number:03d}-size{result['num_groups']}.json"
        path = self.db_dir / filename

        serializable = dict(result)
        serializable["run_number"] = run_number
        serializable["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

        with path.open("w", encoding="utf-8") as handle:
            json.dump(serializable, handle, indent=2, ensure_ascii=False)

        LOGGER.info(f"Saved result: {path.name}")
        return path

    def list_all(self) -> List[Dict[str, object]]:
        items = []
        for path in sorted(self.db_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            items.append(
                {
                    "filename": path.name,
                    "num_groups": data.get("num_groups"),
                    "coverage_mode": data.get("coverage_mode")
                    or data.get("params", {}).get("coverage_mode"),
                    "aggregation_mode": data.get("aggregation_mode")
                    or data.get("params", {}).get("aggregation_mode"),
                    "timestamp": data.get("timestamp"),
                    "exact_size": data.get("exact_size"),
                    "validation_status": self._validation_snapshot(data)["status"],
                }
            )
        return items

    def print_all(self) -> None:
        items = self.list_all()
        if not items:
            LOGGER.info("No saved results.")
            return

        for item in items:
            exact = (
                f", exact={item['exact_size']}"
                if item.get("exact_size") is not None
                else ""
            )
            LOGGER.info(
                f"{item['filename']} | size={item['num_groups']} | "
                f"{item['coverage_mode']} / {item['aggregation_mode']}{exact} | "
                f"validation={item['validation_status']} | "
                f"{item['timestamp']}"
            )

    def load(self, filename: str) -> Dict[str, object]:
        path = self._resolve_filename(filename)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def delete(self, filename: str) -> None:
        path = self._resolve_filename(filename)
        if not path.exists():
            raise FileNotFoundError(f"No saved result named {filename}.")
        path.unlink()
        LOGGER.info(f"Deleted result: {filename}")

    def print_result(self, filename: str) -> None:
        data = self.load(filename)
        for line in self._result_header_lines(filename, data):
            LOGGER.info(line)
        LOGGER.info(json.dumps(data, indent=2, ensure_ascii=False))
