"""Collect unmodified upstream functional pytest outcomes for the CI integrity gate."""

import json
import os
import time
from pathlib import Path

_REPORT = None


def pytest_configure(config):
    global _REPORT
    names = {Path(str(argument)).name for argument in config.args}
    phases = {
        "test_pretraining_regular_pipeline.py": "regular",
        "test_pretraining_resume_checkpoint_pipeline.py": "resume",
    }
    matches = [phase for filename, phase in phases.items() if filename in names]
    if len(matches) != 1:
        raise ValueError("Functional evidence plugin requires one upstream pipeline")
    _REPORT = {
        "schema_version": 1,
        "phase": matches[0],
        "started": time.monotonic(),
        "identity": json.loads(os.environ["PPU_FUNCTIONAL_IDENTITY"]),
        "case_id": os.environ["PPU_FUNCTIONAL_CASE_ID"],
        "attempt": int(os.environ["PPU_FUNCTIONAL_ATTEMPT"]),
        "repeat": int(os.environ["PPU_FUNCTIONAL_REPEAT"]),
        "manifest_sha256": os.environ["PPU_FUNCTIONAL_MANIFEST_SHA256"],
        "approximate": bool(config.getoption("--allow-nondeterministic-algo", default=False)),
        "collected": [],
        "tests": {},
        "collection_errors": [],
        "complete": False,
    }


def pytest_collection_finish(session):
    _REPORT["collected"] = [item.nodeid for item in session.items]


def pytest_collectreport(report):
    if report.failed:
        _REPORT["collection_errors"].append(str(report.longrepr))


def pytest_runtest_logreport(report):
    _REPORT["tests"].setdefault(report.nodeid, {})[report.when] = report.outcome


def pytest_sessionfinish(session, exitstatus):
    _REPORT.update(
        complete=True,
        exitstatus=int(exitstatus),
        duration_seconds=time.monotonic() - _REPORT.pop("started"),
    )
    path = Path(os.environ["PPU_FUNCTIONAL_REPEAT_DIR"]) / f"pytest-{_REPORT['phase']}.json"
    with path.open("x") as stream:
        json.dump(_REPORT, stream, indent=2)
        stream.write("\n")
