"""PPU UT."""

import json
import os
import time
from pathlib import Path

import pytest

_tests = {}
_selected = set()
_finished = set()
_collection_errors = []
_execution_errors = []
_collected = set()
_durations = {}
_started = 0.0
_collection_seconds = 0.0


def pytest_addoption(parser):
    parser.addoption(
        "--ppu-exclude", action="append", default=[], help="Exclude an exact UT selector"
    )


def pytest_collection_modifyitems(config, items):
    excluded = config.getoption("--ppu-exclude")
    deselected, kept = [], []
    for item in items:
        if any(
            item.nodeid == selector or item.nodeid.startswith((selector + "::", selector + "["))
            for selector in excluded
        ):
            deselected.append(item)
        else:
            kept.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


def pytest_configure(config):
    global _started, _collection_seconds
    _started = time.monotonic()
    _collection_seconds = 0.0
    _collected.clear()
    _durations.clear()
    _tests.clear()
    _selected.clear()
    _finished.clear()
    _collection_errors.clear()
    _execution_errors.clear()


def pytest_itemcollected(item):
    if item.nodeid in _collected:
        _collection_errors.append(f"Duplicate nodeid: {item.nodeid}")
    _collected.add(item.nodeid)


def pytest_collection_finish(session):
    global _collection_seconds
    _collection_seconds = time.monotonic() - _started
    _selected.update(item.nodeid for item in session.items)


def pytest_collectreport(report):
    if report.failed:
        _collection_errors.append(report.nodeid)


def pytest_runtest_logreport(report):
    _durations[report.nodeid] = _durations.get(report.nodeid, 0.0) + report.duration
    if report.failed:
        if report.when != "call":
            _execution_errors.append(f"{report.when}: {report.nodeid}")
        _tests[report.nodeid] = "failed"
    elif _tests.get(report.nodeid) != "failed":
        if report.skipped:
            _tests[report.nodeid] = "skipped"
        elif report.when == "call" and report.passed:
            _tests[report.nodeid] = "passed"
    if report.when == "teardown":
        _finished.add(report.nodeid)


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    yield
    complete = (
        not _collection_errors
        and not _execution_errors
        and exitstatus in (0, 1, 5)
        and _selected == set(_tests)
        and _selected <= _finished
        and (exitstatus != 1 or "failed" in _tests.values())
    )
    rank = int(os.environ.get("RANK", "0"))
    result = {
        "rank": rank,
        "case_id": os.environ["PPU_CASE_ID"],
        "phase": os.environ["PPU_TEST_PHASE"],
        "iteration": int(os.environ["PPU_TEST_ITERATION"]),
        "target_sha": os.environ.get("PPU_TARGET_SHA", ""),
        "control_sha": os.environ.get("PPU_CONTROL_SHA", ""),
        "duration_seconds": time.monotonic() - _started,
        "collection_seconds": _collection_seconds,
        "durations": _durations,
        "collected": sorted(_collected),
        "selected": sorted(_selected),
        "exitstatus": int(exitstatus),
        "complete": complete,
        "collection_errors": _collection_errors,
        "execution_errors": _execution_errors,
        "tests": _tests,
    }
    directory = Path(os.environ["PPU_REPORT_DIR"])
    directory.mkdir(parents=True, exist_ok=True)
    path = (
        directory
        / f"{os.environ['PPU_TEST_PHASE']}-{os.environ['PPU_TEST_ITERATION']}-rank-{rank}.json"
    )
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result), encoding="utf-8")
    temporary.replace(path)
    if complete:
        session.exitstatus = pytest.ExitCode.OK
    else:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
