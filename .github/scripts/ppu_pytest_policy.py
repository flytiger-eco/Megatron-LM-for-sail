"""PPU UT."""

import json
import os
from pathlib import Path

import pytest

_tests = {}
_selected = set()
_finished = set()
_collection_errors = []


def pytest_configure(config):
    _tests.clear()
    _selected.clear()
    _finished.clear()
    _collection_errors.clear()


def pytest_collection_finish(session):
    _selected.update(item.nodeid for item in session.items)


def pytest_collectreport(report):
    if report.failed:
        _collection_errors.append(report.nodeid)


def pytest_runtest_logreport(report):
    if report.failed:
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
        and exitstatus in (0, 1, 5)
        and _selected == set(_tests)
        and _selected <= _finished
        and (exitstatus != 1 or "failed" in _tests.values())
    )
    rank = int(os.environ.get("RANK", "0"))
    result = {
        "rank": rank,
        "exitstatus": int(exitstatus),
        "complete": complete,
        "collection_errors": _collection_errors,
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
