"""PPU PR."""

import argparse
import importlib.util
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

BRANCH = "refs/heads/ci/ppu-validation"
STATUS_CONTEXT = "PPU PR Quick"


def github_api(path: str, payload: dict | None = None) -> dict:
    request = urllib.request.Request(
        f"https://api.github.com/repos/{os.environ['GITHUB_REPOSITORY']}/{path}",
        data=None if payload is None else json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def full_sha(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise ValueError("A full 40-character commit SHA is required")
    return value.lower()


def docs_only(number: int, pr: dict, api=github_api) -> bool:
    count = pr.get("changed_files")
    if type(count) is not int or not 0 < count < 3000:
        return False
    files = []
    try:
        for page in range(1, (count + 99) // 100 + 1):
            rows = api(f"pulls/{number}/files?per_page=100&page={page}")
            if not isinstance(rows, list):
                return False
            files.extend(rows)
        if len(files) != count or len({row["filename"] for row in files}) != count:
            return False
        for row in files:
            paths = [row["filename"]]
            if row.get("status") == "renamed":
                paths.append(row["previous_filename"])
            if any(not (p.startswith("docs/") or p.endswith(".md")) for p in paths):
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def snapshot(pr: dict) -> tuple:
    return full_sha(pr["head"]["sha"]), full_sha(pr["base"]["sha"])


class StaleSnapshot(ValueError):
    """The event no longer describes the current PR."""


def pr_identity(pr: dict) -> tuple:
    return (
        *snapshot(pr),
        pr["base"]["repo"]["full_name"],
        pr["base"]["ref"],
        (pr["head"].get("repo") or {}).get("full_name"),
        pr["state"],
        bool(pr.get("draft")),
    )


def current_pr(pr: dict, authorization: dict, repository: str) -> bool:
    return (
        full_sha(pr["head"]["sha"]) == authorization["pr_head_sha"]
        and pr["base"]["repo"]["full_name"] == repository
        and pr["base"]["ref"] == BRANCH.removeprefix("refs/heads/")
        and bool(pr["head"].get("repo"))
        and pr["head"]["repo"]["full_name"]
        == authorization.get("head_repository", pr["head"]["repo"]["full_name"])
    )


def current_base_sha(api) -> str:
    branch = BRANCH.removeprefix("refs/heads/")
    return full_sha(api(f"git/ref/heads/{branch}")["object"]["sha"])


def base_is_current(base_sha: str, api) -> bool:
    return current_base_sha(api) == base_sha


def blocked(result: dict, decision: str, reason: str) -> dict:
    return {**result, "allowed": False, "decision": decision, "reason": reason}


def authorize(event: dict, inputs: dict, context: dict, api=github_api, sleep=time.sleep) -> dict:
    repository = context["repository"]
    event_name, ref = context["event_name"], context["ref"]
    spec = importlib.util.spec_from_file_location(
        "ppu_functional_route", Path(__file__).with_name("ppu_functional.py")
    )
    functional = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(functional)
    test_kind = functional.validate_route(event_name, ref, inputs)
    if test_kind in ("functional", "compiler_diagnostic"):
        if test_kind == "functional":
            functional.select_config(inputs)
        elif str(context["run_attempt"]) != "1":
            raise ValueError(
                "Compiler diagnostics cannot be rerun; require a new explicit dispatch"
            )
        if full_sha(context["sha"]) != full_sha(context["control_sha"]):
            raise ValueError("Functional dispatch must use the same trusted source/control commit")
    number = str(inputs.get("pr_number", ""))
    requested_sha, requested_base = inputs.get("pr_sha", ""), inputs.get("pr_base_sha", "")
    manual_pr = event_name == "workflow_dispatch" and bool(
        number or requested_sha or requested_base
    )
    result = {
        "allowed": True,
        "decision": "run",
        "reason": "Authorized; awaiting tests",
        "target_sha": full_sha(context["sha"]),
        "control_sha": full_sha(context["control_sha"]),
        "pr_head_sha": "",
        "base_sha": "",
        "pr_number": 0,
        "manual_pr": manual_pr,
        "actor": context["actor"],
        "run_id": context["run_id"],
        "run_attempt": int(context["run_attempt"]),
        "event_name": event_name,
        "test_kind": test_kind,
    }
    if event_name == "workflow_dispatch":
        expected_workflow = f"{repository}/.github/workflows/ppu-build.yml@{BRANCH}"
        if ref != BRANCH or context["workflow_ref"] != expected_workflow:
            raise ValueError("Authorization must originate from the trusted validation branch")
        for actor in {context["actor"], context["triggering_actor"]}:
            if not re.fullmatch(r"[A-Za-z0-9_\[\]-]+", actor):
                raise ValueError("Invalid actor")
            permission = api(f"collaborators/{actor}/permission")["permission"]
            if permission not in ("write", "maintain", "admin"):
                raise ValueError("The actor must have write/maintain/admin permission")
        if manual_pr:
            if str(context["run_attempt"]) != "1":
                raise ValueError(
                    "Authorized PR runs cannot be rerun; resubmit the PR number and SHAs"
                )
            requested_sha, requested_base = full_sha(requested_sha), full_sha(requested_base)
    elif event_name == "pull_request_target":
        if ref != BRANCH:
            raise ValueError("Only PRs targeting the validation branch are allowed")
        if str(context["run_attempt"]) != "1":
            raise ValueError("PR runs cannot be rerun; use a new event or full-SHA dispatch")
        number = str(event["number"])
        requested_sha, requested_base = snapshot(event["pull_request"])
    elif event_name == "merge_group":
        group = event["merge_group"]
        if (
            event.get("action") != "checks_requested"
            or group["base_ref"] != BRANCH
            or event["repository"]["full_name"] != repository
            or group["head_ref"] != ref
            or not ref.startswith("refs/heads/gh-readonly-queue/")
        ):
            raise ValueError("Unsupported merge queue event")
        result.update(target_sha=full_sha(group["head_sha"]), base_sha=full_sha(group["base_sha"]))
        if result["target_sha"] != full_sha(context["sha"]):
            raise ValueError("Merge queue SHA mismatch")
    elif event_name == "push":
        if event.get("deleted") or not (ref == BRANCH or ref.startswith("refs/tags/v")):
            raise ValueError("Unsupported push event")
    else:
        raise ValueError("Unsupported authorization event")

    if not (manual_pr or event_name == "pull_request_target"):
        return result
    if not re.fullmatch(r"[1-9][0-9]{0,8}", number):
        raise ValueError("Invalid PR number")
    pr = api(f"pulls/{number}")
    if pr["base"]["repo"]["full_name"] != repository or pr["base"]["ref"] != BRANCH.removeprefix(
        "refs/heads/"
    ):
        raise ValueError("The PR must target this repository's validation branch")
    if not pr["head"].get("repo"):
        raise ValueError("The PR head repository does not exist")
    if manual_pr:
        if full_sha(pr["head"]["sha"]) != requested_sha or not base_is_current(requested_base, api):
            raise StaleSnapshot("PR head/base changed; revalidate the full snapshot")
    else:
        if pr_identity(pr) != pr_identity(event["pull_request"]):
            raise StaleSnapshot("PR identity or lifecycle state changed since the event")
        requested_base = current_base_sha(api)
    result.update(
        pr_number=int(number),
        pr_head_sha=requested_sha,
        base_sha=requested_base,
        head_repository=pr["head"]["repo"]["full_name"],
    )
    if pr["state"] != "open":
        if manual_pr:
            raise ValueError("The authorized PR must be open")
        return blocked(result, "closed", "PR closed; tests not run")
    if pr.get("draft"):
        return blocked(result, "draft", "Awaiting a ready PR; tests not run")
    if (
        event_name == "pull_request_target"
        and event.get("action") == "edited"
        and "base" not in event.get("changes", {})
    ):
        return blocked(result, "ignore", "Title or body edits do not rerun tests")
    try:
        return resolve_pr(result, pr, inputs, repository, api, sleep)
    except StaleSnapshot:
        raise
    except (OSError, ValueError, KeyError, TypeError):
        return blocked(
            result, "preflight_error", "PR preflight failed; retry with a fresh snapshot"
        )


def resolve_pr(result: dict, pr: dict, inputs: dict, repository: str, api, sleep) -> dict:
    number = result["pr_number"]
    requested_sha, requested_base = result["pr_head_sha"], result["base_sha"]
    only_docs = docs_only(number, pr, api)
    checked = api(f"pulls/{number}")
    if pr_identity(checked) != pr_identity(pr) or not base_is_current(requested_base, api):
        raise StaleSnapshot("PR snapshot changed during preflight")
    if only_docs and inputs.get("scope", "auto") == "auto":
        return blocked(result, "docs_only", "Docs-only changes; UT exempt under policy")
    if pr["head"]["repo"]["full_name"] != repository and not result["manual_pr"]:
        return blocked(
            result,
            "authorization_required",
            "Awaiting maintainer authorization of full head/base SHAs",
        )
    for attempt in range(3):
        current = api(f"pulls/{number}")
        if pr_identity(current) != pr_identity(pr) or not base_is_current(requested_base, api):
            raise StaleSnapshot("PR changed while resolving the synthetic merge commit")
        if current.get("mergeable") is False:
            return blocked(result, "conflict", "PR has merge conflicts; PPU not started")
        merge = current.get("merge_commit_sha")
        if current.get("mergeable") is True and merge:
            merge = full_sha(merge)
            try:
                commit = api(f"git/commits/{merge}")
            except OSError:
                commit = {"parents": []}
            if [full_sha(p["sha"]) for p in commit["parents"]] == [requested_base, requested_sha]:
                checked = api(f"pulls/{number}")
                if (
                    pr_identity(checked) != pr_identity(pr)
                    or checked.get("merge_commit_sha") != merge
                    or not base_is_current(requested_base, api)
                ):
                    raise StaleSnapshot(
                        "PR changed during verification of the synthetic merge commit"
                    )
                result["target_sha"] = merge
                return result
        if attempt < 2:
            sleep(2)
    return blocked(
        result, "merge_unavailable", "Synthetic merge commit not ready; revalidate later"
    )


def run_url(run_id: str) -> str:
    return f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{run_id}"


def latest_status(sha: str, api=github_api) -> dict:
    for page in range(1, 101):
        rows = api(f"commits/{sha}/statuses?per_page=100&page={page}")
        for row in rows:
            if row["context"] == STATUS_CONTEXT:
                return row
        if len(rows) < 100:
            return {}
    raise ValueError("Status pagination limit exceeded; refusing to overwrite unknown status")


def status_description(reason: str, authorization: dict) -> str:
    suffix = (
        f" attempt={int(authorization.get('run_attempt', 1))}"
        f" base={full_sha(authorization['base_sha'])}"
    )
    return reason[: 140 - len(suffix)] + suffix


def initial_state(decision: str) -> str:
    return {
        "run": "pending",
        "docs_only": "success",
        "draft": "pending",
        "closed": "error",
        "authorization_required": "pending",
        "merge_unavailable": "pending",
        "conflict": "failure",
        "preflight_error": "error",
    }[decision]


def valid_lifecycle(pr: dict, decision: str) -> bool:
    if decision == "closed":
        return pr["state"] == "closed"
    if decision == "draft":
        return pr["state"] == "open" and bool(pr.get("draft"))
    return pr["state"] == "open" and not pr.get("draft")


def run_is_current(authorization: dict, api, cancelled: bool = False) -> bool:
    run = api(f"actions/runs/{authorization['run_id']}")
    active = run["status"] == "in_progress" and run.get("conclusion") is None
    cancelled_run = run["status"] == "completed" and run.get("conclusion") == "cancelled"
    return run["run_attempt"] == authorization.get("run_attempt", 1) and (
        active or cancelled and cancelled_run
    )


def status_owner(status: dict) -> tuple[int, int] | None:
    match = re.fullmatch(re.escape(run_url("")) + r"([0-9]+)", status.get("target_url", "") or "")
    if not match:
        return None
    attempt = re.search(r"(?:^| )attempt=([1-9][0-9]*)(?: |$)", status.get("description", ""))
    return int(match[1]), int(attempt[1]) if attempt else 1


def owns_pending(status: dict, authorization: dict) -> bool:
    return (
        status_owner(status) == (int(authorization["run_id"]), authorization.get("run_attempt", 1))
        and status.get("state") == "pending"
        and f"base={authorization['base_sha']}" in status.get("description", "").split()
    )


def publish_status(
    authorization: dict, state: str, description: str, api=github_api, *, cancelled: bool = False
) -> str:
    number, decision = authorization["pr_number"], authorization.get("decision")
    if authorization.get("test_kind", "unit") != "unit":
        return "rejected"
    if not number or decision not in (
        "run",
        "docs_only",
        "draft",
        "closed",
        "authorization_required",
        "merge_unavailable",
        "conflict",
        "preflight_error",
    ):
        return "rejected"
    if state not in ("pending", "success", "failure", "error") or cancelled and state != "error":
        return "rejected"
    if decision != "run" and state != initial_state(decision):
        return "rejected"
    sha = full_sha(authorization["pr_head_sha"])
    pr = api(f"pulls/{number}")
    if (
        not current_pr(pr, authorization, os.environ["GITHUB_REPOSITORY"])
        or not valid_lifecycle(pr, decision)
        or not base_is_current(authorization["base_sha"], api)
    ):
        return "stale"
    if not run_is_current(authorization, api, cancelled):
        return "stale"
    latest = latest_status(sha, api)
    owner = status_owner(latest)
    identity = (int(authorization["run_id"]), authorization.get("run_attempt", 1))
    if latest and (owner is None or owner > identity):
        return "stale"
    payload = {
        "state": state,
        "context": STATUS_CONTEXT,
        "description": status_description(description, authorization),
        "target_url": run_url(authorization["run_id"]),
    }
    if all(latest.get(key) == value for key, value in payload.items()):
        return "published"
    if decision == "run" and state != "pending" and not owns_pending(latest, authorization):
        return "stale"
    if owner == identity and latest.get("state") != "pending":
        return "stale"
    if decision == "run" and state == "success":
        if (
            pr.get("merge_commit_sha") != authorization["target_sha"]
            or pr.get("mergeable") is not True
        ):
            return "stale"
        commit = api(f"git/commits/{authorization['target_sha']}")
        if [full_sha(parent["sha"]) for parent in commit["parents"]] != [
            authorization["base_sha"],
            sha,
        ]:
            return "stale"
    checked = api(f"pulls/{number}")
    if (
        pr_identity(checked) != pr_identity(pr)
        or (
            decision == "run"
            and state == "success"
            and (
                checked.get("merge_commit_sha") != authorization["target_sha"]
                or checked.get("mergeable") is not True
            )
        )
        or latest_status(sha, api) != latest
        or not run_is_current(authorization, api, cancelled)
        or not base_is_current(authorization["base_sha"], api)
    ):
        return "stale"
    api(f"statuses/{sha}", payload)
    return "published"


def execution_guard(authorization: dict, api=github_api) -> bool:
    if not authorization.get("allowed"):
        return False
    if not authorization["pr_number"]:
        return True
    if (
        int(os.environ["GITHUB_RUN_ID"]) != int(authorization["run_id"])
        or int(os.environ["GITHUB_RUN_ATTEMPT"]) != authorization["run_attempt"]
    ):
        return False
    pr = api(f"pulls/{authorization['pr_number']}")
    if (
        not current_pr(pr, authorization, os.environ["GITHUB_REPOSITORY"])
        or not valid_lifecycle(pr, "run")
        or not base_is_current(authorization["base_sha"], api)
        or not run_is_current(authorization, api)
        or not owns_pending(latest_status(authorization["pr_head_sha"], api), authorization)
        or pr.get("mergeable") is not True
        or pr.get("merge_commit_sha") != authorization["target_sha"]
    ):
        return False
    commit = api(f"git/commits/{authorization['target_sha']}")
    checked = api(f"pulls/{authorization['pr_number']}")
    return (
        [full_sha(parent["sha"]) for parent in commit["parents"]]
        == [authorization["base_sha"], authorization["pr_head_sha"]]
        and pr_identity(checked) == pr_identity(pr)
        and checked.get("merge_commit_sha") == authorization["target_sha"]
        and checked.get("mergeable") is True
        and owns_pending(latest_status(authorization["pr_head_sha"], api), authorization)
        and run_is_current(authorization, api)
        and base_is_current(authorization["base_sha"], api)
    )


def invalidate_base(base_sha: str, api=github_api) -> None:
    branch = BRANCH.removeprefix("refs/heads/")
    if full_sha(api(f"git/ref/heads/{branch}")["object"]["sha"]) != base_sha:
        return
    for page in range(1, 101):
        rows = api(f"pulls?state=open&base={branch}&per_page=100&page={page}")
        for pull in rows:
            pr = api(f"pulls/{pull['number']}")
            sha, _ = snapshot(pr)
            if (
                pr["state"] != "open"
                or pr["base"]["ref"] != branch
                or pr["base"]["repo"]["full_name"] != os.environ["GITHUB_REPOSITORY"]
                or not pr["head"].get("repo")
            ):
                continue
            latest = latest_status(sha, api)
            match = re.fullmatch(
                re.escape(run_url("")) + r"([0-9]+)", latest.get("target_url", "") or ""
            )
            if not match or int(match[1]) >= int(os.environ["GITHUB_RUN_ID"]):
                continue
            # A fresh run can share a head; its base is encoded in the description.
            if base_sha in latest.get("description", ""):
                continue
            if (
                full_sha(api(f"git/ref/heads/{branch}")["object"]["sha"]) != base_sha
                or pr_identity(api(f"pulls/{pull['number']}")) != pr_identity(pr)
                or latest_status(sha, api) != latest
            ):
                continue
            api(
                f"statuses/{sha}",
                {
                    "state": "pending",
                    "context": STATUS_CONTEXT,
                    "description": f"Target branch advanced; awaiting revalidation base={base_sha}",
                    "target_url": run_url(os.environ["GITHUB_RUN_ID"]),
                },
            )
        if len(rows) < 100:
            return
    raise ValueError("Open PR pagination limit exceeded")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("authorize", "status", "invalidate-base", "guard"))
    parser.add_argument("--output", type=Path, default=Path(".ppu-ci"))
    parser.add_argument("--state", choices=("pending", "success", "failure", "error"))
    parser.add_argument("--description", default="PPU PR check")
    parser.add_argument("--cancelled", action="store_true")
    args = parser.parse_args()
    path = args.output / "authorization.json"
    if args.mode == "invalidate-base":
        invalidate_base(full_sha(os.environ["GITHUB_SHA"]), github_api)
        return
    if args.mode == "guard":
        if not execution_guard(json.loads(path.read_text()), github_api):
            raise SystemExit("PR execution no longer owns the current snapshot; PPU not started")
        return
    if args.mode == "status":
        authorization = json.loads(path.read_text())
        state = args.state or initial_state(authorization["decision"])
        description = args.description if args.state else authorization["reason"]
        outcome = publish_status(
            authorization, state, description, github_api, cancelled=args.cancelled
        )
        print(f"PPU status: {outcome}")
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
                stream.write(f"status_result={outcome}\n")
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
                stream.write(f"PPU status: {outcome} ({state}) - {description}\n")
        if outcome == "rejected":
            raise SystemExit(1)
        return
    env_keys = {
        "repository": "GITHUB_REPOSITORY",
        "event_name": "GITHUB_EVENT_NAME",
        "ref": "GITHUB_REF",
        "sha": "GITHUB_SHA",
        "control_sha": "PPU_CONTROL_SHA",
        "workflow_ref": "PPU_WORKFLOW_REF",
        "actor": "GITHUB_ACTOR",
        "triggering_actor": "GITHUB_TRIGGERING_ACTOR",
        "run_id": "GITHUB_RUN_ID",
        "run_attempt": "GITHUB_RUN_ATTEMPT",
    }
    context = {key: os.environ[value] for key, value in env_keys.items()}
    result = authorize(
        json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()),
        json.loads(os.environ.get("PPU_INPUTS", "{}")),
        context,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            for key in (
                "allowed",
                "target_sha",
                "control_sha",
                "pr_number",
                "manual_pr",
                "decision",
                "test_kind",
            ):
                value = result[key]
                stream.write(f"{key}={str(value).lower() if isinstance(value, bool) else value}\n")
    print(json.dumps(result, ensure_ascii=False))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(f"PPU preflight: {result['decision']} - {result['reason']}\n")


if __name__ == "__main__":
    main()
