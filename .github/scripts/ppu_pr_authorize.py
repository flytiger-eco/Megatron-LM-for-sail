"""PPU PR."""

import argparse
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


def blocked(result: dict, decision: str, reason: str) -> dict:
    return {**result, "allowed": False, "decision": decision, "reason": reason}


def authorize(event: dict, inputs: dict, context: dict, api=github_api, sleep=time.sleep) -> dict:
    repository = context["repository"]
    event_name, ref = context["event_name"], context["ref"]
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
    if snapshot(pr) != (requested_sha, requested_base):
        raise ValueError("PR head/base changed; revalidate the full snapshot")
    result.update(pr_number=int(number), pr_head_sha=requested_sha, base_sha=requested_base)
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
    only_docs = docs_only(int(number), pr, api)
    checked = api(f"pulls/{number}")
    if snapshot(checked) != snapshot(pr):
        raise ValueError("PR snapshot changed during preflight")
    if checked["state"] != "open" or checked.get("draft"):
        return blocked(result, "changed", "PR state changed during preflight; revalidate")
    if only_docs and inputs.get("scope", "auto") == "auto":
        return blocked(result, "docs_only", "Docs-only changes; UT exempt under policy")
    if pr["head"]["repo"]["full_name"] != repository and not manual_pr:
        return blocked(
            result,
            "authorization_required",
            "Awaiting maintainer authorization of full head/base SHAs",
        )
    for attempt in range(3):
        current = api(f"pulls/{number}")
        if (
            snapshot(current) != (requested_sha, requested_base)
            or current["state"] != "open"
            or current.get("draft")
        ):
            raise ValueError("PR changed while resolving the synthetic merge commit")
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
                    snapshot(checked) != (requested_sha, requested_base)
                    or checked["state"] != "open"
                    or checked.get("draft")
                    or checked.get("merge_commit_sha") != merge
                ):
                    raise ValueError("PR changed during verification of the synthetic merge commit")
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


def publish_status(authorization: dict, state: str, description: str, api=github_api) -> None:
    number = authorization["pr_number"]
    if not number or authorization.get("decision") == "ignore":
        return
    sha = full_sha(authorization["pr_head_sha"])
    pr = api(f"pulls/{number}")
    if (
        snapshot(pr) != (sha, authorization["base_sha"])
        or pr["base"]["ref"] != BRANCH.removeprefix("refs/heads/")
        or pr["base"]["repo"]["full_name"] != os.environ["GITHUB_REPOSITORY"]
    ):
        return
    if authorization["decision"] in ("run", "docs_only") and (
        pr["state"] != "open" or pr.get("draft")
    ):
        return
    run = api(f"actions/runs/{authorization['run_id']}")
    if (
        run["status"] != "in_progress"
        or run.get("conclusion") is not None
        or run["run_attempt"] != authorization.get("run_attempt", 1)
    ):
        return
    latest = latest_status(sha, api)
    url = run_url(authorization["run_id"])
    latest_url = latest.get("target_url", "") or ""
    match = re.fullmatch(re.escape(run_url("")) + r"([0-9]+)", latest_url)
    if match and int(match[1]) > int(authorization["run_id"]):
        return
    if state != "pending" and authorization["decision"] == "run" and latest_url != url:
        return
    checked = api(f"pulls/{number}")
    if (
        snapshot(checked) != snapshot(pr)
        or checked["state"] != pr["state"]
        or checked.get("draft") != pr.get("draft")
        or checked["base"]["ref"] != pr["base"]["ref"]
        or latest_status(sha, api) != latest
    ):
        return
    api(
        f"statuses/{sha}",
        {
            "state": state,
            "context": STATUS_CONTEXT,
            "description": description[:140],
            "target_url": url,
        },
    )


def invalidate_base(base_sha: str, api=github_api) -> None:
    branch = BRANCH.removeprefix("refs/heads/")
    if full_sha(api(f"git/ref/heads/{branch}")["object"]["sha"]) != base_sha:
        return
    for page in range(1, 101):
        rows = api(f"pulls?state=open&base={branch}&per_page=100&page={page}")
        for pull in rows:
            pr = api(f"pulls/{pull['number']}")
            sha, current_base = snapshot(pr)
            if pr["state"] != "open" or current_base != base_sha:
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
                or snapshot(api(f"pulls/{pull['number']}")) != (sha, base_sha)
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
    parser.add_argument("mode", choices=("authorize", "status", "invalidate-base"))
    parser.add_argument("--output", type=Path, default=Path(".ppu-ci"))
    parser.add_argument("--state", choices=("pending", "success", "failure", "error"))
    parser.add_argument("--description", default="PPU PR check")
    args = parser.parse_args()
    path = args.output / "authorization.json"
    if args.mode == "invalidate-base":
        invalidate_base(full_sha(os.environ["GITHUB_SHA"]), github_api)
        return
    if args.mode == "status":
        authorization = json.loads(path.read_text())
        state = args.state or ("success" if authorization["decision"] == "docs_only" else "pending")
        description = args.description if args.state else authorization["reason"]
        publish_status(
            authorization, state, f"{description} base={authorization['base_sha']}", github_api
        )
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
            ):
                value = result[key]
                stream.write(f"{key}={str(value).lower() if isinstance(value, bool) else value}\n")
    print(json.dumps(result, ensure_ascii=False))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as stream:
            stream.write(f"PPU preflight: {result['decision']} - {result['reason']}\n")


if __name__ == "__main__":
    main()
