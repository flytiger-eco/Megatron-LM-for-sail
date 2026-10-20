"""PPU PR."""

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

BRANCH = "refs/heads/ci/ppu-validation"


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
        raise ValueError("必须使用完整的 40 位提交 SHA")
    return value.lower()


def authorize(event: dict, inputs: dict, context: dict, api=github_api) -> dict:
    repository = context["repository"]
    event_name, ref = context["event_name"], context["ref"]
    control_sha = full_sha(context["control_sha"])
    number, requested_sha = str(inputs.get("pr_number", "")), inputs.get("pr_sha", "")
    manual_pr = event_name == "workflow_dispatch" and bool(number or requested_sha)
    result = {
        "allowed": True,
        "target_sha": full_sha(context["sha"]),
        "control_sha": control_sha,
        "pr_number": 0,
        "manual_pr": manual_pr,
        "actor": context["actor"],
        "run_id": context["run_id"],
    }
    if event_name == "workflow_dispatch":
        expected_workflow = f"{repository}/.github/workflows/ppu-build.yml@{BRANCH}"
        if ref != BRANCH or context["workflow_ref"] != expected_workflow:
            raise ValueError("授权入口必须来自可信验证分支")
        for actor in {context["actor"], context["triggering_actor"]}:
            if not re.fullmatch(r"[A-Za-z0-9_\[\]-]+", actor):
                raise ValueError("非法触发者")
            permission = api(f"collaborators/{actor}/permission")["permission"]
            if permission not in ("write", "maintain", "admin"):
                raise ValueError("触发者必须具有 write/maintain/admin 权限")
        if manual_pr:
            if str(context["run_attempt"]) != "1":
                raise ValueError("授权 PR 不允许 rerun，请重新提交 PR 编号和 SHA")
            if not re.fullmatch(r"[1-9][0-9]{0,8}", number):
                raise ValueError("非法 PR 编号")
            requested_sha = full_sha(requested_sha)
    elif event_name == "pull_request_target":
        if ref != BRANCH:
            raise ValueError("仅允许面向验证分支的 PR")
        number = str(event["number"])
        if not re.fullmatch(r"[1-9][0-9]{0,8}", number):
            raise ValueError("非法事件 PR 编号")
        requested_sha = full_sha(event["pull_request"]["head"]["sha"])
    elif event_name == "push":
        if event.get("deleted") or not (ref == BRANCH or ref.startswith("refs/tags/v")):
            raise ValueError("不支持的 push 事件")
    else:
        raise ValueError("不支持的授权事件")

    if manual_pr or event_name == "pull_request_target":
        pr = api(f"pulls/{number}")
        if (
            pr["state"] != "open"
            or pr["base"]["repo"]["full_name"] != repository
            or pr["base"]["ref"] != "ci/ppu-validation"
        ):
            raise ValueError("PR 必须为本仓库验证分支上的 open PR")
        if not pr["head"].get("repo"):
            raise ValueError("PR head 仓库不存在")
        target_sha = full_sha(pr["head"]["sha"])
        if target_sha != requested_sha:
            raise ValueError("PR 已更新，必须重新授权当前完整 SHA")
        result.update(pr_number=int(number), target_sha=target_sha)
        if pr["head"]["repo"]["full_name"] != repository and not manual_pr:
            result["allowed"] = False
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("authorize", "status"))
    parser.add_argument("--output", type=Path, default=Path(".ppu-ci"))
    parser.add_argument("--state", choices=("pending", "success", "failure", "error"))
    parser.add_argument("--description", default="PPU PR 检查")
    args = parser.parse_args()
    path = args.output / "authorization.json"
    if args.mode == "status":
        authorization = json.loads(path.read_text())
        if not authorization["pr_number"]:
            return
        if not args.state:
            parser.error("status 必须指定 --state")
        sha = full_sha(authorization["target_sha"])
        github_api(
            f"statuses/{sha}",
            {
                "state": args.state,
                "context": "PPU PR Quick",
                "description": args.description,
                "target_url": f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}",
            },
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
            for key in ("allowed", "target_sha", "control_sha", "pr_number", "manual_pr"):
                value = result[key]
                stream.write(f"{key}={str(value).lower() if isinstance(value, bool) else value}\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
