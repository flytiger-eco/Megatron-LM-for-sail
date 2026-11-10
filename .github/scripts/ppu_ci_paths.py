"""Trusted runtime path configuration; public plans contain logical references only."""

import argparse
import os
import re
from pathlib import Path, PurePosixPath

ROOT_NAMES = (
    "PPU_CPU_RUN_ROOT",
    "PPU_POD_RUN_ROOT",
    "PPU_CPU_ASSETS_ROOT",
    "PPU_POD_ASSETS_ROOT",
    "PPU_TOKENIZER_RECEIPTS_ROOT",
)
LOGICAL = {"cpu": "ppu://assets/cpu", "pod": "ppu://assets/pod"}


def absolute(value: str, name: str) -> Path:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+", value)
        or any(part in (".", "..") for part in value.split("/")[1:])
        or str(PurePosixPath(value)) != value
    ):
        raise ValueError(f"Missing or invalid trusted directory: {name}")
    return Path(value)


def root(name: str, env=None) -> Path:
    if name not in ROOT_NAMES:
        raise ValueError("Unknown trusted directory key")
    return absolute((os.environ if env is None else env).get(name, ""), name)


def mounts(env=None) -> list:
    value = (os.environ if env is None else env).get("PPU_HOST_VOLUMES", "")
    pairs = []
    for item in value.split(","):
        fields = item.split(":")
        if len(fields) != 2:
            raise ValueError("Missing or invalid trusted host mounts")
        pairs.append(tuple(absolute(field, "PPU_HOST_VOLUMES") for field in fields))
    if len({host for host, _ in pairs}) != len(pairs) or len({pod for _, pod in pairs}) != len(
        pairs
    ):
        raise ValueError("Ambiguous trusted host mounts")
    return pairs


def validate(route: str, env=None) -> None:
    if route not in ("run", "assets", "functional"):
        raise ValueError("Unknown runtime route")
    pairs = mounts(env)
    selected = (
        ("RUN",) if route == "run" else (("ASSETS",) if route == "assets" else ("RUN", "ASSETS"))
    )
    for kind in selected:
        cpu, pod = (root(f"PPU_{side}_{kind}_ROOT", env) for side in ("CPU", "POD"))
        candidates = sorted(
            ((host, target) for host, target in pairs if cpu.is_relative_to(host)),
            key=lambda pair: len(pair[0].parts),
            reverse=True,
        )
        inverse = sorted(
            ((host, target) for host, target in pairs if pod.is_relative_to(target)),
            key=lambda pair: len(pair[1].parts),
            reverse=True,
        )
        if (
            not candidates
            or not inverse
            or candidates[0][1] / cpu.relative_to(candidates[0][0]) != pod
            or inverse[0][0] / pod.relative_to(inverse[0][1]) != cpu
        ):
            raise ValueError(f"CPU and Pod trusted {kind.lower()} roots do not share a mount")
    if route == "assets":
        root("PPU_TOKENIZER_RECEIPTS_ROOT", env)


def asset_root(manifest: dict, location: str, env=None) -> Path:
    if location not in LOGICAL:
        raise ValueError("Unknown asset location")
    key = "cpu_data_path" if location == "cpu" else "data_path"
    if manifest.get("schema_version", 1) == 2:
        if manifest["config"].get(key) != LOGICAL[location]:
            raise ValueError("Invalid logical asset reference")
        return root(f"PPU_{location.upper()}_ASSETS_ROOT", env)
    if manifest.get("schema_version", 1) != 1:
        raise ValueError("Unsupported functional manifest schema")
    return absolute(manifest["config"][key], "historical asset root")


def require_current(manifest: dict) -> None:
    if manifest.get("schema_version") != 2 or any(
        manifest["config"].get(key) != LOGICAL[side]
        for side, key in (("cpu", "cpu_data_path"), ("pod", "data_path"))
    ):
        raise ValueError("New training requires a logical-path functional manifest")


def replacements(env=None) -> dict:
    env = os.environ if env is None else env
    values = {}
    for name in ROOT_NAMES:
        if env.get(name):
            path = str(root(name, env))
            reference = {
                "PPU_CPU_RUN_ROOT": "ppu://runs/cpu",
                "PPU_POD_RUN_ROOT": "ppu://runs/pod",
                "PPU_CPU_ASSETS_ROOT": LOGICAL["cpu"],
                "PPU_POD_ASSETS_ROOT": LOGICAL["pod"],
                "PPU_TOKENIZER_RECEIPTS_ROOT": "ppu://receipts/cpu",
            }[name]
            values[path] = reference
    if env.get("PPU_HOST_VOLUMES"):
        for index, (host, pod) in enumerate(mounts(env)):
            values.setdefault(str(host), f"ppu://mount/{index}/host")
            values.setdefault(str(pod), f"ppu://mount/{index}/pod")
    return dict(sorted(values.items(), key=lambda item: len(item[0]), reverse=True))


def redact(text: str, env=None) -> str:
    for value, reference in replacements(env).items():
        text = text.replace(value, reference)
    return text


def register_masks(env=None) -> None:
    env = os.environ if env is None else env
    if env.get("GITHUB_ACTIONS") == "true":
        for value in (env.get("PPU_HOST_VOLUMES", ""), *replacements(env)):
            if value:
                print(f"::add-mask::{value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("route", choices=("run", "assets", "functional", "mask"))
    args = parser.parse_args()
    try:
        register_masks()
        if args.route != "mask":
            validate(args.route)
    except ValueError:
        raise SystemExit("Invalid or missing trusted runtime path configuration") from None
    print("Trusted runtime path configuration checked")


if __name__ == "__main__":
    main()
