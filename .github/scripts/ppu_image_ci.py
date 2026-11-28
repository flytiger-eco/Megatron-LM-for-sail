"""Build-time preparation and runtime checks for the published PPU training image."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
from pathlib import Path

CASE = "gpt/gpt3_mcore_te_tp1_pp1_dist_optimizer_no_mmap_bin_files"
CASE_DIR = Path("tests/functional_tests/test_cases") / CASE
ENTRYPOINTS = ("pretrain_gpt.py", "gpt_builders.py", "model_provider.py")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def prepare(source, control, output, sdk, target_sha, control_sha):
    import ppu_functional as functional
    import ppu_functional_hf as hf
    import yaml

    if sdk not in {"2.2.0", "2.2.1"} or not all(
        re.fullmatch(r"[0-9a-f]{40}", value) for value in (target_sha, control_sha)
    ):
        raise ValueError("Expected supported SDK and full source/control commit SHAs")
    dockerfile = source / f"docker/Dockerfile.sdk.{sdk}-ppu"
    config_path = source / CASE_DIR / "model_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    reference = yaml.safe_load((control / CASE_DIR / "model_config.yaml").read_text())
    # A release branch can use a different launcher, but must not silently borrow
    # a golden for a different model, optimizer, dataset, seed, or numerical mode.
    if {k: v for k, v in config.items() if k != "LAUNCHER"} != {
        k: v for k, v in reference.items() if k != "LAUNCHER"
    }:
        raise ValueError("Image FT configuration differs from the trusted M890 reference")
    if config["TEST_TYPE"] != "regular" or config["MODEL_ARGS"]["--train-iters"] != 50:
        raise ValueError("The image acceptance case must run the full 50-step regular FT")
    golden = control / CASE_DIR / "golden_values_dev_m890.json"
    metrics = functional.read_metrics(golden)
    for name in ("lm loss", "num-zeros"):
        if name not in metrics or metrics[name]["end_step"] < 50:
            raise ValueError("The trusted M890 golden does not cover the acceptance case")
    dependencies = functional.tokenizer_dependencies(source, config, "pretrain_gpt.py")
    sources = hf.load_sources(control)
    for dependency in dependencies:
        dependency.update(status="fixed_source", source=sources[dependency["repository"]])
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("shell_test_utils", "python_test_utils"):
        relative = Path("tests/functional_tests") / directory
        shutil.copytree(source / relative, output / relative)
    for relative in (Path("tests/__init__.py"), Path("tests/functional_tests/__init__.py")):
        if (source / relative).exists():
            shutil.copy2(source / relative, output / relative)
    shutil.copy2(config_path, output / "model_config.yaml")
    shutil.copy2(golden, output / "golden_values_dev_m890.json")
    (output / "control").mkdir()
    for name in (
        "ppu_image_ci.py",
        "ppu_image_ft.sh",
        "ppu_ci_paths.py",
        "ppu_ci_export.py",
        "ppu_functional_hf.py",
        "ppu_functional_pytest.py",
    ):
        shutil.copy2(control / ".github/scripts" / name, output / "control" / name)
    manifest = {
        "schema_version": 1,
        "target_sha": target_sha,
        "control_sha": control_sha,
        "sdk": sdk,
        "case": CASE,
        "train_iters": 50,
        "gpus": 8,
        "repeat": 1,
        "dockerfile_sha256": sha256(dockerfile),
        "entrypoints": {name: sha256(source / name) for name in ENTRYPOINTS},
        "assets": functional.input_assets(config),
        "tokenizer_dependencies": dependencies,
        "files": {
            p.relative_to(output).as_posix(): sha256(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    write_json(output / "manifest.json", manifest)


def verify_files(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = root / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or sha256(path) != expected
        ):
            raise ValueError(f"Image FT input changed: {name}")
    return manifest


def assets(root, data):
    import ppu_functional_hf as hf

    manifest = verify_files(root)
    for row in manifest["assets"]:
        path = data / row["relative_path"]
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing FT asset: {row['relative_path']}")
    hf.verify_cache(data / hf.CACHE_RELATIVE, hf.sources_for(manifest["tokenizer_dependencies"]))
    return manifest


def runtime(root, data, result):
    import ppu_functional_hf as hf

    manifest = assets(root, data)
    image = os.environ["PPU_IMAGE_DIGEST"]
    if not re.fullmatch(r"ghcr\.io/[a-z0-9._/-]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("FT must run the published image by immutable digest")
    if os.environ.get("PPU_VERSION") != manifest["sdk"]:
        raise ValueError("Image SDK does not match the requested build")
    for name, expected in manifest["entrypoints"].items():
        if sha256(Path("/opt/megatron-lm") / name) != expected:
            raise ValueError(f"Image entry point does not match the source commit: {name}")
    import torch

    import megatron.core

    if "/site-packages/" not in str(megatron.core.__file__) and "/dist-packages/" not in str(
        megatron.core.__file__
    ):
        raise ValueError("Training must import the wheel installed in the image")
    versions = {
        name: importlib.metadata.version(name)
        for name in ("torch", "triton", "transformer-engine", "megatron-core")
    }
    if (
        versions["torch"].split("+")[0] != "2.13.0"
        or versions["transformer-engine"] != "2.18.0+v0.2.0.ppu2.2.0"
    ):
        raise ValueError("Image Torch/TE versions do not match the Dockerfile contract")
    if torch.cuda.device_count() != 8:
        raise ValueError("The image acceptance case requires exactly eight PPU devices")
    result.mkdir(parents=True, exist_ok=False)
    shutil.copy2(root / "model_config.yaml", result / "model_config.yaml")
    hf.populate(data, result / "cache/huggingface", manifest["tokenizer_dependencies"])
    write_json(
        result / "image.json",
        {
            "image": image,
            "sdk": manifest["sdk"],
            "versions": versions,
            "megatron_path": str(megatron.core.__file__),
            "target_sha": manifest["target_sha"],
            "control_sha": manifest["control_sha"],
            "dockerfile_sha256": manifest["dockerfile_sha256"],
        },
    )


def check_result(root, result):
    manifest = verify_files(root)
    report = json.loads((result / "pytest-regular.json").read_text())
    actual = result / "golden_values_dev_m890.json"
    if not (
        report["complete"] is True
        and report["exitstatus"] == 0
        and report["train_iters"] == manifest["train_iters"]
        and report["case_id"] == manifest["case"]
        and report["repeat"] == 1
        and report["identity"]
        == {
            "target_sha": manifest["target_sha"],
            "control_sha": manifest["control_sha"],
            "image_digest": os.environ["PPU_IMAGE_DIGEST"],
        }
        and report["manifest_sha256"] == sha256(root / "manifest.json")
        and report["approximate"] is False
        and len(report["collected"]) == 1
        and not report["collection_errors"]
        and set(report["tests"]) == set(report["collected"])
        and all(
            phases == {"setup": "passed", "call": "passed", "teardown": "passed"}
            for phases in report["tests"].values()
        )
        and report["inputs"]["golden-values-path"] == sha256(root / "golden_values_dev_m890.json")
        and report["inputs"]["actual-values-path"] == sha256(actual)
        and report["inputs"]["model-config-path"] == sha256(result / "model_config.yaml")
    ):
        raise ValueError(
            "Image FT did not produce a complete, passing upstream numerical comparison"
        )
    values = json.loads(actual.read_text())
    for name in ("lm loss", "num-zeros"):
        row = values[name]
        if row["end_step"] != 50 or not all(str(step) in row["values"] for step in range(1, 51)):
            raise ValueError("Image FT did not record every required training step")
    write_json(
        result / "summary.json",
        {
            "passed": True,
            "case": manifest["case"],
            "train_iters": 50,
            "gpus": 8,
            "image": os.environ["PPU_IMAGE_DIGEST"],
            "target_sha": manifest["target_sha"],
            "control_sha": manifest["control_sha"],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "assets", "runtime", "check"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--sdk", choices=("2.2.0", "2.2.1"))
    parser.add_argument("--target-sha")
    parser.add_argument("--control-sha")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.control, args.root, args.sdk, args.target_sha, args.control_sha)
    elif args.command == "assets":
        assets(args.root, args.data)
    elif args.command == "runtime":
        runtime(args.root, args.data, args.result)
    else:
        check_result(args.root, args.result)


if __name__ == "__main__":
    main()
