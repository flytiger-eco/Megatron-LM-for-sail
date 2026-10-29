"""Prepare and attest functional-only tools without changing the fixed PPU image."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

YQ_VERSION = "4.47.2"
YQ = {
    "x86_64": ("amd64", "1bb99e1019e23de33c7e6afc23e93dad72aad6cf2cb03c797f068ea79814ddb0"),
    "aarch64": ("arm64", "05df1f6aed334f223bb3e6a967db259f7185e33650c3b6447625e16fea0ed31f"),
}


def command_output(command):
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30).strip()


def prepare(bundle: Path, manifest: dict):
    tools = bundle / "functional-tools"
    tools.mkdir(exist_ok=False)
    arch, digest = YQ[platform.machine()]
    url = f"https://github.com/mikefarah/yq/releases/download/v{YQ_VERSION}/yq_linux_{arch}"
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Official Mike Farah yq binary digest mismatch")
    (tools / "yq").write_bytes(content)
    (tools / "yq").chmod(0o755)
    envsubst = shutil.which("envsubst")
    if not envsubst:
        raise ValueError("GNU envsubst must be installed by the Build preparation step")
    shutil.copy2(envsubst, tools / "envsubst")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--index-url",
            os.environ["PIP_INDEX_URL"],
            "tensorboard",
            "click",
            "pyyaml",
        ],
        check=True,
    )
    if any(case["launcher"] == "ft_launcher" for case in manifest["functional"].values()):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--index-url",
                os.environ["PIP_INDEX_URL"],
                "nvidia-resiliency-ext==0.6.0",
            ],
            check=True,
        )
    site = command_output(
        [sys.executable, "-c", 'import sysconfig; print(sysconfig.get_path("purelib"))']
    )
    shutil.copy2(
        bundle / "control/ppu_functional_pytest.py", Path(site) / "ppu_functional_pytest.py"
    )


def check(bundle: Path, manifest: dict, output: Path):
    tools = bundle / "functional-tools"
    arch, expected = YQ[platform.machine()]
    if hashlib.sha256((tools / "yq").read_bytes()).hexdigest() != expected:
        raise ValueError("Functional yq binary digest mismatch")
    for name in ("yq", "envsubst"):
        shutil.copy2(tools / name, Path("/usr/local/bin") / name)
    bash = command_output(["bash", "--version"])
    if "GNU bash, version" not in bash or int(bash.split("version ")[1].split(".")[0]) < 4:
        raise ValueError("Functional tests require GNU Bash >=4")
    yq = command_output(["/usr/local/bin/yq", "--version"])
    if "mikefarah/yq" not in yq or f"v{YQ_VERSION}" not in yq:
        raise ValueError("Functional tests require the attested Mike Farah yq v4")
    envsubst = command_output(["envsubst", "--version"])
    if "GNU gettext" not in envsubst:
        raise ValueError("Functional tests require GNU envsubst")
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "tensorboard",
            "uv",
            "pytest",
            "numpy",
            "pydantic",
            "transformer-engine",
        )
    }
    for name in ("tensorboard", "yaml", "click", "ppu_functional_pytest"):
        importlib.import_module(name)
    if any(case["launcher"] == "ft_launcher" for case in manifest["functional"].values()):
        packages["nvidia-resiliency-ext"] = importlib.metadata.version("nvidia-resiliency-ext")
        if not shutil.which("ft_launcher"):
            raise ValueError("Selected case requires the original ft_launcher")
        command_output(["ft_launcher", "--help"])
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
        raise ValueError("Functional tests require exactly eight visible PPU devices")
    if int(os.environ.get("NPROC_PER_NODE", "8")) != 8 or int(os.environ.get("NNODES", "1")) != 1:
        raise ValueError("Functional tests require one eight-card node")
    # Upstream uses this path for the empty initial load only, never to save these cases.
    empty = Path("/tmp/checkpoints")
    if empty.is_symlink() or (empty.exists() and (not empty.is_dir() or any(empty.iterdir()))):
        raise ValueError("Upstream initial checkpoint path must be empty; refusing stale state")
    output.mkdir(parents=True, exist_ok=True)
    (output / "functional-environment.json").write_text(
        json.dumps(
            {
                "identity": manifest["identity"],
                "image": manifest["config"]["image"],
                "python": sys.version,
                "packages": packages,
                "bash": bash,
                "yq": yq,
                "envsubst": envsubst,
                "uv": command_output(["uv", "--version"]),
                "architecture": arch,
                "devices": torch.cuda.device_count(),
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check"))
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.bundle / "manifest.json").read_text())
    if manifest.get("test_kind") != "functional" or not manifest["config"]["build"]:
        raise ValueError("Functional environment setup requires a regression build")
    if args.command == "prepare":
        prepare(args.bundle, manifest)
    check(args.bundle, manifest, args.output or args.bundle)


if __name__ == "__main__":
    main()
