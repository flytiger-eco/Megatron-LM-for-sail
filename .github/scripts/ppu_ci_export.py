"""Collect CI reports and text logs, redacting private mount paths before upload."""

import argparse
import importlib.util
import json
import os
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ppu_export_paths", Path(__file__).with_name("ppu_ci_paths.py")
)
paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)
OMITTED_DIRS = {"checkpoints", "initial-checkpoints", "cache", "tensorboard"}
BUNDLE_FILES = {
    ".complete",
    "manifest.json",
    "functional-manifest.json",
    "commit.txt",
    "control.txt",
    "image.txt",
    "build-timing.json",
    "build.log",
    "bundle-integrity.json",
    "functional-environment.json",
    "functional-package-install.json",
    "packaging-install.json",
    "build-tools-install.json",
    "source-package-install.json",
    "installed-packages.json",
    "docker-build.json",
    "runtime-image.json",
    "compatibility-patches.json",
    "base-packages.json",
    "vendor-bytes.json",
    "export-files.json",
    "constraints.txt",
    "ppu-requirements.txt",
    "test-requirements.txt",
}
ACTION_FIELDS = {
    "job_status",
    "duration_seconds",
    "queue_duration_seconds",
    "occupied_ppu_card_hours",
    "execution_ppu_card_hours",
}


def action_outputs(value: str) -> str:
    raw = json.loads(value)
    if not isinstance(raw, dict):
        raise ValueError("Malformed action outputs")
    return json.dumps({key: val for key, val in raw.items() if key in ACTION_FIELDS})


class Redactor:
    def __init__(self, env=None):
        self.mapping = paths.replacements(env)

    def text(self, value: str) -> str:
        for private, public in self.mapping.items():
            value = value.replace(private, public)
        return value

    def structure(self, value):
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.structure(item) for item in value]
        if isinstance(value, dict):
            result = {self.structure(key): self.structure(item) for key, item in value.items()}
            if len(result) != len(value):
                raise ValueError("Redaction produced duplicate public keys")
            return result
        if value is not None and not isinstance(value, (bool, int, float)):
            raise ValueError("Unsupported structured report value")
        return value


def selected(relative: Path, kind: str) -> bool:
    if any(part in OMITTED_DIRS for part in relative.parts):
        return False
    if kind == "bundle":
        return relative.as_posix() in BUNDLE_FILES or (
            len(relative.parts) == 2
            and relative.parts[0] == "control"
            and relative.suffix in {".py", ".sh"}
        )
    return (
        relative.suffix in {".json", ".log", ".txt", ".xml", ".md"}
        or relative.name in {"model_config.yaml", ".complete"}
        or ("barriers" in relative.parts and relative.name == "node_0")
        or (
            len(relative.parts) == 2
            and relative.parts[0] == "control"
            and relative.suffix in {".py", ".sh"}
        )
    )


def export_file(source: Path, destination: Path, redactor: Redactor) -> None:
    raw = source.read_bytes()
    content = raw.decode("utf-8")
    if "\0" in content:
        raise ValueError("Public reports must be UTF-8 text")
    public = redactor.text(content)
    if source.suffix == ".json" or source.name == "model_config.yaml":
        if source.suffix == ".json":
            value = json.loads(content)
        else:
            import yaml

            value = yaml.safe_load(content)
        safe = redactor.structure(value)
        immutable = source.name == "model_config.yaml" or source.name.startswith(
            ("golden_values", "observation")
        )
        if immutable:
            if safe != value or public != content:
                raise ValueError("Immutable metric/configuration evidence contains a private path")
        elif safe != value:
            public = json.dumps(safe, indent=2) + "\n"
    # Keep comparison inputs byte-for-byte: their hashes bind them to upstream pytest results.
    destination.write_bytes(raw if public == content else public.encode("utf-8"))


def export(source: Path, destination: Path, *, kind="evidence", env=None, overlay=None) -> int:
    redactor = Redactor(env)
    source, destination = source.absolute(), destination.absolute()
    if (
        destination.exists()
        or destination.is_symlink()
        or destination.resolve().is_relative_to(source.resolve())
    ):
        raise ValueError("Public staging must be new and outside private evidence")
    if source.is_symlink() or (not source.exists() and overlay is None):
        raise ValueError("Private evidence is missing or is a symlink")
    if any(parent.is_symlink() for parent in (*source.parents, *destination.parents)):
        raise ValueError("Evidence roots must not have symlink ancestors")
    candidates = (
        {Path(source.name): source}
        if source.is_file()
        else {path.relative_to(source): path for path in source.rglob("*")}
    )
    if overlay is not None:
        overlay = Path(overlay).absolute()
        if not overlay.is_dir() or any(p.is_symlink() for p in (overlay, *overlay.parents)):
            raise ValueError("Local action report must be an unlinked directory")
        for path in overlay.iterdir():
            if path.name != "action-timing.json" or path.is_symlink() or not path.is_file():
                raise ValueError("Unapproved local action report")
            candidates[Path(path.name)] = path
    destination.mkdir(parents=True, exist_ok=False)
    count = 0
    for relative, path in sorted(candidates.items()):
        if not selected(relative, kind):
            continue
        if path.is_symlink():
            raise ValueError("Symlinks are not allowed in public evidence")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Special files are not allowed in public evidence")
        if redactor.text("/" + relative.as_posix()) != "/" + relative.as_posix():
            raise ValueError("Private directory in public evidence filename")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        export_file(path, target, redactor)
        count += 1
    if not count:
        raise ValueError("No public evidence files were exported")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--kind", choices=("evidence", "bundle"), default="evidence")
    parser.add_argument("--action-outputs", action="store_true")
    parser.add_argument("--overlay", type=Path)
    args = parser.parse_args()
    try:
        if args.action_outputs:
            print(paths.redact(action_outputs(os.environ.get("ACTION_OUTPUTS", "{}"))))
            return
        paths.register_masks()
        if args.source is None or args.destination is None:
            parser.error("--source and --destination are required")
        count = export(args.source, args.destination, kind=args.kind, overlay=args.overlay)
    except Exception as error:
        # Parser/IO exceptions can contain private paths; do not print their values or tracebacks.
        raise SystemExit(f"Public evidence export failed ({type(error).__name__})") from None
    print(f"Public evidence exported: {count} files")


if __name__ == "__main__":
    main()
