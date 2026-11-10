"""Export a fail-closed, path-redacted public view without changing private evidence."""

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import struct
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ppu_export_paths", Path(__file__).with_name("ppu_ci_paths.py")
)
paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)
TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".log",
    ".txt",
    ".xml",
    ".csv",
    ".md",
    ".py",
    ".sh",
    ".pth",
    ".ready",
}
OMITTED_DIRS = {"checkpoints", "initial-checkpoints"}
ARCHIVES = {
    "source.tar.gz",
    "environment.tar.gz",
    "compiled-source.tar.gz",
    "control-runtime.tar.gz",
}
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
    "compatibility-patches.json",
    "base-packages.json",
    "vendor-bytes.json",
    "export-files.json",
    "constraints.txt",
    "ppu-requirements.txt",
    "test-requirements.txt",
}
OVERLAY_FILES = {"action-timing.json", "checkpoint-omissions.json"}
ACTION_FIELDS = {
    "job_status",
    "duration_seconds",
    "queue_duration_seconds",
    "occupied_ppu_card_hours",
    "execution_ppu_card_hours",
}
FAILURE_CODES = {
    "Private directory remained in public evidence": "private-path-bytes",
    "Unsupported private binary field in structured evidence": "structured-private-binary",
    "Unsupported structured evidence collection": "structured-collection",
    "Redaction produced duplicate public JSON keys": "duplicate-public-keys",
    "A scalar/event tag cannot be redacted": "tensorboard-private-tag",
    "Corrupt TensorBoard record header": "tensorboard-header-crc",
    "TensorBoard record exceeds export limit": "tensorboard-record-limit",
    "Corrupt TensorBoard record payload": "tensorboard-payload-crc",
    "TensorBoard export changed event identity": "tensorboard-event-identity",
    "TensorBoard export changed scalar evidence": "tensorboard-scalar-changed",
    "TensorBoard export changed numeric tensor evidence": "tensorboard-numeric-changed",
    "Unsupported private text in TensorBoard record": "tensorboard-private-field",
    "Immutable numeric/configuration/cache evidence contains a private path": "immutable-private-path",
    "Public staging must be new and outside private evidence": "invalid-public-staging",
    "Private evidence is missing or is a symlink": "invalid-private-root",
    "Evidence roots must not have symlink ancestors": "linked-evidence-ancestor",
    "Local envelope overlay must be a regular directory": "invalid-overlay-root",
    "Unapproved local envelope overlay": "unapproved-overlay",
    "Unsupported encoded private path in expanded model configuration": "expanded-config-encoded-path",
    "Symlinks are not allowed in public evidence": "public-evidence-symlink",
    "Private directory in public evidence filename": "private-evidence-filename",
    "Special files are not allowed in public evidence": "public-special-file",
    "No public evidence files were exported": "empty-public-evidence",
    "Reserved public export filename": "reserved-public-filename",
    "Public evidence root or inventory is not a regular, unlinked path": "invalid-public-root",
    "Invalid public evidence inventory": "invalid-public-inventory",
    "Public evidence contains an escaping link": "public-escaping-link",
    "Public evidence contains a special file": "public-special-file",
    "Public evidence file inventory mismatch": "public-inventory-mismatch",
    "Malformed public evidence file metadata": "invalid-public-metadata",
    "Public evidence failed size or digest verification": "public-digest-mismatch",
}


class ExportFileError(ValueError):
    """Retain the original error internally, with only an allowlisted public file category."""

    def __init__(self, category: str, error: Exception):
        super().__init__(str(error))
        self.category = category
        self.error = error


def failure_summary(error: Exception) -> str:
    """Never expose exception messages, filenames, paths or traceback locals."""
    category = "export-root"
    if isinstance(error, ExportFileError):
        category, error = error.category, error.error
    code = FAILURE_CODES.get(str(error), "export-failed")
    if isinstance(error, json.JSONDecodeError):
        code = "invalid-json"
    elif isinstance(error, UnicodeDecodeError):
        code = "invalid-utf8"
    elif isinstance(error, OSError):
        code = "evidence-io-error"
    return (
        f"Public evidence export failed ({type(error).__name__}; code={code}; evidence={category})"
    )


def action_outputs(value: str) -> str:
    raw = json.loads(value)
    if not isinstance(raw, dict):
        raise ValueError("Malformed action outputs")
    return json.dumps({key: val for key, val in raw.items() if key in ACTION_FIELDS})


class Redactor:
    def __init__(self, env=None):
        self.mapping = paths.replacements(env)
        self.needles = tuple(
            value.encode(encoding)
            for value in self.mapping
            for encoding in ("utf-8", "utf-16-le", "utf-16-be")
        )

    def text(self, value: str) -> str:
        for private, public in self.mapping.items():
            value = value.replace(private, public)
        return value

    def contains(self, value: bytes) -> bool:
        return any(needle in value for needle in self.needles)

    def scan(self, path: Path) -> None:
        overlap = max(map(len, self.needles), default=1)
        tail = b""
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                combined = tail + block
                if self.contains(combined):
                    raise ValueError("Private directory remained in public evidence")
                tail = combined[-overlap:]

    def structure(self, value):
        if isinstance(value, bytes) and self.contains(value):
            raise ValueError("Unsupported private binary field in structured evidence")
        if isinstance(value, (set, tuple)):
            raise ValueError("Unsupported structured evidence collection")
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.structure(item) for item in value]
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                safe = self.structure(key)
                if safe in result:
                    raise ValueError("Redaction produced duplicate public JSON keys")
                result[safe] = self.structure(item)
            return result
        return value


def crc32c(data: bytes) -> int:
    try:
        import google_crc32c
    except ImportError:
        return slow_crc32c(data)
    crc = google_crc32c.value(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def slow_crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for value in data:
        crc ^= value
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    crc ^= 0xFFFFFFFF
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def redact_event(event, redactor: Redactor) -> None:
    """Change only descriptive text; tags, steps, wall times and numeric tensors stay intact."""
    if event.HasField("log_message"):
        event.log_message.message = redactor.text(event.log_message.message)
    if event.HasField("source_metadata"):
        event.source_metadata.writer = redactor.text(event.source_metadata.writer)
    for value in event.summary.value:
        if redactor.text(value.tag) != value.tag:
            raise ValueError("A scalar/event tag cannot be redacted")
        value.metadata.display_name = redactor.text(value.metadata.display_name)
        value.metadata.summary_description = redactor.text(value.metadata.summary_description)
        if value.HasField("tensor") and value.tensor.dtype == 7:
            for index, text in enumerate(value.tensor.string_val):
                if redactor.contains(text):
                    value.tensor.string_val[index] = redactor.text(text.decode("utf-8")).encode()


def tensorboard(source: Path, destination: Path, redactor: Redactor) -> None:
    from tensorboard.compat.proto.event_pb2 import Event

    with source.open("rb") as stream, destination.open("xb") as output:
        while header := stream.read(12):
            if len(header) != 12 or crc32c(header[:8]) != struct.unpack("<I", header[8:])[0]:
                raise ValueError("Corrupt TensorBoard record header")
            length = struct.unpack("<Q", header[:8])[0]
            if length > 64 * 1024 * 1024:
                raise ValueError("TensorBoard record exceeds export limit")
            data, checksum = stream.read(length), stream.read(4)
            if (
                len(data) != length
                or len(checksum) != 4
                or crc32c(data) != struct.unpack("<I", checksum)[0]
            ):
                raise ValueError("Corrupt TensorBoard record payload")
            event = Event.FromString(data)
            if redactor.contains(data):
                before = Event()
                before.CopyFrom(event)
                redact_event(event, redactor)
                # A second application must be idempotent; only approved descriptive fields change.
                expected = Event()
                expected.CopyFrom(before)
                redact_event(expected, redactor)
                if (
                    event != expected
                    or event.step != before.step
                    or event.wall_time != before.wall_time
                ):
                    raise ValueError("TensorBoard export changed event identity")
                for old, new in zip(before.summary.value, event.summary.value):
                    if old.tag != new.tag or old.simple_value != new.simple_value:
                        raise ValueError("TensorBoard export changed scalar evidence")
                    if old.tensor.dtype != 7 and old.tensor != new.tensor:
                        raise ValueError("TensorBoard export changed numeric tensor evidence")
                data = event.SerializeToString()
                if redactor.contains(data):
                    raise ValueError("Unsupported private text in TensorBoard record")
            length_bytes = struct.pack("<Q", len(data))
            output.write(length_bytes + struct.pack("<I", crc32c(length_bytes)))
            output.write(data + struct.pack("<I", crc32c(data)))


def immutable(path: Path) -> bool:
    return (
        "huggingface" in path.parts
        or path.name == "model_config.yaml"
        or (path.name.startswith(("golden_values", "observation")) and path.suffix == ".json")
    )


def export_file(source: Path, destination: Path, redactor: Redactor) -> None:
    if source.name.startswith("events.out.tfevents."):
        tensorboard(source, destination, redactor)
    elif source.suffix in {".json", ".jsonl", ".yaml", ".yml"}:
        content = source.read_bytes()
        if source.suffix == ".jsonl":
            value = [json.loads(row) for row in content.splitlines() if row.strip()]
        elif source.suffix == ".json":
            value = json.loads(content)
        else:
            import yaml

            value = yaml.safe_load(content)
        safe = redactor.structure(value)
        if immutable(source):
            redactor.scan(source)
            if safe != value:
                raise ValueError(
                    "Immutable numeric/configuration/cache evidence contains a private path"
                )
            copy_bytes(source, destination)
        elif safe == value and not redactor.contains(content):
            copy_bytes(source, destination)
        elif source.suffix == ".jsonl":
            destination.write_text(
                "".join(json.dumps(row) + "\n" for row in safe), encoding="utf-8"
            )
        elif source.suffix == ".json":
            destination.write_text(json.dumps(safe, indent=2) + "\n", encoding="utf-8")
        else:
            destination.write_text(yaml.safe_dump(safe), encoding="utf-8")
    elif source.name == "model_config.yaml.tmp" and not immutable(source):
        import yaml

        # Upstream envsubst leaves this descriptive expanded copy beside the immutable config.
        # Preserve every non-path byte; reject encodings that would bypass structural redaction.
        content = source.read_bytes().decode("utf-8")
        safe = redactor.structure(yaml.safe_load(content))
        public = redactor.text(content)
        try:
            public_value = yaml.safe_load(public)
        except yaml.YAMLError as error:
            raise ValueError(
                "Unsupported encoded private path in expanded model configuration"
            ) from error
        if public_value != safe:
            raise ValueError("Unsupported encoded private path in expanded model configuration")
        destination.write_bytes(public.encode("utf-8"))
    elif source.suffix in TEXT_SUFFIXES and not immutable(source):
        content = source.read_bytes()
        destination.write_bytes(redactor.text(content.decode("utf-8")).encode("utf-8"))
    else:
        # Unknown binaries are copied only if they contain no known private directory bytes.
        redactor.scan(source)
        copy_bytes(source, destination)
    redactor.scan(destination)


def export_checked_file(source: Path, destination: Path, redactor: Redactor) -> None:
    if source.name.startswith("events.out.tfevents."):
        category = "tensorboard"
    elif immutable(source):
        category = "immutable-evidence"
    elif source.name == "model_config.yaml.tmp":
        category = "expanded-model-config"
    elif source.suffix in {".json", ".jsonl", ".yaml", ".yml"}:
        category = "structured-evidence"
    elif source.suffix in TEXT_SUFFIXES:
        category = "text-evidence"
    else:
        category = "unknown-binary"
    try:
        export_file(source, destination, redactor)
    except Exception as error:
        raise ExportFileError(category, error) from error


def copy_bytes(source: Path, destination: Path) -> None:
    with source.open("rb") as stream, destination.open("xb") as output:
        while block := stream.read(1024 * 1024):
            output.write(block)


def omitted(relative: Path, kind: str) -> bool:
    if relative.name == "public-export.json":
        return True
    parts = relative.parts
    if any(part in OMITTED_DIRS for part in parts):
        return True
    if "cache" in parts:
        index = parts.index("cache")
        if len(parts) <= index + 1 or parts[index + 1] != "huggingface":
            return True
    if relative.name in ARCHIVES:
        return True
    if kind == "bundle" and not (
        relative.as_posix() in BUNDLE_FILES
        or (len(parts) == 2 and parts[0] == "control" and relative.suffix in {".py", ".sh"})
    ):
        return True
    return False


def digest_file(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            sha.update(data)
    return sha.hexdigest()


def export(source: Path, destination: Path, *, kind="evidence", env=None, overlay=None) -> dict:
    redactor = Redactor(env)
    source = source.absolute()
    destination = destination.absolute()
    if (
        destination.exists()
        or destination.is_symlink()
        or destination.resolve().is_relative_to(source.resolve())
    ):
        raise ValueError("Public staging must be new and outside private evidence")
    if source.is_symlink() or (not source.exists() and overlay is None):
        raise ValueError("Private evidence is missing or is a symlink")
    if any(parent.is_symlink() for parent in tuple(source.parents) + tuple(destination.parents)):
        raise ValueError("Evidence roots must not have symlink ancestors")
    destination.mkdir(parents=True, exist_ok=False)
    files, omissions = {}, []
    candidates = [source] if source.is_file() else sorted(source.rglob("*"))
    overlay_paths = {}
    if overlay is not None:
        overlay = Path(overlay).absolute()
        if (
            not overlay.is_dir()
            or overlay.is_symlink()
            or any(p.is_symlink() for p in overlay.parents)
        ):
            raise ValueError("Local envelope overlay must be a regular directory")
        for path in overlay.iterdir():
            if path.name not in OVERLAY_FILES or path.is_symlink() or not path.is_file():
                raise ValueError("Unapproved local envelope overlay")
            overlay_paths[path.name] = path
    for path in candidates:
        relative = Path(path.name) if source.is_file() else path.relative_to(source)
        if path.is_symlink() and not omitted(relative, kind):
            raise ValueError("Symlinks are not allowed in public evidence")
        if relative.as_posix() in overlay_paths:
            continue
        if redactor.text("/" + relative.as_posix()) != "/" + relative.as_posix():
            raise ValueError("Private directory in public evidence filename")
        if omitted(relative, kind):
            # Excluded links are neither exported nor followed, including dangling links.
            if not path.is_symlink() and path.is_file():
                omissions.append(relative.as_posix())
            continue
        mode = path.stat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError("Special files are not allowed in public evidence")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        export_checked_file(path, target, redactor)
        files[relative.as_posix()] = {"sha256": digest_file(target), "size": target.stat().st_size}
    for name, path in overlay_paths.items():
        target = destination / name
        export_checked_file(path, target, redactor)
        files[name] = {"sha256": digest_file(target), "size": target.stat().st_size}
    if not files:
        raise ValueError("No public evidence files were exported")
    report = {
        "schema_version": 1,
        "view": "path-redacted-public-evidence",
        "original_digest_applies_to": "private NAS originals only",
        "files": files,
        "omitted": omissions,
    }
    report_path = destination / "public-export.json"
    if report_path.exists():
        raise ValueError("Reserved public export filename")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    redactor.scan(report_path)
    verify_public(destination)
    return report


def verify_public(directory: Path) -> dict:
    """Verify the public view's own inventory; it never replaces the private Build digest."""
    report_path = directory / "public-export.json"
    if (
        directory.is_symlink()
        or any(parent.is_symlink() for parent in directory.parents)
        or report_path.is_symlink()
        or not report_path.is_file()
    ):
        raise ValueError("Public evidence root or inventory is not a regular, unlinked path")
    report = json.loads(report_path.read_text())
    if (
        not isinstance(report, dict)
        or type(report.get("schema_version")) is not int
        or report.get("schema_version") != 1
        or report.get("view") != "path-redacted-public-evidence"
        or report.get("original_digest_applies_to") != "private NAS originals only"
        or not isinstance(report.get("files"), dict)
        or not report["files"]
    ):
        raise ValueError("Invalid public evidence inventory")
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Public evidence contains an escaping link")
        if path.is_file() and path.relative_to(directory).as_posix() != "public-export.json":
            actual.add(path.relative_to(directory).as_posix())
        elif not path.is_dir() and path.relative_to(directory).as_posix() != "public-export.json":
            raise ValueError("Public evidence contains a special file")
    if actual != set(report["files"]):
        raise ValueError("Public evidence file inventory mismatch")
    for name, expected in report["files"].items():
        if (
            not isinstance(expected, dict)
            or set(expected) != {"size", "sha256"}
            or type(expected.get("size")) is not int
            or expected["size"] < 0
            or not isinstance(expected.get("sha256"), str)
        ):
            raise ValueError("Malformed public evidence file metadata")
        path = directory / name
        if path.stat().st_size != expected.get("size") or digest_file(path) != expected.get(
            "sha256"
        ):
            raise ValueError("Public evidence failed size or digest verification")
    return report


def checkpoint_audit(source: Path, destination: Path) -> None:
    """Advisory, read-only payload inventory; never creates or alters private run directories."""
    if not source.is_dir():
        return
    if destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Checkpoint audit must not write into private evidence")
    rows = []
    try:
        for directory in sorted(source.rglob("*")):
            if (
                directory.name not in OMITTED_DIRS
                or directory.is_symlink()
                or not directory.is_dir()
            ):
                continue
            sizes = []
            for path in directory.rglob("*"):
                try:
                    if path.is_file() and not path.is_symlink():
                        sizes.append(path.stat().st_size)
                except OSError:
                    continue
            rows.append(
                {
                    "path": directory.relative_to(source).as_posix(),
                    "files": len(sizes),
                    "bytes": sum(sizes),
                }
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"omitted": rows}) + "\n")
    except OSError:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        print("::warning::Unable to record omitted checkpoint payloads")
    print(f"Recorded {len(rows)} omitted checkpoint payload groups")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--kind", choices=("evidence", "bundle"), default="evidence")
    parser.add_argument("--action-outputs", action="store_true")
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--checkpoint-audit", action="store_true")
    args = parser.parse_args()
    if args.action_outputs:
        print(paths.redact(action_outputs(os.environ.get("ACTION_OUTPUTS", "{}"))))
        return
    paths.register_masks()
    if args.source is None or args.destination is None:
        parser.error("--source and --destination are required")
    if args.checkpoint_audit:
        checkpoint_audit(args.source, args.destination)
        return
    try:
        report = export(args.source, args.destination, kind=args.kind, overlay=args.overlay)
    except Exception as error:
        # Only fixed failure codes and file categories reach the public log.
        raise SystemExit(failure_summary(error)) from None
    print(f"Public evidence export checked: {len(report['files'])} files")


if __name__ == "__main__":
    main()
