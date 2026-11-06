"""Lock and verify the byte-preserved, user-supplied PPU baselines."""

import argparse
import hashlib
import json
from pathlib import Path

BASELINES = Path("tests/functional_tests/zhenwu_test_cases")
LOCK = Path(".github/ppu-functional-baselines.json")
FILENAMES = {"model_config.yaml", "golden_values_dev_ppu.json", "golden_values_dev_ppu_2nd.json"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root: Path) -> dict:
    base = root / BASELINES
    files = {}
    for path in sorted(base.rglob("*")):
        if path.name == ".DS_Store":
            continue
        if path.is_symlink():
            raise ValueError(f"Baseline symlinks are not allowed: {path}")
        if path.is_dir():
            continue
        relative = path.relative_to(base)
        if len(relative.parts) != 3 or path.name not in FILENAMES:
            raise ValueError(f"Unexpected baseline file: {relative}")
        if not path.stat().st_size:
            raise ValueError(f"Empty baseline file: {relative}")
        files[relative.as_posix()] = {"sha256": digest(path), "size": path.stat().st_size}
    if not files:
        raise ValueError("PPU baseline inventory is empty")
    return files


def verify(root: Path, control: Path) -> dict:
    lock = json.loads((control / LOCK).read_text())
    actual = inventory(root)
    if lock.get("schema_version") != 1 or lock.get("files") != actual:
        raise ValueError("PPU baseline inventory does not match the trusted snapshot")
    return actual


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    files = inventory(args.source)
    if inventory(args.destination) != files:
        raise ValueError("Imported PPU files differ from the user-supplied source")
    lock = {"schema_version": 1, "files": files}
    destination = args.destination / LOCK
    if destination.exists():
        raise ValueError("Refusing to overwrite an existing baseline lock")
    destination.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    counts = {name: sum(Path(path).name == name for path in files) for name in sorted(FILENAMES)}
    print(json.dumps({"verified_files": len(files), "counts": counts, "lock": str(destination)}))


if __name__ == "__main__":
    main()
