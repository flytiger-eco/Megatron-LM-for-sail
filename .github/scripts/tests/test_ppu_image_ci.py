"""Image acceptance must reject changed inputs, skipped tests, and partial training."""

import copy
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "ppu_image_ci", ROOT / ".github/scripts/ppu_image_ci.py"
)
image_ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(image_ci)
DIGEST = "ghcr.io/flytiger-eco/megatron-lm-ppu-nightly@sha256:" + "c" * 64


@pytest.fixture(autouse=True)
def script_imports(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / ".github/scripts"))


@pytest.fixture
def source(tmp_path):
    result = tmp_path / "source"
    relatives = [
        *image_ci.ENTRYPOINTS,
        str(image_ci.CASE_DIR / "model_config.yaml"),
        "megatron/core/tokenizers/text/libraries/megatron_hf_tokenizer.py",
    ]
    for relative in relatives:
        target = result / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    for directory in ("shell_test_utils", "python_test_utils"):
        relative = Path("tests/functional_tests") / directory
        shutil.copytree(ROOT / relative, result / relative)
    (result / "docker").mkdir()
    (result / "docker/Dockerfile.sdk.2.2.1-ppu").write_text("FROM test-image\n")
    return result


def prepare(source, destination):
    image_ci.prepare(source, ROOT, destination, "2.2.1", "a" * 40, "b" * 40)
    return image_ci.verify_files(destination)


def test_release_launcher_can_differ_without_changing_training(source, tmp_path):
    config_path = source / image_ci.CASE_DIR / "model_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config.pop("LAUNCHER", None)
    config_path.write_text(yaml.safe_dump(config))
    manifest = prepare(source, tmp_path / "prepared")
    assert manifest["train_iters"] == 50
    assert {row["repository"] for row in manifest["tokenizer_dependencies"]} == {"gpt2"}
    assert not (tmp_path / "prepared/megatron").exists()


@pytest.mark.parametrize(
    "key,value", [("--train-iters", 4), ("--hidden-size", 128), ("--mock-data", True)]
)
def test_changed_training_cannot_reuse_reference(source, tmp_path, key, value):
    config_path = source / image_ci.CASE_DIR / "model_config.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["MODEL_ARGS"][key] = value
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="differs"):
        prepare(source, tmp_path / "prepared")


def test_changed_harness_is_rejected(source, tmp_path):
    root = tmp_path / "prepared"
    prepare(source, root)
    (root / "model_config.yaml").write_text("TEST_TYPE: regular\n")
    with pytest.raises(ValueError, match="input changed"):
        image_ci.verify_files(root)


@pytest.fixture
def completed(source, tmp_path, monkeypatch):
    root, result = tmp_path / "prepared", tmp_path / "result"
    manifest = prepare(source, root)
    result.mkdir()
    monkeypatch.setenv("PPU_IMAGE_DIGEST", DIGEST)
    shutil.copy2(root / "model_config.yaml", result / "model_config.yaml")
    values = {
        name: {"end_step": 50, "values": {str(step): 1.0 for step in range(1, 51)}}
        for name in ("lm loss", "num-zeros")
    }
    image_ci.write_json(result / "golden_values_dev_m890.json", values)
    report = {
        "complete": True,
        "exitstatus": 0,
        "train_iters": 50,
        "case_id": manifest["case"],
        "repeat": 1,
        "identity": {
            "target_sha": manifest["target_sha"],
            "control_sha": manifest["control_sha"],
            "image_digest": DIGEST,
        },
        "manifest_sha256": image_ci.sha256(root / "manifest.json"),
        "approximate": False,
        "collected": ["pipeline"],
        "collection_errors": [],
        "tests": {"pipeline": {"setup": "passed", "call": "passed", "teardown": "passed"}},
        "inputs": {
            "golden-values-path": image_ci.sha256(root / "golden_values_dev_m890.json"),
            "actual-values-path": image_ci.sha256(result / "golden_values_dev_m890.json"),
            "model-config-path": image_ci.sha256(result / "model_config.yaml"),
        },
    }
    image_ci.write_json(result / "pytest-regular.json", report)
    return root, result, report


def test_complete_upstream_evidence_passes(completed):
    root, result, _ = completed
    image_ci.check_result(root, result)
    assert json.loads((result / "summary.json").read_text())["image"] == DIGEST


@pytest.mark.parametrize(
    "change", ["skipped", "empty", "partial", "different-image", "approximate"]
)
def test_incomplete_or_different_test_cannot_promote(completed, change):
    root, result, report = completed
    report = copy.deepcopy(report)
    if change == "skipped":
        report["tests"]["pipeline"]["call"] = "skipped"
    elif change == "empty":
        report["collected"], report["tests"] = [], {}
    elif change == "partial":
        report["complete"] = False
    elif change == "different-image":
        report["identity"]["image_digest"] = DIGEST[:-1] + "d"
    else:
        report["approximate"] = True
    image_ci.write_json(result / "pytest-regular.json", report)
    with pytest.raises(ValueError, match="complete, passing"):
        image_ci.check_result(root, result)
    assert not (result / "summary.json").exists()


def test_missing_training_step_is_not_accepted(completed):
    root, result, report = completed
    actual = result / "golden_values_dev_m890.json"
    values = json.loads(actual.read_text())
    del values["lm loss"]["values"]["25"]
    image_ci.write_json(actual, values)
    report["inputs"]["actual-values-path"] = image_ci.sha256(actual)
    image_ci.write_json(result / "pytest-regular.json", report)
    with pytest.raises(ValueError, match="every required training step"):
        image_ci.check_result(root, result)
