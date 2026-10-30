"""Exercise the original shell with mocked training and real TensorBoard/pytest tools."""

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest
import yaml
from test_ppu_functional import ROOT, SCRIPTS, functional
from test_ppu_functional_hf import bind_case, official, provisioned
from test_ppu_functional_results import envelope, results
from test_ppu_functional_runtime import runtime

BOOTSTRAP = r'''
function /usr/local/bin/yq() {
    "$FIXTURE_PYTHON" "$FIXTURE_SCRIPT" yq "$@"
}
bash() {
    if [[ "$1" == */tests/functional_tests/shell_test_utils/_run_training.sh ]]; then
        "$FIXTURE_PYTHON" "$FIXTURE_SCRIPT" training
    else
        command bash "$@"
    fi
}
uv() {
    [[ "$1" == run && "$2" == --no-sync ]] || return 91
    shift 2
    PYTHONPATH="$FIXTURE_PYTHONPATH" "$FIXTURE_PYTHON" "$FIXTURE_SCRIPT" uv "$@"
}
'''


def fixture_yq(arguments):
    inplace = arguments[0] == "-i"
    if inplace:
        arguments = arguments[1:]
    query = arguments[0]
    path = Path(arguments[1]) if len(arguments) > 1 else None
    config = yaml.safe_load(path.read_text() if path else sys.stdin.read())
    assignments = {
        '.ENV_VARS."SKIP_PYTEST" = 1': ("ENV_VARS", "SKIP_PYTEST", 1),
        '.MODEL_ARGS."--exit-interval" = 4': ("MODEL_ARGS", "--exit-interval", 4),
        '.MODEL_ARGS."--save-interval" = 2': ("MODEL_ARGS", "--save-interval", 2),
        '.MODEL_ARGS."--exit-interval" = .MODEL_ARGS."--train-iters"': (
            "MODEL_ARGS",
            "--exit-interval",
            config["MODEL_ARGS"]["--train-iters"],
        ),
    }
    if inplace:
        section, key, value = assignments[query]
        config.setdefault(section, {})[key] = value
        path.write_text(yaml.safe_dump(config))
        return
    reads = {
        ".TEST_TYPE": ("TEST_TYPE", None),
        '.TEST_EVALUATION // "pass"': ("TEST_EVALUATION", "pass"),
        '.ENV_VARS.ENABLE_LIGHTWEIGHT_MODE // "false"': (
            "ENV_VARS.ENABLE_LIGHTWEIGHT_MODE",
            "false",
        ),
        '.ENV_VARS.N_REPEAT // "1"': ("ENV_VARS.N_REPEAT", 1),
        '.MODE // "pretraining"': ("MODE", "pretraining"),
        '.MODEL_ARGS."--exit-interval" // "100"': ("MODEL_ARGS.--exit-interval", 100),
        ".ENV_VARS.NVTE_ALLOW_NONDETERMINISTIC_ALGO": (
            "ENV_VARS.NVTE_ALLOW_NONDETERMINISTIC_ALGO",
            None,
        ),
        '.ENV_VARS.NON_DETERMINSTIC_RESULTS // "0"': ("ENV_VARS.NON_DETERMINSTIC_RESULTS", "0"),
        ".ENV_VARS.SKIP_PYTEST": ("ENV_VARS.SKIP_PYTEST", None),
    }
    dotted, default = reads[query]
    value = config
    for key in dotted.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    value = default if value is None else value
    print("null" if value is None else str(value).lower() if isinstance(value, bool) else value)


def fixture_training():
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    directory = Path(os.environ["OUTPUT_PATH"])
    phase = int(os.environ["RUN_NUMBER"])
    config = yaml.safe_load(Path(os.environ["TRAINING_PARAMS_PATH"]).read_text())
    final = config["MODEL_ARGS"]["--exit-interval"]
    load = Path(os.environ["CHECKPOINT_LOAD_PATH"])
    start = 1
    if phase == 2:
        start = int((load / "latest_checkpointed_iteration.txt").read_text()) + 1
        assert start == final // 2 + 1
        assert load == Path(os.environ["CHECKPOINT_SAVE_PATH"])
    else:
        assert str(load) == "/tmp/checkpoints"
    selected = (
        f"MODEL_ARGS_{phase}" if phase > 1 and f"MODEL_ARGS_{phase}" in config else "MODEL_ARGS"
    )
    (directory / f"mock-training-{phase}.json").write_text(
        json.dumps(
            {
                "phase": phase,
                "start": start,
                "final": final,
                "load": str(load),
                "save": os.environ["CHECKPOINT_SAVE_PATH"],
                "model_args": config[selected],
            }
        )
    )
    writer = EventFileWriter(os.environ["TENSORBOARD_PATH"])
    golden = json.loads(Path(os.environ["GOLDEN_VALUES_PATH"]).read_text())
    for step in range(start, final + 1):
        values = []
        for metric, row in golden.items():
            value = row["values"].get(str(step), 1.0)
            if not math.isfinite(float(value)):
                continue
            if os.environ.get("FIXTURE_BAD_NUMERIC") == "1" and metric == "lm loss":
                value = 999.0
            values.append(Summary.Value(tag=metric, simple_value=float(value)))
        writer.add_event(Event(step=step, summary=Summary(value=values)))
    writer.close()
    for rank in range(8):
        logs = directory / "logs/1" / f"mock_{phase}" / "attempt_0" / str(rank)
        logs.mkdir(parents=True)
        (logs / "stdout.log").write_text("Mock training completed\n")
        (logs / "stderr.log").write_text("")
    raise SystemExit(int(os.environ.get("FIXTURE_TRAINING_EXIT", "0")))


class ShellRunner(runtime.ProcessRunner):
    def __init__(self, bootstrap, **overrides):
        super().__init__()
        self.bootstrap = bootstrap
        self.overrides = overrides

    def run(self, command, *, cwd, env, log, deadline):
        env = {
            **env,
            **self.overrides,
            "BASH_ENV": str(self.bootstrap),
            "FIXTURE_PYTHON": sys.executable,
            "FIXTURE_SCRIPT": str(Path(__file__).resolve()),
            "FIXTURE_PYTHONPATH": os.pathsep.join((str(ROOT), str(SCRIPTS))),
        }
        return super().run(command, cwd=cwd, env=env, log=log, deadline=deadline)


@pytest.fixture
def shell_case(tmp_path):
    def prepare(name, **overrides):
        config = functional.select_config(
            {"functional_mode": "regression", "functional_repeat": 1, "functional_cases": name}
        )
        catalogue = functional.catalogue(ROOT, ROOT, config)
        case = next(iter(catalogue["functional"].values()))
        # training 边界已替换为合成 TensorBoard 生成器，不会实例化原 tokenizer。
        case["tokenizer_dependencies"] = []
        case["assets"] = [asset for asset in case["assets"] if not asset.get("tokenizer_cache")]
        assets = tmp_path / "assets"
        assets.mkdir()
        for asset in case["assets"]:
            path = assets / asset["relative_path"]
            if asset["kind"] == "directory":
                path.mkdir(parents=True, exist_ok=True)
                (path / "fixture").write_text("Mock checkpoint")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("Mock input")
        config["data_path"] = str(assets)
        manifest = {
            **catalogue,
            "schema_version": 1,
            "test_kind": "functional",
            "source_files": {},
            "config": config,
            "identity": {
                "target_sha": "a" * 40,
                "control_sha": "a" * 40,
                "run_id": "123",
                "run_attempt": 1,
            },
        }
        bootstrap = tmp_path / "mock-boundaries.sh"
        bootstrap.write_text(BOOTSTRAP)
        directory = tmp_path / case["id"]
        before = functional.sha256(ROOT / case["config_path"])
        record = runtime.run_case(
            manifest, ROOT, case["id"], directory, runner=ShellRunner(bootstrap, **overrides)
        )
        assert functional.sha256(ROOT / case["config_path"]) == before
        return manifest, case, directory, record

    return prepare


@pytest.mark.parametrize(
    "kind", ["regular", "resume", "long_run", "approximate", "skip", "lightweight"]
)
def test_original_shell_tensorboard_extraction_and_comparison(shell_case, kind):
    catalogue = functional.catalogue(ROOT, ROOT, functional.select_config({}))
    cases = list(catalogue["functional"].values())
    if kind == "regular":
        name = "bert/bert_mcore_tp1_pp2"
    elif kind == "resume":
        name = next(
            row["name"]
            for row in cases
            if row["test_type"] == "ckpt-resume" and not row["approximate"]
        )
    elif kind == "long_run":
        name = next(
            row["name"] for row in cases if row["train_iters"] == 2000 and not row["observation"]
        )
    elif kind == "approximate":
        name = next(row["name"] for row in cases if row["approximate"] and not row["observation"])
    else:
        name = next(
            row["name"]
            for row in cases
            if row["observation"] == ("lightweight" if kind == "lightweight" else "SKIP_PYTEST")
        )
    manifest, case, directory, record = shell_case(name)
    repeat = directory / "attempt-1/repeat-1"
    assert record["success"], (
        (repeat / "driver.log").read_text()
        + str(record)
        + (repeat / "verification.json").read_text()
    )
    envelope(manifest, case, directory)
    assert results.verify_case(manifest, ROOT, case, directory)["success"]
    summary = results.summarize(manifest, ROOT, directory.parent)
    assert summary["success"]
    assert summary["numerical_validation"] == (not bool(case["observation"]))
    assert summary["numeric_passed"] == (0 if case["observation"] else 1)
    phases = results.phase_count(case)
    assert len(list(repeat.glob("mock-training-*.json"))) == phases
    assert len(list(repeat.glob("pytest-*.json"))) == (0 if case["observation"] else phases)
    if phases == 2:
        second = json.loads((repeat / "mock-training-2.json").read_text())
        assert second["start"] == case["effective_train_iters"] // 2 + 1
    assert json.loads((repeat / "execution.json").read_text())["shell_exit_code"] == 0
    if kind == "long_run":
        first = json.loads((repeat / "mock-training-1.json").read_text())
        assert first["final"] == 2000
    if case["observation"] and phases == 2:
        (repeat / "observation_2nd.json").rename(repeat / "absent-observation.json")
        assert not results.verify_case(manifest, ROOT, case, directory)["success"]


def test_original_shell_preserves_synthetic_reshard_configuration(tmp_path, provisioned):
    config = functional.select_config({"functional_repeat": 1})
    catalogue = functional.catalogue(ROOT, ROOT, config)
    case = next(
        row
        for row in catalogue["functional"].values()
        if row["test_type"] == "ckpt-resume" and not row["approximate"]
    )
    original = yaml.safe_load((ROOT / case["config_path"]).read_text())
    original["MODEL_ARGS_2"] = {
        **original["MODEL_ARGS"],
        "--tensor-model-parallel-size": 2,
        "--pipeline-model-parallel-size": 4,
    }
    synthetic = tmp_path / "synthetic-reshard.yaml"
    synthetic.write_text(yaml.safe_dump(original))
    config["data_path"] = str(provisioned[0])
    manifest = {"identity": {}, "config": config}
    repeat = tmp_path / "repeat"
    repeat.mkdir()
    case = {**case, "config_path": str(synthetic)}
    bind_case(case, provisioned[2])
    assert {row["section"] for row in case["tokenizer_dependencies"]} == {
        "MODEL_ARGS",
        "MODEL_ARGS_2",
    }
    bootstrap = tmp_path / "mock-boundaries.sh"
    bootstrap.write_text(BOOTSTRAP)
    with runtime.repeat_environment(manifest, ROOT, case, repeat, 1, 1) as env:
        code = ShellRunner(bootstrap).run(
            ["bash", str(ROOT / "tests/functional_tests/shell_test_utils/run_ci_test.sh")],
            cwd=ROOT,
            env=env,
            log=repeat / "driver.log",
            deadline=time.monotonic() + 120,
        )
    assert code == 0, (repeat / "driver.log").read_text()
    second = json.loads((repeat / "mock-training-2.json").read_text())
    assert second["model_args"] == original["MODEL_ARGS_2"]
    assert second["start"] == case["effective_train_iters"] // 2 + 1
    assert yaml.safe_load(synthetic.read_text()) == original
    cache = functional.hf.verify_cache(
        repeat / "cache/huggingface", functional.hf.sources_for(case["tokenizer_dependencies"])
    )
    assert json.loads((repeat / "tokenizer-cache.json").read_text()) == cache
    results.verify_pytest(repeat, "resume", manifest, case, 1, 1)


@pytest.mark.parametrize(
    "overrides", [{"FIXTURE_TRAINING_EXIT": "23"}, {"FIXTURE_BAD_NUMERIC": "1"}]
)
def test_original_shell_keeps_training_and_comparison_failures(shell_case, overrides):
    manifest, case, directory, record = shell_case("bert/bert_mcore_tp1_pp2", **overrides)
    assert not record["success"] and record["attempts"] == 1
    repeat = directory / "attempt-1/repeat-1"
    execution = json.loads((repeat / "execution.json").read_text())
    assert execution["shell_exit_code"] != 0
    if "FIXTURE_TRAINING_EXIT" in overrides:
        assert (repeat / "barriers/repeat_1_run_1_status/node_0").read_text().strip() == "23"
    else:
        assert json.loads((repeat / "pytest-regular.json").read_text())["exitstatus"] == 1


if __name__ == "__main__":
    mode, *arguments = sys.argv[1:]
    if mode == "yq":
        fixture_yq(arguments)
    elif mode == "training":
        fixture_training()
    elif mode == "uv":
        if arguments[0] == "python":
            command = [sys.executable, *arguments[1:]]
        elif arguments[0] == "pytest":
            command = [sys.executable, "-m", "pytest", *arguments[1:]]
        else:
            raise ValueError(arguments)
        raise SystemExit(subprocess.run(command, check=False).returncode)
    else:
        raise ValueError(mode)
