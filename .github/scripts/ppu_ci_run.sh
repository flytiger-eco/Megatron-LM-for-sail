#!/usr/bin/env bash
set -euo pipefail
selection_id="${1:?缺少 UT bucket 或 suite 标识}"
[[ "$selection_id" == quick || "$selection_id" =~ ^(dev|lts)-[0-9]+$ ]]
: "${PPU_RESULTS_DIR:?缺少本次作业结果目录}"
: "${PPU_SOURCE_ROOT:?缺少待测源码目录}"
: "${PPU_CONTROL_DIR:?缺少可信控制脚本目录}"
: "${PPU_ARTIFACT_DIR:?缺少构建产物目录}"
results_root="$PPU_RESULTS_DIR"
suite_started=$(date +%s)
cd "$PPU_SOURCE_ROOT"
[[ ! -e .ppu-ci && ! -L .ppu-ci ]]
ln -s "$PPU_ARTIFACT_DIR" .ppu-ci
prepare_results_for_upload() {
    local exit_code="$1"
    trap - EXIT
    if [[ -d "$results_root" ]]; then
        if ! chmod -R -- a+rX "$results_root"; then
            printf '::error::无法设置 UT 结果目录的上传读取权限: %s\n' "$PPU_RESULTS_DIR" >&2
            if ((exit_code == 0)); then
                exit_code=1
            fi
        fi
    fi
    exit "$exit_code"
}
trap 'prepare_results_for_upload "$?"' EXIT
umask 022
mkdir -p "$PPU_RESULTS_DIR" /opt/data /workspace/data/cache /tmp/checkpoints
export PPU_REPORT_DIR="$PPU_RESULTS_DIR"
python3 -m pip uninstall --yes cloud-tpu-client protobuf
tar -xzf .ppu-ci/environment.tar.gz -C /opt
export PATH="/opt/ppu-ci-venv/bin:$PATH"
export VIRTUAL_ENV=/opt/ppu-ci-venv UV_PROJECT_ENVIRONMENT=/opt/ppu-ci-venv
python -m pip check
export PYTHONPATH="$PPU_CONTROL_DIR:$PWD"
export PYTHONSAFEPATH=1
export UV_NO_BUILD=1 PIP_ONLY_BINARY=:all: UV_PYTHON_DOWNLOADS=never
export TORCH_CPP_LOG_LEVEL=ERROR
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
export GPUS_PER_NODE="${NPROC_PER_NODE:?}" NUM_NODES="${NNODES:?}" SLURM_NODEID="${NODE_RANK:-0}"
unset WORLD_SIZE RANK LOCAL_RANK
printf '=== PPU 环境自检 ===\n'
printf 'hostname=%s node=%s suite=%s\n' "$(hostname)" "${NODE_NAME:-unknown}" "$selection_id"
python --version
if command -v ppu-smi >/dev/null 2>&1; then
    ppu-smi || printf '%s\n' '警告：ppu-smi 执行失败，继续检查 torch 设备状态' >&2
else
    printf '%s\n' '警告：镜像中没有 ppu-smi，继续检查 torch 设备状态' >&2
fi
python - <<'PY'
import os
import sys
import torch
expected = int(os.environ['NPROC_PER_NODE'])
available = torch.cuda.is_available()
count = torch.cuda.device_count()
print('Python:', sys.version, '解释器:', sys.executable, flush=True)
print('torch:', torch.__version__, '路径:', torch.__file__, flush=True)
print('CUDA/PPU 可用:', available, '可见卡数:', count, '要求卡数:', expected, flush=True)
if not available:
    raise RuntimeError('torch 无法使用 CUDA/PPU 设备')
if count < expected:
    raise RuntimeError(f'可见卡数不足: {count} < {expected}')
import megatron.core
from megatron.core.datasets import helpers_cpp
import google.protobuf
from importlib.metadata import version
from pathlib import Path
assert Path(google.protobuf.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert google.protobuf.__version__ == version('protobuf')
print('protobuf:', google.protobuf.__version__, google.protobuf.__file__)
print('megatron.core:', megatron.core.__version__, '路径:', megatron.core.__file__)
print('预编译扩展:', helpers_cpp.__file__)
PY
make -q -C megatron/core/datasets
if [[ ! -f .ppu-ci/data/.ready ]]; then
    echo '缺少 Build 阶段生成的 UT 数据就绪标记' >&2
    exit 1
fi
cp -a .ppu-ci/data/. /opt/data/
export HF_HOME=/opt/data/huggingface-cache HF_HUB_CACHE=/opt/data/huggingface-cache/hub
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
python - <<'PY'
import json
from pathlib import Path
from transformers import AutoTokenizer
for name in json.loads(Path('/opt/data/hf-tokenizers.json').read_text()):
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True, trust_remote_code=False)
    assert tokenizer.encode('PPU CI tokenizer cache')
    print('UT tokenizer 离线检查通过:', name)
PY
install -D -m 0644 /opt/data/tokenizers/megatron/gpt2-vocab.json \
    /home/gitlab-runner/data/gpt3_data/gpt2-vocab.json
install -D -m 0644 /opt/data/tokenizers/megatron/gpt2-merges.txt \
    /home/gitlab-runner/data/gpt3_data/gpt2-merges.txt
export PYTEST_PLUGINS=ppu_pytest_policy ONE_LOGGER_JOB_CATEGORY=test
python - "$selection_id" > .ppu-ci/suite-settings.txt <<'PY'
import json
import os
import sys
from pathlib import Path
manifest = json.loads(Path('.ppu-ci/manifest.json').read_text())
identity = manifest['identity']
assert identity['target_sha'] == os.environ['PPU_TARGET_SHA']
assert identity['control_sha'] == os.environ['PPU_CONTROL_SHA']
if sys.argv[1] == 'quick':
    assert manifest['config']['scope'] == 'quick'
    cases = list(manifest['unit'].values())
else:
    assert manifest['config']['scope'] == 'full'
    cases = [manifest['unit'][sys.argv[1]]]
assert cases
print(sum(case['minutes'] for case in cases) * 60)
print('\n'.join(case['id'] for case in cases))
PY
suite_settings=()
while IFS= read -r setting; do suite_settings+=("$setting"); done < .ppu-ci/suite-settings.txt
suite_deadline=$((SECONDS + suite_settings[0]))
prepared_at=$(date +%s)
overall=0
run_with_budget() {
    local remaining=$((deadline - SECONDS))
    if ((remaining <= 0)); then
        echo '测试总时间预算已耗尽' >&2
        return 124
    fi
    timeout --signal=TERM --kill-after=30s "$remaining" "$@"
}
for case_id in "${suite_settings[@]:1}"; do
    export PPU_CASE_ID="$case_id"
    export PPU_RESULTS_DIR="$results_root"
    if [[ "$selection_id" == quick ]]; then
        export PPU_RESULTS_DIR="$results_root/$case_id"
    fi
    mkdir -p "$PPU_RESULTS_DIR"
    export PPU_REPORT_DIR="$PPU_RESULTS_DIR" COVERAGE_FILE="$PPU_RESULTS_DIR/.coverage"
    python - "$case_id" > .ppu-ci/settings.txt <<'PY'
import json
import os
import sys
from pathlib import Path
case = json.loads(Path('.ppu-ci/manifest.json').read_text())['unit'][sys.argv[1]]
assert case['environment'] in ('dev', 'lts')
assert case['environment'] == sys.argv[1].split('-', 1)[0]
assert case['cards'] == int(os.environ['NPROC_PER_NODE'])
print(case['minutes'] * 60)
print(case['repeat'])
print('not flaky' if case['environment'] == 'lts' else 'not flaky_in_dev')
print('\n'.join(case['args']))
PY
    settings=()
    while IFS= read -r setting; do settings+=("$setting"); done < .ppu-ci/settings.txt
    seconds="${settings[0]}"
    repeat="${settings[1]}"
    marker="${settings[2]}"
    args=("${settings[@]:3}")
    status=0
    started_at=$(date +%s)
    deadline=$((SECONDS + seconds))
    if ((deadline > suite_deadline)); then deadline=$suite_deadline; fi
    for ((iteration = 1; iteration <= repeat; iteration++)); do
        export PPU_TEST_ITERATION="$iteration"
        for phase in prod experimental; do
            export PPU_TEST_PHASE="$phase"
            selection="not experimental and $marker"
            extra=(-m "$selection")
            if [[ "$phase" == experimental ]]; then
                selection="experimental and $marker"
                extra=(-m "$selection" --experimental)
            fi
            run_with_budget \
                python -m torch.distributed.run --standalone --nproc-per-node "$NPROC_PER_NODE" \
                --log-dir "$PPU_RESULTS_DIR/logs/$iteration/$phase" --tee 0:3 --redirects 3 \
                -m coverage run --parallel-mode --source=megatron/core -m pytest \
                "${args[@]}" -vs --tb=short -o faulthandler_timeout=300 \
                "${extra[@]}" || status=$?
        done
    done
    if python -m coverage combine --keep "$PPU_RESULTS_DIR"; then
        python -m coverage xml -o "$PPU_RESULTS_DIR/coverage.xml" || true
    fi
    ended_at=$(date +%s)
    printf '{"started_at":%s,"completed_at":%s,"elapsed_seconds":%s}\n' \
        "$started_at" "$ended_at" "$((ended_at - started_at))" > "$PPU_RESULTS_DIR/timing.json"
    python "$PPU_CONTROL_DIR/ppu_ci.py" case-result --case-id "$case_id" \
        --results "$PPU_RESULTS_DIR" --runner-exit "$status" || overall=1
done
ended_at=$(date +%s)
printf '{"started_at":%s,"prepared_at":%s,"completed_at":%s,"elapsed_seconds":%s}\n' \
    "$suite_started" "$prepared_at" "$ended_at" "$((ended_at - suite_started))" > "$results_root/suite-timing.json"
cp .ppu-ci/build-timing.json "$results_root/build-timing.json"
prepare_results_for_upload "$overall"
