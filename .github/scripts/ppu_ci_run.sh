#!/usr/bin/env bash
set -euo pipefail
case_id="${1:?缺少 UT bucket 标识}"
[[ "$case_id" =~ ^(dev|lts)-[0-9]+$ ]]
: "${PPU_RESULTS_DIR:?缺少本次作业结果目录}"
prepare_results_for_upload() {
    local exit_code=$?
    trap - EXIT
    if [[ -d "$PPU_RESULTS_DIR" ]]; then
        if ! chmod -R -- a+rX "$PPU_RESULTS_DIR"; then
            printf '::error::无法设置 UT 结果目录的上传读取权限: %s\n' "$PPU_RESULTS_DIR" >&2
            if ((exit_code == 0)); then
                exit_code=1
            fi
        fi
    fi
    exit "$exit_code"
}
trap prepare_results_for_upload EXIT
umask 022
mkdir -p "$PPU_RESULTS_DIR" /opt/data /workspace/data/cache /tmp/checkpoints
export PPU_REPORT_DIR="$PPU_RESULTS_DIR"
tar -xzf .ppu-ci/environment.tar.gz -C /opt
export PATH="/opt/ppu-ci-venv/bin:$PATH"
export VIRTUAL_ENV=/opt/ppu-ci-venv UV_PROJECT_ENVIRONMENT=/opt/ppu-ci-venv
export PYTHONPATH="$PWD:$PWD/.github/scripts:${PYTHONPATH:-}"
export UV_NO_BUILD=1 PIP_ONLY_BINARY=:all: UV_PYTHON_DOWNLOADS=never
export TORCH_CPP_LOG_LEVEL=ERROR
export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
export GPUS_PER_NODE="${NPROC_PER_NODE:?}" NUM_NODES="${NNODES:?}" SLURM_NODEID="${NODE_RANK:-0}"
unset WORLD_SIZE RANK LOCAL_RANK
printf '=== PPU 环境自检 ===\n'
printf 'hostname=%s node=%s bucket=%s\n' "$(hostname)" "${NODE_NAME:-unknown}" "$case_id"
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
print('megatron.core:', megatron.core.__version__, '路径:', megatron.core.__file__)
print('预编译扩展:', helpers_cpp.__file__)
PY
make -q -C megatron/core/datasets
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
mapfile -t settings < .ppu-ci/settings.txt
seconds="${settings[0]}"
repeat="${settings[1]}"
marker="${settings[2]}"
args=("${settings[@]:3}")
status=0
if [[ ! -f .ppu-ci/data/.ready ]]; then
    echo '缺少 Build 阶段生成的 UT 数据就绪标记' >&2
    exit 1
fi
cp -a .ppu-ci/data/. /opt/data/
export COVERAGE_FILE="$PPU_RESULTS_DIR/.coverage"
export PYTEST_PLUGINS=ppu_pytest_policy ONE_LOGGER_JOB_CATEGORY=test
deadline=$((SECONDS + seconds))
run_with_budget() {
    local remaining=$((deadline - SECONDS))
    if ((remaining <= 0)); then
        echo '测试总时间预算已耗尽' >&2
        return 124
    fi
    timeout --signal=TERM --kill-after=30s "$remaining" "$@"
}
for ((iteration = 1; iteration <= repeat; iteration++)); do
    export PPU_TEST_ITERATION="$iteration"
    for phase in prod experimental; do
        export PPU_TEST_PHASE="$phase"
        selection="not experimental and $marker"
        extra=()
        if [[ "$phase" == experimental ]]; then
            selection="experimental and $marker"
            extra=(--experimental)
        fi
        run_with_budget \
            python -m torch.distributed.run --standalone --nproc-per-node "$NPROC_PER_NODE" \
            --log-dir "$PPU_RESULTS_DIR/logs/$iteration/$phase" --tee 0:3 --redirects 3 \
            -m coverage run --parallel-mode --source=megatron/core -m pytest \
            "${args[@]}" -vs --tb=short -m "$selection" "${extra[@]}" || status=$?
    done
done
if python -m coverage combine --keep "$PPU_RESULTS_DIR"; then
    python -m coverage xml -o "$PPU_RESULTS_DIR/coverage.xml" || true
fi
python .github/scripts/ppu_ci.py case-result --case-id "$case_id" \
    --results "$PPU_RESULTS_DIR" --runner-exit "$status"
