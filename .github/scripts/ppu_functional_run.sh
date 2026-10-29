#!/usr/bin/env bash
set -euo pipefail
case_id="${1:?Missing functional case ID}"
[[ "$case_id" =~ ^functional-[0-9]{3}$ ]]
: "${PPU_RESULTS_DIR:?Missing isolated case output}"
: "${PPU_SOURCE_ROOT:?Missing source under test}"
: "${PPU_CONTROL_DIR:?Missing trusted control scripts}"
: "${PPU_ARTIFACT_DIR:?Missing verified build artifacts}"
[[ ! -e "$PPU_RESULTS_DIR" && ! -L "$PPU_RESULTS_DIR" ]]
started_at=$(date +%s)
setup_dir="$(mktemp -d)"
finish() {
    local code="$1" completed_at
    trap - EXIT
    completed_at=$(date +%s)
    mkdir -p "$PPU_RESULTS_DIR" || code=1
    cp -R "$setup_dir/." "$PPU_RESULTS_DIR/" || code=1
    printf '{"started_at":%s,"completed_at":%s,"elapsed_seconds":%s,"exit_code":%s}\n' \
        "$started_at" "$completed_at" "$((completed_at - started_at))" "$code" > "$PPU_RESULTS_DIR/pod-timing.json" || code=1
    chmod -R a+rX "$PPU_RESULTS_DIR" || code=1
    exit "$code"
}
trap 'finish "$?"' EXIT
trap 'exit 143' TERM INT
umask 022
cd "$PPU_SOURCE_ROOT"
python3 -m pip uninstall --yes cloud-tpu-client protobuf
tar -xzf "$PPU_ARTIFACT_DIR/environment.tar.gz" -C /opt
export PATH="/opt/ppu-ci-venv/bin:$PATH"
export VIRTUAL_ENV=/opt/ppu-ci-venv UV_PROJECT_ENVIRONMENT=/opt/ppu-ci-venv
export PYTHONSAFEPATH=1 PYTHONPATH="$PPU_SOURCE_ROOT"
export PIP_INDEX_URL=https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple
export PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL='' PIP_FIND_LINKS=''
export PIP_NO_INPUT=1 PIP_DISABLE_PIP_VERSION_CHECK=1
unset WORLD_SIZE RANK LOCAL_RANK
python -m pip check
make -q -C megatron/core/datasets
python -I "$PPU_CONTROL_DIR/ppu_functional_environment.py" check \
    --bundle "$PPU_ARTIFACT_DIR" --output "$setup_dir"
python -I "$PPU_CONTROL_DIR/ppu_functional_runtime.py" \
    --manifest "$PPU_ARTIFACT_DIR/manifest.json" --root "$PPU_SOURCE_ROOT" \
    --case-id "$case_id" --output "$PPU_RESULTS_DIR" &
runtime_pid=$!
cancel_runtime() {
    trap - TERM INT
    kill -TERM "$runtime_pid" 2>/dev/null || true
    wait "$runtime_pid" || true
    exit 143
}
trap cancel_runtime TERM INT
wait "$runtime_pid"
