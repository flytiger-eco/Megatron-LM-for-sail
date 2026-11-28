#!/bin/bash
# Exercise the installed training image; only the upstream test harness is staged.
set -euo pipefail
root=$(realpath "$1")
result=$PPU_RESULTS_DIR
export PYTHONPATH="$root/control"
python3 "$root/control/ppu_image_ci.py" runtime --root "$root" --data "$PPU_POD_ASSETS_ROOT" --result "$result"

# Test utilities are isolated from the image's native training dependencies.
python3 -m venv --system-site-packages "$root/test-venv"
"$root/test-venv/bin/python" -m pip install --no-index --no-deps --find-links "$root/wheels" \
    pytest==8.4.2 iniconfig==2.3.0 pluggy==1.6.0 pygments==2.19.2 uv==0.13.0
echo "1bb99e1019e23de33c7e6afc23e93dad72aad6cf2cb03c797f068ea79814ddb0  $root/yq" | sha256sum --check
install -m 755 "$root/yq" /usr/local/bin/yq
export VIRTUAL_ENV="$root/test-venv" PATH="$root/test-venv/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV" UV_PYTHON="$VIRTUAL_ENV/bin/python" UV_NO_SYNC=1 UV_PYTHON_DOWNLOADS=never
export PYTHONPATH="$root:$root/control"
export PYTEST_PLUGINS=ppu_functional_pytest PYTEST_ADDOPTS=""
export HF_HOME="$result/cache/huggingface" HF_HUB_CACHE="$result/cache/huggingface/hub"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export PPU_FUNCTIONAL_IDENTITY
PPU_FUNCTIONAL_IDENTITY=$(python -c 'import json,os,sys; m=json.load(open(sys.argv[1])); print(json.dumps({"target_sha":m["target_sha"],"control_sha":m["control_sha"],"image_digest":os.environ["PPU_IMAGE_DIGEST"]}))' "$root/manifest.json")
export PPU_FUNCTIONAL_CASE_ID=gpt/gpt3_mcore_te_tp1_pp1_dist_optimizer_no_mmap_bin_files
export PPU_FUNCTIONAL_ATTEMPT=1 PPU_FUNCTIONAL_REPEAT=1 PPU_FUNCTIONAL_REPEAT_DIR="$result"
export PPU_FUNCTIONAL_MANIFEST_SHA256
PPU_FUNCTIONAL_MANIFEST_SHA256=$(sha256sum "$root/manifest.json" | cut -d ' ' -f1)
export SLURM_NODEID=0 NODE_RANK=0 NUM_NODES=1 GPUS_PER_NODE=8 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
export ALLOW_NONDETERMINISTIC_ALGO_ARG="" OMP_NUM_THREADS=4
export TORCHINDUCTOR_QUIESCE_ASYNC_COMPILE_POOL=0
export TRAINING_SCRIPT_PATH=/opt/megatron-lm/pretrain_gpt.py TRAINING_PARAMS_PATH="$result/model_config.yaml"
export GOLDEN_VALUES_PATH="$root/golden_values_dev_m890.json" OUTPUT_PATH="$result"
export TENSORBOARD_PATH="$result/tensorboard" CHECKPOINT_SAVE_PATH="$result/checkpoints" CHECKPOINT_LOAD_PATH="$result/initial-checkpoints"
export DATA_PATH="$PPU_POD_ASSETS_ROOT" DATA_CACHE_PATH="$result/cache/data"
export N_REPEAT=1 ENABLE_LIGHTWEIGHT_MODE=false SKIP_PYTEST=0 RECORD_CHECKPOINTS=false
cd /opt/megatron-lm

# run_ci_test.sh resets PYTHONPATH before pytest; keep the evidence plugin
# importable through a .pth file inside the temporary test venv only.
site=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
printf '%s\n' "$root/control" > "$site/ppu_image_control.pth"
status=0
timeout --kill-after=60s 1800 bash "$root/tests/functional_tests/shell_test_utils/run_ci_test.sh" > "$result/driver.log" 2>&1 || status=$?
if [[ "$status" == 0 ]]; then
    python "$root/control/ppu_image_ci.py" check --root "$root" --result "$result" || status=$?
fi
printf '%s\n' "$status" > "$result/exit-code.txt"
python "$root/control/ppu_ci_export.py" --source "$result" --destination "$result-public" --kind evidence
tail -n 40 "$result-public/driver.log"
exit "$status"
