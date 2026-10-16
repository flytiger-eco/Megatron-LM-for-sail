#!/usr/bin/env bash
set -euo pipefail
[[ -f .ppu-ci/manifest.json ]]
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple}"
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 PIP_NO_CACHE_DIR=1
export DEBIAN_FRONTEND=noninteractive TORCH_CPP_LOG_LEVEL=ERROR
export MAX_JOBS="${MAX_JOBS:-16}"
python3 -m pip uninstall --yes megatron-core
python3 -m venv --system-site-packages /opt/ppu-ci-venv
export PATH="/opt/ppu-ci-venv/bin:$PATH"
export VIRTUAL_ENV=/opt/ppu-ci-venv

python - <<'PY'
import importlib.metadata as md
from pathlib import Path
packages = (
    'torch', 'torchvision', 'triton', 'transformer-engine',
    'transformer-engine-torch', 'flash-attn', 'mamba-ssm', 'causal-conv1d',
)
lines = []
for package in packages:
    try:
        lines.append(f'{package}=={md.version(package)}')
    except md.PackageNotFoundError:
        pass
Path('.ppu-ci/constraints.txt').write_text('\n'.join(lines) + '\n')
PY
export PIP_CONSTRAINT="$PWD/.ppu-ci/constraints.txt"
python -m pip install --index-url "$PIP_INDEX_URL" --upgrade 'packaging>=24.2'
python -m pip install --index-url "$PIP_INDEX_URL" 'setuptools>=80' wheel pybind11 click pyyaml
python - <<'PY'
import tomllib
from pathlib import Path
project = tomllib.loads(Path('pyproject.toml').read_text())
requirements = (
    project['project']['dependencies']
    + project['dependency-groups']['test']
    + project['project']['optional-dependencies']['training']
)
Path('.ppu-ci/test-requirements.txt').write_text('\n'.join(requirements) + '\n')
PY
python -m pip install --index-url "$PIP_INDEX_URL" --no-build-isolation \
    -r .ppu-ci/test-requirements.txt pytest-timeout click wrapt requests tensorstore \
    'pydantic<2.14' boto3 'multi-storage-client~=0.50' 'einops~=0.8' 'uv==0.13.0'
install -m 0644 .github/scripts/ppu_pytest_policy.py \
    "$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/ppu_pytest_policy.py"
python -m pip wheel --index-url "$PIP_INDEX_URL" --no-build-isolation --no-deps --wheel-dir dist .
python setup.py build_ext --inplace
make -C megatron/core/datasets
make -q -C megatron/core/datasets
python -m pip install --no-deps --ignore-installed dist/*.whl
python -I - <<'PY'
import sys
from pathlib import Path
import torch
import megatron.core
import megatron.training
from megatron.core.datasets import helpers_cpp
assert Path(megatron.core.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert Path(helpers_cpp.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
print('megatron.core:', megatron.core.__version__, megatron.core.__file__)
print('torch:', torch.__version__, 'CPU 构建容器可见 PPU 卡数:', torch.cuda.device_count())
print('CPU 构建及导入检查通过:', helpers_cpp.__file__)
PY
mkdir -p .ppu-ci/data
if python -c 'import json; raise SystemExit(not json.load(open(".ppu-ci/manifest.json"))["config"]["unit_test"])'; then
    python -m tests.test_utils.python_scripts.download_unit_tests_dataset --assets-dir "$PWD/.ppu-ci/data"
    touch .ppu-ci/data/.ready
fi
git ls-files -z > .ppu-ci/source-files.txt
tar -czf .ppu-ci/source.tar.gz --null -T .ppu-ci/source-files.txt megatron/core/datasets/helpers_cpp*.so
tar -czf .ppu-ci/environment.tar.gz -C /opt ppu-ci-venv
cp dist/*.whl .ppu-ci/
printf '%s\n' "$PPU_IMAGE" > .ppu-ci/image.txt
printf '%s\n' "$GITHUB_SHA" > .ppu-ci/commit.txt
