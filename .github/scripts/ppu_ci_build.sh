#!/usr/bin/env bash
set -euo pipefail
[[ -f .ppu-ci/manifest.json ]] || exit 1
export PIP_INDEX_URL=https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple
export PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL='' PIP_FIND_LINKS=''
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 PIP_NO_CACHE_DIR=1
export DEBIAN_FRONTEND=noninteractive TORCH_CPP_LOG_LEVEL=ERROR
export MAX_JOBS="${MAX_JOBS:-16}"
python3 -m pip uninstall --yes megatron-core cloud-tpu-client
python3 -m venv --system-site-packages /opt/ppu-ci-venv
export PATH="/opt/ppu-ci-venv/bin:$PATH"
export VIRTUAL_ENV=/opt/ppu-ci-venv

python - <<'PY'
import importlib.metadata as md
from pathlib import Path
packages = ('torch', 'torchvision', 'triton', 'scipy', 'scikit-learn')
lines = []
for package in packages:
    try:
        lines.append(f'{package}=={md.version(package)}')
    except md.PackageNotFoundError:
        if package in ('torch', 'triton'):
            raise RuntimeError(f'基础镜像缺少必需的 PPU runtime: {package}') from None
native_requirements = [
    'transformer-engine==2.16+v0.1.0.ppu2.2.0',
    'flash-attn==2.8.2+v0.2.0.ppu2.2.0',
    'mamba-ssm==2.3.1+v0.1.0.ppu2.2.0',
    'causal-conv1d==1.7.0+v0.1.0.ppu2.2.0',
]
Path('.ppu-ci/ppu-requirements.txt').write_text('\n'.join(native_requirements) + '\n')
Path('.ppu-ci/constraints.txt').write_text(
    '\n'.join(lines + native_requirements + ['numpy==1.26.4']) + '\n'
)
PY
export PIP_CONSTRAINT="$PWD/.ppu-ci/constraints.txt"
python -m pip install --index-url "$PIP_INDEX_URL" --upgrade 'packaging>=24.2'
python -m pip install --index-url "$PIP_INDEX_URL" 'setuptools>=80' wheel pybind11 ninja click pyyaml numpy
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
    -r .ppu-ci/test-requirements.txt -r .ppu-ci/ppu-requirements.txt \
    pytest-timeout click wrapt requests tensorstore scipy scikit-learn \
    'pydantic<2.14' boto3 'multi-storage-client~=0.50' 'einops~=0.8' 'uv==0.13.0' \
    'google-api-python-client>=2'
python -m pip check
python -I - <<'PY'
import importlib
import importlib.metadata as md
from pathlib import Path
from packaging.requirements import Requirement
for line in Path('.ppu-ci/constraints.txt').read_text().splitlines():
    requirement = Requirement(line)
    installed = md.version(requirement.name)
    if not requirement.specifier.contains(installed, prereleases=True):
        raise RuntimeError(f'依赖版本偏离约束: {requirement}, 实际安装 {installed}')
    print(f'{requirement.name}=={installed}')
for name in (
    'numpy', 'scipy.linalg', 'sklearn.utils.murmurhash',
    'transformer_engine.pytorch', 'flash_attn.flash_attn_interface', 'flash_attn_2_cuda',
    'mamba_ssm.ops.triton.layernorm_gated', 'mamba_ssm.ops.triton.ssd_combined',
    'selective_scan_cuda', 'causal_conv1d', 'causal_conv1d_cuda',
):
    module = importlib.import_module(name)
    print('依赖导入检查通过:', name, module.__file__)
PY
install -m 0644 .github/scripts/ppu_pytest_policy.py \
    "$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/ppu_pytest_policy.py"
python -m pip wheel --index-url "$PIP_INDEX_URL" --no-build-isolation --no-deps --wheel-dir dist .
python setup.py build_ext --inplace
make -C megatron/core/datasets
make -q -C megatron/core/datasets
python -m pip install --no-index --no-deps --ignore-installed dist/*.whl
python -m pip check
python -m pip list --format=json > .ppu-ci/installed-packages.json
python -I - <<'PY'
import sys
from pathlib import Path
import torch
import megatron.core
import megatron.training
from megatron.core.datasets import helpers_cpp
from megatron.core.extensions import transformer_engine
from megatron.core.ssm import mamba_mixer
assert transformer_engine.HAVE_TE, 'Megatron 未启用 Transformer Engine'
assert mamba_mixer.HAVE_MAMBA_SSM, 'Megatron 未启用 Mamba SSM'
assert mamba_mixer.causal_conv1d_fn is not None, 'Megatron 未启用 causal-conv1d'
assert Path(megatron.core.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert Path(helpers_cpp.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
print('megatron.core:', megatron.core.__version__, megatron.core.__file__)
print('torch:', torch.__version__, '构建 Pod 可见 PPU 卡数:', torch.cuda.device_count())
print('PPU 构建及导入检查通过:', helpers_cpp.__file__)
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
