#!/usr/bin/env bash
set -euo pipefail
[[ -f .ppu-ci/manifest.json ]] || exit 1
export PIP_INDEX_URL=https://pkg.flytiger-eco.com/artifactory/api/pypi/pypi_index/simple
export PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL='' PIP_FIND_LINKS=''
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 PIP_NO_CACHE_DIR=1
export DEBIAN_FRONTEND=noninteractive TORCH_CPP_LOG_LEVEL=ERROR
export MAX_JOBS="${MAX_JOBS:-16}"
python3 -m pip uninstall --yes megatron-core cloud-tpu-client protobuf
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
    '\n'.join(lines + native_requirements + ['numpy==1.26.4', 'pyarrow==21.0.0']) + '\n'
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
    'google-api-python-client>=2' pyzmq 'openai[aiohttp]' fastapi uvicorn datasets \
    'nvidia-modelopt==0.45.0'
python -m pip check
cat > .ppu-ci/import_diagnostics.py <<'PY'
import sys


def ppu_import_snapshot(label):
    print('[PPU-IMPORT]', label, flush=True)
    print('[PPU-IMPORT] interpreter:', sys.executable, sys.prefix, sys.base_prefix, flush=True)
    print('[PPU-IMPORT] sys.path:', sys.path, flush=True)
    for name in (
        'google', 'google.protobuf', 'google.protobuf.runtime_version',
        'setuptools', 'distutils', 'joblib', 'transformer_engine.pytorch', 'onnx',
    ):
        module = sys.modules.get(name)
        if module is None:
            continue
        data = vars(module)
        path = data.get('__path__')
        raw_path = vars(path) if hasattr(path, '__dict__') else {}
        print('[PPU-IMPORT] module:', name, data.get('__file__'), data.get('__version__'),
              'path:', path if isinstance(path, (list, tuple)) else raw_path.get('_path'),
              'parent:', raw_path.get('_last_parent_path'), flush=True)


def ppu_import_audit(event, args):
    if event != 'import' or args[0] not in (
        'google.protobuf', 'google.protobuf.runtime_version', 'setuptools',
        'distutils', 'joblib', 'transformer_engine.pytorch', 'onnx',
    ):
        return
    ppu_import_snapshot('import ' + args[0])
    frame = sys._getframe(1)
    remaining = 24
    while frame is not None and remaining:
        if not frame.f_code.co_filename.startswith('<frozen importlib'):
            print('[PPU-IMPORT] caller:', frame.f_code.co_filename, frame.f_lineno,
                  frame.f_code.co_name, flush=True)
            remaining -= 1
        frame = frame.f_back


ppu_original_excepthook = sys.excepthook


def ppu_import_excepthook(kind, value, traceback):
    ppu_import_snapshot('uncaught exception')
    ppu_original_excepthook(kind, value, traceback)


sys.addaudithook(ppu_import_audit)
sys.excepthook = ppu_import_excepthook
ppu_import_snapshot('probe startup')
ppu_os = sys.modules['os']
for ppu_directory in tuple(sys.path):
    if not ppu_os.path.isdir(ppu_directory):
        continue
    ppu_entries = ppu_os.listdir(ppu_directory)
    for ppu_name in sorted(ppu_entries):
        if not (ppu_name.startswith('protobuf') and ppu_name.endswith('.pth')):
            continue
        ppu_file = ppu_os.path.join(ppu_directory, ppu_name)
        with open(ppu_file) as ppu_stream:
            ppu_content = ppu_stream.read()
        print('[PPU-IMPORT] protobuf startup file:', ppu_file, repr(ppu_content), flush=True)
        for ppu_owner in ppu_entries:
            if not (ppu_owner.startswith('protobuf') and ppu_owner.endswith('.dist-info')):
                continue
            ppu_record = ppu_os.path.join(ppu_directory, ppu_owner, 'RECORD')
            if ppu_os.path.isfile(ppu_record):
                with open(ppu_record) as ppu_stream:
                    if any(line.split(',', 1)[0] == ppu_name for line in ppu_stream):
                        print('[PPU-IMPORT] startup file owner:', ppu_owner, flush=True)
PY
python -I - <<'PY'
exec(compile(open('.ppu-ci/import_diagnostics.py').read(), '.ppu-ci/import_diagnostics.py', 'exec'))
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
    'zmq', 'openai', 'fastapi', 'uvicorn', 'pyarrow', 'pyarrow.parquet', 'datasets', 'modelopt.torch',
):
    ppu_import_snapshot('before ' + name)
    module = importlib.import_module(name)
    print('依赖导入检查通过:', name, module.__file__)
    ppu_import_snapshot('after ' + name)
import numpy as np
import pyarrow as pa
from datasets import Dataset
values = np.arange(4, dtype=np.int64)
np.testing.assert_array_equal(pa.array(values).to_numpy(), values)
assert Dataset.from_dict({'value': values}).to_dict() == {'value': values.tolist()}
print('NumPy、Arrow 和 datasets 数据转换检查通过')
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
exec(compile(open('.ppu-ci/import_diagnostics.py').read(), '.ppu-ci/import_diagnostics.py', 'exec'))
import sys
from pathlib import Path
import torch
ppu_import_snapshot('before megatron.core')
import megatron.core
ppu_import_snapshot('after megatron.core')
import megatron.training
from megatron.core.datasets import helpers_cpp
from megatron.core.extensions import transformer_engine
from megatron.core.ssm import mamba_mixer
import google.protobuf
from importlib.metadata import version
assert Path(google.protobuf.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert google.protobuf.__version__ == version('protobuf')
print('protobuf:', google.protobuf.__version__, google.protobuf.__file__)
assert transformer_engine.HAVE_TE, 'Megatron 未启用 Transformer Engine'
assert mamba_mixer.HAVE_MAMBA_SSM, 'Megatron 未启用 Mamba SSM'
assert mamba_mixer.causal_conv1d_fn is not None, 'Megatron 未启用 causal-conv1d'
assert Path(megatron.core.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert Path(helpers_cpp.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
print('megatron.core:', megatron.core.__version__, megatron.core.__file__)
print('torch:', torch.__version__, '构建 Pod 可见 PPU 卡数:', torch.cuda.device_count())
print('PPU 构建及导入检查通过:', helpers_cpp.__file__)
PY
python - <<'PY'
from pathlib import Path
import megatron.post_training.model_builder
import megatron.rl.rl_utils
for module in (megatron.post_training.model_builder, megatron.rl.rl_utils):
    assert Path(module.__file__).resolve().is_relative_to(Path.cwd().resolve())
    print('UT 源码导入检查通过:', module.__name__, module.__file__)
PY
mkdir -p .ppu-ci/data
if python -c 'import json; raise SystemExit(not json.load(open(".ppu-ci/manifest.json"))["config"]["unit_test"])'; then
    python -m tests.test_utils.python_scripts.download_unit_tests_dataset --assets-dir "$PWD/.ppu-ci/data"
    export HF_HOME="$PWD/.ppu-ci/data/huggingface-cache"
    export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_DISABLE_TELEMETRY=1
    HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 timeout --signal=TERM --kill-after=30s 900 python - <<'PY'
import json
import time
from pathlib import Path
from transformers import AutoTokenizer
from megatron.core.tokenizers.text.libraries.megatron_hf_tokenizer import MEGATRON_CONFIG_MAP
names = sorted({entry['tokenizer_name'] for entry in MEGATRON_CONFIG_MAP.values()})
for name in names:
    for attempt in range(3):
        try:
            AutoTokenizer.from_pretrained(name, trust_remote_code=False)
            print('UT tokenizer 缓存就绪:', name, flush=True)
            break
        except (OSError, RuntimeError):
            if attempt == 2:
                raise
            time.sleep(15 * (attempt + 1))
Path('.ppu-ci/data/hf-tokenizers.json').write_text(json.dumps(names))
PY
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - <<'PY'
import json
from pathlib import Path
from transformers import AutoTokenizer
for name in json.loads(Path('.ppu-ci/data/hf-tokenizers.json').read_text()):
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True, trust_remote_code=False)
    assert tokenizer.encode('PPU CI tokenizer cache')
    print('UT tokenizer 离线检查通过:', name)
PY
    touch .ppu-ci/data/.ready
fi
git ls-files -z > .ppu-ci/source-files.txt
tar -czf .ppu-ci/source.tar.gz --null -T .ppu-ci/source-files.txt megatron/core/datasets/helpers_cpp*.so
tar -czf .ppu-ci/environment.tar.gz -C /opt ppu-ci-venv
cp dist/*.whl .ppu-ci/
printf '%s\n' "$PPU_IMAGE" > .ppu-ci/image.txt
printf '%s\n' "$GITHUB_SHA" > .ppu-ci/commit.txt
