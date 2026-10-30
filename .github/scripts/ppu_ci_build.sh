#!/usr/bin/env bash
set -euo pipefail
: "${PPU_SOURCE_ROOT:?Missing source directory under test}"
: "${PPU_ARTIFACT_DIR:?Missing build artifact directory}"
: "${PPU_CONTROL_DIR:?Missing trusted control script directory}"
: "${PPU_TARGET_SHA:?Missing target commit SHA}"
: "${PPU_CONTROL_SHA:?Missing trusted control commit SHA}"
started_at=$(date +%s)
cd "$PPU_SOURCE_ROOT"
[[ "$(git rev-parse HEAD)" == "$PPU_TARGET_SHA" ]]
[[ ! -e .ppu-ci && ! -L .ppu-ci ]]
ln -s "$PPU_ARTIFACT_DIR" .ppu-ci
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
import json
from pathlib import Path
packages = ('torch', 'torchvision', 'triton', 'scipy', 'scikit-learn')
lines = []
for package in packages:
    try:
        lines.append(f'{package}=={md.version(package)}')
    except md.PackageNotFoundError:
        if package in ('torch', 'triton'):
            raise RuntimeError(f'Base image is missing a required PPU runtime: {package}') from None
native_requirements = [
    'transformer-engine==2.16+v0.1.0.ppu2.2.0',
    'flash-attn==2.8.2+v0.2.0.ppu2.2.0',
    'mamba-ssm==2.3.1+v0.1.0.ppu2.2.0',
    'causal-conv1d==1.7.0+v0.1.0.ppu2.2.0',
    'pla==1.0.0+v0.1.0.ppu2.2.0',
    'fla-core==0.4.1+v0.1.0.ppu2.2.0',
]
if json.loads(Path('.ppu-ci/manifest.json').read_text()).get('test_kind') == 'functional':
    # 六项依赖当前均可从 Flytiger 获取；不混用额外索引或覆盖 PPU runtime。
    native_requirements += [
        'flash-linear-attention==0.4.1+v0.1.0.ppu2.2.0',
        'emerging-optimizers==0.3.0',
        'apex==24.4.1+v0.1.0.ppu2.2.0',
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
functional_pip_args=()
if python -c 'import json; raise SystemExit(json.load(open(".ppu-ci/manifest.json")).get("test_kind") != "functional")'; then
    functional_pip_args=(--report .ppu-ci/functional-package-install.json)
fi
python -m pip install --index-url "$PIP_INDEX_URL" --no-build-isolation "${functional_pip_args[@]}" \
    -r .ppu-ci/test-requirements.txt -r .ppu-ci/ppu-requirements.txt \
    pytest-timeout click wrapt requests tensorstore scipy scikit-learn \
    'pydantic<2.14' boto3 'multi-storage-client~=0.50' 'einops~=0.8' 'uv==0.13.0' \
    'google-api-python-client>=2' pyzmq msgpack 'openai[aiohttp]' fastapi uvicorn datasets \
    'nvidia-modelopt==0.45.0'
python -m pip check
python -I - <<'PY'
import importlib
import importlib.metadata as md
import sys
from pathlib import Path

version = md.version('pla')
if version != '1.0.0+v0.1.0.ppu2.2.0':
    raise RuntimeError(f'PLA version does not match the PPU constraint: {version}')
for name in ('pla', 'pla.decode', 'pla.prefill.flashqla', 'pla.prefill.flashkdapro'):
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    if not path.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f'PLA module was not loaded from the build virtual environment: {name}: {path}')
    print('PLA import check passed:', name, version, path)
PY
python -I - <<'PY'
import hashlib
import importlib.metadata as md
import sys
from pathlib import Path
distribution = md.distribution('fla-core')
if distribution.version != '0.4.1+v0.1.0.ppu2.2.0':
    raise RuntimeError(f'FLA Core version does not match the PPU constraint: {distribution.version}')
path = Path(distribution.locate_file('fla/modules/l2norm.py')).resolve()
if not path.is_relative_to(Path(sys.prefix).resolve()):
    raise RuntimeError(f'FLA Core patch target is outside the build virtual environment: {path}')
source = path.read_text()
original = '''def l2norm(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    return L2NormFunction.apply(x, eps, output_dtype)'''
replacement = '''def l2norm(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
    *,
    dim: int = -1,
) -> torch.Tensor:
    if not isinstance(dim, int) or isinstance(dim, bool) or dim not in (-1, x.ndim - 1):
        raise ValueError("PPU FLA Core l2norm only supports the last dimension")
    return L2NormFunction.apply(x, eps, output_dtype)'''
if source.count(original) != 1:
    raise RuntimeError('FLA Core l2norm patch target is not unique or the source has changed')
patched = source.replace(original, replacement)
compile(patched, str(path), 'exec')
path.write_text(patched)
print('FLA Core l2norm last dimension argument adaptation:', path, 'sha256:', hashlib.sha256(patched.encode()).hexdigest())
PY
python -I - <<'PY'
import importlib
import importlib.metadata as md
import inspect
import sys
from pathlib import Path

version = md.version('fla-core')
if version != '0.4.1+v0.1.0.ppu2.2.0':
    raise RuntimeError(f'FLA Core version does not match the PPU constraint: {version}')
required = {
    'fla': (),
    'fla.modules.convolution': ('causal_conv1d', 'causal_conv1d_update'),
    'fla.modules.l2norm': ('l2norm',),
    'fla.ops.gated_delta_rule': ('chunk_gated_delta_rule', 'fused_recurrent_gated_delta_rule'),
}
for name, symbols in required.items():
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    if not path.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError(f'FLA Core module was not loaded from the build virtual environment: {name}: {path}')
    for symbol in symbols:
        if not callable(getattr(module, symbol, None)):
            raise RuntimeError(f'FLA Core is missing a callable interface: {name}.{symbol}')
    if name == 'fla.modules.l2norm':
        inspect.signature(module.l2norm).bind(object(), dim=-1, eps=1e-6)
    print('FLA Core import check passed:', name, version, path)
PY
python - <<'PY'
import hashlib
import importlib.metadata as md
import sys
from pathlib import Path
distribution = md.distribution('transformer-engine')
assert distribution.version == '2.16+v0.1.0.ppu2.2.0', distribution.version
path = Path(distribution.locate_file(
    'transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py'
)).resolve()
assert path.is_relative_to(Path(sys.prefix).resolve()), path
source = path.read_text()
original = '''            dq[cu_seqlens_q_padded[-1] :].fill_(0)
            dk[cu_seqlens_kv_padded[-1] :].fill_(0)
            dv[cu_seqlens_kv_padded[-1] :].fill_(0)'''
replacement = '''            _ppu_zero_cp_padding(dq, cu_seqlens_q_padded)
            _ppu_zero_cp_padding(dk, cu_seqlens_kv_padded)
            _ppu_zero_cp_padding(dv, cu_seqlens_kv_padded)'''
assert source.count(original) == 1, 'TE CP patch target is not unique or the source has changed'
assert '_ppu_zero_cp_padding' not in source, 'TE CP patch is already present'
patched = source.replace(original, replacement) + '''

def _ppu_zero_cp_padding(gradient, cu_seqlens):
    indices = torch.arange(gradient.shape[0], device=gradient.device)
    mask = (indices >= cu_seqlens[-1]).view(-1, *([1] * (gradient.ndim - 1)))
    gradient.masked_fill_(mask, 0)
'''
compile(patched, str(path), 'exec')
path.write_text(patched)
print('TE CP graph-capture zero-padding fix:', path, 'sha256:', hashlib.sha256(patched.encode()).hexdigest())
PY
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
        raise RuntimeError(f'Dependency version violates constraint: {requirement}, installed {installed}')
    print(f'{requirement.name}=={installed}')
for name in (
    'numpy', 'scipy.linalg', 'sklearn.utils.murmurhash',
    'transformer_engine.pytorch', 'flash_attn.flash_attn_interface', 'flash_attn_2_cuda',
    'mamba_ssm.ops.triton.layernorm_gated', 'mamba_ssm.ops.triton.ssd_combined',
    'selective_scan_cuda', 'causal_conv1d', 'causal_conv1d_cuda',
    'zmq', 'msgpack', 'openai', 'fastapi', 'uvicorn', 'pyarrow', 'pyarrow.parquet', 'datasets', 'modelopt.torch',
):
    ppu_import_snapshot('before ' + name)
    module = importlib.import_module(name)
    print('Dependency import check passed:', name, module.__file__)
    ppu_import_snapshot('after ' + name)
import numpy as np
import pyarrow as pa
from datasets import Dataset
values = np.arange(4, dtype=np.int64)
np.testing.assert_array_equal(pa.array(values).to_numpy(), values)
assert Dataset.from_dict({'value': values}).to_dict() == {'value': values.tolist()}
print('NumPy, Arrow, and datasets conversion check passed')
import torch
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import _ppu_zero_cp_padding
for dtype in (torch.float32, torch.bfloat16):
    for noncontiguous in (False, True):
        original = torch.arange(54, device='cuda', dtype=torch.float32).reshape(9, 2, 3).to(dtype)
        original[0, 0, 0] = float('nan')
        original[1, 0, 0] = float('inf')
        original[8, 0, 0] = float('nan')
        if noncontiguous:
            original = original.transpose(1, 2)
        gradient = original.clone()
        cu_seqlens = torch.tensor([0, 5], dtype=torch.int32, device='cuda')
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                _ppu_zero_cp_padding(gradient, cu_seqlens)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            _ppu_zero_cp_padding(gradient, cu_seqlens)
        for end in (0, 1, 5, 9, 12):
            gradient.copy_(original)
            cu_seqlens[-1].fill_(end)
            graph.replay()
            expected = original.clone()
            expected[end:].zero_()
            torch.testing.assert_close(gradient, expected, rtol=0, atol=0, equal_nan=True)
        print('TE CP zero-padding graph capture and dynamic-boundary replay checks passed:', dtype, 'noncontiguous=', noncontiguous)
PY
install -m 0644 "$PPU_CONTROL_DIR/ppu_pytest_policy.py" \
    "$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')/ppu_pytest_policy.py"
python -m pip wheel --index-url "$PIP_INDEX_URL" --no-build-isolation --no-deps --wheel-dir dist .
python setup.py build_ext --inplace
make -C megatron/core/datasets
make -q -C megatron/core/datasets
python -m pip install --no-index --no-deps --ignore-installed dist/*.whl
if python -c 'import json; raise SystemExit(json.load(open(".ppu-ci/manifest.json")).get("test_kind") != "functional")'; then
    apt-get update
    apt-get install -y --no-install-recommends gettext-base
    python -I "$PPU_CONTROL_DIR/ppu_functional_environment.py" prepare --bundle "$PPU_ARTIFACT_DIR"
fi
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
from megatron.core.ssm import gated_delta_net, mamba_mixer
from megatron.core.inference.engines import dynamic_engine
import megatron.core.inference.inference_client
import google.protobuf
from importlib.metadata import version
assert Path(google.protobuf.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert google.protobuf.__version__ == version('protobuf')
print('protobuf:', google.protobuf.__version__, google.protobuf.__file__)
assert transformer_engine.HAVE_TE, 'Megatron did not enable Transformer Engine'
assert mamba_mixer.HAVE_MAMBA_SSM, 'Megatron did not enable Mamba SSM'
assert mamba_mixer.causal_conv1d_fn is not None, 'Megatron did not enable causal-conv1d'
assert gated_delta_net.HAVE_FLA, 'Megatron did not enable PPU FLA Core'
print('Megatron FLA capabilities:', 'GDN=', gated_delta_net.HAVE_FLA, 'GDN2=', gated_delta_net.HAVE_FLA_GDN2)
assert dynamic_engine.HAVE_ZMQ and dynamic_engine.HAVE_MSGPACK, 'Inference engine communication dependencies are incomplete'
assert Path(megatron.core.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
assert Path(helpers_cpp.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
print('megatron.core:', megatron.core.__version__, megatron.core.__file__)
print('torch:', torch.__version__, 'PPU device count visible to the build Pod:', torch.cuda.device_count())
print('PPU build and import checks passed:', helpers_cpp.__file__)
PY
python - <<'PY'
from pathlib import Path
import megatron.post_training.model_builder
import megatron.rl.rl_utils
for module in (megatron.post_training.model_builder, megatron.rl.rl_utils):
    assert Path(module.__file__).resolve().is_relative_to(Path.cwd().resolve())
    print('UT source import check passed:', module.__name__, module.__file__)
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
            print('UT tokenizer cache is ready:', name, flush=True)
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
    print('UT tokenizer offline check passed:', name)
PY
    touch .ppu-ci/data/.ready
fi
git ls-files -z > .ppu-ci/source-files.txt
tar -czf .ppu-ci/source.tar.gz --null -T .ppu-ci/source-files.txt megatron/core/datasets/helpers_cpp*.so
tar -czf .ppu-ci/environment.tar.gz -C /opt ppu-ci-venv
cp dist/*.whl .ppu-ci/
printf '%s\n' "$PPU_IMAGE" > .ppu-ci/image.txt
printf '%s\n' "$PPU_TARGET_SHA" > .ppu-ci/commit.txt
printf '%s\n' "$PPU_CONTROL_SHA" > .ppu-ci/control.txt
ended_at=$(date +%s)
printf '{"started_at":%s,"completed_at":%s,"elapsed_seconds":%s}\n' \
    "$started_at" "$ended_at" "$((ended_at - started_at))" > .ppu-ci/build-timing.json
python -I "$PPU_CONTROL_DIR/ppu_ci.py" seal-bundle --output "$PPU_ARTIFACT_DIR"
