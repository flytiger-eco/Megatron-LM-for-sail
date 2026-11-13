#!/usr/bin/env bash
set -euo pipefail
[[ -f /build/export_environment.py && -f /build/pyproject.toml ]]
export VIRTUAL_ENV=/opt/ppu-ci-venv
export PATH="$VIRTUAL_ENV/bin:$PATH"
cd /build
python - <<'PY'
import importlib.metadata as md
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

native = [
    'transformer-engine==2.16+v0.1.0.ppu2.2.0',
    'flash-attn==2.8.2+v0.2.0.ppu2.2.0',
    'mamba-ssm==2.3.1+v0.1.0.ppu2.2.0',
    'causal-conv1d==1.7.0+v0.1.0.ppu2.2.0',
    'pla==1.0.0+v0.1.0.ppu2.2.0',
    'fla-core==0.4.1+v0.1.0.ppu2.2.0',
    'flash-linear-attention==0.4.1+v0.1.0.ppu2.2.0',
    'emerging-optimizers==0.3.0',
    'apex==24.4.1+v0.1.0.ppu2.2.0',
    'nvidia-resiliency-ext==0.6.0',
]
base = []
for name in ('torch', 'torchvision', 'triton', 'scipy', 'scikit-learn'):
    try:
        base.append(f'{name}=={md.version(name)}')
    except md.PackageNotFoundError:
        if name in ('torch', 'triton'):
            raise RuntimeError('Required vendor runtime is missing') from None
Path('ppu-requirements.txt').write_text('\n'.join(native) + '\n')
Path('constraints.txt').write_text('\n'.join(base + native + ['numpy==1.26.4', 'pyarrow==21.0.0']) + '\n')
project = tomllib.loads(Path('pyproject.toml').read_text())
requirements = (project['project']['dependencies'] + project['dependency-groups']['test']
                + project['project']['optional-dependencies']['training'])
for requirement in requirements:
    if not isinstance(requirement, str) or '\n' in requirement or '\r' in requirement:
        raise RuntimeError('Dependency requirements must be single-line package specifications')
    if Requirement(requirement).url is not None:
        raise RuntimeError('Direct dependency URLs bypass the fixed package index')
Path('test-requirements.txt').write_text('\n'.join(requirements) + '\n')
PY
export PIP_CONSTRAINT=/build/constraints.txt
python -m pip install --index-url "$PIP_INDEX_URL" --report /build/packaging-install.json --upgrade 'packaging>=24.2'
python -m pip install --index-url "$PIP_INDEX_URL" --report /build/build-tools-install.json \
    'setuptools>=80' wheel pybind11 ninja click pyyaml numpy
python -m pip install --index-url "$PIP_INDEX_URL" --no-build-isolation \
    --report /build/functional-package-install.json \
    -r /build/test-requirements.txt -r /build/ppu-requirements.txt \
    pytest-timeout click wrapt requests tensorstore scipy scikit-learn \
    'pydantic<2.14' boto3 'multi-storage-client~=0.50' 'einops~=0.8' 'uv==0.13.0' \
    'google-api-python-client>=2' pyzmq msgpack 'openai[aiohttp]' fastapi uvicorn datasets \
    'nvidia-modelopt==0.45.0' 'tensorboard==2.20.0' 'google-crc32c==1.7.1' 'pyyaml==6.0.3'
python -I - <<'PY'
import hashlib
import importlib.metadata as md
import sys
import sysconfig
from pathlib import Path

distribution = md.distribution('fla-core')
if distribution.version != '0.4.1+v0.1.0.ppu2.2.0':
    raise RuntimeError('FLA Core version does not match the PPU constraint')
path = Path(distribution.locate_file('fla/modules/l2norm.py')).resolve()
if not path.is_relative_to(Path(sys.prefix).resolve()):
    raise RuntimeError('FLA Core patch target is outside the build environment')
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
    raise RuntimeError('FLA Core patch target is not unique or the source has changed')
patched = source.replace(original, replacement)
compile(patched, str(path), 'exec')
path.write_text(patched)
patches = {'fla-l2norm': hashlib.sha256(patched.encode()).hexdigest()}

distribution = md.distribution('transformer-engine')
if distribution.version != '2.16+v0.1.0.ppu2.2.0':
    raise RuntimeError('Transformer Engine version does not match the PPU constraint')
path = Path(distribution.locate_file(
    'transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py'
)).resolve()
if not path.is_relative_to(Path(sys.prefix).resolve()):
    raise RuntimeError('TE patch target is outside the build environment')
source = path.read_text()
original = '''            dq[cu_seqlens_q_padded[-1] :].fill_(0)
            dk[cu_seqlens_kv_padded[-1] :].fill_(0)
            dv[cu_seqlens_kv_padded[-1] :].fill_(0)'''
replacement = '''            _ppu_zero_cp_padding(dq, cu_seqlens_q_padded)
            _ppu_zero_cp_padding(dk, cu_seqlens_kv_padded)
            _ppu_zero_cp_padding(dv, cu_seqlens_kv_padded)'''
if source.count(original) != 1 or '_ppu_zero_cp_padding' in source:
    raise RuntimeError('TE CP patch target is not unique or the source has changed')
patched = source.replace(original, replacement) + '''

def _ppu_zero_cp_padding(gradient, cu_seqlens):
    indices = torch.arange(gradient.shape[0], device=gradient.device)
    mask = (indices >= cu_seqlens[-1]).view(-1, *([1] * (gradient.ndim - 1)))
    gradient.masked_fill_(mask, 0)
'''
compile(patched, str(path), 'exec')
path.write_text(patched)
patches['te-cp-padding'] = hashlib.sha256(patched.encode()).hexdigest()
import json
Path('/build/compatibility-patches.json').write_text(json.dumps(patches, sort_keys=True))
site = Path(sysconfig.get_path('purelib'))
(site / '00-ppu-functional-google-namespace.pth').write_text(
    'import importlib; importlib.invalidate_caches(); import google\n'
)
PY
python -m pip check
python - <<'PY'
import hashlib
import urllib.request
from pathlib import Path

url = 'https://github.com/mikefarah/yq/releases/download/v4.47.2/yq_linux_amd64'
with urllib.request.urlopen(url, timeout=60) as response:
    content = response.read()
if hashlib.sha256(content).hexdigest() != '1bb99e1019e23de33c7e6afc23e93dad72aad6cf2cb03c797f068ea79814ddb0':
    raise RuntimeError('Official Mike Farah yq binary digest mismatch')
path = Path('/opt/ppu-ci-venv/bin/yq')
path.write_bytes(content)
path.chmod(0o755)
PY
