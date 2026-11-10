"""Build-only exporters; archives contain materialized regular files, never host links."""

import hashlib
import importlib.metadata as md
import json
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
from pathlib import Path
from urllib.parse import urlsplit

EXPORT = Path('/exports')
VENV = Path('/opt/ppu-ci-venv')


def copy(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, symlinks=False, dirs_exist_ok=True)
    else:
        shutil.copy2(source.resolve(strict=True), destination)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_reports(paths):
    """Reject successful exports sourced outside the fixed package service."""
    from packaging.requirements import Requirement

    for path in paths:
        report = json.loads(Path(path).read_text())
        if report.get('version') != '1' or not isinstance(report.get('install'), list):
            raise RuntimeError('Invalid package installation report')
        for item in report['install']:
            url = urlsplit(item['download_info']['url'])
            if (
                item.get('is_direct') is not False
                or url.scheme != 'https'
                or url.hostname != 'pkg.flytiger-eco.com'
                or url.port not in (None, 443)
                or url.username is not None
                or url.password is not None
            ):
                raise RuntimeError('Package download bypassed the fixed package index')
            for requirement in item['metadata'].get('requires_dist', []):
                if Requirement(requirement).url is not None:
                    raise RuntimeError('Transitive dependency URL bypasses the fixed package index')


def vendor_bytes(base):
    return {
        path.relative_to(base).as_posix(): sha256(path)
        for name in ('torch', 'triton')
        for path in sorted((base / name).rglob('*'))
        if path.is_file() and path.suffix != '.pyc'
    }


def libraries(files, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for path in files:
        if not path.is_file():
            continue
        with path.open('rb') as stream:
            if stream.read(4) != b'\x7fELF':
                continue
        result = subprocess.run(['ldd', str(path)], capture_output=True, text=True, check=False)
        if 'not found' in result.stdout:
            raise RuntimeError('A native dependency is missing in the build container')
        if result.returncode and 'not a dynamic executable' not in result.stderr + result.stdout:
            raise RuntimeError('Unable to inventory native dependencies')
        for name in re.findall(r'(/[^\s()]+)', result.stdout):
            source = Path(name).resolve(strict=True)
            target = directory / Path(name).name
            if target.exists() and sha256(target) != sha256(source):
                raise RuntimeError('Conflicting native dependency SONAMEs')
            copy(source, target)


def archive(root, name, paths):
    EXPORT.mkdir(exist_ok=True)
    with tarfile.open(EXPORT / name, 'w:gz') as stream:
        for path in paths:
            for item in sorted([path, *path.rglob('*')]) if path.is_dir() else [path]:
                if item.is_symlink() or not (item.is_file() or item.is_dir()):
                    raise RuntimeError('Export requires materialized files and directories')
                stream.add(item, arcname=item.relative_to(root).as_posix(), recursive=False)


def isolate():
    base = Path(sysconfig.get_path('purelib'))
    packages = {item.metadata['Name']: item.version for item in md.distributions()}
    subprocess.run([sys.executable, '-m', 'venv', '--copies', str(VENV)], check=True)
    destination = VENV / 'lib/python3.12/site-packages'
    Path('/build/vendor-bytes.json').write_text(json.dumps(vendor_bytes(base), sort_keys=True))
    for item in base.iterdir():
        if item.suffix == '.pth' or item.name.startswith(('pip', 'setuptools', '_distutils_hack')):
            continue
        copy(item, destination / item.name)
    for path in (VENV / 'bin').iterdir():
        if path.is_symlink():
            content = path.resolve(strict=True)
            path.unlink()
            copy(content, path)
    for distribution in md.distributions():
        for entry in distribution.entry_points:
            if entry.group != 'console_scripts':
                continue
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', entry.name):
                raise RuntimeError('Invalid vendor console script name')
            module, symbol = entry.value.split(':', 1)
            symbol = symbol.split('[', 1)[0].strip()
            script = VENV / 'bin' / entry.name
            script.write_text(
                f'#!{VENV}/bin/python\nimport importlib, sys\n'
                f'function = importlib.import_module({module!r})\n'
                f'for part in {symbol!r}.split("."): function = getattr(function, part)\n'
                'sys.exit(function())\n'
            )
            script.chmod(0o755)
    Path('/build/base-packages.json').write_text(json.dumps(packages, sort_keys=True))


def control():
    verify_reports(['/build/control-install.json'])
    root = Path('/build/control-runtime')
    copy('/usr/local', root)
    lib = root / 'lib/ppu-system'
    libraries([root / 'bin/python3.12', *root.rglob('*.so'), *root.rglob('*.so.*')], lib)
    launcher = root / 'bin/python'
    if launcher.exists():
        launcher.unlink()
    launcher.write_text('''#!/bin/bash
set -euo pipefail
prefix="$(cd -- "$(dirname -- "$0")/.." && pwd)"
export LD_LIBRARY_PATH="$prefix/lib/ppu-system:$prefix/lib"
exec "$prefix/lib/ppu-system/ld-linux-x86-64.so.2" --library-path "$LD_LIBRARY_PATH" "$prefix/bin/python3.12" "$@"
''')
    launcher.chmod(0o755)
    archive(root, 'control-runtime.tar.gz', list(root.iterdir()))
    copy('/build/control-install.json', EXPORT / 'control-package-install.json')


def runtime():
    verify_reports(
        Path('/build') / name
        for name in (
            'packaging-install.json',
            'build-tools-install.json',
            'functional-package-install.json',
        )
    )
    site = Path(sysconfig.get_path('purelib'))
    if vendor_bytes(site) != json.loads(Path('/build/vendor-bytes.json').read_text()):
        raise RuntimeError('Dockerfile build changed vendor Torch or Triton bytes')
    for name in ('git', 'wget', 'envsubst', 'make'):
        path = shutil.which(name)
        if not path:
            raise RuntimeError('Required runtime tool is missing')
        copy(path, VENV / 'bin' / name)
    copy('/usr/lib/git-core', VENV / 'libexec/git-core')
    copy('/etc/ssl/certs/ca-certificates.crt', VENV / 'etc/ssl/certs/ca-certificates.crt')
    libraries(
        [VENV / 'bin' / name for name in ('git', 'wget', 'envsubst', 'make', 'yq')]
        + list((VENV / 'libexec/git-core').iterdir()),
        VENV / 'lib/ppu-system',
    )
    archive(Path('/opt'), 'environment.tar.gz', [VENV])
    source = Path('/build/source')
    extension = list((source / 'megatron/core/datasets').glob('helpers_cpp*.so'))
    if len(extension) != 1:
        raise RuntimeError('Exactly one source dataset native extension is required')
    archive(source, 'compiled-source.tar.gz', extension)
    for name in (
        'constraints.txt',
        'ppu-requirements.txt',
        'test-requirements.txt',
        'functional-package-install.json',
        'compatibility-patches.json',
        'base-packages.json',
        'vendor-bytes.json',
        'packaging-install.json',
        'build-tools-install.json',
        'source-package-install.json',
    ):
        copy(Path('/build') / name, EXPORT / name)
    (EXPORT / 'installed-packages.json').write_text(
        json.dumps(
            [
                {'name': item.metadata['Name'], 'version': item.version}
                for item in md.distributions()
            ],
            sort_keys=True,
        )
    )
    copy('/build/wheels', EXPORT / 'wheels')
    tools = EXPORT / 'functional-tools'
    for name in ('yq', 'envsubst'):
        copy(VENV / 'bin' / name, tools / name)
    files = {}
    for path in sorted(EXPORT.rglob('*')):
        if path.is_file():
            files[path.relative_to(EXPORT).as_posix()] = {
                'size': path.stat().st_size,
                'sha256': sha256(path),
            }
    (EXPORT / 'export-files.json').write_text(json.dumps(files, sort_keys=True))


if __name__ == '__main__':
    {'isolate': isolate, 'control': control, 'runtime': runtime}[sys.argv[1]]()
