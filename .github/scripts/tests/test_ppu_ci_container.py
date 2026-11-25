"""CPU-only regression gates for Dockerfile construction and authenticated restoration."""

import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / '.github/scripts'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


container = load('ppu_container_test', SCRIPTS / 'ppu_ci_container.py')
exporter = load('ppu_exporter_test', ROOT / '.github/docker/export_environment.py')
ci = load('ppu_container_bundle_test', SCRIPTS / 'ppu_ci.py')
CONTROL = 'a' * 40
TARGET = 'b' * 40
PINNED = container.PPU_IMAGE.rsplit(':', 1)[0] + '@sha256:' + 'c' * 64
PUBLISHED = container.GHCR_IMAGE + '@sha256:' + 'e' * 64
LOCAL_ID = 'sha256:' + 'f' * 64


def archive(path, rows):
    with tarfile.open(path, 'w:gz') as stream:
        for name, content, kind in rows:
            member = tarfile.TarInfo(name)
            member.mode, member.mtime = 0o6755, 1
            member.type = kind
            if kind == tarfile.REGTYPE:
                member.size = len(content)
                stream.addfile(member, io.BytesIO(content))
            else:
                member.linkname = content.decode()
                stream.addfile(member)


def runtime_export(root, *, schema=1):
    root.mkdir(exist_ok=True)
    if schema == 1:
        archive(
            root / 'environment.tar.gz', [('ppu-ci-venv/bin/python', b'python', tarfile.REGTYPE)]
        )
    for name in (
        'functional-package-install.json',
        'constraints.txt',
        'installed-packages.json',
        'compatibility-patches.json',
        'base-packages.json',
        'vendor-bytes.json',
        'export-files.json',
        'ppu-requirements.txt',
        'test-requirements.txt',
        'packaging-install.json',
        'build-tools-install.json',
        'source-package-install.json',
        'functional-tools/yq',
        'functional-tools/envsubst',
        'wheels/megatron.whl',
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'{}')
    return seal(root, schema=schema)


def seal(root, *, schema=1):
    metadata = dict(
        schema_version=schema,
        kind='runtime',
        platform='linux/amd64',
        control_sha=CONTROL,
        target_sha=TARGET,
        image=container.PPU_IMAGE,
        image_digest=PINNED,
        inputs_sha256='d' * 64,
        files=container.inventory(root, exclude=('docker-build.json', 'runtime-image.json')),
    )
    (root / 'docker-build.json').write_text(json.dumps(metadata))
    return container.digest(root / 'docker-build.json')


def write_image_record(root, trusted):
    metadata = json.loads((root / 'docker-build.json').read_text())
    (root / 'runtime-image.json').write_text(
        json.dumps(container.image_record(metadata, trusted, PUBLISHED))
    )


@pytest.fixture
def image_exported(tmp_path):
    root = tmp_path / 'image-export'
    trusted = runtime_export(root, schema=2)
    write_image_record(root, trusted)
    return root, trusted


def image_args(root, trusted, tmp_path):
    return SimpleNamespace(
        artifact=root,
        digest=trusted,
        control_sha=CONTROL,
        target_sha=TARGET,
        image_digest=PUBLISHED,
        base_image_digest=PINNED,
        source_root=tmp_path / 'source',
        check_only=True,
    )


@pytest.mark.parametrize(
    'name,kind,content',
    [
        ('../escape', tarfile.REGTYPE, b'x'),
        ('/absolute', tarfile.REGTYPE, b'x'),
        ('a/../../escape', tarfile.REGTYPE, b'x'),
        ('a\\escape', tarfile.REGTYPE, b'x'),
        ('a\nfile', tarfile.REGTYPE, b'x'),
        ('a', tarfile.SYMTYPE, b'../escape'),
        ('a', tarfile.LNKTYPE, b'../escape'),
        ('a', tarfile.CHRTYPE, b''),
        ('a', tarfile.FIFOTYPE, b''),
    ],
)
def test_archive_rejects_escaping_and_special_members(tmp_path, name, kind, content):
    path, destination = tmp_path / 'bad.tar.gz', tmp_path / 'restore'
    archive(path, [(name, content, kind)])
    with pytest.raises(container.ValidationError):
        container.extract_archive(path, destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    'rows',
    [
        [('a', b'one', tarfile.REGTYPE), ('a', b'two', tarfile.REGTYPE)],
        [('a/b', b'x', tarfile.REGTYPE), ('a', b'x', tarfile.REGTYPE)],
    ],
)
def test_archive_validates_all_members_before_writes(tmp_path, rows):
    path, destination = tmp_path / 'bad.tar.gz', tmp_path / 'restore'
    archive(path, rows)
    with pytest.raises(container.ValidationError):
        container.extract_archive(path, destination)
    assert not destination.exists()


@pytest.fixture
def build_inputs(tmp_path, monkeypatch):
    control, source = tmp_path / 'control', tmp_path / 'source'
    docker = control / '.github/docker'
    scripts = control / '.github/scripts'
    docker.mkdir(parents=True)
    scripts.mkdir(parents=True)
    source.mkdir()
    for name in container.DOCKER_FILES:
        shutil.copyfile(ROOT / '.github/docker' / name, docker / name)
    for name in ('ppu_ci_container.py', 'ppu_ci_container.sh', 'ppu_functional_pytest.py'):
        shutil.copyfile(SCRIPTS / name, scripts / name)
    shutil.copyfile(ROOT / 'pyproject.toml', source / 'pyproject.toml')
    monkeypatch.setattr(container, 'preflight', lambda root: None)
    monkeypatch.setattr(container, 'checked_git', lambda *args: None)
    tracked = ''.join(f'.github/scripts/{path.name}\0' for path in scripts.glob('*.py'))
    monkeypatch.setattr(container, 'command', lambda *args, **kwargs: tracked)
    return SimpleNamespace(
        kind='runtime',
        mode='build-image',
        local_image='ppu-ci-runtime:run-12-1',
        control_root=control,
        control_sha=CONTROL,
        source_root=source,
        target_sha=TARGET,
        image_digest=PINNED,
        output=tmp_path / 'export',
        cache_root=tmp_path / 'cache',
    )


def test_cache_is_dependency_based_but_build_inputs_bind_full_identities(build_inputs):
    args = build_inputs
    _, _, first_key, first_inputs = container.build_identity(args)
    args.target_sha = 'f' * 40
    _, _, second_key, second_inputs = container.build_identity(args)
    assert first_key == second_key and first_inputs != second_inputs
    args.control_sha = 'e' * 40
    assert container.build_identity(args)[2] == first_key
    for path in (
        args.source_root / 'pyproject.toml',
        args.control_root / '.github/docker/Dockerfile.ppu-ci',
    ):
        path.write_text(path.read_text() + '\n# changed\n')
        assert container.build_identity(args)[2] != first_key
        first_key = container.build_identity(args)[2]
    args.image_digest = PINNED[:-1] + 'e'
    assert container.build_identity(args)[2] != first_key


def assert_build_arguments(argv, args):
    expected = [
        f'{"PPU_IMAGE" if args.kind == "runtime" else "CONTROL_IMAGE"}={args.image_digest}',
        f'TARGET_SHA={args.target_sha or ""}',
        f'CONTROL_SHA={args.control_sha}',
    ]
    actual = [argv[index + 1] for index, value in enumerate(argv) if value == '--build-arg']
    assert actual == expected
    assert not any(value.endswith('--build-arg') for value in argv if value != '--build-arg')


@pytest.mark.parametrize(
    'damage', ['none', 'digest', 'base', 'image', 'record', 'file', 'legacy', 'software']
)
def test_image_handoff_checks_all_identities_without_restoring_software(
    image_exported, tmp_path, damage
):
    root, trusted = image_exported
    args = image_args(root, trusted, tmp_path)
    if damage == 'digest':
        args.digest = 'f' * 64
    elif damage == 'base':
        args.base_image_digest = PUBLISHED
    elif damage == 'image':
        args.image_digest = PINNED
    elif damage == 'record':
        (root / 'runtime-image.json').write_text('{}')
    elif damage == 'file':
        (root / 'constraints.txt').write_text('corrupt')
    elif damage == 'legacy':
        args.digest = seal(root)
        write_image_record(root, args.digest)
    elif damage == 'software':
        (root / 'environment.tar.gz').write_bytes(b'unexpected')
    if damage == 'none':
        container.verify_image(args)
    else:
        with pytest.raises(container.ValidationError):
            container.verify_image(args)
    assert not args.source_root.exists()
    assert not (tmp_path / 'opt').exists()


@pytest.mark.parametrize('stage', ['none', 'export', 'load', 'probe'])
def test_complete_image_build_loads_and_probes_exact_id_without_publication(
    build_inputs, monkeypatch, tmp_path, stage
):
    args = build_inputs
    args.mode, args.local_image = 'build-image', 'ppu-ci-runtime:run-12-1'
    outputs = tmp_path / 'job-outputs'
    monkeypatch.setenv('GITHUB_OUTPUT', str(outputs))
    monkeypatch.setenv('GH_TOKEN', 'fixture-secret')
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        if argv[0] == 'git':
            return (
                '.github/scripts/ppu_ci_container.py\0'
                if argv[2] == str(args.control_root)
                else 'pyproject.toml\0'
            )
        if argv[:3] == ['docker', 'buildx', 'build']:
            assert_build_arguments(argv, args)
        if '--output' in argv:
            if stage == 'export':
                raise container.ValidationError('export failed')
            runtime_export(args.output, schema=2)
        if '--load' in argv:
            assert '--push' not in argv
            metadata = Path(argv[-1]) / 'release/docker-build.json'
            assert metadata.read_bytes() == (args.output / 'docker-build.json').read_bytes()
            if stage == 'load':
                raise container.ValidationError('load failed')
        if argv[:3] == ['docker', 'image', 'inspect']:
            return LOCAL_ID
        if argv[:2] == ['docker', 'run']:
            assert LOCAL_ID in argv and args.local_image not in argv
            assert '--network=none' in argv and '--cap-drop=ALL' in argv
            assert '--security-opt=no-new-privileges' in argv
            exports = next(item for item in argv if 'target=/exports' in item)
            assert exports.endswith('/probe-exports,target=/exports,readonly')
            assert str(args.output) not in exports
            if stage == 'probe':
                raise container.ValidationError('probe failed')
        assert 'fixture-secret' not in str(argv)
        return ''

    monkeypatch.setattr(container, 'command', command)
    if stage == 'none':
        container.build(args)
        assert f'local_image_id={LOCAL_ID}' in outputs.read_text()
        assert 'runtime_image_digest=' not in outputs.read_text()
        assert not (args.output / 'environment.tar.gz').exists()
        assert not (args.output / 'runtime-image.json').exists()
    else:
        with pytest.raises(container.ValidationError):
            container.build(args)
        assert not outputs.exists()
    assert not any(call[:2] in (['docker', 'push'], ['docker', 'login']) for call in calls)
    assert 'GH_TOKEN' not in container.client_env()


@pytest.mark.parametrize(
    'damage',
    [
        'none',
        'id',
        'tag',
        'repository',
        'base',
        'evidence',
        'missing_digest',
        'wrong_digest',
        'ambiguous_digest',
        'invalid_digest_inventory',
        'push',
        'lookup',
    ],
)
def test_publication_only_pushes_verified_image_and_never_builds_or_runs_source(
    image_exported, tmp_path, monkeypatch, damage
):
    root, trusted = image_exported
    # Publication begins with build evidence, before any publication record exists.
    record_path = root / 'runtime-image.json'
    record_path.unlink()
    args = image_args(root, trusted, tmp_path)
    args.local_image, args.local_image_id = 'ppu-ci-runtime:run-12-1', LOCAL_ID
    args.image_tag = f'{container.GHCR_IMAGE}:run-12-1-{TARGET}'
    if damage == 'tag':
        args.image_tag = f'{container.GHCR_IMAGE}:latest'
    elif damage == 'repository':
        args.image_tag = f'ghcr.io/other/ppu-ci:run-12-1-{TARGET}'
    elif damage == 'base':
        args.base_image_digest = PINNED[:-1] + '0'
    elif damage == 'evidence':
        args.digest = '0' * 64
    output = tmp_path / 'outputs'
    monkeypatch.setenv('GITHUB_OUTPUT', str(output))
    monkeypatch.delenv('GH_TOKEN', raising=False)
    monkeypatch.setattr(
        'urllib.request.urlopen',
        lambda *a, **k: pytest.fail('Publication must not query package visibility APIs'),
    )
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        assert argv[:2] not in (['docker', 'run'], ['docker', 'build'])
        assert argv[:3] != ['docker', 'buildx', 'build']
        if '{{.Id}}' in argv:
            return 'sha256:' + '0' * 64 if damage == 'id' else LOCAL_ID
        if argv[:2] == ['docker', 'push'] and damage == 'push':
            raise container.ValidationError('push failed')
        if '{{json .RepoDigests}}' in argv:
            refs = {
                'missing_digest': [],
                'wrong_digest': ['ghcr.io/other/ppu-ci@sha256:' + 'e' * 64],
                'ambiguous_digest': [PUBLISHED, PUBLISHED[:-1] + '0'],
                'invalid_digest_inventory': [None],
            }.get(damage, [PUBLISHED])
            return json.dumps(refs)
        if argv[:3] == ['docker', 'buildx', 'imagetools']:
            assert argv[-1] == PUBLISHED
            if damage == 'lookup':
                raise container.ValidationError('lookup failed')
        return ''

    monkeypatch.setattr(container, 'command', command)
    if damage == 'none':
        container.publish_image(args)
        assert not hasattr(container, 'check_private_package')
        assert f'runtime_image_digest={PUBLISHED}' in output.read_text()
        assert f'base_image_digest={PINNED}' in output.read_text()
        assert f'environment_digest={trusted}' in output.read_text()
        assert [argv[:2] for argv in calls] == [
            ['docker', 'image'],
            ['docker', 'tag'],
            ['docker', 'push'],
            ['docker', 'image'],
            ['docker', 'buildx'],
        ]
        assert calls[1] == ['docker', 'tag', LOCAL_ID, args.image_tag]
        container.verify_image(args)
    else:
        with pytest.raises(container.ValidationError):
            container.publish_image(args)
        assert not output.exists() and not record_path.exists()
    if damage in ('id', 'tag', 'repository', 'base', 'evidence'):
        assert not any(call[:2] in (['docker', 'tag'], ['docker', 'push']) for call in calls)


@pytest.mark.parametrize('damage', ['none', 'image', 'record', 'legacy', 'missing_report'])
def test_schema_two_bundle_is_sealed_without_environment_archive(image_exported, damage):
    root, _ = image_exported
    for name in (
        'commit.txt',
        'control.txt',
        'build-timing.json',
        'control/ppu_ci.py',
        'control/ppu_ci_run.sh',
        'control/ppu_pytest_policy.py',
        'control/ppu_ci_container.py',
        'control/ppu_ci_container.sh',
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
    (root / 'image.txt').write_text(PUBLISHED + '\n')
    (root / 'manifest.json').write_text(
        json.dumps(
            {
                'environment_build_schema': 1 if damage == 'legacy' else 2,
                'identity': {'control_sha': CONTROL, 'target_sha': TARGET},
            }
        )
    )
    if damage == 'image':
        (root / 'image.txt').write_text(PINNED)
    elif damage == 'record':
        (root / 'runtime-image.json').write_text('{}')
    elif damage == 'missing_report':
        (root / 'packaging-install.json').unlink()
    if damage == 'none':
        ci.seal_bundle(root)
        files = json.loads((root / 'bundle-integrity.json').read_text())['files']
        assert 'runtime-image.json' in files
        assert not {'source.tar.gz', 'compiled-source.tar.gz', 'environment.tar.gz'} & files.keys()
    else:
        with pytest.raises((ValueError, KeyError)):
            ci.seal_bundle(root)
        assert not (root / 'bundle-integrity.json').exists()


@pytest.fixture
def subprocess_bundle(tmp_path):
    plan = tmp_path / 'plan'
    control = plan / 'control'
    control.mkdir(parents=True)
    for path in SCRIPTS.glob('ppu_*'):
        if path.suffix in ('.py', '.sh'):
            shutil.copyfile(path, control / path.name)
    (plan / 'manifest.json').write_text(
        json.dumps(
            {
                'schema_version': 2,
                'test_kind': 'functional',
                'runtime_path_schema': 1,
                'environment_build_schema': 2,
                'identity': {'control_sha': CONTROL, 'target_sha': TARGET},
            }
        )
    )
    root = tmp_path / 'bundle'
    trusted = runtime_export(root, schema=2)
    write_image_record(root, trusted)
    shutil.copytree(control, root / 'control')
    shutil.copyfile(plan / 'manifest.json', root / 'manifest.json')
    archive(root / 'source.tar.gz', [('source.py', b'VALUE = 1\n', tarfile.REGTYPE)])
    for name, content in (
        ('commit.txt', TARGET),
        ('control.txt', CONTROL),
        ('.complete', TARGET),
        ('image.txt', PUBLISHED),
        ('build-timing.json', '{"elapsed_seconds":1}'),
        ('functional-environment.json', '{}'),
    ):
        (root / name).write_text(content + '\n')
    return plan, root


def tree_bytes(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob('*')
        if path.is_file()
    }


@pytest.mark.parametrize('flags', [('-I',), ('-I', '-B')])
@pytest.mark.parametrize('executor', ['bundle', 'plan'])
def test_real_bundle_seal_and_verify_do_not_mutate_control(subprocess_bundle, flags, executor):
    plan, root = subprocess_bundle
    before = tree_bytes(root)
    trusted_control = tree_bytes(plan / 'control')
    assert not list(root.rglob('*.pyc'))
    script = (root if executor == 'bundle' else plan) / 'control/ppu_ci.py'
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, *flags, str(script)]
    sealed = subprocess.run(
        [*command, 'seal-bundle', '--output', str(root)],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert sealed.returncode == 0, sealed.stderr
    assert tree_bytes(root / 'control') == trusted_control
    assert tree_bytes(plan / 'control') == trusted_control
    assert not list(root.rglob('*.pyc')) and not list(plan.rglob('*.pyc'))
    files = json.loads((root / 'bundle-integrity.json').read_text())['files']
    assert set(files) == set(before) - {'.complete'}
    after_seal = tree_bytes(root)
    assert {
        name: content for name, content in after_seal.items() if name != 'bundle-integrity.json'
    } == before
    for _ in range(2):
        checked = subprocess.run(
            [
                *command,
                'verify-bundle',
                '--output',
                str(root),
                '--plan',
                str(plan),
                '--bundle-digest',
                container.digest(root / 'bundle-integrity.json'),
            ],
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert checked.returncode == 0, checked.stderr
        assert tree_bytes(root) == after_seal
        assert tree_bytes(plan / 'control') == trusted_control


@pytest.mark.parametrize('reseal', [False, True])
def test_real_bundle_verification_still_rejects_added_bytecode(subprocess_bundle, reseal):
    plan, root = subprocess_bundle
    command = [sys.executable, '-I', '-B', str(plan / 'control/ppu_ci.py')]
    subprocess.run([*command, 'seal-bundle', '--output', str(root)], check=True, timeout=30)
    cache = root / 'control/__pycache__/unexpected.cpython-312.pyc'
    cache.parent.mkdir()
    cache.write_bytes(b'untrusted bytecode')
    if reseal:
        subprocess.run([*command, 'seal-bundle', '--output', str(root)], check=True, timeout=30)
    checked = subprocess.run(
        [
            *command,
            'verify-bundle',
            '--output',
            str(root),
            '--plan',
            str(plan),
            '--bundle-digest',
            container.digest(root / 'bundle-integrity.json'),
        ],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert checked.returncode != 0
    expected = 'control script inventory mismatch' if reseal else 'failed content verification'
    assert expected in checked.stderr
    assert cache.read_bytes() == b'untrusted bytecode'


@pytest.mark.parametrize(
    'damage', ['none', 'missing', 'modified', 'not_executable', 'symlink', 'wrong_output']
)
def test_execution_yq_probe_executes_parse_and_rejects_invalid_copy(tmp_path, monkeypatch, damage):
    calls = []
    real_run = subprocess.run
    monkeypatch.setattr(
        container.subprocess,
        'run',
        lambda argv, **kwargs: calls.append(argv) or SimpleNamespace(returncode=0),
    )
    container.portable_checks(tmp_path, execution_image=True)
    code = next(argv[-1] for argv in calls if 'input="TEST_TYPE: regular' in argv[-1])
    local, baked = tmp_path / 'yq', tmp_path / 'baked-yq'
    payload = '#!/bin/sh\ncat >/dev/null\nprintf "regular\\n"\n'
    if damage == 'wrong_output':
        payload = payload.replace('regular', 'wrong')
    baked.write_text(payload)
    local.write_text(payload)
    local.chmod(0o755)
    if damage == 'missing':
        local.unlink()
    elif damage == 'modified':
        local.write_text(payload + '# changed\n')
    elif damage == 'not_executable':
        local.chmod(0o644)
    elif damage == 'symlink':
        local.unlink()
        local.symlink_to(baked)
    code = code.replace('/usr/local/bin/yq', str(local)).replace(
        '/opt/ppu-ci-venv/bin/yq', str(baked)
    )
    result = real_run([sys.executable, '-I', '-c', code], capture_output=True, text=True)
    assert (result.returncode == 0) == (damage == 'none')


@pytest.mark.parametrize('dependent_first', [True, False])
def test_native_closure_resolves_bundled_transitive_libraries(
    tmp_path, monkeypatch, dependent_first
):
    extension = tmp_path / 'extension.so'
    bundled = tmp_path / 'numpy.libs'
    bundled.mkdir()
    fortran = bundled / 'libgfortran-040039e1.so.5.0.0'
    quadmath = bundled / 'libquadmath-96973f99.so.0.0.0'
    for path in (extension, fortran, quadmath):
        path.write_bytes(b'\x7fELF' + path.name.encode())
    directory = tmp_path / 'ppu-system'
    calls = []
    monkeypatch.setenv('LD_LIBRARY_PATH', '/private/unrelated')
    monkeypatch.setenv('LC_ALL', 'private-locale')

    def run(argv, **kwargs):
        path = Path(argv[1])
        calls.append(path)
        assert argv[0] == 'ldd'
        assert kwargs['env']['LD_LIBRARY_PATH'] == str(directory)
        assert kwargs['env']['LC_ALL'] == 'C'
        if path == extension:
            output = f'{fortran.name} => {fortran} (0x1234)\n'
            output += f'{quadmath.name} => {quadmath} (0x1234)\n'
        elif path == fortran:
            saved = directory / quadmath.name
            output = (
                f'{quadmath.name} => {saved} (0x1234)\n'
                if saved.exists()
                else f'{quadmath.name} => not found\n'
            )
        else:
            output = ''
        return SimpleNamespace(returncode=0, stdout=output, stderr='private-warning')

    monkeypatch.setattr(exporter.subprocess, 'run', run)
    files = [fortran, quadmath, extension] if dependent_first else [extension, fortran, quadmath]
    original = {path: path.read_bytes() for path in files}
    exporter.libraries(files, directory)
    assert calls.count(fortran) == (2 if dependent_first else 1)
    assert (directory / fortran.name).read_bytes() == original[fortran]
    assert (directory / quadmath.name).read_bytes() == original[quadmath]
    assert {path: path.read_bytes() for path in files} == original


def test_native_closure_still_rejects_missing_dependencies_after_progress(tmp_path, monkeypatch):
    extension, missing, library = [
        tmp_path / name for name in ('extension.so', 'missing.so', 'lib.so')
    ]
    for path in (extension, missing, library):
        path.write_bytes(b'\x7fELF' + path.name.encode())
    calls = []

    def run(argv, **kwargs):
        path = Path(argv[1])
        calls.append(path)
        output = (
            f'lib.so => {library} (0x1234)\n'
            if path == extension
            else 'libgcc_s.so.1 => not found\nprivate-library.so => not found\n'
        )
        return SimpleNamespace(returncode=0, stdout=output, stderr='private-token')

    monkeypatch.setattr(exporter.subprocess, 'run', run)
    with pytest.raises(RuntimeError) as error:
        exporter.libraries([missing, extension], tmp_path / 'ppu-system')
    assert str(error.value) == (
        'A native dependency is missing in the build container; '
        'missing-native-libgcc,missing-native-object-other,missing-native-unknown'
    )
    assert calls == [missing, extension, missing]


def test_native_closure_rejects_conflicting_sonames(tmp_path, monkeypatch):
    files = []
    for name in ('first', 'second'):
        root = tmp_path / name
        root.mkdir()
        extension, library = root / 'extension.so', root / 'lib.so'
        extension.write_bytes(b'\x7fELF')
        library.write_bytes(name.encode())
        files.append(extension)

    def run(argv, **kwargs):
        library = Path(argv[1]).parent / 'lib.so'
        return SimpleNamespace(returncode=0, stdout=f'lib.so => {library} (0x1234)\n', stderr='')

    monkeypatch.setattr(exporter.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='^Conflicting native dependency SONAMEs$'):
        exporter.libraries(files, tmp_path / 'ppu-system')


@pytest.mark.parametrize(
    'url,direct,requirements',
    [
        ('https://files.pythonhosted.org/pkg.whl', False, []),
        ('http://pkg.flytiger-eco.com/pkg.whl', False, []),
        ('https://pkg.flytiger-eco.com:444/pkg.whl', False, []),
        ('https://fixture@pkg.flytiger-eco.com/pkg.whl', False, []),
        ('https://pkg.flytiger-eco.com/pkg.whl', True, []),
        ('https://pkg.flytiger-eco.com/pkg.whl', False, ['pkg @ https://example.com/pkg.whl']),
    ],
)
def test_installation_reports_reject_external_or_direct_sources(
    tmp_path, url, direct, requirements
):
    report = tmp_path / 'install.json'
    report.write_text(
        json.dumps(
            {
                'version': '1',
                'install': [
                    {
                        'is_direct': direct,
                        'download_info': {'url': url},
                        'metadata': {'requires_dist': requirements},
                    }
                ],
            }
        )
    )
    with pytest.raises(RuntimeError, match='fixed package index'):
        exporter.verify_reports([report])


@pytest.mark.parametrize(
    'filename', ['ppu_ci_container.py', 'ppu_ci_container.sh', 'ppu_functional_pytest.py']
)
def test_orchestration_changes_do_not_invalidate_dependency_cache(build_inputs, filename):
    args = build_inputs
    before = container.build_identity(args)
    helper = args.control_root / '.github/scripts' / filename
    helper.write_text(helper.read_text() + '\n# orchestration change\n')
    after = container.build_identity(args)
    assert before[2] == after[2]
    assert before[3] != after[3]


def test_control_requirements_do_not_invalidate_runtime_dependencies(build_inputs):
    args = build_inputs
    before = container.build_identity(args)
    requirements = args.control_root / '.github/docker/control-requirements.txt'
    requirements.write_text(requirements.read_text() + '\n# control-only change\n')
    after = container.build_identity(args)
    assert before[2] == after[2] and before[3] != after[3]


def test_cache_population_never_compiles_or_executes_target_source(build_inputs, monkeypatch):
    args = build_inputs
    args.mode, args.cache_root = 'build-cache', None
    args.cache_from = container.GHCR_IMAGE + ':cache-main-runtime-amd64-baseline'
    args.cache_to = container.GHCR_IMAGE + ':cache-main-runtime-amd64-build-' + 'a' * 64
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        if argv[0] == 'git':
            assert argv[2] == str(args.control_root)
            return '.github/scripts/ppu_ci_container.py\0'
        assert argv[:3] == ['docker', 'buildx', 'build']
        assert argv[argv.index('--target') + 1] == 'dependencies'
        assert argv[argv.index('--cache-from') + 1] == f'type=registry,ref={args.cache_from}'
        assert argv[argv.index('--cache-to') + 1] == f'type=registry,ref={args.cache_to},mode=max'
        assert '--load' not in argv and '--output' not in argv and '--push' not in argv
        context = Path(argv[-1])
        assert not (context / 'source').exists()
        assert not (context / 'control').exists()
        assert (context / 'pyproject.toml').read_bytes() == (
            args.source_root / 'pyproject.toml'
        ).read_bytes()
        return ''

    monkeypatch.setattr(container, 'command', command)
    container.build(args)
    assert len([call for call in calls if call[0] == 'docker']) == 1
    assert not args.output.exists()


@pytest.mark.parametrize(
    'ref',
    [
        'other.example/cache:tag',
        container.GHCR_IMAGE + ':run-123',
        container.GHCR_IMAGE + ':cache-main,mode=min',
        container.GHCR_IMAGE + ':cache-main\n',
    ],
)
def test_registry_cache_rejects_invalid_destinations_before_build(build_inputs, monkeypatch, ref):
    args = build_inputs
    args.cache_root, args.cache_to = None, ref
    monkeypatch.setattr(
        container, 'build_identity', lambda *a: pytest.fail('invalid cache reached build')
    )
    with pytest.raises(container.ValidationError, match='cache registry reference'):
        container.build(args)


@pytest.mark.parametrize(
    'pr,available,populate,expected_read,expected_write',
    [
        ('15', ['pr', 'target', 'baseline'], 'false', 'pr', 'pr'),
        ('15', ['target', 'baseline'], 'false', 'target', 'pr'),
        ('15', ['baseline'], 'false', 'baseline', 'pr'),
        ('15', [], 'false', '', 'pr'),
        ('0', ['target', 'baseline'], 'false', 'target', 'baseline'),
        ('0', ['baseline'], 'true', 'baseline', 'target'),
        ('0', ['candidate', 'target'], 'false', 'candidate', 'candidate'),
        ('0', ['baseline'], 'false', 'baseline', 'candidate'),
    ],
)
@pytest.mark.parametrize('branch', ['main', 'dev'])
def test_registry_cache_action_selects_donors_and_isolates_pr_writes(
    tmp_path, pr, available, populate, expected_read, expected_write, branch
):
    action = yaml.safe_load((ROOT / '.github/actions/ppu-build-cache/action.yml').read_text())
    script = action['runs']['steps'][0]['run']
    prefix = container.GHCR_IMAGE + f':cache-{branch}-runtime-amd64'
    key = 'b' * 64
    refs = {
        'pr': f'{prefix}-pr-{pr}-{key}',
        'target': f'{prefix}-build-{key}',
        'baseline': f'{prefix}-baseline',
        'candidate': f'{prefix}-candidate-{key}',
        '': '',
    }
    docker = tmp_path / 'docker'
    docker.write_text(
        '#!/bin/bash\n[[ "$1 $2 $3" == "buildx imagetools inspect" ]] || exit 9\nwhile IFS= read -r ref; do [[ "$4" == "$ref" ]] && exit 0; done < "$AVAILABLE"\nexit 1\n'
    )
    docker.chmod(0o755)
    available_file = tmp_path / 'available'
    available_file.write_text('\n'.join(refs[name] for name in available) + '\n')
    output, summary = tmp_path / 'output', tmp_path / 'summary'
    env = {
        **os.environ,
        'PATH': f'{tmp_path}{os.pathsep}{os.environ["PATH"]}',
        'AVAILABLE': str(available_file),
        'BUILD_KEY': key,
        'BASE_REF': f'refs/heads/{branch}',
        'PR_NUMBER': pr,
        'PUBLISH_TARGET_CACHE': populate,
        'WRITE_BASELINE': 'true' if expected_write == 'baseline' else 'false',
        'GITHUB_OUTPUT': str(output),
        'GITHUB_STEP_SUMMARY': str(summary),
    }
    subprocess.run(
        ['bash', '-e', '-o', 'pipefail', '-c', script],
        env=env,
        check=True,
        timeout=10,
        capture_output=True,
    )
    values = dict(line.split('=', 1) for line in output.read_text().splitlines())
    assert values == {'cache-from': refs[expected_read], 'cache-to': refs[expected_write]}
    assert 'BuildKit validates individual layers' in summary.read_text()


@pytest.mark.parametrize(
    'event,pr,sync_source,expected',
    [
        ('push', 0, None, 'true'),
        ('workflow_dispatch', 0, None, 'true'),
        ('pull_request_target', 15, None, 'false'),
        ('workflow_dispatch', 15, None, 'false'),
        ('merge_group', 0, None, 'false'),
        ('push', 0, 'sync/upstream-main/' + 'a' * 40, 'false'),
        ('push', 0, 'dev', 'true'),
    ],
)
def test_cache_scope_distinguishes_published_branches_from_candidates(
    tmp_path, event, pr, sync_source, expected
):
    workflow = yaml.safe_load((ROOT / '.github/workflows/ppu-execute.yml').read_text())
    step = next(
        s for s in workflow['jobs']['runtime-environment']['steps'] if s.get('id') == 'cache-scope'
    )
    branch = 'dev' if sync_source == 'dev' else 'main'
    authorization = {'event_name': event, 'pr_number': pr, 'base_ref': f'refs/heads/{branch}'}
    if sync_source:
        authorization['sync_request'] = {'source_ref': sync_source, 'target_branch': branch}
    (tmp_path / '.ppu-ci').mkdir()
    (tmp_path / '.ppu-ci/authorization.json').write_text(json.dumps(authorization))
    output = tmp_path / 'outputs'
    subprocess.run(
        ['bash', '-e', '-o', 'pipefail', '-c', step['run']],
        cwd=tmp_path,
        env={**os.environ, 'GITHUB_OUTPUT': str(output)},
        check=True,
        capture_output=True,
        timeout=10,
    )
    values = dict(line.split('=', 1) for line in output.read_text().splitlines())
    assert values == {
        'base_ref': f'refs/heads/{branch}',
        'pr_number': str(pr),
        'write_baseline': expected,
    }


@pytest.mark.parametrize('damage', [None, 'metadata', 'source'])
def test_execution_uses_only_the_source_baked_into_the_verified_image(
    tmp_path, monkeypatch, damage
):
    source = tmp_path / 'baked-source'
    source.mkdir()
    venv = tmp_path / 'venv'
    (venv / 'bin').mkdir(parents=True)
    (venv / 'bin/python').write_text('python')
    metadata = tmp_path / 'docker-build.json'
    metadata.write_text('{}')
    expected = container.digest(metadata)
    if damage == 'metadata':
        metadata.write_text('{"changed": true}')
    roots = {Path('/opt/ppu-ci-venv'): venv, Path('/opt/ppu-ci-image/docker-build.json'): metadata}
    monkeypatch.setattr(container, 'no_links', lambda path: roots.get(Path(path), Path(path)))
    monkeypatch.setattr(container, 'SOURCE_ROOT', source)
    args = SimpleNamespace(
        base_image_digest=PINNED,
        digest=expected,
        source_root=tmp_path / 'uploaded-source' if damage == 'source' else source,
    )
    if damage:
        with pytest.raises(container.ValidationError):
            container.verify_installed_image(args, {'image_digest': PINNED})
    else:
        container.verify_installed_image(args, {'image_digest': PINNED})
    assert not (tmp_path / 'uploaded-source').exists()
    assert list(source.iterdir()) == []
