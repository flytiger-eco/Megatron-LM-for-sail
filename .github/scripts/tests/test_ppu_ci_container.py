"""CPU-only regression gates for Dockerfile construction and authenticated restoration."""

import importlib.util
import io
import json
import os
import re
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
bert = load('ppu_legacy_restore_test', SCRIPTS / 'ppu_bert_diagnostic.py')
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
    archive(
        root / 'compiled-source.tar.gz',
        [
            (
                'megatron/core/datasets/helpers_cpp.cpython-312-x86_64-linux-gnu.so',
                b'extension',
                tarfile.REGTYPE,
            )
        ],
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


@pytest.fixture
def exported(tmp_path):
    root = tmp_path / 'export'
    return root, runtime_export(root)


def restore_args(root, digest, tmp_path):
    return SimpleNamespace(
        artifact=root,
        digest=digest,
        control_sha=CONTROL,
        target_sha=TARGET,
        image_digest=PINNED,
        destination=tmp_path / 'opt',
        source_root=tmp_path / 'source',
        check_only=False,
    )


@pytest.mark.parametrize(
    'mutation',
    ['digest', 'control', 'target', 'image', 'platform', 'file', 'extra', 'symlink', 'schema'],
)
def test_authenticated_export_rejects_mutations(exported, mutation):
    root, trusted = exported
    control, target = CONTROL, TARGET
    metadata = json.loads((root / 'docker-build.json').read_text())
    if mutation == 'digest':
        trusted = 'f' * 64
    elif mutation == 'control':
        control = 'e' * 40
    elif mutation == 'target':
        target = 'e' * 40
    elif mutation in ('image', 'platform', 'schema'):
        field, value = {
            'image': ('image_digest', 'evil/image@sha256:' + 'c' * 64),
            'platform': ('platform', 'linux/arm64'),
            'schema': ('schema_version', True),
        }[mutation]
        metadata[field] = value
        (root / 'docker-build.json').write_text(json.dumps(metadata))
        trusted = container.digest(root / 'docker-build.json')
    elif mutation == 'file':
        (root / 'constraints.txt').write_bytes(b'corrupt')
    elif mutation == 'extra':
        (root / 'unexpected').touch()
    else:
        (root / 'linked').symlink_to(root / 'constraints.txt')
    with pytest.raises(container.ValidationError):
        container.verify_export(root, trusted, control, target)


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


def test_archive_refuses_linked_destination_and_existing_file(tmp_path):
    path, destination = tmp_path / 'ok.tar.gz', tmp_path / 'restore'
    archive(path, [('a', b'x', tarfile.REGTYPE)])
    destination.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(container.ValidationError):
        container.extract_archive(path, destination)
    other = tmp_path / 'existing'
    other.mkdir()
    (other / 'a').write_bytes(b'original')
    with pytest.raises(container.ValidationError):
        container.extract_archive(path, other)
    assert (other / 'a').read_bytes() == b'original'


def test_runtime_restore_checks_image_and_both_archives_before_writes(exported, tmp_path):
    root, digest = exported
    args = restore_args(root, digest, tmp_path)
    args.image_digest = PINNED[:-1] + 'e'
    with pytest.raises(container.ValidationError, match='base image mismatch'):
        container.restore_runtime(args)
    args.image_digest = PINNED
    archive(root / 'compiled-source.tar.gz', [('evil.py', b'exec', tarfile.REGTYPE)])
    args.digest = seal(root)
    with pytest.raises(container.ValidationError, match='compiled source'):
        container.restore_runtime(args)
    assert not args.destination.exists()


def test_runtime_restores_without_installer_and_removes_privilege_bits(
    exported, tmp_path, monkeypatch
):
    root, digest = exported
    monkeypatch.setattr(
        container,
        'command',
        lambda *args, **kwargs: pytest.fail('Restore must not install software'),
    )
    args = restore_args(root, digest, tmp_path)
    container.restore_runtime(args)
    python = args.destination / 'ppu-ci-venv/bin/python'
    assert python.read_bytes() == b'python' and python.stat().st_mode & 0o6000 == 0
    extension = next(args.source_root.rglob('*.so'))
    assert extension.read_bytes() == b'extension' and extension.stat().st_mtime > 1
    with pytest.raises(container.ValidationError, match='overwrite'):
        container.restore_runtime(args)


def test_check_only_does_not_extract(exported, tmp_path):
    root, digest = exported
    args = restore_args(root, digest, tmp_path)
    args.check_only = True
    container.restore_runtime(args)
    assert not args.destination.exists() and not args.source_root.exists()


def test_source_restore_requires_sealed_archive_identity(tmp_path):
    root = tmp_path / 'bundle'
    root.mkdir()
    archive(root / 'source.tar.gz', [('source.py', b'pass', tarfile.REGTYPE)])
    inventory = root / 'bundle-integrity.json'
    inventory.write_text(
        json.dumps({'schema_version': 1, 'files': container.inventory(root, exclude=())})
    )
    args = SimpleNamespace(
        artifact=root, bundle_digest=container.digest(inventory), destination=tmp_path / 'source'
    )
    container.restore_source(args)
    assert (args.destination / 'source.py').read_bytes() == b'pass'
    (root / 'source.tar.gz').write_bytes(b'corrupt')
    with pytest.raises(container.ValidationError):
        container.restore_source(args)


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
        args.control_root / '.github/docker/install_ppu.sh',
    ):
        path.write_text(path.read_text() + '\n# changed\n')
        assert container.build_identity(args)[2] != first_key
        first_key = container.build_identity(args)[2]
    args.image_digest = PINNED[:-1] + 'e'
    assert container.build_identity(args)[2] != first_key


def test_build_uses_only_trusted_dockerfile_and_offline_unprivileged_probe(
    build_inputs, monkeypatch
):
    args = build_inputs
    malicious = args.source_root / 'Dockerfile'
    malicious.write_text('RUN steal-credentials')
    calls = []

    def command(argv, **kwargs):
        calls.append(argv)
        if argv[0] == 'git':
            if argv[2] == str(args.control_root):
                return '.github/scripts/ppu_ci_container.py\0.github/scripts/ppu_functional_pytest.py\0'
            return 'pyproject.toml\0Dockerfile\0'
        if '--output' in argv:
            runtime_export(args.output)
        if argv[:2] == ['docker', 'run']:
            assert '--network=none' in argv and '--cap-drop=ALL' in argv
            assert all('docker.sock' not in item and 'secret' not in item for item in argv)
            return '{"environment_restore_seconds": 0.1}'
        if argv[:3] == ['docker', 'buildx', 'build']:
            context = Path(argv[-1])
            assert (context / 'docker/Dockerfile.ppu-ci').read_bytes() == (
                ROOT / '.github/docker/Dockerfile.ppu-ci'
            ).read_bytes()
            if '--target' in argv and argv[argv.index('--target') + 1] == 'runtime-export':
                assert (context / 'source/Dockerfile').read_text() == 'RUN steal-credentials'
        return ''

    monkeypatch.setenv('GITHUB_TOKEN', 'fixture-token')
    monkeypatch.setenv('PPU_POD_RUN_ROOT', 'fixture-private-root')
    monkeypatch.setattr(container, 'command', command)
    container.build(args)
    assert (
        'GITHUB_TOKEN' not in container.client_env()
        and 'PPU_POD_RUN_ROOT' not in container.client_env()
    )
    metadata = json.loads((args.output / 'docker-build.json').read_text())
    assert (
        metadata['offline_verification']['passed']
        and metadata['offline_verification']['device_validation'] is False
    )
    assert any('--cache-to' in call and 'dependencies' in call for call in calls)
    assert not any('--cache-to' in call and 'runtime-export' in call for call in calls)


@pytest.mark.parametrize(
    'damage', ['none', 'digest', 'base', 'image', 'record', 'file', 'legacy', 'archive', 'software']
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
    elif damage == 'archive':
        archive(root / 'compiled-source.tar.gz', [('evil.py', b'exec', tarfile.REGTYPE)])
        args.digest = seal(root, schema=2)
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
    with pytest.raises(container.ValidationError):
        container.restore_runtime(restore_args(root, trusted, tmp_path))


def test_image_check_only_without_source_does_not_create_paths(
    image_exported, tmp_path, monkeypatch
):
    root, trusted = image_exported
    args = image_args(root, trusted, tmp_path)
    args.source_root = None
    before = container.inventory(root)
    monkeypatch.setattr(
        container,
        'verify_installed_image',
        lambda *a: pytest.fail('CPU check must not access the installed environment'),
    )
    container.verify_image(args)
    assert container.inventory(root) == before
    assert not (root / '.compiled-check').exists()


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
        if '--output' in argv:
            if stage == 'export':
                raise container.ValidationError('export failed')
            runtime_export(args.output, schema=2)
        if '--load' in argv:
            assert '--push' not in argv and 'EXPORT_FORMAT=image' in argv
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


@pytest.mark.parametrize('damage', ['none', 'id', 'tag', 'push', 'lookup', 'public'])
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
    output = tmp_path / 'outputs'
    monkeypatch.setenv('GITHUB_OUTPUT', str(output))
    visibility_calls, calls = [], []

    def visibility(*, allow_missing):
        visibility_calls.append(allow_missing)
        if damage == 'public':
            raise container.ValidationError('GHCR execution package must remain private')

    def command(argv, **kwargs):
        calls.append(argv)
        assert argv[:2] not in (['docker', 'run'], ['docker', 'build'])
        assert argv[:3] != ['docker', 'buildx', 'build']
        if '{{.Id}}' in argv:
            return 'sha256:' + '0' * 64 if damage == 'id' else LOCAL_ID
        if argv[:2] == ['docker', 'push'] and damage == 'push':
            raise container.ValidationError('push failed')
        if '{{json .RepoDigests}}' in argv:
            return json.dumps([PUBLISHED])
        if argv[:3] == ['docker', 'buildx', 'imagetools']:
            assert argv[-1] == PUBLISHED
            if damage == 'lookup':
                raise container.ValidationError('lookup failed')
        return ''

    monkeypatch.setattr(container, 'command', command)
    monkeypatch.setattr(container, 'check_private_package', visibility)
    if damage == 'none':
        container.publish_image(args)
        assert visibility_calls == [True, False]
        assert f'runtime_image_digest={PUBLISHED}' in output.read_text()
        container.verify_image(args)
    else:
        with pytest.raises(container.ValidationError):
            container.publish_image(args)
        assert not output.exists() and not record_path.exists()
    if damage in ('id', 'tag', 'public'):
        assert not any(call[:2] == ['docker', 'push'] for call in calls)


@pytest.mark.parametrize(
    'damage', ['none', 'embedded', 'missing_environment', 'record_schema', 'linked_metadata']
)
def test_baked_image_checks_metadata_before_overlay_and_never_restores_venv(
    image_exported, tmp_path, monkeypatch, damage
):
    root, trusted = image_exported
    args = image_args(root, trusted, tmp_path)
    args.check_only = False
    venv = tmp_path / 'baked-venv'
    (venv / 'bin').mkdir(parents=True)
    (venv / 'bin/python').write_bytes(b'baked interpreter')
    baked = tmp_path / 'baked-metadata.json'
    baked.write_bytes((root / 'docker-build.json').read_bytes())
    original_no_links = container.no_links

    def no_links(path):
        mapping = {'/opt/ppu-ci-venv': venv, '/opt/ppu-ci-image/docker-build.json': baked}
        return original_no_links(mapping.get(str(path), path))

    monkeypatch.setattr(container, 'no_links', no_links)
    monkeypatch.setattr(
        container,
        'command',
        lambda *a, **k: pytest.fail('Activation must never install or run software'),
    )
    if damage == 'embedded':
        baked.write_text('corrupt')
    elif damage == 'missing_environment':
        (venv / 'bin/python').unlink()
    elif damage == 'record_schema':
        record = json.loads((root / 'runtime-image.json').read_text())
        record['schema_version'] = True
        (root / 'runtime-image.json').write_text(json.dumps(record))
    elif damage == 'linked_metadata':
        original = tmp_path / 'original-metadata.json'
        original.write_bytes(baked.read_bytes())
        baked.unlink()
        baked.symlink_to(original)
    if damage == 'none':
        container.verify_image(args)
        assert next(args.source_root.rglob('*.so')).read_bytes() == b'extension'
        assert (venv / 'bin/python').read_bytes() == b'baked interpreter'
        assert len(list(venv.rglob('*'))) == 2
    else:
        with pytest.raises(container.ValidationError):
            container.verify_image(args)
        assert not args.source_root.exists()


@pytest.mark.parametrize(
    'visibility,status,allow_missing,passed',
    [
        ('private', 200, False, True),
        ('public', 200, True, False),
        ('internal', 200, True, False),
        ('private', 404, True, True),
        ('private', 404, False, False),
        ('private', 403, True, False),
    ],
)
def test_private_package_preflight_does_not_change_visibility_or_expose_token(
    monkeypatch, capsys, visibility, status, allow_missing, passed
):
    monkeypatch.setenv('GH_TOKEN', 'fixture-private-token')

    def urlopen(request, **kwargs):
        assert (
            request.full_url
            == 'https://api.github.com/orgs/flytiger-eco/packages/container/megatron-lm-for-sail%2Fppu-ci'
        )
        assert request.get_method() == 'GET'
        assert 'fixture-private-token' not in request.full_url
        assert request.headers['Authorization'] == 'Bearer fixture-private-token'
        if status != 200:
            raise container.HTTPError(request.full_url, status, 'fixture-private-token', None, None)
        return io.BytesIO(json.dumps({'visibility': visibility}).encode())

    monkeypatch.setattr(container, 'urlopen', urlopen)
    if passed:
        container.check_private_package(allow_missing=allow_missing)
    else:
        with pytest.raises(container.ValidationError) as error:
            container.check_private_package(allow_missing=allow_missing)
        assert 'fixture-private-token' not in str(error.value)
    assert capsys.readouterr() == ('', '')


@pytest.mark.parametrize('damage', ['none', 'image', 'record', 'legacy', 'missing_report'])
def test_schema_two_bundle_is_sealed_without_environment_archive(image_exported, damage):
    root, _ = image_exported
    for name in (
        'source.tar.gz',
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
        assert 'runtime-image.json' in files and 'environment.tar.gz' not in files
    else:
        with pytest.raises((ValueError, KeyError)):
            ci.seal_bundle(root)
        assert not (root / 'bundle-integrity.json').exists()


def test_only_guarded_runtime_job_receives_package_publication_permission():
    for filename in (
        'ppu-build',
        'ppu-execute',
        'ppu-ut',
        'ppu-functional',
        'ppu-compiler-diagnostic',
    ):
        jobs = yaml.safe_load((ROOT / f'.github/workflows/{filename}.yml').read_text())['jobs']
        allowed_job = {'ppu-build': 'execute', 'ppu-execute': 'runtime-environment'}.get(filename)
        for name, job in jobs.items():
            assert (job.get('permissions', {}).get('packages') == 'write') is (name == allowed_job)
            if name != allowed_job:
                assert not any(
                    'publish-image' in step.get('run', '') for step in job.get('steps', [])
                )
    for filename in ('ppu_ci_build.sh', 'ppu_ci_run.sh', 'ppu_functional_run.sh'):
        text = (SCRIPTS / filename).read_text()
        assert 'verify-image' in text and 'restore-runtime' not in text
        assert 'PPU_BASE_IMAGE_DIGEST' in text and 'PPU_IMAGE_DIGEST' in text


def test_namespace_legacy_restore_keeps_historical_bytes_and_does_not_run_base_hooks(tmp_path):
    venv, base = tmp_path / 'venv', tmp_path / 'base'
    site = venv / 'lib/python3.12/site-packages'
    (site / 'google/protobuf').mkdir(parents=True)
    (site / 'google/protobuf/current.py').write_text('current')
    (venv / 'pyvenv.cfg').write_text('include-system-site-packages = true\n')
    (base / 'google/protobuf').mkdir(parents=True)
    (base / 'google/protobuf/old.py').write_text('old')
    (base / 'google/auth').mkdir()
    (base / 'torch').mkdir()
    for name in ('protobuf-1.dist-info', 'cloud_tpu_client-1.dist-info'):
        (base / name).mkdir()
    (base / 'startup.pth').write_text('raise RuntimeError("must not execute")')
    before = {
        p.relative_to(base).as_posix(): p.read_bytes() for p in base.rglob('*') if p.is_file()
    }
    bert.isolate_historical_environment(venv, base)
    assert 'include-system-site-packages = false' in (venv / 'pyvenv.cfg').read_text()
    assert not (site / 'startup.pth').exists() and not (site / 'protobuf-1.dist-info').exists()
    assert not (site / 'cloud_tpu_client-1.dist-info').exists()
    assert (site / 'torch').resolve() == base / 'torch'
    assert (site / 'google/auth').resolve() == base / 'google/auth'
    assert (site / 'google/protobuf/current.py').read_text() == 'current'
    assert not (site / 'google/protobuf/old.py').exists()
    assert before == {
        p.relative_to(base).as_posix(): p.read_bytes() for p in base.rglob('*') if p.is_file()
    }


@pytest.mark.parametrize(
    'filename', ['ppu-build', 'ppu-execute', 'ppu-ut', 'ppu-functional', 'ppu-compiler-diagnostic']
)
def test_workflows_never_install_software(filename):
    text = (ROOT / f'.github/workflows/{filename}.yml').read_text()
    assert not re.search(
        r'\b(?:apt(?:-get)?\s+(?:install|update)|pip\s+(?:install|uninstall)|uv\s+(?:sync|pip)|setup-python)',
        text,
    )
    workflow = yaml.safe_load(text)
    for job in workflow['jobs'].values():
        for step in job.get('steps', []):
            if 'restore-control' in step.get('run', ''):
                assert '--digest' in step['run'] and '--control-sha' in step['run']


@pytest.mark.parametrize(
    'filename', ['ppu-execute', 'ppu-ut', 'ppu-functional', 'ppu-compiler-diagnostic']
)
def test_cpu_jobs_declare_required_unprivileged_job_container(filename):
    workflow = yaml.safe_load((ROOT / f'.github/workflows/{filename}.yml').read_text())
    cpu_jobs = [
        job
        for job in workflow['jobs'].values()
        if job.get('runs-on') == 'k8s-runner-group-cpu-flytiger'
    ]
    assert cpu_jobs
    for job in cpu_jobs:
        assert job.get('container') == {'image': 'ghcr.io/actions/actions-runner:latest'}


@pytest.mark.parametrize(
    'filename,job_name,kind',
    [
        ('ppu-build', 'control-environment', 'control'),
        ('ppu-execute', 'runtime-environment', 'runtime'),
    ],
)
def test_docker_build_jobs_use_ubuntu_runner_after_authorization(filename, job_name, kind):
    workflow = yaml.safe_load((ROOT / f'.github/workflows/{filename}.yml').read_text())
    job = workflow['jobs'][job_name]
    assert job['runs-on'] == 'ubuntu-latest'
    assert 'container' not in job
    assert 'secrets.' not in json.dumps(job)
    assert 'PPU_CPU_RUN_ROOT' not in json.dumps(job) and 'cluster-task' not in json.dumps(job)
    steps = job['steps']
    guard = next(
        i
        for i, step in enumerate(steps)
        if step.get('run', '').endswith('ppu_pr_authorize.py guard')
    )
    setup = next(
        i
        for i, step in enumerate(steps)
        if step.get('uses') == 'docker/setup-buildx-action@8d2750c68a42422c14e847fe6c8ac0403b4cbd6f'
    )
    assert guard < setup
    identity = next(i for i, step in enumerate(steps) if step.get('id') == 'identity')
    assert f' fingerprint --kind {kind}' in steps[identity]['run']
    restore = next(
        i for i, step in enumerate(steps) if step.get('uses') == 'actions/cache/restore@v4'
    )
    build = next(
        i
        for i, step in enumerate(steps)
        if (f' build --kind {kind}' if kind == 'control' else ' build-image ')
        in step.get('run', '')
    )
    assert guard < setup < identity < restore < build
    assert '--image-digest' in steps[build]['run']
    assert 'push:' not in json.dumps(job) and '--push' not in json.dumps(job)
    if kind == 'runtime':
        guards = [
            i
            for i, step in enumerate(steps)
            if step.get('run', '').endswith('ppu_pr_authorize.py guard')
        ]
        login = next(
            i
            for i, step in enumerate(steps)
            if step.get('uses') == 'docker/login-action@c94ce9fb468520275223c153574b00df6fe4bcc9'
        )
        publish = next(i for i, step in enumerate(steps) if step.get('id') == 'publish')
        assert build < guards[-1] < login < publish
        assert steps[login]['with']['password'] == '${{ github.token }}'
        assert (
            'publish-image' in steps[publish]['run'] and 'source-root' not in steps[publish]['run']
        )
        assert job['permissions']['packages'] == 'write'
    else:
        assert 'packages' not in job['permissions']
        assert not any('publish-image' in step.get('run', '') for step in steps)


def test_runtime_handoff_preserves_flytiger_orchestration_and_trusted_job_outputs():
    jobs = yaml.safe_load((ROOT / '.github/workflows/ppu-execute.yml').read_text())['jobs']
    producer, consumer = jobs['runtime-environment'], jobs['build']
    assert producer['needs'] == ['configure', 'admit-build']
    assert producer['environment'] == consumer['environment'] == 'ppu-ci'
    assert consumer['needs'] == ['configure', 'admit-build', 'runtime-environment']
    assert "needs.runtime-environment.result == 'success'" in consumer['if']
    assert consumer['runs-on'] == 'k8s-runner-group-cpu-flytiger'
    assert consumer['container'] == {'image': 'ghcr.io/actions/actions-runner:latest'}
    upload = next(
        step for step in producer['steps'] if step.get('uses') == 'actions/upload-artifact@v4'
    )
    download = next(
        step
        for step in consumer['steps']
        if step.get('with', {}).get('name') == 'ppu-runtime-environment'
    )
    assert upload['with']['name'] == download['with']['name'] == 'ppu-runtime-environment'
    assert upload['with']['path'].rstrip('/') == download['with']['path']
    assert upload['with']['if-no-files-found'] == 'error'
    assert (
        producer['outputs']['environment_digest']
        == '${{ steps.publish.outputs.environment_digest }}'
    )
    assert (
        producer['outputs']['runtime_image_digest']
        == '${{ steps.publish.outputs.runtime_image_digest }}'
    )
    steps = consumer['steps']
    verify = next(
        i
        for i, step in enumerate(steps)
        if step.get('name') == 'Verify Dockerfile environment handoff before requesting PPU'
    )
    ppu = next(i for i, step in enumerate(steps) if step.get('id') == 'ppu-build')
    assert verify < ppu and '--check-only' in steps[verify]['run']
    for field in ('environment_digest', 'runtime_image_digest', 'base_image_digest'):
        value = '${{ needs.runtime-environment.outputs.' + field + ' }}'
        assert consumer['outputs'][field] == value and value in steps[verify]['run']
    action = steps[ppu]['with']
    assert action['image'] == '${{ needs.runtime-environment.outputs.runtime_image_digest }}'
    assert (
        'PPU_ENVIRONMENT_DIGEST=${{ needs.runtime-environment.outputs.environment_digest }}'
        in action['extra_env']
    )
    assert (
        'PPU_IMAGE_DIGEST=${{ needs.runtime-environment.outputs.runtime_image_digest }}'
        in action['extra_env']
    )
    assert 'steps.runtime.' not in json.dumps(consumer) and 'steps.identity.' not in json.dumps(
        consumer
    )
    assert action['nnodes'] == 0 and action['nproc_per_node'] == 8
    assert action['node_selector'] == 'board-type=ZW-M890P'
    assert not any(
        ' fingerprint ' in step.get('run', '') or ' build --kind ' in step.get('run', '')
        for step in steps
    )
    assert not any(step.get('uses', '').startswith(('actions/cache/', 'docker/')) for step in steps)


@pytest.mark.parametrize('filename', ['ppu-execute', 'ppu-ut', 'ppu-functional'])
@pytest.mark.parametrize(
    'damage', ['none', 'digest', 'control', 'target', 'image', 'base', 'file', 'archive']
)
def test_runtime_handoff_shell_checks_before_copying_or_restoring(
    image_exported, tmp_path, damage, filename
):
    root, trusted = image_exported
    jobs = yaml.safe_load((ROOT / f'.github/workflows/{filename}.yml').read_text())['jobs']
    steps = jobs['build' if filename == 'ppu-execute' else 'test']['steps']
    verify = next(i for i, step in enumerate(steps) if 'verify-image' in step.get('run', ''))
    ppu = next(i for i, step in enumerate(steps) if step.get('id') in ('ppu-build', 'ppu'))
    assert verify < ppu
    assert not steps[verify].get('continue-on-error', False)
    script = steps[verify]['run']
    assert '--check-only' in script
    action = steps[ppu]['with']
    for field, variable in (
        ('environment_digest', 'PPU_ENVIRONMENT_DIGEST'),
        ('runtime_image_digest', 'PPU_IMAGE_DIGEST'),
        ('base_image_digest', 'PPU_BASE_IMAGE_DIGEST'),
    ):
        prefix = 'needs.runtime-environment.outputs.' if filename == 'ppu-execute' else 'inputs.'
        value = '${{ ' + prefix + field + ' }}'
        assert value in script
        assert variable + '=' + value in action['extra_env']
        if field == 'runtime_image_digest':
            assert action['image'] == value
    runner = tmp_path / 'runner'
    runner.mkdir()
    shutil.copytree(root, runner / 'ppu-runtime-export')
    artifact = runner / 'ppu-runtime-export'
    values = {
        'environment_digest': trusted,
        'runtime_image_digest': PUBLISHED,
        'base_image_digest': PINNED,
        'control_sha': CONTROL,
        'target_sha': TARGET,
    }
    if damage in ('digest', 'control', 'target', 'image', 'base'):
        field = {
            'digest': 'environment_digest',
            'control': 'control_sha',
            'target': 'target_sha',
            'image': 'runtime_image_digest',
            'base': 'base_image_digest',
        }[damage]
        values[field] = {
            'digest': 'f' * 64,
            'control': 'f' * 40,
            'target': 'f' * 40,
            'image': PUBLISHED[:-1] + 'f',
            'base': PUBLISHED,
        }[damage]
    elif damage == 'file':
        (artifact / 'constraints.txt').write_bytes(b'corrupt')
    elif damage == 'archive':
        archive(artifact / 'compiled-source.tar.gz', [('evil.py', b'x', tarfile.REGTYPE)])
        values['environment_digest'] = seal(artifact, schema=2)
        write_image_record(artifact, values['environment_digest'])
    for field in ('environment_digest', 'runtime_image_digest', 'base_image_digest'):
        prefix = 'needs.runtime-environment.outputs.' if filename == 'ppu-execute' else 'inputs.'
        script = script.replace('${{ ' + prefix + field + ' }}', values[field])
    for field in ('control_sha', 'target_sha'):
        script = script.replace('${{ inputs.' + field + ' }}', values[field])
    control = tmp_path / 'control/.github/scripts'
    control.mkdir(parents=True)
    for name in ('ppu_ci_container.py', 'ppu_ci_container.sh'):
        shutil.copyfile(SCRIPTS / name, control / name)
    (tmp_path / 'source').mkdir()
    destination = tmp_path / '.ppu-ci'
    destination.mkdir()
    result = subprocess.run(
        ['bash', '-euo', 'pipefail', '-c', script],
        cwd=tmp_path,
        env={
            'PATH': str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'],
            'RUNNER_TEMP': str(runner),
        },
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) is (damage == 'none'), result.stderr
    assert not (runner / 'ppu-runtime-check').exists()
    assert not (tmp_path / 'source/megatron').exists()
    if damage == 'none' and filename == 'ppu-execute':
        assert container.inventory(destination) == container.inventory(artifact)
        assert (destination / 'docker-build.json').read_bytes() == (
            artifact / 'docker-build.json'
        ).read_bytes()
    else:
        assert not list(destination.iterdir())
    assert not (artifact / '.compiled-check').exists()


@pytest.mark.parametrize(
    'filename',
    [
        'ppu_ci_build.sh',
        'ppu_ci_run.sh',
        'ppu_functional_run.sh',
        'ppu_functional_environment.py',
        'ppu_bert_diagnostic.py',
    ],
)
def test_runtime_helpers_never_install_or_compile(filename):
    text = (SCRIPTS / filename).read_text()
    assert not re.search(
        r'pip\s+(?:install|uninstall|wheel)|apt(?:-get)?\s+(?:install|update)|uv\s+(?:sync|pip)|setup\.py\s+build|make\s+-C',
        text,
    )
    assert 'urllib.request' not in text


def test_tools_vendor_byte_checks_and_native_compilation_stay_in_build():
    dockerfile = (ROOT / '.github/docker/Dockerfile.ppu-ci').read_text()
    assert (
        'RUN --network=none' in dockerfile
        and 'TARGET_SHA' in dockerfile
        and 'CONTROL_SHA' in dockerfile
    )
    assert 'cloud-tpu-client protobuf' in dockerfile
    assert "vendor_bytes(site)" in (ROOT / '.github/docker/export_environment.py').read_text()
    assert (
        "'git', 'wget', 'envsubst', 'make', 'yq'"
        in (ROOT / '.github/docker/export_environment.py').read_text()
    )


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
        'missing-native-libgcc,missing-native-unknown'
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


def test_native_library_closure_catches_missing_dependencies(tmp_path, monkeypatch):
    elf = tmp_path / 'tool'
    elf.write_bytes(b'\x7fELF')
    monkeypatch.setattr(
        exporter.subprocess,
        'run',
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout='lib.so => not found', stderr=''
        ),
    )
    with pytest.raises(RuntimeError, match='native dependency'):
        exporter.libraries([elf], tmp_path / 'libs')


@pytest.mark.parametrize(
    'soname,marker',
    [
        ('libstdc++.so.6', 'libstdcxx'),
        ('libgcc_s.so.1', 'libgcc'),
        ('libgomp.so.1', 'libgomp'),
        ('libatomic.so.1', 'libatomic'),
        ('libgfortran.so.5', 'libgfortran'),
        ('libquadmath.so.0', 'libquadmath'),
        ('libgfortran-040039e1.so.5.0.0', 'libgfortran'),
        ('libquadmath-96973f99.so.0.0.0', 'libquadmath'),
        ('libssl.so.3', 'libssl'),
        ('libcrypto.so.3', 'libcrypto'),
        ('private-library.so', 'unknown'),
        ('/private/library.so', 'unknown'),
    ],
)
def test_missing_native_libraries_only_emit_allowlisted_markers(
    tmp_path, monkeypatch, soname, marker
):
    elf = tmp_path / 'private-extension.so'
    elf.write_bytes(b'\x7fELF')
    monkeypatch.setattr(
        exporter.subprocess,
        'run',
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=f'\t{soname} => not found\n\tprivate-library.so => /private/path (0x1234)\n',
            stderr='private-token',
        ),
    )
    with pytest.raises(RuntimeError) as error:
        exporter.libraries([elf], tmp_path / 'libs')
    message = 'A native dependency is missing in the build container; missing-native-' + marker
    assert str(error.value) == message
    assert container.failure_markers('', str(error.value)) == (
        'export-native-missing,native-' + marker
    )
    assert not list((tmp_path / 'libs').iterdir())


def test_missing_native_markers_are_deduplicated_and_loaded_libraries_are_not_reported(
    tmp_path, monkeypatch
):
    elf = tmp_path / 'extension.so'
    elf.write_bytes(b'\x7fELF')
    monkeypatch.setattr(
        exporter.subprocess,
        'run',
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout='libstdc++.so.6 => not found\nlibgcc_s.so.1 => not found\n'
            'libstdc++.so.6 => not found\nlibgomp.so.1 => /private/library (0x1234)\n',
            stderr='',
        ),
    )
    with pytest.raises(RuntimeError) as error:
        exporter.libraries([elf], tmp_path / 'libs')
    assert str(error.value) == (
        'A native dependency is missing in the build container; '
        'missing-native-libgcc,missing-native-libstdcxx'
    )
    assert container.failure_markers('', str(error.value)) == (
        'export-native-missing,native-libstdcxx,native-libgcc'
    )


def test_preflight_missing_builder_does_not_install_or_launch_daemon(tmp_path, monkeypatch):
    monkeypatch.setattr(container.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(container.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(container.shutil, 'which', lambda name: None)
    monkeypatch.setattr(
        container, 'command', lambda *args: pytest.fail('Missing builder must fail before commands')
    )
    with pytest.raises(container.ValidationError, match='must provide Docker'):
        container.preflight(tmp_path)


@pytest.mark.parametrize(
    'stage,expected',
    [
        ('git_identity', 'Build checkout Git identity read failed'),
        ('git_diff', 'Build checkout tracked-file validation failed'),
        ('docker_daemon', 'CPU builder Docker daemon check failed'),
        ('buildkit', 'CPU builder BuildKit client check failed'),
        ('base_pull', 'CPU builder fixed base image pull failed'),
        ('base_digest', 'CPU builder base image digest lookup failed'),
    ],
)
def test_builder_failures_identify_stage_without_raw_output(tmp_path, monkeypatch, stage, expected):
    def run(argv, **kwargs):
        operation = {
            'git_identity': 'rev-parse',
            'git_diff': 'diff',
            'docker_daemon': 'info',
            'buildkit': 'buildx',
            'base_pull': 'pull',
            'base_digest': 'inspect',
        }[stage]
        failed = operation in argv
        stdout = 'private-output' if failed else 'linux' if 'info' in argv else CONTROL
        return SimpleNamespace(returncode=int(failed), stdout=stdout, stderr='private-error')

    monkeypatch.setattr(container.subprocess, 'run', run)
    monkeypatch.setattr(container.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(container.platform, 'machine', lambda: 'x86_64')
    monkeypatch.setattr(container.shutil, 'which', lambda name: '/usr/bin/docker')
    with pytest.raises(container.ValidationError) as error:
        if stage.startswith('git_'):
            container.checked_git(tmp_path, CONTROL)
        elif stage in ('docker_daemon', 'buildkit'):
            container.preflight(tmp_path)
        else:
            monkeypatch.setattr(container, 'checked_git', lambda *args: None)
            monkeypatch.setattr(container, 'preflight', lambda *args: None)
            container.build_identity(
                SimpleNamespace(control_root=tmp_path, control_sha=CONTROL, kind='control')
            )
    assert str(error.value) == expected


def test_command_launch_failure_does_not_expose_os_error(monkeypatch):
    def run(*args, **kwargs):
        raise OSError('private-error')

    monkeypatch.setattr(container.subprocess, 'run', run)
    with pytest.raises(container.ValidationError, match='^Fixed public stage failed$') as error:
        container.command(['fixture'], failure='Fixed public stage failed')
    assert error.value.__suppress_context__


@pytest.mark.parametrize(
    'text,expected',
    [
        ('No matching distribution found for private-package', 'package-unavailable'),
        ('ResolutionImpossible: private constraints', 'dependency-conflict'),
        ('CERTIFICATE_VERIFY_FAILED private-host', 'tls-verification'),
        ('Temporary failure in name resolution: private-host', 'name-resolution'),
        ('Read timed out: private-host', 'network-timeout'),
        ('Connection refused: private-host', 'connection-failed'),
        ('Permission denied: /private/file', 'access-denied'),
        ('No space left on device: /private/file', 'disk-full'),
        ('Dockerfile parse error: private instruction', 'dockerfile-parse'),
        ('ModuleNotFoundError: private-module', 'python-import'),
        ('Traceback (most recent call last): private-frame', 'python-traceback'),
        ('Package download bypassed the fixed package index', 'export-package-source'),
        ('A native dependency is missing in the build container', 'export-native-missing'),
        ('Unable to inventory native dependencies', 'export-native-inventory'),
        ('Conflicting native dependency SONAMEs', 'export-native-conflict'),
        ('Export requires materialized files and directories', 'export-file-type'),
        ('Restored control interpreter import check failed', 'offline-control-import'),
        ('private error without a known marker', 'unknown'),
    ],
)
@pytest.mark.parametrize('stream', ['stdout', 'stderr'])
def test_command_diagnostics_emit_only_fixed_markers(monkeypatch, capsys, text, expected, stream):
    output = dict(stdout='', stderr='')
    output[stream] = 'https://private-user:private-token@private-host/private/file\n' + text
    monkeypatch.setattr(
        container.subprocess, 'run', lambda *args, **kwargs: SimpleNamespace(returncode=7, **output)
    )
    with pytest.raises(container.ValidationError) as error:
        container.command(
            ['private-command'], failure='Fixed public stage failed', diagnostics=True
        )
    assert (
        str(error.value) == f'Fixed public stage failed; exit_code=7; diagnostic markers={expected}'
    )
    assert capsys.readouterr() == ('', '')


def test_failure_markers_are_bounded_and_unknown_logs_are_not_echoed():
    assert container.failure_markers('No space left on device' + 'x' * 1024**2, '') == 'unknown'
    assert container.failure_markers('', '\x1b[31m::error::private-token\x00') == 'unknown'
    assert container.failure_markers('Connection refused', 'No matching distribution found') == (
        'package-unavailable,connection-failed'
    )


def test_diagnostic_command_success_preserves_stdout_without_emitting_stderr(monkeypatch, capsys):
    monkeypatch.setattr(
        container.subprocess,
        'run',
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout='result\n', stderr='private-warning'
        ),
    )
    assert container.command(['fixture'], diagnostics=True) == 'result'
    assert capsys.readouterr() == ('', '')


def test_diagnostic_command_launch_failure_keeps_os_details_private(monkeypatch):
    def run(*args, **kwargs):
        raise OSError('private-token /private/file')

    monkeypatch.setattr(container.subprocess, 'run', run)
    with pytest.raises(container.ValidationError) as error:
        container.command(['fixture'], failure='Fixed public stage failed', diagnostics=True)
    assert str(error.value) == 'Fixed public stage failed; diagnostic markers=command-launch'
    assert error.value.__suppress_context__


@pytest.mark.parametrize('kind', ['control', 'runtime'])
@pytest.mark.parametrize(
    'stage', ['dependency/cache build', 'environment export build', 'fresh-base offline probe']
)
def test_build_stage_failures_stop_before_publishing_environment_digest(
    build_inputs, monkeypatch, tmp_path, kind, stage
):
    args = build_inputs
    args.kind = kind
    if kind == 'control':
        args.target_sha = None
        args.image_digest = container.CONTROL_IMAGE.rsplit(':', 1)[0] + '@sha256:' + 'c' * 64
    output = tmp_path / 'github-output'
    monkeypatch.setenv('GITHUB_OUTPUT', str(output))
    calls = []

    def command(argv, **kwargs):
        if argv[0] == 'git':
            return (
                '.github/scripts/ppu_ci_container.py\0'
                if argv[2] == str(args.control_root)
                else 'pyproject.toml\0'
            )
        calls.append(kwargs['failure'])
        assert kwargs['diagnostics'] is True
        if kwargs['failure'] == f'Dockerfile {kind} {stage} failed':
            raise container.ValidationError(kwargs['failure'])
        assert argv[:3] == ['docker', 'buildx', 'build']
        assert argv[argv.index('--progress') + 1] == 'plain'
        if '--output' in argv:
            if kind == 'runtime':
                runtime_export(args.output)
            else:
                args.output.mkdir()
                archive(
                    args.output / 'control-runtime.tar.gz',
                    [('bin/python', b'python', tarfile.REGTYPE)],
                )
                (args.output / 'control-package-install.json').write_text('{}')
        return ''

    monkeypatch.setattr(container, 'command', command)
    with pytest.raises(container.ValidationError) as error:
        container.build(args)
    assert str(error.value) == f'Dockerfile {kind} {stage} failed'
    assert calls[-1] == str(error.value)
    assert not output.exists()
    if (args.output / 'docker-build.json').exists():
        assert 'offline_verification' not in json.loads(
            (args.output / 'docker-build.json').read_text()
        )


@pytest.mark.parametrize('damage', ['none', 'target', 'missing_report', 'corrupt'])
def test_new_bundle_inventory_validates_environment_metadata(exported, damage):
    root, _ = exported
    for name in (
        'source.tar.gz',
        'commit.txt',
        'control.txt',
        'image.txt',
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
    (root / 'manifest.json').write_text(
        json.dumps(
            {
                'environment_build_schema': 1,
                'identity': {'control_sha': CONTROL, 'target_sha': TARGET},
            }
        )
    )
    if damage == 'target':
        metadata = json.loads((root / 'docker-build.json').read_text())
        metadata['target_sha'] = 'f' * 40
        (root / 'docker-build.json').write_text(json.dumps(metadata))
    elif damage == 'missing_report':
        (root / 'packaging-install.json').unlink()
    elif damage == 'corrupt':
        (root / 'constraints.txt').write_bytes(b'corrupt')
    if damage == 'none':
        ci.seal_bundle(root)
        assert (root / 'bundle-integrity.json').is_file()
    else:
        with pytest.raises(ValueError):
            ci.seal_bundle(root)
        assert not (root / 'bundle-integrity.json').exists()


def test_untracked_control_helpers_do_not_enter_build_inputs(build_inputs):
    args = build_inputs
    first = container.build_identity(args)[3]
    extra = args.control_root / '.github/scripts/ppu_untracked.py'
    extra.write_text('raise RuntimeError("untrusted")\n')
    assert extra not in container.trusted_controls(args.control_root)
    assert container.build_identity(args)[3] == first


@pytest.mark.parametrize('name', ['../escape.py', 'pyproject.toml', '.github/scripts/missing.py'])
def test_tracked_control_inventory_rejects_wrong_paths(build_inputs, monkeypatch, name):
    monkeypatch.setattr(container, 'command', lambda *args, **kwargs: name + '\0')
    with pytest.raises(container.ValidationError):
        container.trusted_controls(build_inputs.control_root)


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


def test_installation_reports_preserve_real_sources_and_already_satisfied_packages(tmp_path):
    report = tmp_path / 'install.json'
    content = json.dumps(
        {
            'version': '1',
            'install': [
                {
                    'is_direct': False,
                    'download_info': {'url': 'https://pkg.flytiger-eco.com/artifactory/pkg.whl'},
                    'metadata': {'requires_dist': ['numpy>=1']},
                }
            ],
        }
    )
    report.write_text(content)
    exporter.verify_reports([report])
    assert report.read_text() == content
    report.write_text('{"version":"1","install":[]}')
    exporter.verify_reports([report])
