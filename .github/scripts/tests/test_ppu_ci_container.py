"""CPU-only regression gates for Dockerfile construction and authenticated restoration."""

import importlib.util
import io
import json
import re
import shutil
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


def runtime_export(root):
    root.mkdir(exist_ok=True)
    archive(root / 'environment.tar.gz', [('ppu-ci-venv/bin/python', b'python', tarfile.REGTYPE)])
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
    return seal(root)


def seal(root):
    metadata = dict(
        schema_version=1,
        kind='runtime',
        platform='linux/amd64',
        control_sha=CONTROL,
        target_sha=TARGET,
        image=container.PPU_IMAGE,
        image_digest=PINNED,
        inputs_sha256='d' * 64,
        files=container.inventory(root),
    )
    (root / 'docker-build.json').write_text(json.dumps(metadata))
    return container.digest(root / 'docker-build.json')


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
    'filename', ['ppu-build', 'ppu-execute', 'ppu-ut', 'ppu-functional', 'ppu-compiler-diagnostic']
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
