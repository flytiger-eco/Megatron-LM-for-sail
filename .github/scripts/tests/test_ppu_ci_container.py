"""CPU regression gates for Dockerfile construction and runtime handoff."""

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
ARCHIVE_BYTES = b'fixture runtime archive'


def runtime_export(root, *, control=CONTROL, target=TARGET):
    root.mkdir(exist_ok=True)
    for name in (
        'functional-package-install.json',
        'constraints.txt',
        'installed-packages.json',
        'compatibility-patches.json',
        'base-packages.json',
        'vendor-bytes.json',
        'ppu-requirements.txt',
        'test-requirements.txt',
        'packaging-install.json',
        'build-tools-install.json',
        'source-package-install.json',
        'functional-tools/yq',
        'functional-tools/envsubst',
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'{}')
    return seal(root, control=control, target=target)


def seal(root, *, control=CONTROL, target=TARGET):
    metadata = dict(
        schema_version=3,
        kind='runtime',
        platform='linux/amd64',
        control_sha=control,
        target_sha=target,
        image=container.PPU_IMAGE,
        image_digest=PINNED,
        inputs_sha256='d' * 64,
        files=container.inventory(root, exclude=('docker-build.json', 'runtime-image.json', container.ARCHIVE)),
        archive={'size': len(ARCHIVE_BYTES), 'sha256': container.hashlib.sha256(ARCHIVE_BYTES).hexdigest()},
    )
    (root / 'docker-build.json').write_text(json.dumps(metadata))
    return container.digest(root / 'docker-build.json')


def write_image_record(root, trusted):
    metadata = json.loads((root / 'docker-build.json').read_text())
    (root / 'runtime-image.json').write_text(
        json.dumps(container.image_record(metadata, trusted, PINNED))
    )


@pytest.fixture
def image_exported(tmp_path):
    root = tmp_path / 'image-export'
    trusted = runtime_export(root)
    write_image_record(root, trusted)
    return root, trusted


def image_args(root, trusted, tmp_path):
    return SimpleNamespace(
        artifact=root,
        digest=trusted,
        control_sha=CONTROL,
        target_sha=TARGET,
        image_digest=PINNED,
        base_image_digest=PINNED,
        source_root=tmp_path / 'source',
        check_only=True,
    )


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
    for name in ('ppu_ci_container.py', 'ppu_functional_pytest.py'):
        shutil.copyfile(SCRIPTS / name, scripts / name)
    shutil.copyfile(ROOT / 'pyproject.toml', source / 'pyproject.toml')
    monkeypatch.setattr(container, 'preflight', lambda root: None)
    monkeypatch.setattr(container, 'checked_git', lambda *args: None)
    tracked = ''.join(f'.github/scripts/{path.name}\0' for path in scripts.glob('*.py'))
    monkeypatch.setattr(container, 'command', lambda *args, **kwargs: tracked)
    return SimpleNamespace(
        mode='build-runtime',
        control_root=control,
        control_sha=CONTROL,
        source_root=source,
        target_sha=TARGET,
        image_digest=PINNED,
        output=tmp_path / 'export',
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
        f'PPU_IMAGE={args.image_digest}',
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
        args.image_digest = PUBLISHED
    elif damage == 'record':
        (root / 'runtime-image.json').write_text('{}')
    elif damage == 'file':
        (root / 'constraints.txt').write_text('corrupt')
    elif damage == 'legacy':
        metadata = json.loads((root / 'docker-build.json').read_text())
        metadata['schema_version'] = 1
        (root / 'docker-build.json').write_text(json.dumps(metadata))
        args.digest = container.digest(root / 'docker-build.json')
        write_image_record(root, args.digest)
    elif damage == 'software':
        args.archive = root / container.ARCHIVE
        args.archive.write_bytes(b'unexpected')
    if damage == 'none':
        container.verify_image(args)
    else:
        with pytest.raises(container.ValidationError):
            container.verify_image(args)
    assert not args.source_root.exists()
    assert not (tmp_path / 'opt').exists()


@pytest.mark.parametrize('stage', ['none', 'export', 'probe'])
def test_complete_runtime_build_probes_fixed_sdk_without_publishing_an_image(
    build_inputs, monkeypatch, tmp_path, stage
):
    args = build_inputs
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
            runtime_export(args.output)
            (args.output / container.ARCHIVE).write_bytes(ARCHIVE_BYTES)
        if argv[:2] == ['docker', 'run']:
            assert PINNED in argv and 'runtime-probe' in argv
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
        assert f'runtime_image_digest={PINNED}' in outputs.read_text()
        assert (args.output / container.ARCHIVE).is_file()
        assert (args.output / 'runtime-image.json').is_file()
    else:
        with pytest.raises(container.ValidationError):
            container.build(args)
        assert not outputs.exists()
    assert not any(call[:2] in (['docker', 'push'], ['docker', 'login']) for call in calls)
    assert 'GH_TOKEN' not in container.client_env()


@pytest.mark.parametrize('damage', ['none', 'image', 'record', 'legacy', 'missing_report'])
def test_handoff_evidence_is_sealed_without_copying_the_runtime_archive(image_exported, damage):
    root, _ = image_exported
    for name in (
        'commit.txt',
        'control.txt',
        'build-timing.json',
        'control/ppu_ci.py',
        'control/ppu_ci_run.sh',
        'control/ppu_pytest_policy.py',
        'control/ppu_ci_container.py',
        'control/ppu_ci_paths.py',
        'control/ppu_ci_export.py',
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'fixture')
    (root / 'image.txt').write_text(PINNED + '\n')
    (root / 'manifest.json').write_text(
        json.dumps({'identity': {'control_sha': CONTROL, 'target_sha': TARGET}})
    )
    if damage == 'image':
        (root / 'image.txt').write_text(PUBLISHED)
    elif damage == 'record':
        (root / 'runtime-image.json').write_text('{}')
    elif damage == 'legacy':
        metadata = json.loads((root / 'docker-build.json').read_text())
        metadata['schema_version'] = 1
        (root / 'docker-build.json').write_text(json.dumps(metadata))
        write_image_record(root, container.digest(root / 'docker-build.json'))
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
                'identity': {'control_sha': CONTROL, 'target_sha': TARGET},
            }
        )
    )
    root = tmp_path / 'bundle'
    trusted = runtime_export(root)
    write_image_record(root, trusted)
    shutil.copytree(control, root / 'control')
    shutil.copyfile(plan / 'manifest.json', root / 'manifest.json')
    for name, content in (
        ('commit.txt', TARGET),
        ('control.txt', CONTROL),
        ('.complete', TARGET),
        ('image.txt', PINNED),
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
    calls, environments = [], []
    real_run = subprocess.run
    monkeypatch.setattr(
        container.subprocess,
        'run',
        lambda argv, **kwargs: (
            calls.append(argv)
            or environments.append(kwargs['env'])
            or SimpleNamespace(returncode=0)
        ),
    )
    monkeypatch.setenv('LD_LIBRARY_PATH', '/vendor/sdk/lib')
    container.execution_checks(tmp_path)
    assert all(env['LD_LIBRARY_PATH'] == '/vendor/sdk/lib' for env in environments)
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


@pytest.mark.parametrize('filename', ['ppu_ci_container.py', 'ppu_functional_pytest.py'])
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
    requirements.write_text('# control-only change\n')
    after = container.build_identity(args)
    assert before == after


def test_cache_population_never_compiles_or_executes_target_source(build_inputs, monkeypatch):
    args = build_inputs
    args.mode = 'build-cache'
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
    args.cache_to = ref
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


@pytest.mark.parametrize('damage', [None, 'hash', 'escape'])
def test_restore_exposes_only_the_verified_runtime(tmp_path, monkeypatch, damage):
    root, tools = tmp_path / 'opt', tmp_path / 'bin'
    root.mkdir()
    tools.mkdir()
    source, venv = root / 'ppu-ci-source', root / 'ppu-ci-venv'
    monkeypatch.setattr(container, 'SOURCE_ROOT', source)
    monkeypatch.setattr(container, 'VENV_ROOT', venv)
    monkeypatch.setattr(container, 'TOOL_BIN', tools)
    vendor = tmp_path / 'sdk'
    for name in ('torch', 'triton'):
        (vendor / name).mkdir(parents=True)
        (vendor / name / '__init__.py').write_text(name)
    monkeypatch.setattr(container.sysconfig, 'get_path', lambda _: str(vendor))
    evidence = tmp_path / 'evidence'
    runtime_export(evidence)
    archive = tmp_path / container.ARCHIVE
    with tarfile.open(archive, 'w:gz') as stream:
        names = [
            'ppu-ci-source/source.py',
            'ppu-ci-venv/bin/python',
            'ppu-ci-venv/lib/python3.12/site-packages/tool.py',
        ]
        if damage == 'escape':
            names.append('ppu-ci-source/../../escaped')
        for name in names:
            member = tarfile.TarInfo(name)
            payload = b'sample'
            member.size = len(payload)
            stream.addfile(member, io.BytesIO(payload))
    metadata = {'archive': container.file_identity(archive)}
    if damage == 'hash':
        metadata['archive']['sha256'] = '0' * 64
    if damage:
        with pytest.raises(container.ValidationError):
            container.restore_runtime(evidence, archive, metadata)
        assert not list(root.iterdir()) and not list(tools.iterdir())
    else:
        container.restore_runtime(evidence, archive, metadata)
        assert (source / 'source.py').read_bytes() == b'sample'
        assert (venv / 'bin/python').read_bytes() == b'sample'
        assert (venv / 'lib/python3.12/site-packages/torch/__init__.py').read_text() == 'torch'
        assert (tools / 'yq').read_bytes() == (evidence / 'functional-tools/yq').read_bytes()
    assert not (tmp_path / 'escaped').exists()
