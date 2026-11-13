"""Trusted Dockerfile build orchestration and fail-closed environment restoration."""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

GHCR_IMAGE = 'ghcr.io/flytiger-eco/megatron-lm-for-sail/ppu-ci'
PPU_IMAGE = 'pkg.flytiger-eco.com/docker_release/pytorch:sdk2.2.0-2.13.0-ubuntu24.04-cuda13.0-py312-20261001'
CONTROL_IMAGE = 'python:3.12-slim-bookworm'
SHA = re.compile(r'[0-9a-f]{40}')
DIGEST = re.compile(r'[0-9a-f]{64}')
DOCKER_FILES = (
    'Dockerfile.ppu-ci',
    'Dockerfile.ppu-ci.dockerignore',
    'control-requirements.txt',
    'install_ppu.sh',
    'export_environment.py',
)


class ValidationError(ValueError):
    """A fixed, public-safe validation failure."""


def require(value, message):
    if not value:
        raise ValidationError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def relative(name):
    require(isinstance(name, str), 'Invalid environment member path')
    path = PurePosixPath(name)
    require(
        name
        and not path.is_absolute()
        and all(part not in ('', '.', '..') for part in name.split('/'))
        and '\\' not in name
        and not any(ord(char) < 32 for char in name),
        'Invalid environment member path',
    )
    return path


def no_links(path):
    path = Path(path).absolute()
    require(
        not any(item.is_symlink() for item in (path, *path.parents)),
        'Environment path contains a symlink',
    )
    return path


def inventory(directory, exclude=('docker-build.json',)):
    directory = no_links(directory)
    files = {}
    for path in sorted(directory.rglob('*')):
        require(not path.is_symlink(), 'Environment inventory contains a symlink')
        if path.is_dir():
            continue
        require(path.is_file(), 'Environment inventory contains a special file')
        name = path.relative_to(directory).as_posix()
        if name not in exclude:
            files[name] = {'size': path.stat().st_size, 'sha256': digest(path)}
    return files


def verify_export(
    directory, trusted_digest, control_sha, target_sha=None, *, subset=False, schema=1
):
    directory = no_links(directory)
    require(DIGEST.fullmatch(trusted_digest or ''), 'Missing trusted environment digest')
    path = directory / 'docker-build.json'
    require(
        not path.is_symlink() and digest(path) == trusted_digest,
        'Environment metadata does not match the trusted job digest',
    )
    metadata = json.loads(path.read_text())
    kind = 'runtime' if target_sha is not None else 'control'
    require(
        type(metadata.get('schema_version')) is int
        and metadata['schema_version'] == schema
        and schema in (1, 2)
        and (schema == 1 or kind == 'runtime')
        and metadata.get('kind') == kind,
        'Unsupported environment build metadata',
    )
    require(
        SHA.fullmatch(control_sha or '') and metadata['control_sha'] == control_sha,
        'Environment control identity mismatch',
    )
    require(
        (target_sha is None or SHA.fullmatch(target_sha)) and metadata['target_sha'] == target_sha,
        'Environment target identity mismatch',
    )
    image = PPU_IMAGE if kind == 'runtime' else CONTROL_IMAGE
    repository = image.rsplit(':', 1)[0]
    require(
        metadata['image'] == image
        and re.fullmatch(re.escape(repository) + r'@sha256:[0-9a-f]{64}', metadata['image_digest']),
        'Environment base image identity mismatch',
    )
    require(metadata['platform'] == 'linux/amd64', 'Environment platform mismatch')
    files = metadata.get('files')
    require(isinstance(files, dict) and files, 'Missing environment file inventory')
    for name, expected in files.items():
        relative(name)
        path = no_links(directory / name)
        require(
            path.is_file() and expected == {'size': path.stat().st_size, 'sha256': digest(path)},
            'Missing or corrupted environment member',
        )
    if schema == 2:
        require(
            'environment.tar.gz' not in files
            and not no_links(directory / 'environment.tar.gz').exists(),
            'Image evidence must not contain a software environment archive',
        )
    if not subset:
        exclude = (
            ('docker-build.json', 'runtime-image.json') if schema == 2 else ('docker-build.json',)
        )
        require(
            inventory(directory, exclude=exclude) == files, 'Unexpected environment export members'
        )
    required = (
        (
            'environment.tar.gz',
            'compiled-source.tar.gz',
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
        )
        if kind == 'runtime'
        else ('control-runtime.tar.gz', 'control-package-install.json')
    )
    require(
        all(
            files.get(name, {}).get('size', 0) > 0
            for name in required
            if schema == 1 or name != 'environment.tar.gz'
        ),
        'Incomplete Dockerfile environment export',
    )
    require(DIGEST.fullmatch(metadata.get('inputs_sha256', '')), 'Missing build inputs digest')
    return metadata


def extract_archive(
    archive, destination, *, prefix=None, overlay=False, check_only=False, compiled=False
):
    destination = no_links(destination)
    archive = no_links(archive)
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        require(0 < len(members) <= 500000, 'Invalid environment archive size')
        names, regular, total = set(), set(), 0
        for member in members:
            name = relative(member.name).as_posix()
            if compiled:
                require(
                    re.fullmatch(r'megatron/core/datasets/helpers_cpp[^/]*\.so', name)
                    and member.isfile(),
                    'Invalid compiled source members',
                )
            require(
                name not in names and (member.isdir() or member.isfile()) and not member.issparse(),
                'Unsupported or repeated archive member',
            )
            names.add(name)
            if member.isfile():
                regular.add(name)
            require(member.size >= 0, 'Invalid archive member size')
            if prefix:
                require(
                    name == prefix or name.startswith(prefix + '/'),
                    'Environment archive root mismatch',
                )
            target = no_links(destination / name)
            require(target.is_relative_to(destination), 'Archive member escapes the restore root')
            if target.exists():
                require(
                    target.is_dir()
                    and member.isdir()
                    or overlay
                    and target.is_file()
                    and member.isfile(),
                    'Refusing to overwrite an environment member',
                )
            total += member.size
        require(
            not any(
                str(parent) in regular for name in names for parent in PurePosixPath(name).parents
            ),
            'Archive file conflicts with a parent directory',
        )
        require(total <= 64 * 1024**3, 'Environment archive exceeds the size limit')
        if check_only:
            return
        ancestor = destination
        while not ancestor.exists():
            ancestor = ancestor.parent
        require(
            shutil.disk_usage(ancestor).free > total + 256 * 1024**2,
            'Insufficient environment restore space',
        )
        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with (
                    stream.extractfile(member) as source,
                    target.open('wb' if overlay else 'xb') as output,
                ):
                    shutil.copyfileobj(source, output)
                target.chmod(member.mode & 0o755)
                timestamp = time.time() if compiled else member.mtime
                os.utime(target, (timestamp, timestamp))


def client_env():
    keys = (
        'PATH',
        'HOME',
        'TMPDIR',
        'DOCKER_HOST',
        'DOCKER_CONTEXT',
        'DOCKER_CONFIG',
        'DOCKER_TLS_VERIFY',
        'DOCKER_CERT_PATH',
        'BUILDX_BUILDER',
    )
    return {key: os.environ[key] for key in keys if key in os.environ}


def failure_markers(stdout, stderr):
    """Return fixed log markers, not raw excerpts or a claimed root cause."""
    text = '\n'.join(value[-1024 * 1024 :] for value in (stdout, stderr)).lower()
    patterns = (
        (
            'package-unavailable',
            r'no matching distribution found|could not find a version that satisfies',
        ),
        ('dependency-conflict', r'resolutionimpossible|conflicting dependencies'),
        ('tls-verification', r'certificate_verify_failed|certificate verify failed|x509:'),
        (
            'name-resolution',
            r'temporary failure in name resolution|name or service not known|no such host',
        ),
        ('network-timeout', r'read timed out|connecttimeout|connection timed out|i/o timeout'),
        ('connection-failed', r'connection refused|connection reset|network is unreachable'),
        ('access-denied', r'permission denied|unauthorized|403 forbidden|401 unauthorized'),
        ('disk-full', r'no space left on device'),
        ('dockerfile-parse', r'dockerfile parse error|unknown instruction'),
        ('python-import', r'modulenotfounderror:|importerror:'),
        ('python-traceback', r'traceback \(most recent call last\)'),
        (
            'export-package-source',
            r'package download bypassed the fixed package index|transitive dependency url bypasses the fixed package index',
        ),
        ('export-native-missing', r'a native dependency is missing in the build container'),
        ('native-libstdcxx', r'\bmissing-native-libstdcxx\b'),
        ('native-libgcc', r'\bmissing-native-libgcc\b'),
        ('native-libgomp', r'\bmissing-native-libgomp\b'),
        ('native-libatomic', r'\bmissing-native-libatomic\b'),
        ('native-libgfortran', r'\bmissing-native-libgfortran\b'),
        ('native-libquadmath', r'\bmissing-native-libquadmath\b'),
        ('native-libssl', r'\bmissing-native-libssl\b'),
        ('native-libcrypto', r'\bmissing-native-libcrypto\b'),
        ('native-unknown', r'\bmissing-native-unknown\b'),
        ('native-libpython', r'\bmissing-native-libpython\b'),
        ('native-libtk', r'\bmissing-native-libtk\b'),
        ('native-libtcl', r'\bmissing-native-libtcl\b'),
        ('native-version', r'\bmissing-native-version\b'),
        ('native-unparsed', r'\bmissing-native-unparsed\b'),
        ('native-object-python', r'\bmissing-native-object-python\b'),
        ('native-object-numpy', r'\bmissing-native-object-numpy\b'),
        ('native-object-crc32c', r'\bmissing-native-object-crc32c\b'),
        ('native-object-grpc', r'\bmissing-native-object-grpc\b'),
        ('native-object-pydantic', r'\bmissing-native-object-pydantic\b'),
        ('native-object-stdlib', r'\bmissing-native-object-stdlib\b'),
        ('native-object-other', r'\bmissing-native-object-other\b'),
        ('export-native-inventory', r'unable to inventory native dependencies'),
        ('export-native-conflict', r'conflicting native dependency sonames'),
        ('export-file-type', r'export requires materialized files and directories'),
        ('offline-control-import', r'restored control interpreter import check failed'),
        (
            'restore-archive-invalid',
            r'invalid environment member path|invalid environment archive size|unsupported or repeated archive member|invalid archive member size|archive file conflicts with a parent directory|environment archive exceeds the size limit',
        ),
        (
            'restore-path-invalid',
            r'environment path contains a symlink|archive member escapes the restore root|refusing to overwrite an environment member|refusing to overwrite the control interpreter',
        ),
        ('restore-space', r'insufficient environment restore space'),
        (
            'restore-integrity',
            r'environment metadata does not match the trusted job digest|missing or corrupted environment member|unexpected environment export members',
        ),
        ('probe-file-missing', r'ppu environment operation failed: filenotfounderror;'),
        ('probe-file-permission', r'ppu environment operation failed: permissionerror;'),
        ('probe-type-error', r'ppu environment operation failed: typeerror;'),
        ('probe-attribute-error', r'ppu environment operation failed: attributeerror;'),
        ('probe-os-error', r'ppu environment operation failed: oserror;'),
        ('probe-tar-error', r'ppu environment operation failed: (?:readerror|compressionerror);'),
        ('probe-json-error', r'ppu environment operation failed: jsondecodeerror;'),
    )
    return ','.join(label for label, pattern in patterns if re.search(pattern, text)) or 'unknown'


def command(
    args,
    *,
    cwd=None,
    failure='Container operation failed; raw output suppressed',
    diagnostics=False,
):
    try:
        process = subprocess.run(args, cwd=cwd, env=client_env(), capture_output=True, text=True)
    except OSError:
        suffix = '; diagnostic markers=command-launch' if diagnostics else ''
        raise ValidationError(failure + suffix) from None
    if process.returncode:
        suffix = (
            f'; exit_code={process.returncode}; diagnostic markers='
            + failure_markers(process.stdout, process.stderr)
            if diagnostics
            else ''
        )
        raise ValidationError(failure + suffix)
    return process.stdout.rstrip('\n')


def preflight(root):
    require(sys.version_info >= (3, 11), 'CPU runner bootstrap requires Python 3.11 or newer')
    require(
        platform.system() == 'Linux' and platform.machine() in ('x86_64', 'AMD64'),
        'Dockerfile environment builds require a Linux/amd64 CPU runner',
    )
    require(shutil.which('docker'), 'CPU runner must provide Docker and BuildKit')
    require(
        command(
            ['docker', 'info', '--format', '{{.OSType}}'],
            failure='CPU builder Docker daemon check failed',
        )
        == 'linux',
        'CPU runner Docker daemon must run Linux containers',
    )
    command(['docker', 'buildx', 'version'], failure='CPU builder BuildKit client check failed')
    require(
        shutil.disk_usage(root).free >= 20 * 1024**3,
        'CPU runner requires at least 20 GiB free build space',
    )


def checked_git(root, sha):
    require(SHA.fullmatch(sha or ''), 'A full commit identity is required')
    require(
        command(
            ['git', '-C', str(root), 'rev-parse', 'HEAD'],
            failure='Build checkout Git identity read failed',
        )
        == sha,
        'Build checkout does not match the reviewed commit',
    )
    command(
        ['git', '-C', str(root), 'diff', '--quiet', 'HEAD', '--'],
        failure='Build checkout tracked-file validation failed',
    )


def trusted_controls(root):
    names = command(
        ['git', '-C', str(root), 'ls-files', '-z', '--', '.github/scripts/ppu*.py'],
        failure='Build trusted control inventory read failed',
    ).split('\0')
    paths = []
    for name in names:
        if not name:
            continue
        relative(name)
        path = no_links(root / name)
        require(
            path.parent == root / '.github/scripts' and path.is_file(),
            'Invalid tracked control build helper',
        )
        paths.append(path)
    require(paths, 'Missing tracked control build helpers')
    return sorted(paths)


def build_identity(args):
    root = no_links(args.control_root)
    checked_git(root, args.control_sha)
    preflight(root)
    image = PPU_IMAGE if args.kind == 'runtime' else CONTROL_IMAGE
    pinned = getattr(args, 'image_digest', None)
    if not pinned:
        command(
            ['docker', 'pull', '--platform', 'linux/amd64', image],
            failure='CPU builder fixed base image pull failed',
        )
        pinned = command(
            ['docker', 'image', 'inspect', '--format', '{{index .RepoDigests 0}}', image],
            failure='CPU builder base image digest lookup failed',
        )
    require(
        re.fullmatch(re.escape(image.rsplit(':', 1)[0]) + r'@sha256:[0-9a-f]{64}', pinned),
        'Unable to pin the build base image',
    )
    inputs = {'image': pinned, 'platform': 'linux/amd64'}
    for name in DOCKER_FILES:
        path = no_links(root / '.github/docker' / name)
        require(path.is_file(), 'Trusted Dockerfile build input is missing')
        inputs[name] = digest(path)
    if args.kind == 'runtime':
        source = no_links(args.source_root)
        checked_git(source, args.target_sha)
        inputs['pyproject.toml'] = digest(no_links(source / 'pyproject.toml'))
    inputs['ppu_ci_container.py'] = digest(no_links(root / '.github/scripts/ppu_ci_container.py'))
    inputs['ppu_ci_container.sh'] = digest(no_links(root / '.github/scripts/ppu_ci_container.sh'))
    dependency_key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    for path in trusted_controls(root):
        inputs[path.name] = digest(path)
    inputs['control_sha'] = args.control_sha
    inputs['target_sha'] = args.target_sha
    return image, pinned, dependency_key, inputs


def fingerprint(args):
    _, pinned, key, _ = build_identity(args)
    print(json.dumps({'cache_key': key, 'image_digest': pinned}))
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
            stream.write(f'cache_key={key}\nimage_digest={pinned}\n')


def materialize_tracked_source(source, destination, names):
    """仅将仓库内指向已跟踪普通文件的单层别名实体化到空构建目录。"""
    source, destination = no_links(source), no_links(destination)
    require(
        destination.is_dir() and not any(destination.iterdir()),
        'Build source destination must be an empty directory',
    )
    names = sorted({name for name in names if name})
    for name in names:
        relative(name)
    planned = {}
    for name in names:
        original = source / name
        no_links(original.parent)
        rows = [(name, original)]
        if original.is_symlink():
            resolved = Path(os.path.abspath(original.parent / os.readlink(original)))
            require(resolved.is_relative_to(source), 'Build source alias escapes the checkout')
            no_links(resolved)
            try:
                require(
                    original.resolve(strict=True) == resolved,
                    'Build source alias target is not a direct repository member',
                )
            except (OSError, RuntimeError):
                raise ValidationError('Build source alias target cannot be resolved') from None
            alias = resolved.relative_to(source).as_posix()
            if resolved.is_dir():
                rows = [
                    (name + '/' + item[len(alias) + 1 :], source / item)
                    for item in names
                    if item.startswith(alias + '/')
                ]
                require(rows, 'Build source alias has no tracked targets')
            else:
                require(alias in names, 'Build source alias target is not tracked')
                rows = [(name, resolved)]
        for target_name, original in rows:
            original = no_links(original)
            require(original.is_file(), 'Build source must contain only regular tracked files')
            require(target_name not in planned, 'Build source aliases conflict')
            planned[target_name] = original
    require(
        not any(
            str(parent) in planned for name in planned for parent in PurePosixPath(name).parents
        ),
        'Build source file conflicts with a parent directory',
    )
    for name, original in planned.items():
        target = no_links(destination / name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with original.open('rb') as stream, target.open('xb') as output:
            shutil.copyfileobj(stream, output)
        target.chmod(original.stat().st_mode & 0o755)


def readable_probe_export(output, context):
    """Prepare a readable bind root inside the private, temporary build context."""
    snapshot = context / 'probe-exports'
    shutil.copytree(no_links(output), snapshot)
    for path in (snapshot, *snapshot.rglob('*')):
        no_links(path)
        require(path.is_dir() or path.is_file(), 'Invalid offline probe export member')
        path.chmod(0o755 if path.is_dir() else 0o644)
    return snapshot


def build(args):
    image_mode = getattr(args, 'mode', '') == 'build-image'
    root = no_links(args.control_root)
    image, pinned, _, inputs = build_identity(args)
    output = no_links(args.output)
    require(not output.exists(), 'Refusing to overwrite a container export')
    with tempfile.TemporaryDirectory(prefix='ppu-docker-context-', dir=root) as temporary:
        context = Path(temporary)
        docker = context / 'docker'
        docker.mkdir()
        for name in DOCKER_FILES:
            source = no_links(root / '.github/docker' / name)
            require(source.is_file(), 'Trusted Dockerfile build input is missing')
            shutil.copyfile(source, docker / name)
            inputs[name] = digest(source)
        if args.kind == 'runtime':
            source = no_links(args.source_root)
            checked_git(source, args.target_sha)
            shutil.copyfile(source / 'pyproject.toml', context / 'pyproject.toml')
            inputs['pyproject.toml'] = digest(context / 'pyproject.toml')
            (context / 'source').mkdir()
            names = command(
                ['git', '-C', str(source), 'ls-files', '-z'],
                failure='Build tracked source inventory read failed',
            ).split('\0')
            materialize_tracked_source(source, context / 'source', names)
            control = context / 'control'
            control.mkdir()
            for original in trusted_controls(root):
                shutil.copyfile(original, control / original.name)
                inputs[original.name] = digest(original)
        key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        cache = no_links(args.cache_root) if args.cache_root else None
        common = [
            'docker',
            'buildx',
            'build',
            '--progress',
            'plain',
            '--platform',
            'linux/amd64',
            '--file',
            str(docker / 'Dockerfile.ppu-ci'),
            '--build-arg',
            f'{"PPU_IMAGE" if args.kind == "runtime" else "CONTROL_IMAGE"}={pinned}',
            '--build-arg',
            f'TARGET_SHA={args.target_sha or ""}',
            '--build-arg',
            f'CONTROL_SHA={args.control_sha}',
        ]
        if image_mode:
            require(
                re.fullmatch(r'ppu-ci-runtime:run-[0-9]+-[0-9]+', args.local_image),
                'Invalid local execution image tag',
            )
            common += ['--build-arg', 'EXPORT_FORMAT=image']
        if cache and (cache / 'index.json').is_file():
            common += ['--cache-from', f'type=local,src={cache}']
        if cache:
            dependency = 'dependencies' if args.kind == 'runtime' else 'control'
            command(
                common
                + [
                    '--target',
                    dependency,
                    '--cache-to',
                    f'type=local,dest={cache},mode=max',
                    str(context),
                ],
                failure=f'Dockerfile {args.kind} dependency/cache build failed',
                diagnostics=True,
            )
            common += ['--cache-from', f'type=local,src={cache}']
        started = time.time()
        command(
            common
            + [
                '--target',
                args.kind + '-export',
                '--output',
                f'type=local,dest={output}',
                str(context),
            ],
            failure=f'Dockerfile {args.kind} environment export build failed',
            diagnostics=True,
        )
        metadata = {
            'schema_version': 2 if image_mode else 1,
            'kind': args.kind,
            'platform': 'linux/amd64',
            'control_sha': args.control_sha,
            'target_sha': args.target_sha,
            'image': image,
            'image_digest': pinned,
            'inputs_sha256': key,
            'elapsed_seconds': round(time.time() - started, 3),
            'files': inventory(output),
        }
        (output / 'docker-build.json').write_text(json.dumps(metadata, sort_keys=True) + '\n')
        sha = digest(output / 'docker-build.json')
        verify_export(output, sha, args.control_sha, args.target_sha, schema=2 if image_mode else 1)
        # BuildKit local exports may be owned by the runner with mode 0700.
        # A root container without DAC capabilities cannot read that bind mount.
        # Keep original evidence private and mount a readable temporary snapshot.
        probe_export = readable_probe_export(output, context)
        if image_mode:
            release = context / 'release'
            release.mkdir()
            shutil.copyfile(output / 'docker-build.json', release / 'docker-build.json')
            command(
                common
                + ['--target', 'execution', '--load', '--tag', args.local_image, str(context)],
                failure='Dockerfile execution image build failed',
                diagnostics=True,
            )
            local_id = command(
                ['docker', 'image', 'inspect', '--format', '{{.Id}}', args.local_image]
            )
            require(
                re.fullmatch(r'sha256:[0-9a-f]{64}', local_id), 'Invalid local execution image ID'
            )
            command(
                [
                    'docker',
                    'run',
                    '--rm',
                    '--network=none',
                    '--platform',
                    'linux/amd64',
                    '--cap-drop=ALL',
                    '--security-opt=no-new-privileges',
                    '--mount',
                    f'type=bind,source={probe_export},target=/exports,readonly',
                    '--mount',
                    f'type=bind,source={root / ".github/scripts"},target=/trusted,readonly',
                    '--mount',
                    f'type=bind,source={context / "source"},target=/source,readonly',
                    local_id,
                    'python3',
                    '-I',
                    '/trusted/ppu_ci_container.py',
                    'image-probe',
                    '--artifact',
                    '/exports',
                    '--digest',
                    sha,
                    '--control-sha',
                    args.control_sha,
                    '--target-sha',
                    args.target_sha,
                    '--base-image-digest',
                    pinned,
                ],
                failure='Dockerfile execution image offline probe failed',
                diagnostics=True,
            )
            emit_outputs(environment_digest=sha, local_image_id=local_id, base_image_digest=pinned)
            return
        mounts = [
            '--mount',
            f'type=bind,source={probe_export},target=/exports,readonly',
            '--mount',
            f'type=bind,source={root / ".github/scripts"},target=/trusted,readonly',
        ]
        if args.kind == 'runtime':
            mounts += ['--mount', f'type=bind,source={context / "source"},target=/source,readonly']
        probe = [
            'python3',
            '-I',
            '/trusted/ppu_ci_container.py',
            'offline-probe',
            '--artifact',
            '/exports',
            '--digest',
            sha,
            '--control-sha',
            args.control_sha,
            '--destination',
            '/tmp/ppu-control-runtime',
        ]
        if args.kind == 'runtime':
            probe += ['--target-sha', args.target_sha, '--image-digest', pinned]
        started = time.monotonic()
        probe_output = command(
            [
                'docker',
                'run',
                '--rm',
                '--network=none',
                '--platform',
                'linux/amd64',
                '--cap-drop=ALL',
                '--security-opt=no-new-privileges',
                *mounts,
                pinned,
                *probe,
            ],
            failure=f'Dockerfile {args.kind} fresh-base offline probe failed',
            diagnostics=True,
        )
        metadata['offline_verification'] = {
            'passed': True,
            'network': 'none',
            'device_validation': False,
            'elapsed_seconds': round(time.monotonic() - started, 3),
        }
        if args.kind == 'runtime':
            metadata['offline_verification'].update(json.loads(probe_output.splitlines()[-1]))
        (output / 'docker-build.json').write_text(json.dumps(metadata, sort_keys=True) + '\n')
        sha = digest(output / 'docker-build.json')
        print(
            json.dumps(
                {
                    'kind': args.kind,
                    'environment_digest': sha,
                    'environment_bytes': sum(item['size'] for item in metadata['files'].values()),
                }
            )
        )
        if os.environ.get('GITHUB_OUTPUT'):
            with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
                stream.write(f'environment_digest={sha}\n')


def emit_outputs(**values):
    print(json.dumps(values, sort_keys=True))
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
            for name, value in values.items():
                stream.write(f'{name}={value}\n')


def image_record(metadata, environment_digest, runtime_image_digest):
    require(
        re.fullmatch(re.escape(GHCR_IMAGE) + r'@sha256:[0-9a-f]{64}', runtime_image_digest or ''),
        'Invalid published execution image digest',
    )
    return {
        'schema_version': 1,
        'runtime_image_digest': runtime_image_digest,
        'base_image_digest': metadata['image_digest'],
        'environment_digest': environment_digest,
        'control_sha': metadata['control_sha'],
        'target_sha': metadata['target_sha'],
    }


def publish_image(args):
    metadata = verify_export(
        args.artifact, args.digest, args.control_sha, args.target_sha, schema=2
    )
    require(args.base_image_digest == metadata['image_digest'], 'Publication base image mismatch')
    require(
        re.fullmatch(r'ppu-ci-runtime:run-([0-9]+)-([0-9]+)', args.local_image),
        'Invalid local execution image tag',
    )
    run = args.local_image.split(':', 1)[1]
    require(
        args.image_tag == f'{GHCR_IMAGE}:{run}-{args.target_sha}', 'Invalid publication image tag'
    )
    require(
        re.fullmatch(r'sha256:[0-9a-f]{64}', args.local_image_id), 'Invalid trusted local image ID'
    )
    require(
        command(['docker', 'image', 'inspect', '--format', '{{.Id}}', args.local_image])
        == args.local_image_id,
        'Execution image changed after offline validation',
    )
    record_path = no_links(Path(args.artifact) / 'runtime-image.json')
    require(not record_path.exists(), 'Refusing to overwrite an image publication record')
    command(['docker', 'tag', args.local_image_id, args.image_tag])
    command(
        ['docker', 'push', args.image_tag],
        failure='GHCR execution image push failed',
        diagnostics=True,
    )
    refs = json.loads(
        command(['docker', 'image', 'inspect', '--format', '{{json .RepoDigests}}', args.image_tag])
    )
    require(
        isinstance(refs, list) and all(isinstance(ref, str) for ref in refs),
        'Invalid pushed repository digest inventory',
    )
    matches = [
        ref for ref in refs if re.fullmatch(re.escape(GHCR_IMAGE) + r'@sha256:[0-9a-f]{64}', ref)
    ]
    require(len(matches) == 1, 'Unable to resolve pushed execution image digest')
    published = matches[0]
    # Read the immutable registry reference, not the mutable tag or push log.
    command(
        ['docker', 'buildx', 'imagetools', 'inspect', published],
        failure='GHCR immutable image lookup failed',
    )
    record_path.write_text(
        json.dumps(image_record(metadata, args.digest, published), sort_keys=True) + '\n'
    )
    emit_outputs(
        runtime_image_digest=published,
        environment_digest=args.digest,
        base_image_digest=args.base_image_digest,
    )


def verify_installed_image(args, metadata):
    require(
        args.base_image_digest == metadata['image_digest'], 'Execution image base identity mismatch'
    )
    venv = no_links('/opt/ppu-ci-venv')
    require((venv / 'bin/python').is_file(), 'Missing baked execution environment')
    embedded = no_links('/opt/ppu-ci-image/docker-build.json')
    require(embedded.is_file() and digest(embedded) == args.digest, 'Baked image metadata mismatch')
    require(args.source_root is not None, 'Missing execution source directory')
    extract_archive(
        Path(args.artifact) / 'compiled-source.tar.gz',
        args.source_root,
        overlay=True,
        compiled=True,
    )


def verify_image(args):
    metadata = verify_export(
        args.artifact, args.digest, args.control_sha, args.target_sha, subset=True, schema=2
    )
    require(
        args.base_image_digest == metadata['image_digest'], 'Execution image base identity mismatch'
    )
    record = no_links(Path(args.artifact) / 'runtime-image.json')
    published = json.loads(record.read_text())
    require(
        isinstance(published, dict)
        and type(published.get('schema_version')) is int
        and published == image_record(metadata, args.digest, args.image_digest),
        'Execution image publication record mismatch',
    )
    extract_archive(
        Path(args.artifact) / 'compiled-source.tar.gz',
        args.source_root or Path(args.artifact) / '.compiled-check',
        overlay=True,
        check_only=True,
        compiled=True,
    )
    if args.check_only:
        return
    verify_installed_image(args, metadata)


def image_probe(args):
    metadata = verify_export(
        args.artifact, args.digest, args.control_sha, args.target_sha, schema=2
    )
    source = Path('/tmp/ppu-image-source')
    shutil.copytree('/source', source)
    args.source_root = source
    verify_installed_image(args, metadata)
    portable_checks(source)


def restore_control(args):
    verify_export(args.artifact, args.digest, args.control_sha)
    destination = no_links(args.destination)
    require(not destination.exists(), 'Refusing to overwrite the control interpreter')
    extract_archive(Path(args.artifact) / 'control-runtime.tar.gz', destination)
    interpreter = destination / 'bin/python'
    command(
        [
            str(interpreter),
            '-I',
            '-c',
            'import sys, _tkinter, click, yaml, numpy, pydantic, pytest, google_crc32c; '
            'from tensorboard.compat.proto.event_pb2 import Event; '
            'assert sys.version_info[:2] == (3, 12); '
            'assert Event.FromString(Event(step=1).SerializeToString()).step == 1',
        ],
        failure='Restored control interpreter import check failed',
        diagnostics=True,
    )


def restore_runtime(args):
    metadata_path = Path(args.artifact) / 'docker-build.json'
    require(metadata_path.is_file(), 'Missing Dockerfile environment metadata')
    metadata = verify_export(
        args.artifact, args.digest, args.control_sha, args.target_sha, subset=True
    )
    require(
        args.image_digest == metadata['image_digest'], 'Restored environment base image mismatch'
    )
    destination = no_links(args.destination)
    require(not (destination / 'ppu-ci-venv').exists(), 'Refusing to overwrite the PPU environment')
    environment = Path(args.artifact) / 'environment.tar.gz'
    compiled = Path(args.artifact) / 'compiled-source.tar.gz'
    extract_archive(environment, destination, prefix='ppu-ci-venv', check_only=True)
    if args.source_root:
        extract_archive(compiled, args.source_root, overlay=True, check_only=True, compiled=True)
    if getattr(args, 'check_only', False):
        return
    started = time.monotonic()
    extract_archive(environment, destination, prefix='ppu-ci-venv')
    if args.source_root:
        extract_archive(compiled, args.source_root, overlay=True, compiled=True)
    print(json.dumps({'environment_restore_seconds': round(time.monotonic() - started, 3)}))


def offline_probe(args):
    """Fresh-base, network-disabled portability checks; never claim device acceptance."""
    if args.target_sha is None:
        restore_control(args)
        return
    source = Path('/tmp/ppu-offline-source')
    shutil.copytree('/source', source)
    args.source_root, args.destination = source, Path('/opt')
    restore_runtime(args)
    portable_checks(source)


def portable_checks(source):
    environment = client_env()
    environment.update(
        {
            'PATH': '/opt/ppu-ci-venv/bin:' + environment.get('PATH', ''),
            'PYTHONNOUSERSITE': '1',
            'UV_NO_SYNC': '1',
            'UV_PYTHON_DOWNLOADS': 'never',
            'LD_LIBRARY_PATH': '/opt/ppu-ci-venv/lib/ppu-system:'
            + os.environ.get('LD_LIBRARY_PATH', ''),
            'GIT_EXEC_PATH': '/opt/ppu-ci-venv/libexec/git-core',
            'SSL_CERT_FILE': '/opt/ppu-ci-venv/etc/ssl/certs/ca-certificates.crt',
        }
    )
    checks = [
        ['/opt/ppu-ci-venv/bin/python', '-I', '-m', 'pip', 'check'],
        [
            '/opt/ppu-ci-venv/bin/python',
            '-I',
            '-c',
            'import sys, pathlib, torch, triton, google.protobuf, tensorboard, nvidia_resiliency_ext; '
            'assert pathlib.Path(google.protobuf.__file__).resolve().is_relative_to(sys.prefix); '
            'assert "include-system-site-packages = false" in pathlib.Path(sys.prefix, "pyvenv.cfg").read_text()',
        ],
        ['/opt/ppu-ci-venv/bin/git', '--version'],
        ['/opt/ppu-ci-venv/bin/wget', '--version'],
        ['/opt/ppu-ci-venv/bin/envsubst', '--version'],
        ['/opt/ppu-ci-venv/bin/yq', '--version'],
        ['/opt/ppu-ci-venv/bin/make', '-q', '-C', str(source / 'megatron/core/datasets')],
    ]
    for check in checks:
        result = subprocess.run(check, env=environment, capture_output=True, text=True)
        require(
            result.returncode == 0,
            'Offline clean-base portability check failed; device checks remain required',
        )


def archive_source(args):
    """Package materialized tracked source and authenticated compiled extensions."""
    source, directory = no_links(args.source_root), no_links(args.artifact)
    checked_git(source, args.target_sha)
    verify_export(directory, args.digest, args.control_sha, args.target_sha, subset=True, schema=2)
    output = no_links(directory / 'source.tar.gz')
    require(not output.exists(), 'Refusing to overwrite a source archive')
    names = command(
        ['git', '-C', str(source), 'ls-files', '-z'],
        failure='Source archive tracked inventory read failed',
    ).split('\0')
    require(any(names), 'Missing tracked source archive members')
    with tempfile.TemporaryDirectory(prefix='ppu-source-archive-', dir=directory) as temporary:
        context = Path(temporary)
        stage = context / 'source'
        stage.mkdir()
        materialize_tracked_source(source, stage, names)
        extract_archive(directory / 'compiled-source.tar.gz', stage, overlay=True, compiled=True)
        checked_git(source, args.target_sha)
        candidate = context / 'source.tar.gz'
        files = inventory(stage, exclude=())
        with tarfile.open(candidate, 'w:gz', dereference=True) as stream:
            for name in sorted(files):
                stream.add(stage / name, arcname=name, recursive=False)
        extract_archive(candidate, context / 'check', check_only=True)
        with candidate.open('rb') as stream, output.open('xb') as target:
            shutil.copyfileobj(stream, target)
    print('Source archive creation and safety check passed')


def restore_source(args):
    directory = no_links(args.artifact)
    require(DIGEST.fullmatch(args.bundle_digest or ''), 'Missing trusted source bundle digest')
    sealed = no_links(directory / 'bundle-integrity.json')
    require(digest(sealed) == args.bundle_digest, 'Source bundle inventory digest mismatch')
    metadata = json.loads(sealed.read_text())
    source = no_links(directory / 'source.tar.gz')
    require(
        type(metadata.get('schema_version')) is int
        and metadata['schema_version'] == 1
        and metadata['files'].get('source.tar.gz')
        == {'size': source.stat().st_size, 'sha256': digest(source)},
        'Source archive digest mismatch',
    )
    extract_archive(source, args.destination)


def mark_plan(args):
    path = no_links(Path(args.artifact) / 'manifest.json')
    manifest = json.loads(path.read_text())
    manifest['environment_build_schema'] = 2
    if manifest.get('test_kind') == 'combined':
        sidecar = no_links(path.with_name('functional-manifest.json'))
        functional = json.loads(sidecar.read_text())
        functional['environment_build_schema'] = 2
        sidecar.write_text(json.dumps(functional, indent=2) + '\n')
        manifest['functional_plan']['sha256'] = digest(sidecar)
    path.write_text(json.dumps(manifest, indent=2) + '\n')


def main():
    require(sys.version_info >= (3, 11), 'Environment bootstrap requires Python 3.11 or newer')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('build', 'fingerprint', 'build-image'):
        create = sub.add_parser(mode)
        if mode == 'build-image':
            create.set_defaults(kind='runtime')
            create.add_argument('--local-image', required=True)
        else:
            create.add_argument('--kind', choices=('control', 'runtime'), required=True)
        create.add_argument('--control-root', type=Path, required=True)
        create.add_argument('--control-sha', required=True)
        create.add_argument('--source-root', type=Path)
        create.add_argument('--target-sha')
        create.add_argument('--image-digest')
        if mode in ('build', 'build-image'):
            create.add_argument('--output', type=Path, required=True)
            create.add_argument('--cache-root', type=Path)
    for mode in ('restore-control', 'restore-runtime', 'offline-probe'):
        restore = sub.add_parser(mode)
        restore.add_argument('--artifact', type=Path, required=True)
        restore.add_argument('--digest', required=True)
        restore.add_argument('--control-sha', required=True)
        restore.add_argument('--target-sha')
        restore.add_argument('--source-root', type=Path)
        restore.add_argument('--destination', type=Path, required=True)
        if mode in ('restore-runtime', 'offline-probe'):
            restore.add_argument('--image-digest', required=mode == 'restore-runtime')
            restore.add_argument('--check-only', action='store_true')
    for mode in ('publish-image', 'verify-image', 'image-probe'):
        image = sub.add_parser(mode)
        image.add_argument('--artifact', type=Path, required=True)
        image.add_argument('--digest', required=True)
        image.add_argument('--control-sha', required=True)
        image.add_argument('--target-sha', required=True)
        image.add_argument('--base-image-digest', required=True)
        if mode == 'publish-image':
            image.add_argument('--local-image', required=True)
            image.add_argument('--local-image-id', required=True)
            image.add_argument('--image-tag', required=True)
        if mode == 'verify-image':
            image.add_argument('--image-digest', required=True)
            image.add_argument('--source-root', type=Path)
            image.add_argument('--check-only', action='store_true')
    pack = sub.add_parser('archive-source')
    pack.add_argument('--artifact', type=Path, required=True)
    pack.add_argument('--source-root', type=Path, required=True)
    pack.add_argument('--digest', required=True)
    pack.add_argument('--control-sha', required=True)
    pack.add_argument('--target-sha', required=True)
    source = sub.add_parser('restore-source')
    source.add_argument('--artifact', type=Path, required=True)
    source.add_argument('--bundle-digest', required=True)
    source.add_argument('--destination', type=Path, required=True)
    mark = sub.add_parser('mark-plan')
    mark.add_argument('--artifact', type=Path, required=True)
    args = parser.parse_args()
    {
        'build': build,
        'build-image': build,
        'publish-image': publish_image,
        'verify-image': verify_image,
        'image-probe': image_probe,
        'fingerprint': fingerprint,
        'restore-control': restore_control,
        'restore-runtime': restore_runtime,
        'restore-source': restore_source,
        'archive-source': archive_source,
        'offline-probe': offline_probe,
        'mark-plan': mark_plan,
    }[args.mode](args)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Only our fixed validation messages are safe to print; subprocess and OS
        # exceptions may contain credentials or private directory names.
        message = str(error) if isinstance(error, ValidationError) else type(error).__name__
        print(
            f'PPU environment operation failed: {message}; raw details suppressed', file=sys.stderr
        )
        raise SystemExit(1) from None
