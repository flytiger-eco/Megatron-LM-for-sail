"""Build and verify the software bundle used with the fixed SAIL SDK image."""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath

GHCR_IMAGE = 'ghcr.io/flytiger-eco/megatron-lm-for-sail/ppu-ci'
PPU_IMAGE = 'pkg.flytiger-eco.com/docker_release/pytorch:sdk2.2.0-2.13.0-ubuntu24.04-cuda13.0-py312-20261001'
SOURCE_ROOT = Path('/opt/ppu-ci-source')
VENV_ROOT = Path('/opt/ppu-ci-venv')
TOOL_BIN = Path('/usr/local/bin')
ARCHIVE = 'environment.tar.gz'
SHA = re.compile(r'[0-9a-f]{40}')
DIGEST = re.compile(r'[0-9a-f]{64}')
DOCKER_FILES = ('Dockerfile.ppu-ci', 'Dockerfile.ppu-ci.dockerignore', 'export_environment.py')


class ValidationError(ValueError):
    """A fixed, public-safe validation failure."""


def require(value, message):
    if not value:
        raise ValidationError(message)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def file_identity(path):
    return {'size': path.stat().st_size, 'sha256': digest(path)}


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
            files[name] = file_identity(path)
    return files


def verify_export(directory, trusted_digest, control_sha, target_sha, *, subset=False):
    directory = no_links(directory)
    require(DIGEST.fullmatch(trusted_digest or ''), 'Missing trusted environment digest')
    path = directory / 'docker-build.json'
    require(
        not path.is_symlink() and digest(path) == trusted_digest,
        'Environment metadata does not match the trusted job digest',
    )
    metadata = json.loads(path.read_text())
    require(
        type(metadata.get('schema_version')) is int
        and metadata['schema_version'] == 3
        and metadata.get('kind') == 'runtime',
        'Unsupported environment build metadata',
    )
    require(
        SHA.fullmatch(control_sha or '') and metadata['control_sha'] == control_sha,
        'Environment control identity mismatch',
    )
    require(
        SHA.fullmatch(target_sha or '') and metadata['target_sha'] == target_sha,
        'Environment target identity mismatch',
    )
    image = PPU_IMAGE
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
            path.is_file() and expected == file_identity(path),
            'Missing or corrupted environment member',
        )
    archive = metadata.get('archive', {})
    require(
        isinstance(archive, dict) and type(archive.get('size')) is int and archive['size'] > 0
        and isinstance(archive.get('sha256'), str) and DIGEST.fullmatch(archive['sha256'])
        and ARCHIVE not in files,
        'Missing trusted runtime archive identity',
    )
    if not subset:
        exclude = ('docker-build.json', 'runtime-image.json', ARCHIVE)
        require(
            inventory(directory, exclude=exclude) == files, 'Unexpected environment export members'
        )
    required = (
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
    )
    require(
        all(files.get(name, {}).get('size', 0) > 0 for name in required),
        'Incomplete Dockerfile environment export',
    )
    require(DIGEST.fullmatch(metadata.get('inputs_sha256', '')), 'Missing build inputs digest')
    return metadata


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
        (
            'restore-integrity',
            r'environment metadata does not match the trusted job digest|missing or corrupted environment member|unexpected environment export members',
        ),
        ('probe-file-missing', r'ppu environment operation failed: filenotfounderror;'),
        ('probe-file-permission', r'ppu environment operation failed: permissionerror;'),
        ('probe-type-error', r'ppu environment operation failed: typeerror;'),
        ('probe-attribute-error', r'ppu environment operation failed: attributeerror;'),
        ('probe-os-error', r'ppu environment operation failed: oserror;'),
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
    image = PPU_IMAGE
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
    source = no_links(args.source_root)
    checked_git(source, args.target_sha)
    inputs['pyproject.toml'] = digest(no_links(source / 'pyproject.toml'))
    # Only inputs consumed by the dependency stage belong in its cache key.
    # Source/control SHAs and orchestration helpers still bind the final export.
    dependency_key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    inputs['ppu_ci_container.py'] = digest(no_links(root / '.github/scripts/ppu_ci_container.py'))
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
    """Materialize tracked files and single-level internal aliases into an empty build directory."""
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
    shutil.copytree(no_links(output), snapshot, symlinks=True, copy_function=os.link)
    for path in (snapshot, *snapshot.rglob('*')):
        no_links(path)
        require(path.is_dir() or path.is_file(), 'Invalid offline probe export member')
        path.chmod(0o755 if path.is_dir() else 0o644)
    return snapshot


def build(args):
    cache_only = getattr(args, 'mode', '') == 'build-cache'
    cache_from, cache_to = getattr(args, 'cache_from', ''), getattr(args, 'cache_to', '')
    for ref in (cache_from, cache_to):
        require(
            not ref
            or re.fullmatch(re.escape(GHCR_IMAGE) + r':cache-[a-z0-9][a-z0-9_.-]{0,121}', ref),
            'Invalid dependency cache registry reference',
        )
    require(not cache_only or cache_to, 'Cache population requires a registry destination')
    root = no_links(args.control_root)
    image, pinned, _, inputs = build_identity(args)
    output = None if cache_only else no_links(args.output)
    require(cache_only or not output.exists(), 'Refusing to overwrite a container export')
    with tempfile.TemporaryDirectory(prefix='ppu-docker-context-', dir=root) as temporary:
        context = Path(temporary)
        docker = context / 'docker'
        docker.mkdir()
        for name in DOCKER_FILES:
            source = no_links(root / '.github/docker' / name)
            require(source.is_file(), 'Trusted Dockerfile build input is missing')
            shutil.copyfile(source, docker / name)
            inputs[name] = digest(source)
        source = no_links(args.source_root)
        checked_git(source, args.target_sha)
        shutil.copyfile(source / 'pyproject.toml', context / 'pyproject.toml')
        inputs['pyproject.toml'] = digest(context / 'pyproject.toml')
        if not cache_only:
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
            f'PPU_IMAGE={pinned}',
            '--build-arg',
            f'TARGET_SHA={args.target_sha}',
            '--build-arg',
            f'CONTROL_SHA={args.control_sha}',
        ]
        if cache_from:
            common += ['--cache-from', f'type=registry,ref={cache_from}']
        if cache_to:
            command(
                common
                + [
                    '--target',
                    'dependencies',
                    '--cache-to',
                    f'type=registry,ref={cache_to},mode=max',
                    str(context),
                ],
                failure='Dockerfile runtime dependency/cache build failed',
                diagnostics=True,
            )
            if cache_only:
                return
        started = time.time()
        command(
            common
            + ['--target', 'runtime-export', '--output', f'type=local,dest={output}', str(context)],
            failure='Dockerfile runtime environment export build failed',
            diagnostics=True,
        )
        metadata = {
            'schema_version': 3,
            'kind': 'runtime',
            'platform': 'linux/amd64',
            'control_sha': args.control_sha,
            'target_sha': args.target_sha,
            'image': image,
            'image_digest': pinned,
            'inputs_sha256': key,
            'elapsed_seconds': round(time.time() - started, 3),
            'files': inventory(output, exclude=('docker-build.json', ARCHIVE)),
            'archive': file_identity(no_links(output / ARCHIVE)),
        }
        (output / 'docker-build.json').write_text(json.dumps(metadata, sort_keys=True) + '\n')
        sha = digest(output / 'docker-build.json')
        verify_export(output, sha, args.control_sha, args.target_sha)
        (output / 'runtime-image.json').write_text(
            json.dumps(image_record(metadata, sha, pinned), sort_keys=True) + '\n'
        )
        # BuildKit local exports may be owned by the runner with mode 0700.
        # A root container without DAC capabilities cannot read that bind mount.
        # Keep original evidence private and mount a readable temporary snapshot.
        probe_export = readable_probe_export(output, context)
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
                '--entrypoint',
                '/usr/bin/env',
                pinned,
                'python3',
                '-I',
                '/trusted/ppu_ci_container.py',
                'runtime-probe',
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
            failure='SDK runtime bundle offline probe failed',
            diagnostics=True,
        )
        emit_outputs(
            environment_digest=sha, runtime_image_digest=pinned, base_image_digest=pinned
        )


def emit_outputs(**values):
    print(json.dumps(values, sort_keys=True))
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as stream:
            for name, value in values.items():
                stream.write(f'{name}={value}\n')


def image_record(metadata, environment_digest, runtime_image_digest):
    require(
        runtime_image_digest == metadata['image_digest'],
        'Execution must use the SDK digest used to build the runtime bundle',
    )
    return {
        'schema_version': 1,
        'runtime_image_digest': runtime_image_digest,
        'base_image_digest': metadata['image_digest'],
        'environment_digest': environment_digest,
        'control_sha': metadata['control_sha'],
        'target_sha': metadata['target_sha'],
    }


def verify_archive(path, metadata):
    archive = no_links(path)
    require(
        archive.is_file() and file_identity(archive) == metadata['archive'],
        'Runtime archive does not match the trusted build export',
    )


def restore_runtime(directory, path, metadata):
    """Restore the approved environment and compiled source into a fresh SDK worker."""
    root = no_links(SOURCE_ROOT.parent)
    destinations = (VENV_ROOT, SOURCE_ROOT)
    require(
        all(not os.path.lexists(location) for location in destinations),
        'SDK worker already has a CI runtime',
    )
    allowed = {location.name for location in destinations}

    def members(member, target):
        require(relative(member.name).parts[0] in allowed, 'Unexpected runtime archive member')
        return tarfile.data_filter(member, target)

    with tempfile.TemporaryDirectory(prefix='ppu-runtime-', dir=root) as temporary:
        staged = Path(temporary)
        local = staged / ARCHIVE
        shutil.copyfile(no_links(path), local)
        verify_archive(local, metadata)
        try:
            with tarfile.open(local, 'r:gz') as bundle:
                bundle.extractall(staged, filter=members)
        except tarfile.TarError as error:
            raise ValidationError('Unsafe or invalid runtime archive') from error
        for destination in destinations:
            require(no_links(staged / destination.name).is_dir(), 'Runtime archive is incomplete')
        # These trees are byte-identical to the SDK in the CPU build and need not be transferred.
        vendor = Path(sysconfig.get_path('purelib'))
        site = no_links(staged / VENV_ROOT.name / 'lib/python3.12/site-packages')
        require(site.is_dir(), 'Runtime archive is missing the Python environment')
        for name in ('torch', 'triton'):
            shutil.copytree(vendor / name, site / name, symlinks=False)
        for destination in destinations:
            (staged / destination.name).rename(destination)
    for name in ('yq', 'envsubst'):
        tool = no_links(TOOL_BIN / name)
        shutil.copyfile(directory / 'functional-tools' / name, tool)
        tool.chmod(0o755)


def verify_image(args):
    metadata = verify_export(
        args.artifact, args.digest, args.control_sha, args.target_sha, subset=True
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
        'SDK execution record mismatch',
    )
    if args.check_only:
        if getattr(args, 'archive', None):
            verify_archive(args.archive, metadata)
        return
    require(args.source_root == SOURCE_ROOT, 'Invalid runtime source directory')
    require(args.archive is not None, 'Missing verified runtime archive')
    restore_runtime(Path(args.artifact), args.archive, metadata)


def runtime_probe(args):
    args.source_root = SOURCE_ROOT
    args.image_digest = args.base_image_digest
    args.archive = args.artifact / ARCHIVE
    args.check_only = False
    verify_image(args)
    execution_checks(SOURCE_ROOT)


def execution_checks(source):
    environment = client_env()
    environment.update(
        {
            'PATH': '/opt/ppu-ci-venv/bin:' + environment.get('PATH', ''),
            'PYTHONNOUSERSITE': '1',
            'UV_NO_SYNC': '1',
            'UV_PYTHON_DOWNLOADS': 'never',
            'LD_LIBRARY_PATH': os.environ.get('LD_LIBRARY_PATH', ''),
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
        ['/usr/bin/git', '--version'],
        ['/usr/bin/wget', '--version'],
        ['/usr/local/bin/envsubst', '--version'],
        ['/opt/ppu-ci-venv/bin/yq', '--version'],
        ['/usr/bin/make', '-q', '-C', str(source / 'megatron/core/datasets')],
        # Verify the original training driver's absolute path, not just the venv tool in PATH.
        ['/usr/local/bin/yq', '--version'],
        [
            '/opt/ppu-ci-venv/bin/python',
            '-I',
            '-c',
            'import os, pathlib, subprocess; tool = pathlib.Path("/usr/local/bin/yq"); '
            'assert tool.is_file() and not tool.is_symlink() and os.access(tool, os.X_OK); '
            'assert tool.read_bytes() == pathlib.Path("/opt/ppu-ci-venv/bin/yq").read_bytes(); '
            'assert subprocess.check_output([str(tool), ".TEST_TYPE"], '
            'input="TEST_TYPE: regular\\n", text=True, timeout=30).strip() == "regular"',
        ],
    ]
    for check in checks:
        result = subprocess.run(check, env=environment, capture_output=True, text=True)
        require(
            result.returncode == 0,
            'Offline SDK runtime check failed; device checks remain required',
        )


def main():
    require(sys.version_info >= (3, 11), 'Environment bootstrap requires Python 3.11 or newer')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('fingerprint', 'build-runtime', 'build-cache'):
        create = sub.add_parser(mode)
        create.add_argument('--control-root', type=Path, required=True)
        create.add_argument('--control-sha', required=True)
        create.add_argument('--source-root', type=Path, required=True)
        create.add_argument('--target-sha', required=True)
        create.add_argument('--image-digest')
        if mode in ('build-runtime', 'build-cache'):
            create.add_argument('--output', type=Path, required=mode != 'build-cache')
            create.add_argument('--cache-from', default='')
            create.add_argument('--cache-to', default='')
    for mode in ('verify-image', 'runtime-probe'):
        image = sub.add_parser(mode)
        image.add_argument('--artifact', type=Path, required=True)
        image.add_argument('--digest', required=True)
        image.add_argument('--control-sha', required=True)
        image.add_argument('--target-sha', required=True)
        image.add_argument('--base-image-digest', required=True)
        if mode == 'verify-image':
            image.add_argument('--image-digest', required=True)
            image.add_argument('--archive', type=Path)
            image.add_argument('--source-root', type=Path)
            image.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    {
        'build-runtime': build,
        'build-cache': build,
        'verify-image': verify_image,
        'runtime-probe': runtime_probe,
        'fingerprint': fingerprint,
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
