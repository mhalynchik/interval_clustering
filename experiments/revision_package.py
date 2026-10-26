"""Immutable experiment/artifact packages with explicit completion and hashes."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import uuid

SCHEMA = 1


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_manifest(root):
    return json.loads((Path(root) / 'manifest.json').read_text())


def file_hashes(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p != root / 'manifest.json'}


def validate_package(root, kind=None):
    root = Path(root)
    manifest = read_manifest(root)
    if manifest.get('schema') != SCHEMA or manifest.get('status') != 'complete':
        raise ValueError(f'Incomplete or unsupported package: {root}')
    if kind is not None and manifest.get('kind') != kind:
        raise ValueError(f'Expected {kind} package: {root}')
    actual = file_hashes(root)
    if actual != manifest.get('files_sha256'):
        raise ValueError(f'Package file inventory/hash mismatch: {root}')
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError(f'Package contains symlinks: {root}')
    return manifest


def verify_sources(manifest, source_root):
    source_root = Path(source_root).resolve()
    for field in ('code_sha256', 'data_sha256'):
        hashes = manifest.get(field)
        if not hashes:
            raise ValueError(f'Missing provenance: {field}')
        for name, expected in hashes.items():
            path = (source_root / name).resolve()
            if not path.is_relative_to(source_root) or not path.is_file() or sha256(path) != expected:
                raise ValueError(f'Source hash mismatch: {name}')


def seal(root):
    manifest = read_manifest(root)
    manifest.update(status='complete', files_sha256=file_hashes(root))
    write_json(Path(root) / 'manifest.json', manifest)
    return validate_package(root)


@contextmanager
def staging(out, metadata):
    """Leave failures in a named staging directory; never publish mixed results."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'Refusing existing nonempty output directory: {out}')
    stage = Path(tempfile.mkdtemp(prefix=f'.{out.name}.staging-', dir=out.parent))
    write_json(stage / 'manifest.json', dict(metadata, schema=SCHEMA,
               package_id=str(uuid.uuid4()), status='running'))
    try:
        yield stage
        validate_package(stage)
        # rename refuses a nonempty destination, including another completed run.
        stage.rename(out)
    except BaseException as error:
        manifest = read_manifest(stage)
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}',
                        files_sha256=file_hashes(stage))
        write_json(stage / 'manifest.json', manifest)
        error.add_note(f'Failed package retained at {stage}')
        raise
