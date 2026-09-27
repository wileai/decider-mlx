"""Load only the two audited, dependency-free public upstream modules.

Source: https://github.com/Mapika/decider
The file hashes also allow a source archive without Git metadata. Nothing is
added to sys.path and no upstream package initializer or server is executed.
"""
from hashlib import sha256
from pathlib import Path
from types import ModuleType, SimpleNamespace

UPSTREAM_COMMIT = "b44b4c9880a67291206499b86aac89004850134a"
UPSTREAM_URL = "https://github.com/Mapika/decider"
SOURCE_HASHES = {
    "prompt.py": "46f8158cd00c708408f6340340517c5a0c50fb0a339d0e46cd0a30f8801624cd",
    "systemone.py": "627fc365583c92328b7b21d481b307eb4d2bfd27411d38cd6710ab3adf629a9a",
}


def load_upstream(source):
    """Verify both files before executing either; source is a checkout root."""
    root = Path(source).expanduser().resolve() / "decider"
    verified = {}
    for filename, digest in SOURCE_HASHES.items():
        path = root / filename
        content = path.read_bytes()
        if sha256(content).hexdigest() != digest:
            raise ValueError(
                f"{filename} differs from pinned {UPSTREAM_URL}@{UPSTREAM_COMMIT}"
            )
        verified[filename] = content
    modules = {}
    for filename, content in verified.items():
        name = Path(filename).stem
        module = ModuleType(f"decider_mlx._upstream_{name}")
        module.__file__ = str(root / filename)
        exec(compile(content, module.__file__, "exec"), module.__dict__)
        modules[name] = module
    return SimpleNamespace(**modules)


def fetch_source(destination):
    """Explicitly fetch only pinned prompt/assembly code, without Torch deps.

    The destination must not exist. Download and hash-check both files before
    creating it. The upstream Apache-2.0 license applies to these files.
    """
    import urllib.request
    destination = Path(destination).expanduser()
    if destination.exists():
        raise FileExistsError('Source destination already exists; refusing to overwrite')
    verified = {}
    for filename, digest in SOURCE_HASHES.items():
        url = f'https://raw.githubusercontent.com/Mapika/decider/{UPSTREAM_COMMIT}/decider/{filename}'
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()
        if sha256(content).hexdigest() != digest:
            raise ValueError(f'{filename} download differs from pinned source')
        verified[filename] = content
    destination.mkdir(parents=True, exist_ok=False)
    root = destination / 'decider'
    root.mkdir()
    for filename, content in verified.items():
        (root / filename).write_bytes(content)
    load_upstream(destination)
    return destination
