"""Only public, synthetic fixtures; no checkpoint needed."""
from pathlib import Path
import importlib.util
import pytest


def test_upstream_loader_is_available():
    assert importlib.util.find_spec('decider_mlx') is not None, 'installable package missing'
    from decider_mlx.upstream import load_upstream
    with pytest.raises(FileNotFoundError):
        load_upstream(Path('missing-upstream-source'))


def test_unpinned_source_is_rejected_before_execution(tmp_path):
    from decider_mlx.upstream import load_upstream
    source = tmp_path / 'decider'
    source.mkdir()
    (source / 'prompt.py').write_text('raise RuntimeError("must not execute")')
    (source / 'systemone.py').write_text('raise RuntimeError("must not execute")')
    with pytest.raises(ValueError, match='pinned'):
        load_upstream(tmp_path)


def test_fetch_source_rejects_tampering_before_writing(tmp_path, monkeypatch):
    import io
    import urllib.request
    from decider_mlx.upstream import fetch_source
    monkeypatch.setattr(urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(b'corrupt'))
    with pytest.raises(ValueError, match='pinned'):
        fetch_source(tmp_path / 'source')
    assert not (tmp_path / 'source').exists()
