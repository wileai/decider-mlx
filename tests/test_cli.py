import json
import os
from pathlib import Path
import subprocess
import sys


def run_cli(*args):
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'src'),
               PYTHONDONTWRITEBYTECODE='1')
    return subprocess.run([sys.executable, '-m', 'decider_mlx', *args],
                          text=True, capture_output=True, env=env)


def test_cli_example_is_synthetic_and_needs_no_model():
    result = run_cli('example')
    assert result.returncode == 0, result.stderr
    request = json.loads(result.stdout)
    assert set(request) == {'state', 'questions'}
    assert len(request['questions']) == 1
    assert request['questions']['route']['type'] == 'choice'


def test_cli_requires_explicit_paths():
    result = run_cli('decide', '--example')
    assert result.returncode == 2
    assert '--checkpoint' in result.stderr
    assert '--upstream-source' in result.stderr


def test_cli_help_has_download_helper():
    result = run_cli('--help')
    assert result.returncode == 0
    assert 'fetch-source' in result.stdout
